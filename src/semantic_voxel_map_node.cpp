#include "local3d_semantic_voxel_map/semantic_voxel_map.hpp"
#include "local3d_semantic_voxel_map/global_semantic_admission.hpp"
#include "local3d_semantic_voxel_map/obstacle_revocation.hpp"
#include "local3d_semantic_voxel_map/ssmi_semantic_encoding.hpp"
#include "local3d_semantic_voxel_map/terrain_boundary_filter.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_srvs/Empty.h>
#include <std_srvs/Trigger.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/Marker.h>
#include <xmlrpcpp/XmlRpcValue.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace local3d_semantic_voxel_map
{

namespace
{

constexpr int kBoundaryDebugPanelPixels = 260;
constexpr int kBoundaryDebugPanelHeaderPixels = 26;
constexpr int kBoundaryDebugPanelGap = 6;
constexpr int kBoundaryDebugPanelCellHeight =
  kBoundaryDebugPanelPixels + kBoundaryDebugPanelHeaderPixels;
constexpr int kBoundaryDebugFooterPixels = 100;

std::array<std::uint8_t, 3> boundaryDebugReasonRgb(
  const TerrainBoundaryDecisionReason reason)
{
  switch (reason)
  {
    case TerrainBoundaryDecisionReason::OutsideClosedTerrainSupport:
      return {{110u, 110u, 110u}};
    case TerrainBoundaryDecisionReason::NoTerrainReference:
      return {{0u, 128u, 255u}};
    case TerrainBoundaryDecisionReason::HeightDifferenceTooHigh:
      return {{255u, 165u, 0u}};
    case TerrainBoundaryDecisionReason::Recovered:
      return {{0u, 255u, 0u}};
    case TerrainBoundaryDecisionReason::Count:
      break;
  }
  return {{255u, 255u, 255u}};
}

struct FieldView
{
  bool valid = false;
  std::uint32_t offset = 0;
  std::uint8_t datatype = 0;
};

FieldView findField(const sensor_msgs::PointCloud2& cloud, const std::string& name)
{
  for (const auto& field : cloud.fields)
  {
    if (field.name == name)
    {
      return FieldView{true, field.offset, field.datatype};
    }
  }
  return FieldView();
}

template<typename T>
T readRaw(const std::uint8_t* data)
{
  T output;
  std::memcpy(&output, data, sizeof(T));
  return output;
}

bool readNumber(const std::uint8_t* point, const FieldView& field, double& output)
{
  if (!field.valid)
  {
    return false;
  }
  const std::uint8_t* data = point + field.offset;
  switch (field.datatype)
  {
    case sensor_msgs::PointField::INT8: output = readRaw<std::int8_t>(data); return true;
    case sensor_msgs::PointField::UINT8: output = readRaw<std::uint8_t>(data); return true;
    case sensor_msgs::PointField::INT16: output = readRaw<std::int16_t>(data); return true;
    case sensor_msgs::PointField::UINT16: output = readRaw<std::uint16_t>(data); return true;
    case sensor_msgs::PointField::INT32: output = readRaw<std::int32_t>(data); return true;
    case sensor_msgs::PointField::UINT32: output = readRaw<std::uint32_t>(data); return true;
    case sensor_msgs::PointField::FLOAT32: output = readRaw<float>(data); return true;
    case sensor_msgs::PointField::FLOAT64: output = readRaw<double>(data); return true;
    default: return false;
  }
}

bool readSemanticLabel(const std::uint8_t* point, const FieldView& field,
                       const bool packed_float, std::uint32_t& output)
{
  if (!field.valid)
  {
    return false;
  }
  const std::uint8_t* data = point + field.offset;
  if (packed_float && field.datatype == sensor_msgs::PointField::FLOAT32)
  {
    output = readRaw<std::uint32_t>(data) & 0x00ffffffu;
    return true;
  }

  double value = 0.0;
  if (!readNumber(point, field, value) || !std::isfinite(value) || value < 0.0)
  {
    return false;
  }
  output = static_cast<std::uint32_t>(value);
  return true;
}

float clampUnit(const float value)
{
  return std::max(0.0f, std::min(1.0f, value));
}

float packedRgbFloat(const std::uint8_t red, const std::uint8_t green,
                     const std::uint8_t blue)
{
  const std::uint32_t packed = (static_cast<std::uint32_t>(red) << 16u) |
                               (static_cast<std::uint32_t>(green) << 8u) |
                               static_cast<std::uint32_t>(blue);
  float output;
  std::memcpy(&output, &packed, sizeof(float));
  return output;
}

std_msgs::ColorRGBA costColor(const float raw_cost, const float alpha)
{
  const float cost = clampUnit(raw_cost);
  std_msgs::ColorRGBA color;
  color.a = alpha;
  if (cost < 0.5f)
  {
    color.r = 2.0f * cost;
    color.g = 1.0f;
  }
  else
  {
    color.r = 1.0f;
    color.g = 2.0f * (1.0f - cost);
  }
  color.b = 0.0f;
  return color;
}

std::uint32_t parseLabel(const XmlRpc::XmlRpcValue& value)
{
  if (value.getType() == XmlRpc::XmlRpcValue::TypeInt)
  {
    return static_cast<std::uint32_t>(static_cast<int>(value));
  }
  if (value.getType() == XmlRpc::XmlRpcValue::TypeString)
  {
    const std::string text = static_cast<std::string>(value);
    return static_cast<std::uint32_t>(std::stoul(text, nullptr, 0));
  }
  throw std::runtime_error("semantic class label must be an integer or a string");
}

double xmlNumber(const XmlRpc::XmlRpcValue& value)
{
  if (value.getType() == XmlRpc::XmlRpcValue::TypeInt)
  {
    return static_cast<int>(value);
  }
  if (value.getType() == XmlRpc::XmlRpcValue::TypeDouble)
  {
    return static_cast<double>(value);
  }
  throw std::runtime_error("expected a numeric parameter");
}

std::vector<std::uint32_t> loadLabelList(
  const ros::NodeHandle& node, const std::string& name,
  const std::vector<std::uint32_t>& defaults)
{
  XmlRpc::XmlRpcValue values;
  if (!node.getParam(name, values))
  {
    return defaults;
  }
  if (values.getType() != XmlRpc::XmlRpcValue::TypeArray)
  {
    throw std::runtime_error("~" + name + " must be a YAML list");
  }
  std::vector<std::uint32_t> labels;
  labels.reserve(values.size());
  for (int index = 0; index < values.size(); ++index)
  {
    if (values[index].getType() == XmlRpc::XmlRpcValue::TypeInt &&
        static_cast<int>(values[index]) < 0)
    {
      throw std::runtime_error("~" + name + " labels must be non-negative");
    }
    labels.push_back(parseLabel(values[index]));
  }
  return labels;
}

bool xmlBool(const XmlRpc::XmlRpcValue& value, const std::string& context)
{
  if (value.getType() != XmlRpc::XmlRpcValue::TypeBoolean)
  {
    throw std::runtime_error(context + " must be a boolean");
  }
  return static_cast<bool>(value);
}

struct SharedNavigationSemanticClass
{
  std::uint32_t label = 0u;
  std::string name;
  std::string role;
  std::array<std::uint8_t, 3> rgb{{127u, 127u, 127u}};
  float semantic_cost = 0.5f;
};

struct SharedNavigationSemanticProfile
{
  bool enabled = false;
  std::string label_field = "semantic_lable";
  std::string traversability_field = "traversability";
  std::vector<SharedNavigationSemanticClass> classes;
};

SharedNavigationSemanticProfile loadSharedNavigationSemanticProfile(
  const ros::NodeHandle& node)
{
  SharedNavigationSemanticProfile profile;
  XmlRpc::XmlRpcValue root;
  if (!node.getParam("/semantic_schema", root))
  {
    return profile;
  }
  if (root.getType() != XmlRpc::XmlRpcValue::TypeStruct)
  {
    throw std::runtime_error("/semantic_schema must be a YAML mapping");
  }
  if (!root.hasMember("navigation"))
  {
    return profile;
  }
  const XmlRpc::XmlRpcValue& navigation = root["navigation"];
  if (navigation.getType() != XmlRpc::XmlRpcValue::TypeStruct)
  {
    throw std::runtime_error("/semantic_schema/navigation must be a mapping");
  }
  if (!navigation.hasMember("derive_runtime_roles") ||
      !xmlBool(navigation["derive_runtime_roles"],
               "/semantic_schema/navigation/derive_runtime_roles"))
  {
    return profile;
  }

  if (root.hasMember("input"))
  {
    const XmlRpc::XmlRpcValue& input = root["input"];
    if (input.getType() != XmlRpc::XmlRpcValue::TypeStruct)
    {
      throw std::runtime_error("/semantic_schema/input must be a mapping");
    }
    if (input.hasMember("label_field"))
    {
      if (input["label_field"].getType() != XmlRpc::XmlRpcValue::TypeString)
      {
        throw std::runtime_error(
          "/semantic_schema/input/label_field must be a string");
      }
      profile.label_field = static_cast<std::string>(input["label_field"]);
    }
    if (input.hasMember("traversability_field"))
    {
      if (input["traversability_field"].getType() !=
          XmlRpc::XmlRpcValue::TypeString)
      {
        throw std::runtime_error(
          "/semantic_schema/input/traversability_field must be a string");
      }
      profile.traversability_field =
        static_cast<std::string>(input["traversability_field"]);
    }
  }
  if (profile.label_field.empty() || profile.traversability_field.empty())
  {
    throw std::runtime_error(
      "/semantic_schema input field names must not be empty");
  }
  if (!root.hasMember("classes") ||
      root["classes"].getType() != XmlRpc::XmlRpcValue::TypeArray ||
      root["classes"].size() == 0)
  {
    throw std::runtime_error(
      "/semantic_schema/classes must be a non-empty YAML list");
  }

  const XmlRpc::XmlRpcValue& classes = root["classes"];
  std::unordered_set<std::uint32_t> labels;
  profile.classes.reserve(classes.size());
  for (int index = 0; index < classes.size(); ++index)
  {
    const XmlRpc::XmlRpcValue& item = classes[index];
    const std::string context = "/semantic_schema/classes[" +
      std::to_string(index) + "]";
    if (item.getType() != XmlRpc::XmlRpcValue::TypeStruct ||
        !item.hasMember("label") || !item.hasMember("name") ||
        !item.hasMember("rgb") || !item.hasMember("role") ||
        !item.hasMember("semantic_cost"))
    {
      throw std::runtime_error(
        context + " requires label, name, rgb, role, and semantic_cost");
    }

    SharedNavigationSemanticClass semantic_class;
    semantic_class.label = parseLabel(item["label"]);
    if (!labels.insert(semantic_class.label).second)
    {
      throw std::runtime_error(
        context + " duplicates semantic label " +
        std::to_string(semantic_class.label));
    }
    if (item["name"].getType() != XmlRpc::XmlRpcValue::TypeString ||
        item["role"].getType() != XmlRpc::XmlRpcValue::TypeString)
    {
      throw std::runtime_error(context + " name and role must be strings");
    }
    semantic_class.name = static_cast<std::string>(item["name"]);
    semantic_class.role = static_cast<std::string>(item["role"]);
    if (semantic_class.name.empty() ||
        (semantic_class.role != "terrain" &&
         semantic_class.role != "static_obstacle" &&
         semantic_class.role != "dynamic_obstacle" &&
         semantic_class.role != "ignore"))
    {
      throw std::runtime_error(
        context + " has an empty name or unsupported role");
    }

    const XmlRpc::XmlRpcValue& rgb = item["rgb"];
    if (rgb.getType() != XmlRpc::XmlRpcValue::TypeArray || rgb.size() != 3)
    {
      throw std::runtime_error(context + "/rgb must be [r, g, b]");
    }
    for (int channel = 0; channel < 3; ++channel)
    {
      const double value = xmlNumber(rgb[channel]);
      if (value < 0.0 || value > 255.0 || std::floor(value) != value)
      {
        throw std::runtime_error(
          context + "/rgb channels must be integer values in [0, 255]");
      }
      semantic_class.rgb[static_cast<std::size_t>(channel)] =
        static_cast<std::uint8_t>(value);
    }
    const double semantic_cost = xmlNumber(item["semantic_cost"]);
    if (semantic_cost < 0.0 || semantic_cost > 1.0)
    {
      throw std::runtime_error(context + "/semantic_cost must be in [0, 1]");
    }
    semantic_class.semantic_cost = static_cast<float>(semantic_cost);
    profile.classes.push_back(semantic_class);
  }
  profile.enabled = true;
  return profile;
}

}  // namespace

class SemanticVoxelMapNode
{
public:
  SemanticVoxelMapNode()
    : private_nh_("~"), tf_listener_(tf_buffer_)
  {
    SemanticVoxelMapConfig map_config;
    private_nh_.param("voxel_size", map_config.voxel_size, 0.10);
    private_nh_.param("voxel_size_xy", map_config.voxel_size_xy,
                      map_config.voxel_size);
    private_nh_.param("voxel_size_z", map_config.voxel_size_z,
                      map_config.voxel_size);
    private_nh_.param("decay_seconds", map_config.decay_seconds, 0.5);
    int max_voxels = 500000;
    private_nh_.param("max_voxels", max_voxels, 500000);
    map_config.max_voxels = max_voxels <= 0 ? 0u : static_cast<std::size_t>(max_voxels);
    private_nh_.param("unknown_cost", map_config.unknown_cost, 0.5f);
    private_nh_.param("semantic_cost_weight", map_config.semantic_cost_weight, 0.8f);
    private_nh_.param("semantic_risk_alpha", map_config.semantic_risk_alpha, 1.0f);
    private_nh_.param("cost_rise_alpha", map_config.cost_rise_alpha, 0.65f);
    private_nh_.param("cost_fall_alpha", map_config.cost_fall_alpha, 0.15f);
    std::string traversability_fusion_method;
    private_nh_.param<std::string>("traversability_fusion_method",
                                   traversability_fusion_method,
                                   "weighted_average");
    if (traversability_fusion_method == "maximum")
    {
      map_config.traversability_fusion_method = TraversabilityFusionMethod::Maximum;
    }
    else if (traversability_fusion_method == "confidence_weighted_raise")
    {
      map_config.traversability_fusion_method =
        TraversabilityFusionMethod::ConfidenceWeightedRaise;
    }
    else if (traversability_fusion_method != "weighted_average")
    {
      throw std::runtime_error(
        "~traversability_fusion_method must be 'weighted_average', 'maximum', "
        "or 'confidence_weighted_raise'");
    }
    private_nh_.param("semantic_positive_update",
                      map_config.semantic_fusion.positive_update, 1.0f);
    private_nh_.param("semantic_negative_update",
                      map_config.semantic_fusion.negative_update, -0.1f);
    private_nh_.param("semantic_min_log_evidence",
                      map_config.semantic_fusion.min_log_evidence, -4.59512f);
    private_nh_.param("semantic_max_log_evidence",
                      map_config.semantic_fusion.max_log_evidence, 4.59512f);
    private_nh_.param("semantic_new_class_prior",
                      map_config.semantic_fusion.new_class_prior, 0.8f);

    shared_semantic_profile_ = loadSharedNavigationSemanticProfile(nh_);
    map_.reset(new SemanticVoxelMap(map_config));
    map_->setSemanticClasses(loadSemanticClasses());

    private_nh_.param<std::string>("input_topic", input_topic_,
                                   "/semantic_pcl/semantic_pcl");
    private_nh_.param<std::string>("global_frame", global_frame_, "map");
    private_nh_.param("use_initial_pose_reference", use_initial_pose_reference_, false);
    private_nh_.param("initial_pose_reference_yaw_only",
                      initial_pose_reference_yaw_only_, false);
    private_nh_.param<std::string>("reference_frame", reference_frame_, "map_start");
    private_nh_.param<std::string>("semantic_field", semantic_field_,
                                   "semantic_color");
    private_nh_.param<std::string>("confidence_field", confidence_field_,
                                   "semantic_confidence");
    private_nh_.param<std::string>("cost_field", cost_field_, "traversability");
    if (shared_semantic_profile_.enabled)
    {
      semantic_field_ = shared_semantic_profile_.label_field;
      cost_field_ = shared_semantic_profile_.traversability_field;
    }
    private_nh_.param<std::string>("map_file", map_file_,
                                   "/tmp/local3d_semantic_voxel_map.csv");
    private_nh_.param<std::string>("local_cost_frame", local_cost_frame_,
                                   "wuba_base");
    private_nh_.param("publish_local_cost_cloud", publish_local_cost_cloud_, false);
    private_nh_.param<std::string>("global_admission_output_frame",
                                   admission_output_frame_, "");
    private_nh_.param("global_admission_exclude_dynamic",
                      admission_exclude_dynamic_, true);
    private_nh_.param("publish_ssmi_admitted_cloud",
                      publish_ssmi_admitted_cloud_, true);
    private_nh_.param<std::string>("ssmi_admitted_topic",
                                   ssmi_admitted_topic_,
                                   "/semantic_pcl/global_admitted");
    private_nh_.param("ssmi_obstacle_traversability_threshold",
                      ssmi_obstacle_traversability_threshold_, 0.75f);
    private_nh_.param("enable_ssmi_obstacle_revocation",
                      enable_ssmi_obstacle_revocation_, true);
    ObstacleRevocationConfig revocation_config;
    private_nh_.param("ssmi_revocation_voxel_size",
                      revocation_config.voxel_size, 0.10);
    int revocation_minimum_frames = 5;
    private_nh_.param("ssmi_revocation_minimum_free_frames",
                      revocation_minimum_frames, 5);
    revocation_config.minimum_free_frames = revocation_minimum_frames <= 0 ?
      0u : static_cast<std::size_t>(revocation_minimum_frames);
    private_nh_.param("ssmi_revocation_minimum_free_duration",
                      revocation_config.minimum_free_duration, 0.5);
    revocation_config.minimum_free_evidence =
      static_cast<double>(revocation_config.minimum_free_frames);
    private_nh_.param("ssmi_revocation_minimum_free_evidence",
                      revocation_config.minimum_free_evidence,
                      revocation_config.minimum_free_evidence);
    private_nh_.param("ssmi_revocation_free_evidence_decay_per_second",
                      revocation_config.free_evidence_decay_per_second, 0.5);
    private_nh_.param("ssmi_revocation_free_max_traversability",
                      revocation_config.free_max_traversability, 0.45f);
    revocation_config.obstacle_min_traversability =
      ssmi_obstacle_traversability_threshold_;
    private_nh_.param("ssmi_revocation_minimum_semantic_confidence",
                      revocation_config.minimum_semantic_confidence, 0.60f);
    private_nh_.param("ssmi_revocation_ray_endpoint_margin",
                      revocation_config.ray_endpoint_margin, 0.20);
    revocation_config.terrain_labels = loadLabelList(
      private_nh_, "ssmi_revocation_terrain_labels",
      revocation_config.terrain_labels);
    revocation_config.obstacle_labels = loadLabelList(
      private_nh_, "ssmi_revocation_obstacle_labels",
      revocation_config.obstacle_labels);
    revocation_config.ambiguous_obstacle_labels = loadLabelList(
      private_nh_, "ssmi_revocation_ambiguous_obstacle_labels",
      revocation_config.ambiguous_obstacle_labels);
    private_nh_.param(
      "ssmi_revocation_ambiguous_obstacle_reset_min_traversability",
      revocation_config.ambiguous_obstacle_reset_min_traversability, 0.65f);
    revocation_config.dynamic_labels = loadLabelList(
      private_nh_, "ssmi_revocation_dynamic_labels",
      revocation_config.dynamic_labels);
    if (shared_semantic_profile_.enabled)
    {
      revocation_config.terrain_labels.clear();
      revocation_config.obstacle_labels.clear();
      revocation_config.dynamic_labels.clear();
      for (const SharedNavigationSemanticClass& semantic_class :
           shared_semantic_profile_.classes)
      {
        if (semantic_class.role == "terrain")
        {
          revocation_config.terrain_labels.push_back(semantic_class.label);
        }
        else if (semantic_class.role == "static_obstacle")
        {
          revocation_config.obstacle_labels.push_back(semantic_class.label);
        }
        else if (semantic_class.role == "dynamic_obstacle")
        {
          revocation_config.dynamic_labels.push_back(semantic_class.label);
        }
      }
    }
    private_nh_.param("ssmi_revocation_ray_evidence_enabled",
                      ssmi_revocation_ray_evidence_enabled_, false);
    private_nh_.param("ssmi_revocation_ray_point_stride",
                      ssmi_revocation_ray_point_stride_, 2);
    if (ssmi_revocation_ray_point_stride_ <= 0)
    {
      throw std::runtime_error(
        "~ssmi_revocation_ray_point_stride must be positive");
    }
    if (enable_ssmi_obstacle_revocation_)
    {
      obstacle_revocation_tracker_.reset(
        new ObstacleRevocationTracker(revocation_config));
    }
    TerrainBoundaryFilterConfig boundary_filter_config;
    private_nh_.param("terrain_boundary_filter_enabled",
                      boundary_filter_config.enabled, false);
    private_nh_.param("terrain_boundary_debug_opencv_enabled",
                      terrain_boundary_debug_opencv_enabled_, false);
    private_nh_.param("terrain_boundary_debug_rviz_enabled",
                      terrain_boundary_debug_rviz_enabled_, false);
    boundary_filter_config.debug_enabled =
      terrain_boundary_debug_opencv_enabled_ ||
      terrain_boundary_debug_rviz_enabled_;
    private_nh_.param("terrain_boundary_debug_statistics_interval_frames",
                      terrain_boundary_debug_statistics_interval_frames_, 10);
    private_nh_.param("terrain_boundary_debug_max_samples_per_reason",
                      terrain_boundary_debug_max_samples_per_reason_, 1);
    private_nh_.param<std::string>("terrain_boundary_debug_clicked_point_topic",
                      terrain_boundary_debug_clicked_point_topic_,
                      "/clicked_point");
    private_nh_.param("terrain_boundary_debug_clicked_point_radius",
                      terrain_boundary_debug_clicked_point_radius_, 0.20);
    private_nh_.param<std::string>("terrain_boundary_debug_csv_path",
                      terrain_boundary_debug_csv_path_, "");
    if (terrain_boundary_debug_opencv_enabled_)
    {
      const char* display = std::getenv("DISPLAY");
      const char* wayland_display = std::getenv("WAYLAND_DISPLAY");
      const bool have_x_display = display != nullptr && display[0] != '\0';
      const bool have_wayland_display =
        wayland_display != nullptr && wayland_display[0] != '\0';
      if (!have_x_display && !have_wayland_display)
      {
        ROS_ERROR("~terrain_boundary_debug_opencv_enabled is true, "
                  "but neither DISPLAY nor WAYLAND_DISPLAY is available; "
                  "disabling the OpenCV window");
        terrain_boundary_debug_opencv_enabled_ = false;
        boundary_filter_config.debug_enabled =
          terrain_boundary_debug_rviz_enabled_;
      }
    }
    if (terrain_boundary_debug_statistics_interval_frames_ <= 0)
    {
      throw std::runtime_error(
        "~terrain_boundary_debug_statistics_interval_frames must be positive");
    }
    if (terrain_boundary_debug_max_samples_per_reason_ < 0)
    {
      throw std::runtime_error(
        "~terrain_boundary_debug_max_samples_per_reason must be non-negative");
    }
    if (!std::isfinite(terrain_boundary_debug_clicked_point_radius_) ||
        terrain_boundary_debug_clicked_point_radius_ <= 0.0)
    {
      throw std::runtime_error(
        "~terrain_boundary_debug_clicked_point_radius must be positive");
    }
    boundary_filter_config.terrain_labels = loadLabelList(
      private_nh_, "terrain_boundary_terrain_labels",
      boundary_filter_config.terrain_labels);
    boundary_filter_config.recoverable_labels = loadLabelList(
      private_nh_, "terrain_boundary_recoverable_labels",
      boundary_filter_config.recoverable_labels);
    boundary_filter_config.excluded_labels = loadLabelList(
      private_nh_, "terrain_boundary_excluded_labels",
      boundary_filter_config.excluded_labels);
    if (shared_semantic_profile_.enabled)
    {
      boundary_filter_config.terrain_labels.clear();
      boundary_filter_config.recoverable_labels.clear();
      boundary_filter_config.excluded_labels.clear();
      for (const SharedNavigationSemanticClass& semantic_class :
           shared_semantic_profile_.classes)
      {
        if (semantic_class.role == "terrain")
        {
          boundary_filter_config.terrain_labels.push_back(semantic_class.label);
        }
        else if (semantic_class.role == "static_obstacle")
        {
          boundary_filter_config.recoverable_labels.push_back(
            semantic_class.label);
        }
        else
        {
          boundary_filter_config.excluded_labels.push_back(
            semantic_class.label);
        }
      }
      if (boundary_filter_config.terrain_labels.empty())
      {
        throw std::runtime_error(
          "/semantic_schema must define at least one terrain role");
      }
      if (boundary_filter_config.recoverable_labels.empty())
      {
        throw std::runtime_error(
          "/semantic_schema must define at least one static_obstacle role");
      }
    }
    private_nh_.param("terrain_boundary_closing_radius",
                      boundary_filter_config.closing_radius, 0.15);
    private_nh_.param("terrain_boundary_max_height_difference",
                      boundary_filter_config.maximum_height_difference, 0.10);
    terrain_boundary_filter_.reset(
      new TerrainBoundaryFilter(boundary_filter_config));
    terrain_boundary_filter_config_ = boundary_filter_config;
    private_nh_.param("terrain_height_cost_enabled",
                      terrain_height_cost_config_.enabled, false);
    private_nh_.param("terrain_height_difference_threshold",
                      terrain_height_cost_config_.height_difference_threshold, 0.15);
    private_nh_.param("terrain_height_neighborhood_radius",
                      terrain_height_cost_config_.neighborhood_radius, 0.20);
    private_nh_.param("terrain_height_comparison_epsilon",
                      terrain_height_cost_config_.comparison_epsilon, 1e-6);
    private_nh_.param("terrain_height_obstacle_cost",
                      terrain_height_cost_config_.obstacle_cost, 1.0f);
    if (shared_semantic_profile_.enabled)
    {
      terrain_height_cost_config_.terrain_labels.clear();
      for (const SharedNavigationSemanticClass& semantic_class :
           shared_semantic_profile_.classes)
      {
        if (semantic_class.role == "terrain")
        {
          terrain_height_cost_config_.terrain_labels.push_back(
            semantic_class.label);
        }
      }
    }
    private_nh_.param("default_semantic_confidence", default_confidence_, 1.0f);
    private_nh_.param("max_range", max_range_, 15.0);
    private_nh_.param("min_z", min_z_, -std::numeric_limits<double>::max());
    private_nh_.param("max_z", max_z_, std::numeric_limits<double>::max());
    private_nh_.param("local_radius", local_radius_, -1.0);
    private_nh_.param("local_box_enabled", local_box_enabled_, false);
    private_nh_.param("local_box_min_x", local_box_min_x_, -12.0);
    private_nh_.param("local_box_max_x", local_box_max_x_, 12.0);
    private_nh_.param("local_box_min_y", local_box_min_y_, -12.0);
    private_nh_.param("local_box_max_y", local_box_max_y_, 12.0);
    private_nh_.param("local_box_min_z", local_box_min_z_, -2.0);
    private_nh_.param("local_box_max_z", local_box_max_z_, 4.0);
    private_nh_.param("robot_body_exclusion_enabled",
                      robot_body_exclusion_enabled_, false);
    private_nh_.param("robot_body_exclusion_min_x",
                      robot_body_exclusion_min_x_, -0.5);
    private_nh_.param("robot_body_exclusion_max_x",
                      robot_body_exclusion_max_x_, 0.3);
    private_nh_.param("robot_body_exclusion_min_y",
                      robot_body_exclusion_min_y_, -0.3);
    private_nh_.param("robot_body_exclusion_max_y",
                      robot_body_exclusion_max_y_, 0.3);
    private_nh_.param("transform_timeout", transform_timeout_, 0.2);
    private_nh_.param("publish_rate", publish_rate_, 2.0);
    private_nh_.param("marker_alpha", marker_alpha_, 0.85f);
    private_nh_.param("input_image_width", input_image_width_, 640);
    private_nh_.param("input_image_height", input_image_height_, 480);
    private_nh_.param("pixel_stride_x", pixel_stride_x_, 4);
    private_nh_.param("pixel_stride_y", pixel_stride_y_, 4);
    private_nh_.param("pixel_offset_x", pixel_offset_x_, 2);
    private_nh_.param("pixel_offset_y", pixel_offset_y_, 2);
    private_nh_.param<std::string>("input_layout", input_layout_, "image");
    private_nh_.param("point_stride", point_stride_, 1);
    private_nh_.param("timing_report_frames", timing_report_frames_, 50);
    if (input_image_width_ <= 0 || input_image_height_ <= 0 ||
        pixel_stride_x_ <= 0 || pixel_stride_y_ <= 0)
    {
      throw std::runtime_error(
        "input image dimensions and pixel sampling strides must be positive");
    }
    semantic_label_remap_ = loadSemanticLabelRemap();
    if (input_layout_ != "image" && input_layout_ != "point_cloud" &&
        input_layout_ != "auto")
    {
      throw std::runtime_error(
        "~input_layout must be 'image', 'point_cloud', or 'auto'");
    }
    if (point_stride_ <= 0)
    {
      throw std::runtime_error("~point_stride must be positive");
    }
    if (local_box_enabled_ &&
        (local_box_min_x_ >= local_box_max_x_ ||
         local_box_min_y_ >= local_box_max_y_ ||
         local_box_min_z_ >= local_box_max_z_))
    {
      throw std::runtime_error(
        "local box minimum bounds must be smaller than maximum bounds");
    }
    if (local_box_enabled_ && local_radius_ > 0.0)
    {
      ROS_WARN("Both local_box_enabled and local_radius are set; the local box "
               "takes precedence");
    }
    if (robot_body_exclusion_enabled_ &&
        (robot_body_exclusion_min_x_ >= robot_body_exclusion_max_x_ ||
         robot_body_exclusion_min_y_ >= robot_body_exclusion_max_y_))
    {
      throw std::runtime_error(
        "robot body exclusion minimum bounds must be smaller than maximum bounds");
    }
    if (terrain_height_cost_config_.height_difference_threshold < 0.0 ||
        terrain_height_cost_config_.neighborhood_radius < 0.0 ||
        terrain_height_cost_config_.comparison_epsilon < 0.0 ||
        terrain_height_cost_config_.obstacle_cost < 0.0f ||
        terrain_height_cost_config_.obstacle_cost > 1.0f)
    {
      throw std::runtime_error(
        "terrain height thresholds/radius must be non-negative and obstacle cost "
        "must be in [0, 1]");
    }
    if (ssmi_obstacle_traversability_threshold_ < 0.0f ||
        ssmi_obstacle_traversability_threshold_ > 1.0f)
    {
      throw std::runtime_error(
        "~ssmi_obstacle_traversability_threshold must be in [0, 1]");
    }
    if (publish_ssmi_admitted_cloud_ && ssmi_admitted_topic_.empty())
    {
      throw std::runtime_error(
        "~ssmi_admitted_topic must be non-empty when direct SSMI output is enabled");
    }
    if (use_initial_pose_reference_ &&
        (reference_frame_.empty() || reference_frame_ == global_frame_))
    {
      throw std::runtime_error(
        "~reference_frame must be non-empty and different from ~global_frame");
    }
    pixel_offset_x_ = std::max(0, std::min(pixel_offset_x_, pixel_stride_x_ - 1));
    pixel_offset_y_ = std::max(0, std::min(pixel_offset_y_, pixel_stride_y_ - 1));

    semantic_marker_pub_ = private_nh_.advertise<visualization_msgs::Marker>(
      "semantic_voxels", 1, true);
    cost_marker_pub_ = private_nh_.advertise<visualization_msgs::Marker>(
      "traversability_voxels", 1, true);
    cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("voxel_cloud", 1, true);
    cost_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>(
      "traversability_cost_cloud", 1, true);
    local_cost_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>(
      "traversability_cost_cloud_wuba", 1, true);
    global_admission_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>(
      "global_semantic_admission_grid", 1, true);
    if (publish_ssmi_admitted_cloud_)
    {
      ssmi_admitted_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
        ssmi_admitted_topic_, 1, true);
    }
    candidates_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>(
      "candidates", 1, true);
    confirmed_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>(
      "confirmed", 1, true);
    rejected_dynamic_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>(
      "rejected_dynamic", 1, true);
    rejected_unknown_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>(
      "rejected_unknown", 1, true);
    rejected_rear_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>(
      "rejected_rear", 1, true);
    revocation_candidates_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>(
      "revocation_candidates", 1, true);
    revoked_free_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>(
      "revoked_free", 1, true);
    revoked_reclassified_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>(
      "revoked_reclassified", 1, true);
    if (terrain_boundary_debug_rviz_enabled_)
    {
      terrain_boundary_debug_decision_cloud_pub_ =
        private_nh_.advertise<sensor_msgs::PointCloud2>(
          "terrain_boundary_debug/decision_cloud", 1, true);
      terrain_boundary_debug_image_pub_ = private_nh_.advertise<sensor_msgs::Image>(
        "terrain_boundary_debug/stages_image", 1, true);
    }
    cloud_sub_ = nh_.subscribe(input_topic_, 1,
                               &SemanticVoxelMapNode::cloudCallback, this);
    if (terrain_boundary_debug_rviz_enabled_)
    {
      terrain_boundary_debug_clicked_point_sub_ = nh_.subscribe(
        terrain_boundary_debug_clicked_point_topic_, 1,
        &SemanticVoxelMapNode::terrainBoundaryClickedPointCallback, this);
    }
    reset_service_ = private_nh_.advertiseService(
      "reset", &SemanticVoxelMapNode::resetCallback, this);
    save_service_ = private_nh_.advertiseService(
      "save_map", &SemanticVoxelMapNode::saveCallback, this);
    load_service_ = private_nh_.advertiseService(
      "load_map", &SemanticVoxelMapNode::loadCallback, this);

    if ((terrain_boundary_debug_opencv_enabled_ ||
         terrain_boundary_debug_rviz_enabled_) &&
        !terrain_boundary_debug_csv_path_.empty())
    {
      terrain_boundary_debug_csv_.open(
        terrain_boundary_debug_csv_path_, std::ios::out | std::ios::app);
      if (!terrain_boundary_debug_csv_)
      {
        ROS_ERROR("Cannot open terrain-boundary debug CSV '%s'; CSV output disabled",
                  terrain_boundary_debug_csv_path_.c_str());
      }
      else if (terrain_boundary_debug_csv_.tellp() == std::streampos(0))
      {
        terrain_boundary_debug_csv_
          << "stamp,key_x,key_y,key_z,x,y,z,first_failure,failure_mask,"
             "original_label,closed,proposed,reference_found,reference_key_x,"
             "reference_key_y,reference_key_z,reference_x,reference_y,"
             "reference_z,reference_label,reference_distance_xy,"
             "height_difference,replacement_label\n";
      }
    }

    const double timer_period = publish_rate_ > 0.0 ? 1.0 / publish_rate_ : 0.5;
    publish_timer_ = nh_.createTimer(ros::Duration(timer_period),
                                     &SemanticVoxelMapNode::timerCallback, this);
    ROS_INFO("Semantic voxel map: input=%s tracking_frame=%s map_frame=%s "
             "voxel_size_xy=%.3f voxel_size_z=%.3f, "
             "layout=%s pixel sampling=%dx%d stride=(%d,%d) offset=(%d,%d), "
             "point_stride=%d",
             input_topic_.c_str(), global_frame_.c_str(), mapFrame().c_str(),
             map_->voxelSizeXY(), map_->voxelSizeZ(),
             input_layout_.c_str(),
             input_image_width_, input_image_height_, pixel_stride_x_,
             pixel_stride_y_, pixel_offset_x_, pixel_offset_y_, point_stride_);
    if (local_box_enabled_)
    {
      ROS_INFO("Rolling local box in input sensor frame: "
               "x=[%.2f, %.2f], y=[%.2f, %.2f], z=[%.2f, %.2f] m; "
               "max_range=%s",
               local_box_min_x_, local_box_max_x_,
               local_box_min_y_, local_box_max_y_,
               local_box_min_z_, local_box_max_z_,
               max_range_ > 0.0 ? std::to_string(max_range_).c_str() : "disabled");
    }
    if (robot_body_exclusion_enabled_)
    {
      ROS_INFO("Robot body input exclusion in cloud frame: "
               "x=[%.2f, %.2f], y=[%.2f, %.2f] m (all z)",
               robot_body_exclusion_min_x_, robot_body_exclusion_max_x_,
               robot_body_exclusion_min_y_, robot_body_exclusion_max_y_);
    }
    ROS_INFO("Local semantic admission filter: output_frame=%s, "
             "voxel=(%.2f, %.2f, %.2f) m, "
             "all non-dynamic voxels are retained, exclude_dynamic=%s",
             (admission_output_frame_.empty() ? local_cost_frame_ :
              admission_output_frame_).c_str(),
             map_->voxelSizeXY(), map_->voxelSizeXY(), map_->voxelSizeZ(),
             admission_exclude_dynamic_ ? "true" : "false");
    if (publish_ssmi_admitted_cloud_)
    {
      ROS_INFO("Direct SSMI semantic cloud: topic=%s, canonical semantic encoding, "
               "cost >= %.2f encoded as an obstacle",
               ssmi_admitted_topic_.c_str(),
               ssmi_obstacle_traversability_threshold_);
    }
    ROS_INFO("SSMI obstacle revocation: %s, free_frames>=%zu, "
             "free_evidence>=%.2f, evidence_decay=%.2f/s, "
             "ambiguous_obstacles=%zu reset_cost>=%.2f, "
             "ray evidence=%s, ray stride=%d",
             enable_ssmi_obstacle_revocation_ ? "enabled" : "disabled",
             revocation_config.minimum_free_frames,
             revocation_config.minimum_free_evidence,
             revocation_config.free_evidence_decay_per_second,
             revocation_config.ambiguous_obstacle_labels.size(),
             revocation_config.ambiguous_obstacle_reset_min_traversability,
             ssmi_revocation_ray_evidence_enabled_ ? "enabled" : "disabled",
             ssmi_revocation_ray_point_stride_);
    ROS_INFO("Static-terrain morphology recovery: %s, terrain_classes=%zu, "
             "recoverable_static_classes=%zu, excluded_classes=%zu, "
             "closing_radius=%.2f m, height_difference<=%.2f m",
             terrain_boundary_filter_config_.enabled ? "enabled" : "disabled",
             terrain_boundary_filter_config_.terrain_labels.size(),
             terrain_boundary_filter_config_.recoverable_labels.size(),
             terrain_boundary_filter_config_.excluded_labels.size(),
             terrain_boundary_filter_config_.closing_radius,
             terrain_boundary_filter_config_.maximum_height_difference);
    ROS_INFO("Terrain-boundary debug: OpenCV=%s, RViz=%s, "
             "statistics=%d frames, clicked_point=(%s, %.2f m), CSV=%s; "
             "resolved closing radius=%d cells, "
             "height threshold=%d cells",
             terrain_boundary_debug_opencv_enabled_ ? "on" : "off",
             terrain_boundary_debug_rviz_enabled_ ? "on" : "off",
             terrain_boundary_debug_statistics_interval_frames_,
             terrain_boundary_debug_clicked_point_topic_.c_str(),
             terrain_boundary_debug_clicked_point_radius_,
             terrain_boundary_debug_csv_path_.empty() ? "off" :
               terrain_boundary_debug_csv_path_.c_str(),
             static_cast<int>(std::ceil(
               terrain_boundary_filter_config_.closing_radius /
               map_->voxelSizeXY())),
             static_cast<int>(std::ceil(
               terrain_boundary_filter_config_.maximum_height_difference /
               map_->voxelSizeZ())));
    ROS_INFO("Missing-cost terrain height inference: %s, dz>%.2f m + %.1e "
             "within %.2f m -> cost %.2f (%zu configured terrain labels)",
             terrain_height_cost_config_.enabled ? "enabled" : "disabled",
             terrain_height_cost_config_.height_difference_threshold,
             terrain_height_cost_config_.comparison_epsilon,
             terrain_height_cost_config_.neighborhood_radius,
             terrain_height_cost_config_.obstacle_cost,
             terrain_height_cost_config_.terrain_labels.size());
  }

  ~SemanticVoxelMapNode()
  {
    if (!terrain_boundary_opencv_window_initialized_)
    {
      return;
    }
    try
    {
      cv::destroyWindow(terrain_boundary_opencv_window_name_);
    }
    catch (const cv::Exception&)
    {
      // ROS is already shutting down; there is nothing useful to recover here.
    }
  }

private:
  const std::string& mapFrame() const
  {
    return use_initial_pose_reference_ ? reference_frame_ : global_frame_;
  }

  bool isTerrainBoundaryTerrainLabel(const std::uint32_t label) const
  {
    return std::find(terrain_boundary_filter_config_.terrain_labels.begin(),
                     terrain_boundary_filter_config_.terrain_labels.end(),
                     label) != terrain_boundary_filter_config_.terrain_labels.end();
  }

  bool isTerrainBoundaryRecoverableLabel(const std::uint32_t label) const
  {
    return std::find(terrain_boundary_filter_config_.recoverable_labels.begin(),
                     terrain_boundary_filter_config_.recoverable_labels.end(),
                     label) !=
      terrain_boundary_filter_config_.recoverable_labels.end();
  }

  bool isConfiguredDynamicLabel(const std::uint32_t label) const
  {
    if (!shared_semantic_profile_.enabled)
    {
      return label >= 11u && label <= 18u;
    }
    return std::any_of(
      shared_semantic_profile_.classes.begin(),
      shared_semantic_profile_.classes.end(),
      [label](const SharedNavigationSemanticClass& semantic_class)
      {
        return semantic_class.label == label &&
               semantic_class.role == "dynamic_obstacle";
      });
  }

  static cv::Mat boundaryDebugMaskImage(
    const TerrainBoundaryLayerDebug& layer,
    const std::vector<std::uint8_t>& mask,
    const std::array<std::uint8_t, 3>& rgb)
  {
    cv::Mat image(layer.rows, layer.columns, CV_8UC3,
                  cv::Scalar(24, 24, 24));
    const std::size_t expected = static_cast<std::size_t>(layer.rows) *
      static_cast<std::size_t>(layer.columns);
    if (mask.size() == expected)
    {
      const cv::Vec3b bgr(rgb[2], rgb[1], rgb[0]);
      for (int row = 0; row < layer.rows; ++row)
      {
        for (int column = 0; column < layer.columns; ++column)
        {
          const std::size_t index = static_cast<std::size_t>(row) *
            layer.columns + column;
          if (mask[index] != 0u)
          {
            image.at<cv::Vec3b>(row, column) = bgr;
          }
        }
      }
    }
    cv::flip(image, image, 0);
    return image;
  }

  static cv::Mat boundaryDebugPanel(
    const cv::Mat& image, const std::string& title)
  {
    cv::Mat panel(kBoundaryDebugPanelCellHeight,
                  kBoundaryDebugPanelPixels, CV_8UC3,
                  cv::Scalar(24, 24, 24));
    cv::Mat resized;
    cv::resize(image, resized,
               cv::Size(kBoundaryDebugPanelPixels, kBoundaryDebugPanelPixels),
               0.0, 0.0, cv::INTER_NEAREST);
    resized.copyTo(panel(cv::Rect(
      0, kBoundaryDebugPanelHeaderPixels,
      kBoundaryDebugPanelPixels, kBoundaryDebugPanelPixels)));
    cv::putText(panel, title, cv::Point(5, 19),
                cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
    return panel;
  }

  cv::Mat buildTerrainBoundaryDebugCanvas(
    const TerrainBoundaryFilterResult& result, const std::size_t layer_index)
  {
    if (!terrain_boundary_debug_opencv_enabled_ &&
        !terrain_boundary_debug_rviz_enabled_)
    {
      return cv::Mat();
    }

    if (result.debug_layers.empty())
    {
      cv::Mat empty(150, 900, CV_8UC3, cv::Scalar(24, 24, 24));
      cv::putText(empty, "No recoverable static voxels in current local snapshot",
                  cv::Point(24, 82), cv::FONT_HERSHEY_SIMPLEX, 0.70,
                  cv::Scalar(220, 220, 220), 2, cv::LINE_AA);
      return empty;
    }
    const TerrainBoundaryLayerDebug& layer = result.debug_layers[
      std::min(layer_index, result.debug_layers.size() - 1u)];
    const std::array<std::uint8_t, 3> white{{255u, 255u, 255u}};
    const std::array<std::uint8_t, 3> red{{255u, 0u, 0u}};
    const std::array<std::uint8_t, 3> yellow{{255u, 255u, 0u}};
    const std::array<std::uint8_t, 3> blue{{0u, 128u, 255u}};
    const std::array<std::uint8_t, 3> green{{0u, 255u, 0u}};

    std::array<cv::Mat, 9> images{{
      boundaryDebugMaskImage(layer, layer.terrain_original, white),
      boundaryDebugMaskImage(layer, layer.terrain_after_dilation, white),
      boundaryDebugMaskImage(layer, layer.terrain_after_erosion, white),
      boundaryDebugMaskImage(layer, layer.recoverable_static, red),
      boundaryDebugMaskImage(layer, layer.excluded, red),
      boundaryDebugMaskImage(layer, layer.newly_filled, yellow),
      boundaryDebugMaskImage(layer, layer.proposed, yellow),
      boundaryDebugMaskImage(layer, layer.reference_found, blue),
      boundaryDebugMaskImage(layer, layer.recovered, green)
    }};

    const std::array<std::string, 9> titles{{
      "1 terrain projection", "2 after dilation", "3 after erosion",
      "4 recoverable static", "5 dynamic/ignore excluded", "6 newly white",
      "7 supported static candidates", "8 terrain reference found",
      "9 recovered"
    }};
    const int grid_width = 3 * kBoundaryDebugPanelPixels +
      2 * kBoundaryDebugPanelGap;
    const int grid_height = 3 * kBoundaryDebugPanelCellHeight +
      2 * kBoundaryDebugPanelGap;
    cv::Mat canvas(grid_height + kBoundaryDebugFooterPixels, grid_width,
                   CV_8UC3, cv::Scalar(24, 24, 24));
    for (int index = 0; index < 9; ++index)
    {
      const int grid_column = index % 3;
      const int grid_row = index / 3;
      const int x = grid_column *
        (kBoundaryDebugPanelPixels + kBoundaryDebugPanelGap);
      const int y = grid_row *
        (kBoundaryDebugPanelCellHeight + kBoundaryDebugPanelGap);
      boundaryDebugPanel(images[static_cast<std::size_t>(index)],
                         titles[static_cast<std::size_t>(index)]).copyTo(
        canvas(cv::Rect(x, y, kBoundaryDebugPanelPixels,
                        kBoundaryDebugPanelCellHeight)));
    }

    bool have_robot_position = false;
    double robot_x = 0.0;
    double robot_y = 0.0;
    double robot_z = 0.0;
    {
      std::lock_guard<std::mutex> lock(sensor_origin_mutex_);
      have_robot_position = have_sensor_origin_;
      robot_x = latest_sensor_x_;
      robot_y = latest_sensor_y_;
      robot_z = latest_sensor_z_;
    }
    VoxelKey robot_key;
    bool robot_inside_raster = false;
    if (have_robot_position)
    {
      robot_key = map_->worldToKey(robot_x, robot_y, robot_z);
      const std::int64_t robot_column =
        static_cast<std::int64_t>(robot_key.x) - layer.minimum_x;
      const std::int64_t robot_row =
        static_cast<std::int64_t>(robot_key.y) - layer.minimum_y;
      robot_inside_raster = robot_column >= 0 && robot_column < layer.columns &&
        robot_row >= 0 && robot_row < layer.rows;
      if (robot_inside_raster)
      {
        const int panel_x = std::max(0, std::min(
          kBoundaryDebugPanelPixels - 1,
          static_cast<int>((robot_column + 0.5) *
            kBoundaryDebugPanelPixels / layer.columns)));
        const int displayed_row = layer.rows - 1 - static_cast<int>(robot_row);
        const int panel_y = std::max(0, std::min(
          kBoundaryDebugPanelPixels - 1,
          static_cast<int>((displayed_row + 0.5) *
            kBoundaryDebugPanelPixels / layer.rows)));
        for (int index = 0; index < 9; ++index)
        {
          const int grid_column = index % 3;
          const int grid_row = index / 3;
          const cv::Point marker(
            grid_column *
              (kBoundaryDebugPanelPixels + kBoundaryDebugPanelGap) + panel_x,
            grid_row *
              (kBoundaryDebugPanelCellHeight + kBoundaryDebugPanelGap) +
              kBoundaryDebugPanelHeaderPixels + panel_y);
          // A black outline plus white cross stays visible on every mask;
          // the red center identifies it as the robot/sensor XY origin.
          cv::drawMarker(canvas, marker, cv::Scalar(0, 0, 0),
                         cv::MARKER_CROSS, 21, 5, cv::LINE_AA);
          cv::drawMarker(canvas, marker, cv::Scalar(255, 255, 255),
                         cv::MARKER_CROSS, 21, 2, cv::LINE_AA);
          cv::circle(canvas, marker, 3, cv::Scalar(0, 0, 255), cv::FILLED,
                     cv::LINE_AA);
        }
      }
    }

    const TerrainBoundaryDebugStatistics& stats = result.debug_statistics;
    const std::size_t recovered = stats.first_decision_counts[
      static_cast<std::size_t>(TerrainBoundaryDecisionReason::Recovered)];
    std::ostringstream status;
    status << "projected XY grid | terrain=" << stats.terrain_voxels
           << " | recoverable static=" << stats.recoverable_static_voxels
           << " | excluded dynamic/ignore=" << stats.excluded_voxels
           << " | newly white cells=" << stats.opencv_newly_filled_cells
           << " | proposed cells/voxels=" << stats.opencv_proposed_cells
           << "/" << stats.opencv_proposed_voxels
           << " | recovered=" << recovered;
    cv::putText(canvas, status.str(), cv::Point(6, grid_height + 26),
                cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(225, 225, 225), 1, cv::LINE_AA);
    std::ostringstream kernels;
    kernels << "resolved cells: closing radius="
            << stats.closing_radius_cells
            << " height threshold="
            << stats.maximum_height_difference_cells
            << " | click any panel or use RViz /clicked_point for details";
    cv::putText(canvas, kernels.str(), cv::Point(6, grid_height + 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.43,
                cv::Scalar(205, 205, 205), 1, cv::LINE_AA);
    cv::putText(canvas,
      "decision cloud: gray=outside closed support | blue=no terrain reference | orange=height too high | green=recovered",
      cv::Point(6, grid_height + 70), cv::FONT_HERSHEY_SIMPLEX, 0.35,
      cv::Scalar(190, 190, 190), 1, cv::LINE_AA);
    std::ostringstream robot_status;
    if (!have_robot_position)
    {
      robot_status << "robot marker: pose unavailable";
    }
    else
    {
      robot_status << std::fixed << std::setprecision(2)
                   << "robot/sensor origin xyz=(" << robot_x << ","
                   << robot_y << "," << robot_z << ") key=("
                   << robot_key.x << "," << robot_key.y << ","
                   << robot_key.z << ") | XY marker "
                   << (robot_inside_raster ? "shown on every panel" :
                       "outside current raster");
    }
    cv::putText(canvas, robot_status.str(), cv::Point(6, grid_height + 92),
                cv::FONT_HERSHEY_SIMPLEX, 0.40,
                cv::Scalar(225, 225, 225), 1, cv::LINE_AA);
    return canvas;
  }

  sensor_msgs::Image makeTerrainBoundaryDebugImage(
    const cv::Mat& canvas, const ros::Time& stamp) const
  {
    sensor_msgs::Image image;
    image.header.stamp = stamp;
    image.header.frame_id = mapFrame();
    image.height = static_cast<std::uint32_t>(canvas.rows);
    image.width = static_cast<std::uint32_t>(canvas.cols);
    image.encoding = "bgr8";
    image.is_bigendian = false;
    image.step = image.width * 3u;
    image.data.resize(static_cast<std::size_t>(image.height) * image.step);
    for (int row = 0; row < canvas.rows; ++row)
    {
      std::memcpy(image.data.data() + static_cast<std::size_t>(row) * image.step,
                  canvas.ptr(row), image.step);
    }
    return image;
  }

  sensor_msgs::PointCloud2 makeTerrainBoundaryDebugCloud(
    const std::vector<TerrainBoundaryDebugRecord>& records,
    const ros::Time& stamp) const
  {
    sensor_msgs::PointCloud2 cloud;
    cloud.header.stamp = stamp;
    cloud.header.frame_id = mapFrame();
    cloud.height = 1u;
    cloud.is_dense = false;
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2Fields(
      16,
      "x", 1, sensor_msgs::PointField::FLOAT32,
      "y", 1, sensor_msgs::PointField::FLOAT32,
      "z", 1, sensor_msgs::PointField::FLOAT32,
      "rgb", 1, sensor_msgs::PointField::FLOAT32,
      "reason", 1, sensor_msgs::PointField::UINT32,
      "failure_mask", 1, sensor_msgs::PointField::UINT32,
      "original_label", 1, sensor_msgs::PointField::UINT32,
      "proposed", 1, sensor_msgs::PointField::UINT8,
      "reference_found", 1, sensor_msgs::PointField::UINT8,
      "reference_x", 1, sensor_msgs::PointField::FLOAT32,
      "reference_y", 1, sensor_msgs::PointField::FLOAT32,
      "reference_z", 1, sensor_msgs::PointField::FLOAT32,
      "reference_label", 1, sensor_msgs::PointField::UINT32,
      "reference_distance_xy", 1, sensor_msgs::PointField::FLOAT32,
      "height_difference", 1, sensor_msgs::PointField::FLOAT32,
      "replacement_label", 1, sensor_msgs::PointField::UINT32);
    modifier.resize(records.size());
    sensor_msgs::PointCloud2Iterator<float> x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> rgb(cloud, "rgb");
    sensor_msgs::PointCloud2Iterator<std::uint32_t> reason(cloud, "reason");
    sensor_msgs::PointCloud2Iterator<std::uint32_t> failure_mask(
      cloud, "failure_mask");
    sensor_msgs::PointCloud2Iterator<std::uint32_t> original_label(
      cloud, "original_label");
    sensor_msgs::PointCloud2Iterator<std::uint8_t> proposed(cloud, "proposed");
    sensor_msgs::PointCloud2Iterator<std::uint8_t> reference_found(
      cloud, "reference_found");
    sensor_msgs::PointCloud2Iterator<float> reference_x(cloud, "reference_x");
    sensor_msgs::PointCloud2Iterator<float> reference_y(cloud, "reference_y");
    sensor_msgs::PointCloud2Iterator<float> reference_z(cloud, "reference_z");
    sensor_msgs::PointCloud2Iterator<std::uint32_t> reference_label(
      cloud, "reference_label");
    sensor_msgs::PointCloud2Iterator<float> reference_distance(
      cloud, "reference_distance_xy");
    sensor_msgs::PointCloud2Iterator<float> height_difference(
      cloud, "height_difference");
    sensor_msgs::PointCloud2Iterator<std::uint32_t> replacement_label(
      cloud, "replacement_label");
    for (const TerrainBoundaryDebugRecord& record : records)
    {
      const auto color = boundaryDebugReasonRgb(record.first_failure);
      *x = static_cast<float>(record.x);
      *y = static_cast<float>(record.y);
      *z = static_cast<float>(record.z);
      *rgb = packedRgbFloat(color[0], color[1], color[2]);
      *reason = static_cast<std::uint32_t>(record.first_failure);
      *failure_mask = record.failure_mask;
      *original_label = record.original_label;
      *proposed = record.proposed ? 1u : 0u;
      *reference_found = record.reference_found ? 1u : 0u;
      *reference_x = static_cast<float>(record.reference_x);
      *reference_y = static_cast<float>(record.reference_y);
      *reference_z = static_cast<float>(record.reference_z);
      *reference_label = record.reference_label;
      *reference_distance = static_cast<float>(record.reference_distance_xy);
      *height_difference = static_cast<float>(record.height_difference);
      *replacement_label = record.replacement_label;
      ++x; ++y; ++z; ++rgb; ++reason; ++failure_mask; ++original_label;
      ++proposed; ++reference_found; ++reference_x; ++reference_y; ++reference_z;
      ++reference_label; ++reference_distance; ++height_difference;
      ++replacement_label;
    }
    return cloud;
  }

  void logTerrainBoundaryDebugRecord(
    const TerrainBoundaryDebugRecord& record, const std::string& source) const
  {
    ROS_INFO_STREAM(
      "Terrain-boundary debug [" << source << "] key=(" << record.key.x
      << "," << record.key.y << "," << record.key.z << ") xyz=("
      << std::fixed << std::setprecision(3) << record.x << "," << record.y
      << "," << record.z << ") decision="
      << terrainBoundaryDecisionReasonName(record.first_failure)
      << " failure_mask=0x" << std::hex << record.failure_mask << std::dec
      << " | original_label=" << record.original_label
      << " | morphology closed=" << record.closed_terrain
      << " proposed=" << record.proposed
      << " | reference_found=" << record.reference_found
      << " reference_xyz=(" << record.reference_x << ","
      << record.reference_y << "," << record.reference_z << ")"
      << " reference_label=" << record.reference_label
      << " distance_xy=" << record.reference_distance_xy
      << " | height_difference=" << record.height_difference
      << " <= " << terrain_boundary_filter_config_.maximum_height_difference
      << " replacement=" << record.replacement_label);
  }

  void writeTerrainBoundaryDebugCsv(
    const TerrainBoundaryFilterResult& result, const ros::Time& stamp)
  {
    if (!terrain_boundary_debug_csv_)
    {
      return;
    }
    terrain_boundary_debug_csv_ << std::setprecision(9);
    for (const TerrainBoundaryDebugRecord& record : result.debug_records)
    {
      terrain_boundary_debug_csv_
        << stamp.toSec() << ',' << record.key.x << ',' << record.key.y << ','
        << record.key.z << ',' << record.x << ',' << record.y << ',' << record.z
        << ',' << terrainBoundaryDecisionReasonName(record.first_failure) << ','
        << record.failure_mask << ',' << record.original_label << ','
        << record.closed_terrain << ',' << record.proposed << ','
        << record.reference_found << ',' << record.reference_key.x << ','
        << record.reference_key.y << ',' << record.reference_key.z << ','
        << record.reference_x << ',' << record.reference_y << ','
        << record.reference_z << ',' << record.reference_label << ','
        << record.reference_distance_xy << ',' << record.height_difference
        << ',' << record.replacement_label << '\n';
    }
    terrain_boundary_debug_csv_.flush();
  }

  void logTerrainBoundaryDebugStatistics(
    const TerrainBoundaryFilterResult& result)
  {
    const TerrainBoundaryDebugStatistics& stats = result.debug_statistics;
    std::ostringstream summary;
    summary << "Terrain-boundary debug frame "
            << terrain_boundary_debug_frame_count_
            << ": terrain=" << stats.terrain_voxels
            << ", recoverable_static=" << stats.recoverable_static_voxels
            << ", excluded_dynamic_or_ignore=" << stats.excluded_voxels
            << ", newly_white_cells=" << stats.opencv_newly_filled_cells
            << ", proposed_cells/voxels=" << stats.opencv_proposed_cells
            << '/' << stats.opencv_proposed_voxels
            << ", reference_found=" << stats.reference_found_voxels;
    for (std::size_t index = 0u;
         index < stats.first_decision_counts.size(); ++index)
    {
      if (stats.first_decision_counts[index] == 0u)
      {
        continue;
      }
      summary << ", " << terrainBoundaryDecisionReasonName(
        static_cast<TerrainBoundaryDecisionReason>(index)) << '='
              << stats.first_decision_counts[index];
    }
    ROS_INFO_STREAM(summary.str());

    if (terrain_boundary_debug_max_samples_per_reason_ == 0)
    {
      return;
    }
    std::array<int, kTerrainBoundaryDecisionReasonCount> samples{{}};
    for (const TerrainBoundaryDebugRecord& record : result.debug_records)
    {
      const std::size_t index = static_cast<std::size_t>(record.first_failure);
      if (index >= samples.size() ||
          samples[index] >= terrain_boundary_debug_max_samples_per_reason_)
      {
        continue;
      }
      ++samples[index];
      logTerrainBoundaryDebugRecord(record, "periodic sample");
    }
  }

  void publishTerrainBoundaryDebug(
    const TerrainBoundaryFilterResult& result, const ros::Time& stamp)
  {
    if (!terrain_boundary_debug_opencv_enabled_ &&
        !terrain_boundary_debug_rviz_enabled_)
    {
      return;
    }
    ++terrain_boundary_debug_frame_count_;
    {
      std::lock_guard<std::mutex> lock(terrain_boundary_debug_mutex_);
      latest_terrain_boundary_debug_records_ = result.debug_records;
      latest_terrain_boundary_debug_layers_ = result.debug_layers;
    }
    if (terrain_boundary_debug_rviz_enabled_)
    {
      terrain_boundary_debug_decision_cloud_pub_.publish(
        makeTerrainBoundaryDebugCloud(result.debug_records, stamp));
    }
    writeTerrainBoundaryDebugCsv(result, stamp);
    if (terrain_boundary_debug_frame_count_ % static_cast<std::size_t>(
          terrain_boundary_debug_statistics_interval_frames_) == 0u)
    {
      logTerrainBoundaryDebugStatistics(result);
    }

    // The simplified recovery works on one projected XY grid, not per-Z
    // layers, so debug always displays index zero.
    terrain_boundary_debug_selected_layer_index_ = 0;

    try
    {
      if (terrain_boundary_debug_opencv_enabled_ &&
          !terrain_boundary_opencv_window_initialized_)
      {
        cv::namedWindow(terrain_boundary_opencv_window_name_, cv::WINDOW_NORMAL);
        terrain_boundary_opencv_window_initialized_ = true;
        cv::setMouseCallback(
          terrain_boundary_opencv_window_name_,
          &SemanticVoxelMapNode::terrainBoundaryOpenCvMouseCallback, this);
      }

      const cv::Mat canvas = buildTerrainBoundaryDebugCanvas(
        result, static_cast<std::size_t>(
          terrain_boundary_debug_selected_layer_index_));
      if (terrain_boundary_debug_rviz_enabled_)
      {
        terrain_boundary_debug_image_pub_.publish(
          makeTerrainBoundaryDebugImage(canvas, stamp));
      }
      if (terrain_boundary_debug_opencv_enabled_)
      {
        cv::imshow(terrain_boundary_opencv_window_name_, canvas);
        cv::waitKey(1);
      }
    }
    catch (const cv::Exception& exception)
    {
      ROS_ERROR("Terrain-boundary debug visualization failed: %s; "
                "disabling OpenCV and RViz debug output",
                exception.what());
      terrain_boundary_debug_opencv_enabled_ = false;
      terrain_boundary_debug_rviz_enabled_ = false;
    }
  }

  static void terrainBoundaryOpenCvMouseCallback(
    const int event, const int x, const int y, const int, void* user_data)
  {
    if (event != cv::EVENT_LBUTTONDOWN || user_data == nullptr)
    {
      return;
    }
    static_cast<SemanticVoxelMapNode*>(user_data)->
      handleTerrainBoundaryOpenCvClick(x, y);
  }

  void handleTerrainBoundaryOpenCvClick(const int x, const int y)
  {
    const int cell_width = kBoundaryDebugPanelPixels + kBoundaryDebugPanelGap;
    const int cell_height =
      kBoundaryDebugPanelCellHeight + kBoundaryDebugPanelGap;
    const int grid_column = x / cell_width;
    const int grid_row = y / cell_height;
    if (grid_column < 0 || grid_column >= 3 ||
        grid_row < 0 || grid_row >= 3)
    {
      return;
    }
    const int local_x = x - grid_column * cell_width;
    const int local_y = y - grid_row * cell_height -
      kBoundaryDebugPanelHeaderPixels;
    if (local_x < 0 || local_x >= kBoundaryDebugPanelPixels ||
        local_y < 0 || local_y >= kBoundaryDebugPanelPixels)
    {
      return;
    }

    std::lock_guard<std::mutex> lock(terrain_boundary_debug_mutex_);
    if (latest_terrain_boundary_debug_layers_.empty())
    {
      return;
    }
    const std::size_t layer_index = std::min<std::size_t>(
      static_cast<std::size_t>(std::max(
        0, terrain_boundary_debug_selected_layer_index_)),
      latest_terrain_boundary_debug_layers_.size() - 1u);
    const TerrainBoundaryLayerDebug& layer =
      latest_terrain_boundary_debug_layers_[layer_index];
    const int raster_column = std::min(
      layer.columns - 1, local_x * layer.columns / kBoundaryDebugPanelPixels);
    const int displayed_row = std::min(
      layer.rows - 1, local_y * layer.rows / kBoundaryDebugPanelPixels);
    const int raster_row = layer.rows - 1 - displayed_row;
    const std::int32_t selected_x = layer.minimum_x + raster_column;
    const std::int32_t selected_y = layer.minimum_y + raster_row;
    std::size_t matches = 0u;
    for (const TerrainBoundaryDebugRecord& record :
         latest_terrain_boundary_debug_records_)
    {
      if (record.key.x == selected_x && record.key.y == selected_y)
      {
        logTerrainBoundaryDebugRecord(record, "OpenCV click");
        ++matches;
      }
    }
    if (matches == 0u)
    {
      ROS_INFO("Terrain-boundary OpenCV click XY key=(%d,%d): "
               "no recoverable static voxel",
               selected_x, selected_y);
      return;
    }
  }

  void terrainBoundaryClickedPointCallback(
    const geometry_msgs::PointStampedConstPtr& message)
  {
    tf2::Vector3 query(message->point.x, message->point.y, message->point.z);
    const std::string source_frame = message->header.frame_id.empty() ?
      mapFrame() : message->header.frame_id;
    if (source_frame != mapFrame())
    {
      try
      {
        const geometry_msgs::TransformStamped transform =
          tf_buffer_.lookupTransform(
            mapFrame(), source_frame, message->header.stamp,
            ros::Duration(transform_timeout_));
        const tf2::Quaternion rotation(
          transform.transform.rotation.x, transform.transform.rotation.y,
          transform.transform.rotation.z, transform.transform.rotation.w);
        const tf2::Vector3 translation(
          transform.transform.translation.x, transform.transform.translation.y,
          transform.transform.translation.z);
        query = tf2::Transform(rotation, translation) * query;
      }
      catch (const tf2::TransformException& exception)
      {
        ROS_WARN("Terrain-boundary clicked-point TF error: %s", exception.what());
        return;
      }
    }

    std::lock_guard<std::mutex> lock(terrain_boundary_debug_mutex_);
    const TerrainBoundaryDebugRecord* nearest = nullptr;
    double nearest_distance_squared =
      terrain_boundary_debug_clicked_point_radius_ *
      terrain_boundary_debug_clicked_point_radius_;
    for (const TerrainBoundaryDebugRecord& record :
         latest_terrain_boundary_debug_records_)
    {
      const double dx = record.x - query.x();
      const double dy = record.y - query.y();
      const double dz = record.z - query.z();
      const double distance_squared = dx * dx + dy * dy + dz * dz;
      if (distance_squared <= nearest_distance_squared)
      {
        nearest_distance_squared = distance_squared;
        nearest = &record;
      }
    }
    if (nearest == nullptr)
    {
      ROS_INFO("Terrain-boundary clicked point (%.3f,%.3f,%.3f): "
               "no recoverable static voxel within %.2f m",
               query.x(), query.y(), query.z(),
               terrain_boundary_debug_clicked_point_radius_);
      return;
    }
    logTerrainBoundaryDebugRecord(*nearest, "RViz clicked_point");
  }

  void showTerrainBoundaryRecovery(
    const std::vector<VoxelSnapshot>& before,
    const std::vector<VoxelSnapshot>& after,
    const std::vector<TerrainBoundaryRelabel>& relabeled)
  {
    if (!terrain_boundary_debug_opencv_enabled_)
    {
      return;
    }
    ++terrain_boundary_opencv_frame_count_;

    try
    {
      if (before.empty())
      {
        cv::Mat empty_view(120, 720, CV_8UC3, cv::Scalar(24, 24, 24));
        std::ostringstream empty_message;
        empty_message << "Input frame " << terrain_boundary_opencv_frame_count_
                      << ": no voxels in the current local map";
        cv::putText(empty_view, empty_message.str(), cv::Point(28, 68),
                    cv::FONT_HERSHEY_SIMPLEX, 0.70,
                    cv::Scalar(220, 220, 220), 2, cv::LINE_AA);
        if (!terrain_boundary_opencv_window_initialized_)
        {
          cv::namedWindow(terrain_boundary_opencv_window_name_, cv::WINDOW_NORMAL);
          terrain_boundary_opencv_window_initialized_ = true;
        }
        cv::imshow(terrain_boundary_opencv_window_name_, empty_view);
        cv::waitKey(1);
        return;
      }

      std::int32_t min_x = before.front().key.x;
      std::int32_t max_x = before.front().key.x;
      std::int32_t min_y = before.front().key.y;
      std::int32_t max_y = before.front().key.y;
      for (const VoxelSnapshot& voxel : before)
      {
        min_x = std::min(min_x, voxel.key.x);
        max_x = std::max(max_x, voxel.key.x);
        min_y = std::min(min_y, voxel.key.y);
        max_y = std::max(max_y, voxel.key.y);
      }

      const std::int64_t width64 =
        static_cast<std::int64_t>(max_x) - min_x + 1;
      const std::int64_t height64 =
        static_cast<std::int64_t>(max_y) - min_y + 1;
      constexpr std::int64_t kMaximumRasterCells = 16000000;
      if (width64 <= 0 || height64 <= 0 ||
          width64 > std::numeric_limits<int>::max() ||
          height64 > std::numeric_limits<int>::max() ||
          width64 > kMaximumRasterCells / height64)
      {
        ROS_ERROR_THROTTLE(
          2.0, "Cannot draw terrain recovery raster with bounds %lld x %lld; "
          "disabling the OpenCV window",
          static_cast<long long>(width64), static_cast<long long>(height64));
        terrain_boundary_debug_opencv_enabled_ = false;
        return;
      }

      const int width = static_cast<int>(width64);
      const int height = static_cast<int>(height64);
      cv::Mat before_image(height, width, CV_8UC3, cv::Scalar(24, 24, 24));
      cv::Mat after_image(height, width, CV_8UC3, cv::Scalar(24, 24, 24));
      cv::Mat before_priority(height, width, CV_8UC1, cv::Scalar(0));
      cv::Mat after_priority(height, width, CV_8UC1, cv::Scalar(0));

      const auto draw_projection = [this, min_x, max_y](
        const std::vector<VoxelSnapshot>& voxels, cv::Mat& image,
        cv::Mat& priority)
      {
        for (const VoxelSnapshot& voxel : voxels)
        {
          const int column = static_cast<int>(
            static_cast<std::int64_t>(voxel.key.x) - min_x);
          const int row = static_cast<int>(
            static_cast<std::int64_t>(max_y) - voxel.key.y);
          std::uint8_t voxel_priority = 1u;
          cv::Vec3b color(96u, 96u, 96u);
          if (voxel.label == kInvalidSemanticLabel)
          {
            voxel_priority = 0u;
            color = cv::Vec3b(48u, 48u, 48u);
          }
          else if (isTerrainBoundaryTerrainLabel(voxel.label))
          {
            voxel_priority = 2u;
            color = cv::Vec3b(255u, 255u, 255u);
          }
          else if (isTerrainBoundaryRecoverableLabel(voxel.label))
          {
            voxel_priority = 3u;
            color = cv::Vec3b(0u, 0u, 255u);
          }
          if (voxel_priority >= priority.at<std::uint8_t>(row, column))
          {
            priority.at<std::uint8_t>(row, column) = voxel_priority;
            image.at<cv::Vec3b>(row, column) = color;
          }
        }
      };
      draw_projection(before, before_image, before_priority);
      draw_projection(after, after_image, after_priority);

      for (const TerrainBoundaryRelabel& recovered : relabeled)
      {
        const std::int64_t column64 =
          static_cast<std::int64_t>(recovered.key.x) - min_x;
        const std::int64_t row64 =
          static_cast<std::int64_t>(max_y) - recovered.key.y;
        if (column64 >= 0 && column64 < width64 &&
            row64 >= 0 && row64 < height64)
        {
          after_image.at<cv::Vec3b>(static_cast<int>(row64),
                                    static_cast<int>(column64)) =
            cv::Vec3b(0u, 255u, 0u);
        }
      }

      const std::size_t before_recoverable_static = static_cast<std::size_t>(
        std::count_if(before.begin(), before.end(), [this](const VoxelSnapshot& voxel)
        {
          return isTerrainBoundaryRecoverableLabel(voxel.label);
        }));
      const std::size_t after_recoverable_static = static_cast<std::size_t>(
        std::count_if(after.begin(), after.end(), [this](const VoxelSnapshot& voxel)
        {
          return isTerrainBoundaryRecoverableLabel(voxel.label);
        }));

      constexpr int kPanelPixels = 620;
      const double scale = std::min(
        static_cast<double>(kPanelPixels) / width,
        static_cast<double>(kPanelPixels) / height);
      const cv::Size raster_size(
        std::max(1, static_cast<int>(std::lround(width * scale))),
        std::max(1, static_cast<int>(std::lround(height * scale))));
      cv::Mat before_raster;
      cv::Mat after_raster;
      cv::resize(before_image, before_raster, raster_size, 0.0, 0.0,
                 cv::INTER_NEAREST);
      cv::resize(after_image, after_raster, raster_size, 0.0, 0.0,
                 cv::INTER_NEAREST);
      cv::Mat before_panel(kPanelPixels, kPanelPixels, CV_8UC3,
                           cv::Scalar(24, 24, 24));
      cv::Mat after_panel(kPanelPixels, kPanelPixels, CV_8UC3,
                          cv::Scalar(24, 24, 24));
      const int raster_x = (kPanelPixels - raster_size.width) / 2;
      const int raster_y = (kPanelPixels - raster_size.height) / 2;
      before_raster.copyTo(before_panel(cv::Rect(
        raster_x, raster_y, raster_size.width, raster_size.height)));
      after_raster.copyTo(after_panel(cv::Rect(
        raster_x, raster_y, raster_size.width, raster_size.height)));

      constexpr int kHeaderHeight = 42;
      constexpr int kFooterHeight = 54;
      constexpr int kPanelGap = 12;
      cv::Mat canvas(kPanelPixels + kHeaderHeight + kFooterHeight,
                     kPanelPixels * 2 + kPanelGap, CV_8UC3,
                     cv::Scalar(24, 24, 24));
      before_panel.copyTo(canvas(cv::Rect(
        0, kHeaderHeight, kPanelPixels, kPanelPixels)));
      after_panel.copyTo(canvas(cv::Rect(
        kPanelPixels + kPanelGap, kHeaderHeight,
        kPanelPixels, kPanelPixels)));
      cv::putText(canvas, "BEFORE: recoverable static = red", cv::Point(8, 27),
                  cv::FONT_HERSHEY_SIMPLEX, 0.52,
                  cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
      cv::putText(canvas, "AFTER: recovered = green",
                  cv::Point(kPanelPixels + kPanelGap + 8, 27),
                  cv::FONT_HERSHEY_SIMPLEX, 0.52,
                  cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
      std::ostringstream status;
      status << "Input frame: " << terrain_boundary_opencv_frame_count_
             << " | 3D recoverable static voxels: "
             << before_recoverable_static << " -> "
             << after_recoverable_static
             << " | recovered: " << relabeled.size();
      cv::putText(canvas, status.str(),
                  cv::Point(8, canvas.rows - 30), cv::FONT_HERSHEY_SIMPLEX,
                  0.45, cv::Scalar(210, 210, 210), 1, cv::LINE_AA);
      cv::putText(canvas,
                  "white = YAML terrain | gray = other labels | X right, Y up",
                  cv::Point(8, canvas.rows - 10), cv::FONT_HERSHEY_SIMPLEX,
                  0.45, cv::Scalar(210, 210, 210), 1, cv::LINE_AA);

      if (!terrain_boundary_opencv_window_initialized_)
      {
        cv::namedWindow(terrain_boundary_opencv_window_name_, cv::WINDOW_NORMAL);
        terrain_boundary_opencv_window_initialized_ = true;
      }
      cv::imshow(terrain_boundary_opencv_window_name_, canvas);
      cv::waitKey(1);
    }
    catch (const cv::Exception& exception)
    {
      ROS_ERROR("OpenCV terrain-recovery visualization failed: %s; "
                "disabling the window", exception.what());
      terrain_boundary_debug_opencv_enabled_ = false;
    }
  }

  struct ScanVoxel
  {
    std::unordered_map<std::uint32_t, float> label_weights;
    float semantic_weight = 0.0f;
    float cost_sum = 0.0f;
    float cost_weight = 0.0f;
  };

  std::vector<SemanticClass> loadSemanticClasses()
  {
    std::vector<SemanticClass> output;
    if (shared_semantic_profile_.enabled)
    {
      output.reserve(shared_semantic_profile_.classes.size());
      for (const SharedNavigationSemanticClass& shared_class :
           shared_semantic_profile_.classes)
      {
        SemanticClass semantic_class;
        semantic_class.label = shared_class.label;
        semantic_class.name = shared_class.name;
        semantic_class.red = shared_class.rgb[0];
        semantic_class.green = shared_class.rgb[1];
        semantic_class.blue = shared_class.rgb[2];
        semantic_class.traversability_cost = shared_class.semantic_cost;
        output.push_back(semantic_class);
        ROS_INFO("Shared semantic class %s label=%u role=%s cost=%.2f",
                 shared_class.name.c_str(), shared_class.label,
                 shared_class.role.c_str(), shared_class.semantic_cost);
      }
      return output;
    }

    XmlRpc::XmlRpcValue classes;
    if (!private_nh_.getParam("semantic_classes", classes))
    {
      ROS_WARN("No semantic_classes configured; packed semantic RGB will be used "
               "for display and all labels will use unknown_cost");
      return output;
    }
    if (classes.getType() != XmlRpc::XmlRpcValue::TypeArray)
    {
      throw std::runtime_error("~semantic_classes must be a YAML list");
    }

    for (int index = 0; index < classes.size(); ++index)
    {
      const XmlRpc::XmlRpcValue& item = classes[index];
      if (item.getType() != XmlRpc::XmlRpcValue::TypeStruct ||
          !item.hasMember("label") || !item.hasMember("cost"))
      {
        throw std::runtime_error("each semantic class requires label and cost");
      }
      SemanticClass semantic_class;
      semantic_class.label = parseLabel(item["label"]);
      semantic_class.traversability_cost =
        clampUnit(static_cast<float>(xmlNumber(item["cost"])));
      semantic_class.name = item.hasMember("name") ?
        static_cast<std::string>(item["name"]) : std::to_string(semantic_class.label);

      if (item.hasMember("color"))
      {
        const XmlRpc::XmlRpcValue& color = item["color"];
        if (color.getType() != XmlRpc::XmlRpcValue::TypeArray || color.size() != 3)
        {
          throw std::runtime_error("semantic class color must be [r, g, b]");
        }
        semantic_class.red = static_cast<std::uint8_t>(xmlNumber(color[0]));
        semantic_class.green = static_cast<std::uint8_t>(xmlNumber(color[1]));
        semantic_class.blue = static_cast<std::uint8_t>(xmlNumber(color[2]));
      }
      else
      {
        semantic_class.red = (semantic_class.label >> 16u) & 0xffu;
        semantic_class.green = (semantic_class.label >> 8u) & 0xffu;
        semantic_class.blue = semantic_class.label & 0xffu;
      }
      output.push_back(semantic_class);
      ROS_INFO("Semantic class %s label=%u cost=%.2f",
               semantic_class.name.c_str(), semantic_class.label,
               semantic_class.traversability_cost);
    }
    return output;
  }

  std::unordered_map<std::uint32_t, std::uint32_t> loadSemanticLabelRemap()
  {
    std::unordered_map<std::uint32_t, std::uint32_t> output;
    XmlRpc::XmlRpcValue remap;
    if (!private_nh_.getParam("semantic_label_remap", remap))
    {
      return output;
    }
    if (remap.getType() != XmlRpc::XmlRpcValue::TypeArray)
    {
      throw std::runtime_error("~semantic_label_remap must be a YAML list");
    }
    for (int index = 0; index < remap.size(); ++index)
    {
      const XmlRpc::XmlRpcValue& item = remap[index];
      if (item.getType() != XmlRpc::XmlRpcValue::TypeStruct ||
          !item.hasMember("from") || !item.hasMember("to"))
      {
        throw std::runtime_error(
          "each semantic_label_remap entry requires 'from' and 'to'");
      }
      const std::uint32_t source = parseLabel(item["from"]);
      const std::uint32_t target = parseLabel(item["to"]);
      const auto insertion = output.emplace(source, target);
      if (!insertion.second)
      {
        throw std::runtime_error(
          "semantic_label_remap contains duplicate source label " +
          std::to_string(source));
      }
      ROS_INFO("Semantic input label remap: %u (0x%06x) -> %u",
               source, source & 0x00ffffffu, target);
    }
    return output;
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& message)
  {
    const ros::WallTime callback_start = ros::WallTime::now();
    if (message->header.stamp.isZero())
    {
      ROS_ERROR_THROTTLE(2.0,
        "Input cloud has a zero acquisition timestamp; frame-time decay requires "
        "the depth camera timestamp");
      return;
    }

    if (!latest_processed_frame_stamp_.isZero() &&
        message->header.stamp < latest_processed_frame_stamp_)
    {
      ROS_WARN("Input time rewound from %.6f to %.6f; clearing local map, "
               "cached admission snapshot, and initial reference",
               latest_processed_frame_stamp_.toSec(), message->header.stamp.toSec());
      clearRuntimeState(true);
    }
    else if (!latest_processed_frame_stamp_.isZero() &&
             message->header.stamp == latest_processed_frame_stamp_)
    {
      ROS_WARN_THROTTLE(2.0,
        "Dropping duplicate input frame at stamp %.6f; it is not a different frame",
        message->header.stamp.toSec());
      return;
    }

    geometry_msgs::TransformStamped transform;
    try
    {
      transform = tf_buffer_.lookupTransform(global_frame_, message->header.frame_id,
                                             message->header.stamp,
                                             ros::Duration(transform_timeout_));
    }
    catch (const tf2::TransformException& exception)
    {
      ROS_WARN_THROTTLE(2.0, "Semantic voxel map TF error: %s", exception.what());
      return;
    }

    const FieldView x_field = findField(*message, "x");
    const FieldView y_field = findField(*message, "y");
    const FieldView z_field = findField(*message, "z");
    std::string resolved_semantic_field = semantic_field_;
    FieldView semantic;
    if (semantic_field_ == "auto")
    {
      const std::array<std::string, 3> candidates{{
        "semantic_lable", "semantic_color", "label"}};
      for (const std::string& candidate : candidates)
      {
        semantic = findField(*message, candidate);
        if (semantic.valid)
        {
          resolved_semantic_field = candidate;
          break;
        }
      }
    }
    else
    {
      semantic = findField(*message, semantic_field_);
    }
    FieldView fallback_label;
    if (!semantic.valid && resolved_semantic_field != "label")
    {
      fallback_label = findField(*message, "label");
      if (fallback_label.valid)
      {
        resolved_semantic_field = "label";
      }
    }
    const FieldView confidence = findField(*message, confidence_field_);
    FieldView cost = findField(*message, cost_field_);
    if (!cost.valid && cost_field_ != "cost")
    {
      cost = findField(*message, "cost");
    }

    if (!x_field.valid || !y_field.valid || !z_field.valid ||
        (!semantic.valid && !fallback_label.valid))
    {
      ROS_ERROR_THROTTLE(2.0,
        "Input cloud requires x/y/z and semantic field '%s' (or 'label')",
        semantic_field_.c_str());
      return;
    }

    const std::size_t point_count = static_cast<std::size_t>(message->width) *
                                    static_cast<std::size_t>(message->height);
    const bool organized_cloud = message->height > 1u;
    const bool configured_image_shape = point_count ==
      static_cast<std::size_t>(input_image_width_) *
      static_cast<std::size_t>(input_image_height_);
    const bool image_layout = input_layout_ == "image" ||
      (input_layout_ == "auto" && (organized_cloud || configured_image_shape));
    const std::uint32_t image_width = organized_cloud ? message->width :
      static_cast<std::uint32_t>(input_image_width_);
    const std::uint32_t image_height = organized_cloud ? message->height :
      static_cast<std::uint32_t>(input_image_height_);
    if (image_layout && !organized_cloud && point_count !=
        static_cast<std::size_t>(image_width) * image_height)
    {
      ROS_ERROR_THROTTLE(2.0,
        "Flattened input cloud has %zu points, but configured image shape is %ux%u",
        point_count, image_width, image_height);
      return;
    }

    const tf2::Quaternion rotation(
      transform.transform.rotation.x, transform.transform.rotation.y,
      transform.transform.rotation.z, transform.transform.rotation.w);
    const tf2::Vector3 translation(
      transform.transform.translation.x, transform.transform.translation.y,
      transform.transform.translation.z);
    const tf2::Transform sensor_to_global(rotation, translation);
    if (use_initial_pose_reference_ && !have_initial_reference_)
    {
      const double yaw = std::atan2(
        2.0 * (rotation.w() * rotation.z() + rotation.x() * rotation.y()),
        1.0 - 2.0 * (rotation.y() * rotation.y() + rotation.z() * rotation.z()));
      tf2::Quaternion reference_rotation = rotation;
      if (initial_pose_reference_yaw_only_)
      {
        reference_rotation.setRPY(0.0, 0.0, yaw);
        reference_rotation.normalize();
      }
      initial_reference_to_global_ = tf2::Transform(reference_rotation, translation);
      have_initial_reference_ = true;
      initial_reference_stamp_ = message->header.stamp;

      geometry_msgs::TransformStamped reference_transform;
      reference_transform.header.stamp = initial_reference_stamp_;
      reference_transform.header.frame_id = global_frame_;
      reference_transform.child_frame_id = reference_frame_;
      reference_transform.transform.translation.x = translation.x();
      reference_transform.transform.translation.y = translation.y();
      reference_transform.transform.translation.z = translation.z();
      reference_transform.transform.rotation.x = reference_rotation.x();
      reference_transform.transform.rotation.y = reference_rotation.y();
      reference_transform.transform.rotation.z = reference_rotation.z();
      reference_transform.transform.rotation.w = reference_rotation.w();
      reference_tf_broadcaster_.sendTransform(reference_transform);

      ROS_INFO("Captured initial cloud pose at %.9f: %s <- %s, "
               "xyz=(%.6f, %.6f, %.6f), yaw=%.3f deg; map origin is now '%s' "
               "with %s rotation",
               initial_reference_stamp_.toSec(), global_frame_.c_str(),
               message->header.frame_id.c_str(), translation.x(), translation.y(),
               translation.z(), yaw * 180.0 / 3.14159265358979323846,
               reference_frame_.c_str(),
               initial_pose_reference_yaw_only_ ? "yaw-only" : "full 3D");
    }

    const tf2::Transform sensor_to_map = use_initial_pose_reference_ ?
      initial_reference_to_global_.inverseTimes(sensor_to_global) : sensor_to_global;
    const tf2::Vector3 sensor_origin = sensor_to_map.getOrigin();
    const double origin_x = sensor_origin.x();
    const double origin_y = sensor_origin.y();
    const double origin_z = sensor_origin.z();
    const double max_range_squared = max_range_ > 0.0 ?
      max_range_ * max_range_ : std::numeric_limits<double>::max();
    std::unordered_map<VoxelKey, ScanVoxel, VoxelKeyHash> scan_voxels;
    const std::size_t sampled_capacity = image_layout ?
      (image_width + static_cast<std::uint32_t>(pixel_stride_x_) - 1u) /
        static_cast<std::uint32_t>(pixel_stride_x_) *
        ((image_height + static_cast<std::uint32_t>(pixel_stride_y_) - 1u) /
         static_cast<std::uint32_t>(pixel_stride_y_)) :
      (point_count + static_cast<std::size_t>(point_stride_) - 1u) /
        static_cast<std::size_t>(point_stride_);
    scan_voxels.reserve(sampled_capacity / 2u + 1u);
    std::unordered_set<VoxelKey, VoxelKeyHash> ray_free_evidence;
    std::size_t sampled_points = 0u;
    std::size_t valid_points = 0u;
    std::size_t robot_body_rejected_points = 0u;

    auto process_point = [&](const std::size_t point_offset)
    {
      if (point_offset + message->point_step > message->data.size())
      {
        return false;
      }
      const std::uint8_t* point = message->data.data() + point_offset;
      ++sampled_points;
      double sensor_x, sensor_y, sensor_z;
      if (!readNumber(point, x_field, sensor_x) ||
          !readNumber(point, y_field, sensor_y) ||
          !readNumber(point, z_field, sensor_z) ||
          !std::isfinite(sensor_x) || !std::isfinite(sensor_y) ||
          !std::isfinite(sensor_z))
      {
        return true;
      }
      if (sensor_x * sensor_x + sensor_y * sensor_y + sensor_z * sensor_z >
          max_range_squared)
      {
        return true;
      }
      // Apply this in the incoming cloud frame before voxel fusion. For the
      // simulation profile that frame is base_link, so articulated robot-body
      // returns cannot affect semantic evidence or terrain-height inference.
      if (robot_body_exclusion_enabled_ && isInsidePlanarExclusion(
            sensor_x, sensor_y,
            robot_body_exclusion_min_x_, robot_body_exclusion_max_x_,
            robot_body_exclusion_min_y_, robot_body_exclusion_max_y_))
      {
        ++robot_body_rejected_points;
        return true;
      }
      if (local_box_enabled_ &&
          (sensor_x < local_box_min_x_ || sensor_x > local_box_max_x_ ||
           sensor_y < local_box_min_y_ || sensor_y > local_box_max_y_ ||
           sensor_z < local_box_min_z_ || sensor_z > local_box_max_z_))
      {
        return true;
      }

      const tf2::Vector3 global_point = sensor_to_map *
        tf2::Vector3(sensor_x, sensor_y, sensor_z);
      const double x = global_point.x();
      const double y = global_point.y();
      const double z = global_point.z();
      if (z < min_z_ || z > max_z_)
      {
        return true;
      }

      if (obstacle_revocation_tracker_ &&
          ssmi_revocation_ray_evidence_enabled_ &&
          sampled_points %
            static_cast<std::size_t>(ssmi_revocation_ray_point_stride_) == 0u)
      {
        obstacle_revocation_tracker_->collectTrackedRayEvidence(
          origin_x, origin_y, origin_z, x, y, z, ray_free_evidence);
      }

      std::uint32_t label = kInvalidSemanticLabel;
      const FieldView& label_field = semantic.valid ? semantic : fallback_label;
      const bool packed_float = semantic.valid &&
        resolved_semantic_field == "semantic_color";
      bool valid_semantic =
        readSemanticLabel(point, label_field, packed_float, label) &&
        label != kInvalidSemanticLabel;
      if (valid_semantic)
      {
        const auto remapped = semantic_label_remap_.find(label);
        if (remapped != semantic_label_remap_.end())
        {
          label = remapped->second;
        }
      }

      double confidence_value = default_confidence_;
      if (confidence.valid)
      {
        readNumber(point, confidence, confidence_value);
      }
      const float point_confidence = std::isfinite(confidence_value) ?
        clampUnit(static_cast<float>(confidence_value)) : default_confidence_;

      double point_cost = 0.0;
      const bool valid_cost = cost.valid && readNumber(point, cost, point_cost) &&
                              std::isfinite(point_cost);
      if ((!valid_semantic || point_confidence <= 0.0f) && !valid_cost)
      {
        return true;
      }

      ++valid_points;
      ScanVoxel& aggregated = scan_voxels[map_->worldToKey(x, y, z)];
      if (valid_semantic && point_confidence > 0.0f)
      {
        aggregated.label_weights[label] += point_confidence;
        aggregated.semantic_weight += point_confidence;
      }
      if (valid_cost)
      {
        aggregated.cost_sum += clampUnit(static_cast<float>(point_cost));
        aggregated.cost_weight += 1.0f;
      }
      return true;
    };

    bool valid_buffer = true;
    if (image_layout)
    {
      for (std::uint32_t pixel_y = static_cast<std::uint32_t>(pixel_offset_y_);
           pixel_y < image_height && valid_buffer;
           pixel_y += static_cast<std::uint32_t>(pixel_stride_y_))
      {
        for (std::uint32_t pixel_x = static_cast<std::uint32_t>(pixel_offset_x_);
             pixel_x < image_width;
             pixel_x += static_cast<std::uint32_t>(pixel_stride_x_))
        {
          const std::size_t point_offset = organized_cloud ?
            static_cast<std::size_t>(pixel_y) * message->row_step +
              static_cast<std::size_t>(pixel_x) * message->point_step :
            (static_cast<std::size_t>(pixel_y) * image_width + pixel_x) *
              message->point_step;
          if (!process_point(point_offset))
          {
            valid_buffer = false;
            break;
          }
        }
      }
    }
    else
    {
      for (std::size_t point_index = 0u; point_index < point_count;
           point_index += static_cast<std::size_t>(point_stride_))
      {
        const std::size_t row = point_index / message->width;
        const std::size_t column = point_index % message->width;
        const std::size_t point_offset = row * message->row_step +
                                         column * message->point_step;
        if (!process_point(point_offset))
        {
          valid_buffer = false;
          break;
        }
      }
    }
    if (!valid_buffer)
    {
      ROS_ERROR_THROTTLE(2.0, "Input PointCloud2 data buffer is malformed");
      return;
    }

    for (const auto& item : scan_voxels)
    {
      auto winning = std::max_element(
        item.second.label_weights.begin(), item.second.label_weights.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
      VoxelObservation observation;
      if (winning != item.second.label_weights.end())
      {
        observation.label = winning->first;
        // Winner weight divided by all valid semantic weight measures consensus.
        observation.semantic_confidence = clampUnit(
          winning->second / std::max(1e-6f, item.second.semantic_weight));
      }
      observation.has_traversability_cost = item.second.cost_weight > 0.0f;
      if (observation.has_traversability_cost)
      {
        observation.traversability_cost = item.second.cost_sum / item.second.cost_weight;
      }
      observation.stamp = message->header.stamp;
      map_->integrate(item.first, observation);
    }

    // Temporal decay is defined entirely in the depth-camera acquisition-time
    // domain: t(current frame) - t(last frame that hit this voxel).
    const std::size_t temporally_removed = map_->prune(message->header.stamp);
    const tf2::Quaternion sensor_rotation = sensor_to_map.getRotation().normalized();
    if (local_box_enabled_)
    {
      map_->pruneOutsideBox(
        origin_x, origin_y, origin_z,
        {{sensor_rotation.x(), sensor_rotation.y(), sensor_rotation.z(),
          sensor_rotation.w()}},
        {{local_box_min_x_, local_box_min_y_, local_box_min_z_}},
        {{local_box_max_x_, local_box_max_y_, local_box_max_z_}});
    }
    else if (local_radius_ > 0.0)
    {
      map_->pruneOutside(origin_x, origin_y, origin_z, local_radius_);
    }

    // Commit the acquisition time only after the complete local-map update for
    // the frame succeeded. All timer publications reuse exactly this stamp.
    latest_processed_frame_stamp_ = message->header.stamp;
    // End the measurement at completion of this frame's map update. Publishing
    // is intentionally measured outside this interval.
    const double elapsed_ms =
      (ros::WallTime::now() - callback_start).toSec() * 1000.0;

    {
      std::lock_guard<std::mutex> lock(sensor_origin_mutex_);
      latest_sensor_x_ = origin_x;
      latest_sensor_y_ = origin_y;
      latest_sensor_z_ = origin_z;
      latest_sensor_qx_ = sensor_rotation.x();
      latest_sensor_qy_ = sensor_rotation.y();
      latest_sensor_qz_ = sensor_rotation.z();
      latest_sensor_qw_ = sensor_rotation.w();
      have_sensor_origin_ = true;
    }
    // Materialize and commit one complete PointCloud2 after the frame update.
    // The timer only republishes this cached message.
    pending_ray_free_evidence_ = std::move(ray_free_evidence);

    const ros::WallTime postprocess_start = ros::WallTime::now();
    publish(message->header.stamp, true);
    const double postprocess_publish_ms =
      (ros::WallTime::now() - postprocess_start).toSec() * 1000.0;
    const double callback_total_ms =
      (ros::WallTime::now() - callback_start).toSec() * 1000.0;
    const double source_to_publish_ms =
      (ros::Time::now() - message->header.stamp).toSec() * 1000.0;

    ++timing_frame_count_;
    timing_elapsed_sum_ms_ += elapsed_ms;
    timing_elapsed_min_ms_ = std::min(timing_elapsed_min_ms_, elapsed_ms);
    timing_elapsed_max_ms_ = std::max(timing_elapsed_max_ms_, elapsed_ms);
    timing_postprocess_sum_ms_ += postprocess_publish_ms;
    timing_postprocess_min_ms_ =
      std::min(timing_postprocess_min_ms_, postprocess_publish_ms);
    timing_postprocess_max_ms_ =
      std::max(timing_postprocess_max_ms_, postprocess_publish_ms);
    timing_callback_total_sum_ms_ += callback_total_ms;
    timing_callback_total_min_ms_ =
      std::min(timing_callback_total_min_ms_, callback_total_ms);
    timing_callback_total_max_ms_ =
      std::max(timing_callback_total_max_ms_, callback_total_ms);
    timing_source_to_publish_sum_ms_ += source_to_publish_ms;
    timing_source_to_publish_min_ms_ =
      std::min(timing_source_to_publish_min_ms_, source_to_publish_ms);
    timing_source_to_publish_max_ms_ =
      std::max(timing_source_to_publish_max_ms_, source_to_publish_ms);
    if (timing_report_frames_ > 0 &&
        timing_frame_count_ >= static_cast<std::size_t>(timing_report_frames_))
    {
      ROS_INFO(
        "Pipeline timing over %zu input frames: "
        "map_update avg/min/max=%.2f/%.2f/%.2f ms; "
        "postprocess_publish avg/min/max=%.2f/%.2f/%.2f ms; "
        "callback_total avg/min/max=%.2f/%.2f/%.2f ms; "
        "source_to_publish avg/min/max=%.2f/%.2f/%.2f ms; "
        "latest sampled=%zu/%zu, valid=%zu, scan_voxels=%zu, "
        "body_rejected=%zu, expired=%zu, map_voxels=%zu",
        timing_frame_count_, timing_elapsed_sum_ms_ / timing_frame_count_,
        timing_elapsed_min_ms_, timing_elapsed_max_ms_,
        timing_postprocess_sum_ms_ / timing_frame_count_,
        timing_postprocess_min_ms_, timing_postprocess_max_ms_,
        timing_callback_total_sum_ms_ / timing_frame_count_,
        timing_callback_total_min_ms_, timing_callback_total_max_ms_,
        timing_source_to_publish_sum_ms_ / timing_frame_count_,
        timing_source_to_publish_min_ms_, timing_source_to_publish_max_ms_,
        sampled_points, point_count, valid_points, scan_voxels.size(),
        robot_body_rejected_points, temporally_removed, map_->size());
      timing_frame_count_ = 0u;
      timing_elapsed_sum_ms_ = 0.0;
      timing_elapsed_min_ms_ = std::numeric_limits<double>::max();
      timing_elapsed_max_ms_ = 0.0;
      timing_postprocess_sum_ms_ = 0.0;
      timing_postprocess_min_ms_ = std::numeric_limits<double>::max();
      timing_postprocess_max_ms_ = 0.0;
      timing_callback_total_sum_ms_ = 0.0;
      timing_callback_total_min_ms_ = std::numeric_limits<double>::max();
      timing_callback_total_max_ms_ = 0.0;
      timing_source_to_publish_sum_ms_ = 0.0;
      timing_source_to_publish_min_ms_ = std::numeric_limits<double>::max();
      timing_source_to_publish_max_ms_ =
        -std::numeric_limits<double>::max();
    }
  }

  void timerCallback(const ros::TimerEvent&)
  {
    if (latest_processed_frame_stamp_.isZero())
    {
      return;
    }
    // A timer is only a transport retry. It must not advance decay, mutate the
    // map, or make an old snapshot look newer than its source acquisition.
    publish(latest_processed_frame_stamp_, false);
  }

  void publish(const ros::Time& stamp, const bool commit_voxel_snapshot)
  {
    if (use_initial_pose_reference_ && !have_initial_reference_)
    {
      return;
    }

    std::vector<VoxelSnapshot> voxels = map_->snapshot();
    const std::size_t derived_height_obstacles =
      applyTerrainHeightDiscontinuityCost(
        voxels, map_->voxelSizeXY(), terrain_height_cost_config_);
    TerrainBoundaryFilterResult boundary_filter_result;
    if (terrain_boundary_filter_)
    {
      boundary_filter_result = terrain_boundary_filter_->filter(
        voxels, map_->voxelSizeXY(), map_->voxelSizeZ());
      if (commit_voxel_snapshot)
      {
        publishTerrainBoundaryDebug(boundary_filter_result, stamp);
      }
      voxels = std::move(boundary_filter_result.voxels);
    }
    // This assignment is the single publication cut-over. Every consumer
    // below—FAR's voxel_cloud, localPlanner's projected cost cloud, semantic
    // markers, and the global SemanticOctomap admission cloud—is derived from
    // this same recovered snapshot. The persistent fusion map stays raw and
    // is re-filtered on each acquisition so morphology cannot feed itself.
    const std::vector<TraversabilityColumnSnapshot> cost_columns =
      projectTraversabilityColumns(voxels);
    if (commit_voxel_snapshot && derived_height_obstacles > 0u)
    {
      ROS_DEBUG("Raised %zu missing-cost terrain voxels for height discontinuities",
                derived_height_obstacles);
    }
    if (commit_voxel_snapshot && !boundary_filter_result.relabeled.empty())
    {
      ROS_INFO_THROTTLE(
        2.0, "Static-terrain morphology recovery reclassified %zu observed "
        "static voxels using OpenCV closing and the relative-height gate",
        boundary_filter_result.relabeled.size());
    }
    visualization_msgs::Marker semantic_marker;
    semantic_marker.header.frame_id = mapFrame();
    semantic_marker.header.stamp = stamp;
    semantic_marker.ns = "semantic_voxels";
    semantic_marker.id = 0;
    semantic_marker.type = visualization_msgs::Marker::CUBE_LIST;
    semantic_marker.action = visualization_msgs::Marker::ADD;
    semantic_marker.pose.orientation.w = 1.0;
    semantic_marker.scale.x = map_->voxelSizeXY();
    semantic_marker.scale.y = map_->voxelSizeXY();
    semantic_marker.scale.z = map_->voxelSizeZ();

    visualization_msgs::Marker cost_marker = semantic_marker;
    cost_marker.ns = "traversability_voxels";
    semantic_marker.points.reserve(voxels.size());
    semantic_marker.colors.reserve(voxels.size());
    cost_marker.points.reserve(voxels.size());
    cost_marker.colors.reserve(voxels.size());

    sensor_msgs::PointCloud2 cloud;
    cloud.header.frame_id = mapFrame();
    cloud.header.stamp = stamp;
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2Fields(
      13,
      "x", 1, sensor_msgs::PointField::FLOAT32,
      "y", 1, sensor_msgs::PointField::FLOAT32,
      "z", 1, sensor_msgs::PointField::FLOAT32,
      "rgb", 1, sensor_msgs::PointField::FLOAT32,
      "label", 1, sensor_msgs::PointField::UINT32,
      "semantic_confidence", 1, sensor_msgs::PointField::FLOAT32,
      "semantic_cost", 1, sensor_msgs::PointField::FLOAT32,
      "measured_traversability", 1, sensor_msgs::PointField::FLOAT32,
      "traversability", 1, sensor_msgs::PointField::FLOAT32,
      "intensity", 1, sensor_msgs::PointField::FLOAT32,
      "observations", 1, sensor_msgs::PointField::UINT32,
      "semantic_observations", 1, sensor_msgs::PointField::UINT32,
      "traversability_observations", 1, sensor_msgs::PointField::UINT32);
    modifier.resize(voxels.size());
    sensor_msgs::PointCloud2Iterator<float> x_iterator(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> y_iterator(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> z_iterator(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> rgb_iterator(cloud, "rgb");
    sensor_msgs::PointCloud2Iterator<std::uint32_t> label_iterator(cloud, "label");
    sensor_msgs::PointCloud2Iterator<float> confidence_iterator(cloud, "semantic_confidence");
    sensor_msgs::PointCloud2Iterator<float> semantic_cost_iterator(cloud, "semantic_cost");
    sensor_msgs::PointCloud2Iterator<float> measured_cost_iterator(
      cloud, "measured_traversability");
    sensor_msgs::PointCloud2Iterator<float> cost_iterator(cloud, "traversability");
    sensor_msgs::PointCloud2Iterator<float> intensity_iterator(cloud, "intensity");
    sensor_msgs::PointCloud2Iterator<std::uint32_t> observations_iterator(cloud, "observations");
    sensor_msgs::PointCloud2Iterator<std::uint32_t> semantic_observations_iterator(
      cloud, "semantic_observations");
    sensor_msgs::PointCloud2Iterator<std::uint32_t> traversability_observations_iterator(
      cloud, "traversability_observations");

    for (const auto& voxel : voxels)
    {
      const SemanticClass semantic_class = map_->classDescription(voxel.label);
      geometry_msgs::Point point;
      point.x = voxel.x;
      point.y = voxel.y;
      point.z = voxel.z;
      semantic_marker.points.push_back(point);
      cost_marker.points.push_back(point);

      std_msgs::ColorRGBA semantic_color;
      semantic_color.r = semantic_class.red / 255.0f;
      semantic_color.g = semantic_class.green / 255.0f;
      semantic_color.b = semantic_class.blue / 255.0f;
      semantic_color.a = marker_alpha_;
      semantic_marker.colors.push_back(semantic_color);
      cost_marker.colors.push_back(costColor(voxel.traversability_cost, marker_alpha_));

      *x_iterator = static_cast<float>(voxel.x);
      *y_iterator = static_cast<float>(voxel.y);
      *z_iterator = static_cast<float>(voxel.z);
      *rgb_iterator = packedRgbFloat(semantic_class.red, semantic_class.green,
                                    semantic_class.blue);
      *label_iterator = voxel.label;
      *confidence_iterator = voxel.semantic_confidence;
      *semantic_cost_iterator = voxel.semantic_cost;
      *measured_cost_iterator = voxel.has_measured_traversability ?
        voxel.measured_traversability_cost : std::numeric_limits<float>::quiet_NaN();
      *cost_iterator = voxel.traversability_cost;
      *intensity_iterator = voxel.traversability_cost;
      *observations_iterator = voxel.observation_count;
      *semantic_observations_iterator = voxel.semantic_observation_count;
      *traversability_observations_iterator =
        voxel.traversability_observation_count;
      ++x_iterator; ++y_iterator; ++z_iterator; ++rgb_iterator; ++label_iterator;
      ++confidence_iterator; ++semantic_cost_iterator; ++measured_cost_iterator;
      ++cost_iterator; ++intensity_iterator; ++observations_iterator;
      ++semantic_observations_iterator; ++traversability_observations_iterator;
    }
    // A semantic-only voxel uses NaN for measured_traversability to indicate
    // that no direct traversability observation exists.
    cloud.is_dense = false;
    semantic_marker_pub_.publish(semantic_marker);
    cost_marker_pub_.publish(cost_marker);
    if (commit_voxel_snapshot)
    {
      std::lock_guard<std::mutex> lock(snapshot_mutex_);
      voxel_cloud_snapshot_ = cloud;
      have_voxel_cloud_snapshot_ = true;
    }
    else
    {
      std::lock_guard<std::mutex> lock(snapshot_mutex_);
      if (!have_voxel_cloud_snapshot_)
      {
        return;
      }
      cloud = voxel_cloud_snapshot_;
    }
    cloud_pub_.publish(cloud);
    publishTraversabilityCostClouds(cost_columns, stamp);
    publishAdmissionClouds(voxels, boundary_filter_result.relabeled, stamp,
                           commit_voxel_snapshot);
  }

  sensor_msgs::PointCloud2 makeAdmissionCloud(
    const std::vector<AdmissionPoint>& points, const std::string& frame_id,
    const ros::Time& stamp) const
  {
    sensor_msgs::PointCloud2 cloud;
    cloud.header.frame_id = frame_id;
    cloud.header.stamp = stamp;
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    // Keep this schema byte-for-byte compatible with grid_semantic_adapter_node.
    modifier.setPointCloud2Fields(
      5,
      "x", 1, sensor_msgs::PointField::FLOAT32,
      "y", 1, sensor_msgs::PointField::FLOAT32,
      "z", 1, sensor_msgs::PointField::FLOAT32,
      "traversability", 1, sensor_msgs::PointField::FLOAT32,
      "semantic_lable", 1, sensor_msgs::PointField::UINT32);
    modifier.resize(points.size());
    sensor_msgs::PointCloud2Iterator<float> x_iterator(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> y_iterator(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> z_iterator(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> cost_iterator(cloud, "traversability");
    sensor_msgs::PointCloud2Iterator<std::uint32_t> label_iterator(
      cloud, "semantic_lable");
    for (const AdmissionPoint& point : points)
    {
      *x_iterator = static_cast<float>(point.x);
      *y_iterator = static_cast<float>(point.y);
      *z_iterator = static_cast<float>(point.z);
      *cost_iterator = point.traversability;
      *label_iterator = point.label;
      ++x_iterator;
      ++y_iterator;
      ++z_iterator;
      ++cost_iterator;
      ++label_iterator;
    }
    cloud.is_dense = true;
    return cloud;
  }

  sensor_msgs::PointCloud2 makeSsmiAdmissionCloud(
    const std::vector<AdmissionPoint>& points, const std::string& frame_id,
    const ros::Time& stamp) const
  {
    // Match PointXYZRGBSemantic's aligned PCL wire layout exactly:
    // xyz at 0/4/8, the PointXYZ padding word at 12, rgb at 16,
    // semantic_color at 20, and an aligned 32-byte point stride.
    sensor_msgs::PointCloud2 cloud;
    cloud.header.frame_id = frame_id;
    cloud.header.stamp = stamp;
    cloud.height = 1u;
    cloud.width = static_cast<std::uint32_t>(points.size());
    cloud.fields.resize(5u);
    const std::array<std::string, 5> names =
      {{"x", "y", "z", "rgb", "semantic_color"}};
    const std::array<std::uint32_t, 5> offsets = {{0u, 4u, 8u, 16u, 20u}};
    for (std::size_t index = 0u; index < cloud.fields.size(); ++index)
    {
      cloud.fields[index].name = names[index];
      cloud.fields[index].offset = offsets[index];
      cloud.fields[index].datatype = sensor_msgs::PointField::FLOAT32;
      cloud.fields[index].count = 1u;
    }
    cloud.is_bigendian = false;
    cloud.point_step = 32u;
    cloud.row_step = cloud.point_step * cloud.width;
    cloud.data.assign(cloud.row_step, 0u);
    cloud.is_dense = true;

    for (std::size_t index = 0u; index < points.size(); ++index)
    {
      const AdmissionPoint& point = points[index];
      const float x = static_cast<float>(point.x);
      const float y = static_cast<float>(point.y);
      const float z = static_cast<float>(point.z);
      const SemanticRgb color = ssmiSemanticColor(
        point.label, point.traversability,
        ssmi_obstacle_traversability_threshold_);
      const std::uint32_t packed_bits = packSemanticRgb(color);
      float packed_color = 0.0f;
      std::memcpy(&packed_color, &packed_bits, sizeof(float));

      std::uint8_t* output = cloud.data.data() + index * cloud.point_step;
      std::memcpy(output + 0u, &x, sizeof(float));
      std::memcpy(output + 4u, &y, sizeof(float));
      std::memcpy(output + 8u, &z, sizeof(float));
      std::memcpy(output + 16u, &packed_color, sizeof(float));
      std::memcpy(output + 20u, &packed_color, sizeof(float));
    }
    return cloud;
  }

  void updateAdmissionSnapshot(
    const std::vector<VoxelSnapshot>& voxels,
    const std::vector<TerrainBoundaryRelabel>& boundary_relabels,
    const ros::Time& stamp)
  {
    AdmissionFrameResult result;
    const std::string output_frame = admission_output_frame_.empty() ?
      local_cost_frame_ : admission_output_frame_;
    tf2::Transform map_to_output;
    map_to_output.setIdentity();
    if (output_frame != mapFrame())
    {
      try
      {
        // Freeze the robot-relative coordinates at this acquisition time. A
        // timer publication reuses these values and never queries a newer pose.
        const geometry_msgs::TransformStamped transform = tf_buffer_.lookupTransform(
          output_frame, global_frame_, stamp, ros::Duration(transform_timeout_));
        const tf2::Quaternion rotation(
          transform.transform.rotation.x, transform.transform.rotation.y,
          transform.transform.rotation.z, transform.transform.rotation.w);
        const tf2::Vector3 translation(
          transform.transform.translation.x, transform.transform.translation.y,
          transform.transform.translation.z);
        map_to_output = tf2::Transform(rotation, translation);
        if (use_initial_pose_reference_)
        {
          map_to_output *= initial_reference_to_global_;
        }
      }
      catch (const tf2::TransformException& exception)
      {
        ROS_WARN_THROTTLE(
          2.0, "Local semantic admission TF error at %.6f (%s <- %s): %s",
          stamp.toSec(), output_frame.c_str(), mapFrame().c_str(), exception.what());
        latest_admission_result_ = AdmissionFrameResult();
        latest_admission_frame_ = output_frame;
        pending_ray_free_evidence_.clear();
        return;
      }
    }

    ObstacleRevocationResult revocations;
    if (obstacle_revocation_tracker_)
    {
      std::unordered_map<VoxelKey, const VoxelSnapshot*, VoxelKeyHash>
        filtered_voxels;
      filtered_voxels.reserve(voxels.size());
      for (const VoxelSnapshot& voxel : voxels)
      {
        filtered_voxels[voxel.key] = &voxel;
      }
      std::unordered_set<VoxelKey, VoxelKeyHash> reclassified_evidence;
      reclassified_evidence.reserve(boundary_relabels.size());
      for (const TerrainBoundaryRelabel& relabel : boundary_relabels)
      {
        const auto voxel_iterator = filtered_voxels.find(relabel.key);
        if (voxel_iterator == filtered_voxels.end())
        {
          continue;
        }
        const VoxelSnapshot& voxel = *voxel_iterator->second;
        reclassified_evidence.insert(obstacle_revocation_tracker_->keyFor(
          voxel.x, voxel.y, voxel.z));
      }
      revocations = obstacle_revocation_tracker_->update(
        voxels, pending_ray_free_evidence_, stamp, reclassified_evidence);
    }
    pending_ray_free_evidence_.clear();

    const auto transform_admission_point = [&map_to_output](
      const double x, const double y, const double z,
      const float traversability, const std::uint32_t label)
    {
      const tf2::Vector3 transformed = map_to_output * tf2::Vector3(x, y, z);
      AdmissionPoint point;
      point.x = transformed.x();
      point.y = transformed.y();
      point.z = transformed.z();
      point.traversability = traversability;
      point.label = label;
      return point;
    };

    result.confirmed.reserve(voxels.size());
    result.rejected_dynamic.reserve(voxels.size() / 8u);
    result.rejected_unknown.reserve(voxels.size() / 2u);
    for (const VoxelSnapshot& voxel : voxels)
    {
      if (obstacle_revocation_tracker_ &&
          revocations.revoked_keys.count(
            obstacle_revocation_tracker_->keyFor(
              voxel.x, voxel.y, voxel.z)) != 0u)
      {
        // Avoid a same-stamp race between independent insertion and revocation
        // topics. This key is omitted for the revocation frame and may be
        // inserted with its new terrain semantic on the next acquisition.
        continue;
      }
      const AdmissionPoint point = transform_admission_point(
        voxel.x, voxel.y, voxel.z, voxel.traversability_cost, voxel.label);

      if (admission_exclude_dynamic_ && isConfiguredDynamicLabel(voxel.label))
      {
        result.rejected_dynamic.push_back(point);
        continue;
      }
      result.confirmed.push_back(point);
    }
    result.revocation_candidates.reserve(revocations.candidates.size());
    for (const ObstacleRevocationPoint& candidate : revocations.candidates)
    {
      result.revocation_candidates.push_back(transform_admission_point(
        candidate.x, candidate.y, candidate.z,
        candidate.traversability, candidate.label));
    }
    result.revoked_free.reserve(revocations.revoked_free.size());
    for (const ObstacleRevocationPoint& revoked : revocations.revoked_free)
    {
      result.revoked_free.push_back(transform_admission_point(
        revoked.x, revoked.y, revoked.z,
        revoked.traversability, revoked.label));
    }
    result.revoked_reclassified.reserve(
      revocations.revoked_reclassified.size());
    for (const ObstacleRevocationPoint& revoked :
         revocations.revoked_reclassified)
    {
      result.revoked_reclassified.push_back(transform_admission_point(
        revoked.x, revoked.y, revoked.z,
        revoked.traversability, revoked.label));
    }
    if (!result.revoked_free.empty())
    {
      ROS_INFO("Confirmed %zu stale SSMI obstacle voxels as free at %.6f",
               result.revoked_free.size(), stamp.toSec());
    }
    if (!result.revoked_reclassified.empty())
    {
      ROS_INFO("Confirmed %zu stale SSMI obstacle voxels as terrain-boundary "
               "reclassifications at %.6f",
               result.revoked_reclassified.size(), stamp.toSec());
    }
    latest_admission_result_ = std::move(result);
    latest_admission_frame_ = output_frame;
  }

  void publishAdmissionClouds(
    const std::vector<VoxelSnapshot>& voxels,
    const std::vector<TerrainBoundaryRelabel>& boundary_relabels,
    const ros::Time& stamp, const bool commit_voxel_snapshot)
  {
    if (commit_voxel_snapshot)
    {
      updateAdmissionSnapshot(voxels, boundary_relabels, stamp);
    }
    const std::string& frame_id = latest_admission_frame_.empty() ?
      mapFrame() : latest_admission_frame_;
    global_admission_pub_.publish(
      makeAdmissionCloud(latest_admission_result_.confirmed, frame_id, stamp));
    if (publish_ssmi_admitted_cloud_)
    {
      // This is one complete PointCloud2 snapshot. Both color fields, the
      // robot-relative coordinates, frame, and acquisition stamp come from the
      // same committed local-map frame; timer repeats do not refresh any part.
      ssmi_admitted_pub_.publish(makeSsmiAdmissionCloud(
        latest_admission_result_.confirmed, frame_id, stamp));
    }
    candidates_pub_.publish(
      makeAdmissionCloud(latest_admission_result_.candidates, frame_id, stamp));
    confirmed_pub_.publish(
      makeAdmissionCloud(latest_admission_result_.confirmed, frame_id, stamp));
    rejected_dynamic_pub_.publish(
      makeAdmissionCloud(latest_admission_result_.rejected_dynamic, frame_id, stamp));
    rejected_unknown_pub_.publish(
      makeAdmissionCloud(latest_admission_result_.rejected_unknown, frame_id, stamp));
    rejected_rear_pub_.publish(
      makeAdmissionCloud(latest_admission_result_.rejected_rear, frame_id, stamp));
    revocation_candidates_pub_.publish(
      makeAdmissionCloud(
        latest_admission_result_.revocation_candidates, frame_id, stamp));
    revoked_free_pub_.publish(
      makeAdmissionCloud(latest_admission_result_.revoked_free, frame_id, stamp));
    revoked_reclassified_pub_.publish(
      makeAdmissionCloud(
        latest_admission_result_.revoked_reclassified, frame_id, stamp));
  }

  sensor_msgs::PointCloud2 makeTraversabilityCostCloud(
    const std::vector<TraversabilityColumnSnapshot>& columns,
    const std::string& frame_id, const ros::Time& stamp,
    const tf2::Transform* map_to_output = nullptr) const
  {
    sensor_msgs::PointCloud2 cloud;
    cloud.header.frame_id = frame_id;
    cloud.header.stamp = stamp;
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2Fields(
      4,
      "x", 1, sensor_msgs::PointField::FLOAT32,
      "y", 1, sensor_msgs::PointField::FLOAT32,
      "z", 1, sensor_msgs::PointField::FLOAT32,
      "intensity", 1, sensor_msgs::PointField::FLOAT32);
    modifier.resize(columns.size());
    sensor_msgs::PointCloud2Iterator<float> x_iterator(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> y_iterator(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> z_iterator(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> intensity_iterator(cloud, "intensity");
    for (const auto& column : columns)
    {
      tf2::Vector3 point(column.x, column.y, column.z);
      if (map_to_output != nullptr)
      {
        point = (*map_to_output) * point;
      }
      *x_iterator = static_cast<float>(point.x());
      *y_iterator = static_cast<float>(point.y());
      *z_iterator = static_cast<float>(point.z());
      *intensity_iterator = column.traversability_cost;
      ++x_iterator;
      ++y_iterator;
      ++z_iterator;
      ++intensity_iterator;
    }
    cloud.is_dense = true;
    return cloud;
  }

  void publishTraversabilityCostClouds(
    const std::vector<TraversabilityColumnSnapshot>& columns,
    const ros::Time& stamp)
  {
    if (use_initial_pose_reference_)
    {
      cost_cloud_pub_.publish(makeTraversabilityCostCloud(
        columns, global_frame_, stamp, &initial_reference_to_global_));
    }
    else
    {
      cost_cloud_pub_.publish(
        makeTraversabilityCostCloud(columns, global_frame_, stamp));
    }
    if (!publish_local_cost_cloud_)
    {
      return;
    }

    const ros::Time local_stamp = latest_processed_frame_stamp_;
    if (local_stamp.isZero())
    {
      // Reset must clear the latched body-frame cloud too. A zero stamp denotes
      // that there is currently no successfully processed acquisition.
      local_cost_cloud_pub_.publish(makeTraversabilityCostCloud(
        std::vector<TraversabilityColumnSnapshot>(), local_cost_frame_, stamp));
      return;
    }

    if (local_cost_frame_ == mapFrame())
    {
      local_cost_cloud_pub_.publish(
        makeTraversabilityCostCloud(columns, local_cost_frame_, local_stamp));
      return;
    }

    try
    {
      // Express the accumulated global map in the robot frame at the most recent
      // input-cloud time. The transform time and output header time must match.
      const geometry_msgs::TransformStamped transform = tf_buffer_.lookupTransform(
        local_cost_frame_, global_frame_, local_stamp,
        ros::Duration(transform_timeout_));
      const tf2::Quaternion rotation(
        transform.transform.rotation.x, transform.transform.rotation.y,
        transform.transform.rotation.z, transform.transform.rotation.w);
      const tf2::Vector3 translation(
        transform.transform.translation.x, transform.transform.translation.y,
        transform.transform.translation.z);
      tf2::Transform map_to_local(rotation, translation);
      if (use_initial_pose_reference_)
      {
        map_to_local *= initial_reference_to_global_;
      }
      local_cost_cloud_pub_.publish(makeTraversabilityCostCloud(
        columns, local_cost_frame_, local_stamp, &map_to_local));
    }
    catch (const tf2::TransformException& exception)
    {
      ROS_WARN_THROTTLE(2.0, "Local traversability cost cloud TF error: %s",
                        exception.what());
      // Clear a previously latched body-frame map instead of leaving stale data.
      local_cost_cloud_pub_.publish(makeTraversabilityCostCloud(
        std::vector<TraversabilityColumnSnapshot>(), local_cost_frame_, local_stamp));
    }
  }

  bool resetCallback(std_srvs::Empty::Request&, std_srvs::Empty::Response&)
  {
    // Preserve the already-published map -> map_start reference across a map
    // reset. Re-parenting map_start would invalidate downstream coordinates.
    clearRuntimeState(false);
    publish(ros::Time(), true);
    ROS_INFO("Semantic voxel map and cached local admission snapshot reset");
    return true;
  }

  void clearRuntimeState(const bool clear_reference)
  {
    map_->clear();
    latest_admission_result_ = AdmissionFrameResult();
    latest_admission_frame_.clear();
    latest_processed_frame_stamp_ = ros::Time();
    pending_ray_free_evidence_.clear();
    if (clear_reference && obstacle_revocation_tracker_)
    {
      // SSMI also starts a new mapping session after a rosbag time rewind.
      obstacle_revocation_tracker_->clear();
    }
    {
      std::lock_guard<std::mutex> snapshot_lock(snapshot_mutex_);
      have_voxel_cloud_snapshot_ = false;
      voxel_cloud_snapshot_ = sensor_msgs::PointCloud2();
    }
    {
      std::lock_guard<std::mutex> debug_lock(terrain_boundary_debug_mutex_);
      latest_terrain_boundary_debug_records_.clear();
      latest_terrain_boundary_debug_layers_.clear();
      terrain_boundary_debug_selected_layer_index_ = 0;
    }
    if (clear_reference && use_initial_pose_reference_)
    {
      have_initial_reference_ = false;
      initial_reference_stamp_ = ros::Time();
    }
    std::lock_guard<std::mutex> lock(sensor_origin_mutex_);
    have_sensor_origin_ = false;
  }

  bool saveCallback(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& response)
  {
    std::string error;
    response.success = map_->saveCsv(map_file_, error);
    response.message = response.success ? "saved map to " + map_file_ : error;
    return true;
  }

  bool loadCallback(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& response)
  {
    std::string error;
    response.success = map_->loadCsv(map_file_, error);
    response.message = response.success ? "loaded map from " + map_file_ : error;
    // A file load is not a /grids_points acquisition. The cached atomic output
    // therefore remains unchanged until the next successful input frame.
    return true;
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  tf2_ros::StaticTransformBroadcaster reference_tf_broadcaster_;
  std::unique_ptr<SemanticVoxelMap> map_;
  std::unique_ptr<ObstacleRevocationTracker> obstacle_revocation_tracker_;
  std::unique_ptr<TerrainBoundaryFilter> terrain_boundary_filter_;
  ros::Subscriber cloud_sub_;
  ros::Subscriber terrain_boundary_debug_clicked_point_sub_;
  ros::Publisher semantic_marker_pub_;
  ros::Publisher cost_marker_pub_;
  ros::Publisher cloud_pub_;
  ros::Publisher cost_cloud_pub_;
  ros::Publisher local_cost_cloud_pub_;
  ros::Publisher global_admission_pub_;
  ros::Publisher ssmi_admitted_pub_;
  ros::Publisher candidates_pub_;
  ros::Publisher confirmed_pub_;
  ros::Publisher rejected_dynamic_pub_;
  ros::Publisher rejected_unknown_pub_;
  ros::Publisher rejected_rear_pub_;
  ros::Publisher revocation_candidates_pub_;
  ros::Publisher revoked_free_pub_;
  ros::Publisher revoked_reclassified_pub_;
  ros::Publisher terrain_boundary_debug_decision_cloud_pub_;
  ros::Publisher terrain_boundary_debug_image_pub_;
  ros::ServiceServer reset_service_;
  ros::ServiceServer save_service_;
  ros::ServiceServer load_service_;
  ros::Timer publish_timer_;

  std::string input_topic_;
  std::string global_frame_;
  std::string reference_frame_ = "map_start";
  std::string semantic_field_;
  std::string confidence_field_;
  std::string cost_field_;
  std::string map_file_;
  std::string local_cost_frame_ = "wuba_base";
  std::string admission_output_frame_;
  std::string ssmi_admitted_topic_ = "/semantic_pcl/global_admitted";
  std::string latest_admission_frame_;
  std::string input_layout_ = "image";
  std::unordered_map<std::uint32_t, std::uint32_t> semantic_label_remap_;
  bool publish_local_cost_cloud_ = false;
  bool use_initial_pose_reference_ = false;
  bool initial_pose_reference_yaw_only_ = false;
  bool admission_exclude_dynamic_ = true;
  bool publish_ssmi_admitted_cloud_ = true;
  bool enable_ssmi_obstacle_revocation_ = true;
  bool ssmi_revocation_ray_evidence_enabled_ = false;
  int ssmi_revocation_ray_point_stride_ = 2;
  TerrainHeightCostConfig terrain_height_cost_config_;
  TerrainBoundaryFilterConfig terrain_boundary_filter_config_;
  SharedNavigationSemanticProfile shared_semantic_profile_;
  bool terrain_boundary_debug_opencv_enabled_ = false;
  bool terrain_boundary_debug_rviz_enabled_ = false;
  int terrain_boundary_debug_statistics_interval_frames_ = 10;
  int terrain_boundary_debug_max_samples_per_reason_ = 1;
  double terrain_boundary_debug_clicked_point_radius_ = 0.20;
  std::string terrain_boundary_debug_clicked_point_topic_ = "/clicked_point";
  std::string terrain_boundary_debug_csv_path_;
  std::ofstream terrain_boundary_debug_csv_;
  std::size_t terrain_boundary_debug_frame_count_ = 0u;
  int terrain_boundary_debug_selected_layer_index_ = 0;
  std::mutex terrain_boundary_debug_mutex_;
  std::vector<TerrainBoundaryDebugRecord>
    latest_terrain_boundary_debug_records_;
  std::vector<TerrainBoundaryLayerDebug> latest_terrain_boundary_debug_layers_;
  bool terrain_boundary_opencv_window_initialized_ = false;
  std::size_t terrain_boundary_opencv_frame_count_ = 0u;
  const std::string terrain_boundary_opencv_window_name_ =
    "Terrain boundary recovery debug";
  float default_confidence_ = 1.0f;
  float ssmi_obstacle_traversability_threshold_ = 0.75f;
  double max_range_ = 15.0;
  double min_z_ = -std::numeric_limits<double>::max();
  double max_z_ = std::numeric_limits<double>::max();
  double local_radius_ = -1.0;
  bool local_box_enabled_ = false;
  double local_box_min_x_ = -12.0;
  double local_box_max_x_ = 12.0;
  double local_box_min_y_ = -12.0;
  double local_box_max_y_ = 12.0;
  double local_box_min_z_ = -2.0;
  double local_box_max_z_ = 4.0;
  bool robot_body_exclusion_enabled_ = false;
  double robot_body_exclusion_min_x_ = -0.5;
  double robot_body_exclusion_max_x_ = 0.3;
  double robot_body_exclusion_min_y_ = -0.3;
  double robot_body_exclusion_max_y_ = 0.3;
  double transform_timeout_ = 0.2;
  double publish_rate_ = 2.0;
  float marker_alpha_ = 0.85f;
  int input_image_width_ = 640;
  int input_image_height_ = 480;
  int pixel_stride_x_ = 4;
  int pixel_stride_y_ = 4;
  int pixel_offset_x_ = 2;
  int pixel_offset_y_ = 2;
  int point_stride_ = 1;
  int timing_report_frames_ = 50;
  ros::Time latest_processed_frame_stamp_;
  ros::Time initial_reference_stamp_;
  tf2::Transform initial_reference_to_global_;
  bool have_initial_reference_ = false;
  AdmissionFrameResult latest_admission_result_;
  std::unordered_set<VoxelKey, VoxelKeyHash> pending_ray_free_evidence_;
  std::mutex snapshot_mutex_;
  sensor_msgs::PointCloud2 voxel_cloud_snapshot_;
  bool have_voxel_cloud_snapshot_ = false;
  std::size_t timing_frame_count_ = 0u;
  double timing_elapsed_sum_ms_ = 0.0;
  double timing_elapsed_min_ms_ = std::numeric_limits<double>::max();
  double timing_elapsed_max_ms_ = 0.0;
  double timing_postprocess_sum_ms_ = 0.0;
  double timing_postprocess_min_ms_ = std::numeric_limits<double>::max();
  double timing_postprocess_max_ms_ = 0.0;
  double timing_callback_total_sum_ms_ = 0.0;
  double timing_callback_total_min_ms_ = std::numeric_limits<double>::max();
  double timing_callback_total_max_ms_ = 0.0;
  double timing_source_to_publish_sum_ms_ = 0.0;
  double timing_source_to_publish_min_ms_ = std::numeric_limits<double>::max();
  double timing_source_to_publish_max_ms_ =
    -std::numeric_limits<double>::max();

  std::mutex sensor_origin_mutex_;
  bool have_sensor_origin_ = false;
  double latest_sensor_x_ = 0.0;
  double latest_sensor_y_ = 0.0;
  double latest_sensor_z_ = 0.0;
  double latest_sensor_qx_ = 0.0;
  double latest_sensor_qy_ = 0.0;
  double latest_sensor_qz_ = 0.0;
  double latest_sensor_qw_ = 1.0;
};

}  // namespace local3d_semantic_voxel_map

int main(int argc, char** argv)
{
  ros::init(argc, argv, "local_3d_semantic_voxel_map");
  try
  {
    local3d_semantic_voxel_map::SemanticVoxelMapNode node;
    ros::spin();
  }
  catch (const std::exception& exception)
  {
    ROS_FATAL("Failed to start semantic voxel map: %s", exception.what());
    return 1;
  }
  return 0;
}
