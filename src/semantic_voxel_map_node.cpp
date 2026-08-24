#include "local3d_semantic_voxel_map/semantic_voxel_map.hpp"
#include "local3d_semantic_voxel_map/global_semantic_admission.hpp"
#include "local3d_semantic_voxel_map/obstacle_revocation.hpp"
#include "local3d_semantic_voxel_map/ssmi_semantic_encoding.hpp"
#include "local3d_semantic_voxel_map/terrain_boundary_filter.hpp"

#include <geometry_msgs/Point.h>
#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
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
#include <cstdint>
#include <cstring>
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
    revocation_config.dynamic_labels = loadLabelList(
      private_nh_, "ssmi_revocation_dynamic_labels",
      revocation_config.dynamic_labels);
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
    int boundary_obstacle_label = static_cast<int>(
      boundary_filter_config.obstacle_label);
    private_nh_.param("terrain_boundary_obstacle_label",
                      boundary_obstacle_label, boundary_obstacle_label);
    if (boundary_obstacle_label < 0)
    {
      throw std::runtime_error(
        "~terrain_boundary_obstacle_label must be non-negative");
    }
    boundary_filter_config.obstacle_label =
      static_cast<std::uint32_t>(boundary_obstacle_label);
    boundary_filter_config.terrain_labels = loadLabelList(
      private_nh_, "terrain_boundary_terrain_labels",
      boundary_filter_config.terrain_labels);
    int boundary_opening_radius = static_cast<int>(
      boundary_filter_config.opening_radius_cells);
    private_nh_.param("terrain_boundary_opening_radius_cells",
                      boundary_opening_radius, boundary_opening_radius);
    if (boundary_opening_radius < 0)
    {
      throw std::runtime_error(
        "~terrain_boundary_opening_radius_cells must be non-negative");
    }
    boundary_filter_config.opening_radius_cells =
      static_cast<std::size_t>(boundary_opening_radius);
    private_nh_.param("terrain_boundary_neighborhood_radius",
                      boundary_filter_config.neighborhood_radius, 0.30);
    private_nh_.param("terrain_boundary_vertical_tolerance",
                      boundary_filter_config.vertical_tolerance, 0.15);
    int boundary_minimum_neighbors = static_cast<int>(
      boundary_filter_config.minimum_terrain_neighbors);
    private_nh_.param("terrain_boundary_minimum_terrain_neighbors",
                      boundary_minimum_neighbors, boundary_minimum_neighbors);
    if (boundary_minimum_neighbors <= 0)
    {
      throw std::runtime_error(
        "~terrain_boundary_minimum_terrain_neighbors must be positive");
    }
    boundary_filter_config.minimum_terrain_neighbors =
      static_cast<std::size_t>(boundary_minimum_neighbors);
    int boundary_minimum_labels = static_cast<int>(
      boundary_filter_config.minimum_distinct_terrain_labels);
    private_nh_.param("terrain_boundary_minimum_distinct_terrain_labels",
                      boundary_minimum_labels, boundary_minimum_labels);
    if (boundary_minimum_labels <= 0)
    {
      throw std::runtime_error(
        "~terrain_boundary_minimum_distinct_terrain_labels must be positive");
    }
    boundary_filter_config.minimum_distinct_terrain_labels =
      static_cast<std::size_t>(boundary_minimum_labels);
    private_nh_.param("terrain_boundary_minimum_terrain_ratio",
                      boundary_filter_config.minimum_terrain_ratio, 0.70);
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
    cloud_sub_ = nh_.subscribe(input_topic_, 1,
                               &SemanticVoxelMapNode::cloudCallback, this);
    reset_service_ = private_nh_.advertiseService(
      "reset", &SemanticVoxelMapNode::resetCallback, this);
    save_service_ = private_nh_.advertiseService(
      "save_map", &SemanticVoxelMapNode::saveCallback, this);
    load_service_ = private_nh_.advertiseService(
      "load_map", &SemanticVoxelMapNode::loadCallback, this);

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
    ROS_INFO("SSMI obstacle revocation: %s, ray evidence=%s, ray stride=%d",
             enable_ssmi_obstacle_revocation_ ? "enabled" : "disabled",
             ssmi_revocation_ray_evidence_enabled_ ? "enabled" : "disabled",
             ssmi_revocation_ray_point_stride_);
    ROS_INFO("Terrain-boundary obstacle cleanup: %s, obstacle=%u, "
             "terrain_classes=%zu, opening=%zu cell(s), neighborhood=%.2f m, "
             "vertical=%.2f m, terrain_ratio>=%.2f, distinct_classes>=%zu",
             terrain_boundary_filter_config_.enabled ? "enabled" : "disabled",
             terrain_boundary_filter_config_.obstacle_label,
             terrain_boundary_filter_config_.terrain_labels.size(),
             terrain_boundary_filter_config_.opening_radius_cells,
             terrain_boundary_filter_config_.neighborhood_radius,
             terrain_boundary_filter_config_.vertical_tolerance,
             terrain_boundary_filter_config_.minimum_terrain_ratio,
             terrain_boundary_filter_config_.minimum_distinct_terrain_labels);
    ROS_INFO("Missing-cost terrain height inference: %s, dz>%.2f m + %.1e "
             "within %.2f m -> cost %.2f (terrain labels 0/1/9 only)",
             terrain_height_cost_config_.enabled ? "enabled" : "disabled",
             terrain_height_cost_config_.height_difference_threshold,
             terrain_height_cost_config_.comparison_epsilon,
             terrain_height_cost_config_.neighborhood_radius,
             terrain_height_cost_config_.obstacle_cost);
  }

private:
  const std::string& mapFrame() const
  {
    return use_initial_pose_reference_ ? reference_frame_ : global_frame_;
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
      voxels = std::move(boundary_filter_result.voxels);
    }
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
        2.0, "Terrain-boundary cleanup reclassified %zu thin label-%u voxels "
        "using neighboring terrain classes",
        boundary_filter_result.relabeled.size(),
        terrain_boundary_filter_config_.obstacle_label);
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

      const LocalAdmissionDecision decision = classifyLocalAdmissionVoxel(
        voxel.label, admission_exclude_dynamic_);
      if (decision == LocalAdmissionDecision::RejectedDynamic)
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
