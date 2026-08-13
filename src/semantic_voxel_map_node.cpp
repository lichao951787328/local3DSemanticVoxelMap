#include "local3d_semantic_voxel_map/semantic_voxel_map.hpp"

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
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/Marker.h>
#include <xmlrpcpp/XmlRpcValue.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
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

}  // namespace

class SemanticVoxelMapNode
{
public:
  SemanticVoxelMapNode()
    : private_nh_("~"), tf_listener_(tf_buffer_)
  {
    SemanticVoxelMapConfig map_config;
    private_nh_.param("voxel_size", map_config.voxel_size, 0.10);
    private_nh_.param("decay_seconds", map_config.decay_seconds, 0.5);
    int max_voxels = 500000;
    private_nh_.param("max_voxels", max_voxels, 500000);
    map_config.max_voxels = max_voxels <= 0 ? 0u : static_cast<std::size_t>(max_voxels);
    private_nh_.param("unknown_cost", map_config.unknown_cost, 0.5f);
    private_nh_.param("semantic_cost_weight", map_config.semantic_cost_weight, 0.8f);
    private_nh_.param("cost_rise_alpha", map_config.cost_rise_alpha, 0.65f);
    private_nh_.param("cost_fall_alpha", map_config.cost_fall_alpha, 0.15f);
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
    private_nh_.param<std::string>("semantic_field", semantic_field_,
                                   "semantic_color");
    private_nh_.param<std::string>("confidence_field", confidence_field_,
                                   "semantic_confidence");
    private_nh_.param<std::string>("cost_field", cost_field_, "traversability");
    private_nh_.param<std::string>("map_file", map_file_,
                                   "/tmp/local3d_semantic_voxel_map.csv");
    private_nh_.param("default_semantic_confidence", default_confidence_, 1.0f);
    private_nh_.param("max_range", max_range_, 15.0);
    private_nh_.param("min_z", min_z_, -std::numeric_limits<double>::max());
    private_nh_.param("max_z", max_z_, std::numeric_limits<double>::max());
    private_nh_.param("local_radius", local_radius_, -1.0);
    private_nh_.param("transform_timeout", transform_timeout_, 0.2);
    private_nh_.param("publish_rate", publish_rate_, 2.0);
    private_nh_.param("marker_alpha", marker_alpha_, 0.85f);
    private_nh_.param("input_image_width", input_image_width_, 640);
    private_nh_.param("input_image_height", input_image_height_, 480);
    private_nh_.param("pixel_stride_x", pixel_stride_x_, 4);
    private_nh_.param("pixel_stride_y", pixel_stride_y_, 4);
    private_nh_.param("pixel_offset_x", pixel_offset_x_, 2);
    private_nh_.param("pixel_offset_y", pixel_offset_y_, 2);
    private_nh_.param("timing_report_frames", timing_report_frames_, 50);
    if (input_image_width_ <= 0 || input_image_height_ <= 0 ||
        pixel_stride_x_ <= 0 || pixel_stride_y_ <= 0)
    {
      throw std::runtime_error(
        "input image dimensions and pixel sampling strides must be positive");
    }
    pixel_offset_x_ = std::max(0, std::min(pixel_offset_x_, pixel_stride_x_ - 1));
    pixel_offset_y_ = std::max(0, std::min(pixel_offset_y_, pixel_stride_y_ - 1));

    semantic_marker_pub_ = private_nh_.advertise<visualization_msgs::Marker>(
      "semantic_voxels", 1, true);
    cost_marker_pub_ = private_nh_.advertise<visualization_msgs::Marker>(
      "traversability_voxels", 1, true);
    cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>("voxel_cloud", 1, true);
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
    ROS_INFO("Semantic voxel map: input=%s frame=%s voxel_size=%.3f, "
             "pixel sampling=%dx%d stride=(%d,%d) offset=(%d,%d)",
             input_topic_.c_str(), global_frame_.c_str(), map_->voxelSize(),
             input_image_width_, input_image_height_, pixel_stride_x_,
             pixel_stride_y_, pixel_offset_x_, pixel_offset_y_);
  }

private:
  struct ScanVoxel
  {
    std::unordered_map<std::uint32_t, float> label_weights;
    float sample_count = 0.0f;
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
      ROS_WARN_THROTTLE(2.0,
        "Dropping out-of-order depth cloud (stamp %.6f < latest %.6f)",
        message->header.stamp.toSec(), latest_processed_frame_stamp_.toSec());
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
    const FieldView semantic = findField(*message, semantic_field_);
    FieldView fallback_label;
    if (!semantic.valid && semantic_field_ != "label")
    {
      fallback_label = findField(*message, "label");
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
    const std::uint32_t image_width = organized_cloud ? message->width :
      static_cast<std::uint32_t>(input_image_width_);
    const std::uint32_t image_height = organized_cloud ? message->height :
      static_cast<std::uint32_t>(input_image_height_);
    if (!organized_cloud && point_count !=
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
    const double origin_x = translation.x();
    const double origin_y = translation.y();
    const double origin_z = translation.z();
    const double max_range_squared = max_range_ > 0.0 ?
      max_range_ * max_range_ : std::numeric_limits<double>::max();
    std::unordered_map<VoxelKey, ScanVoxel, VoxelKeyHash> scan_voxels;
    const std::size_t sampled_capacity =
      (image_width + static_cast<std::uint32_t>(pixel_stride_x_) - 1u) /
      static_cast<std::uint32_t>(pixel_stride_x_) *
      ((image_height + static_cast<std::uint32_t>(pixel_stride_y_) - 1u) /
       static_cast<std::uint32_t>(pixel_stride_y_));
    scan_voxels.reserve(sampled_capacity / 2u + 1u);
    std::size_t sampled_points = 0u;
    std::size_t valid_points = 0u;

    for (std::uint32_t pixel_y = static_cast<std::uint32_t>(pixel_offset_y_);
         pixel_y < image_height;
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
        if (point_offset + message->point_step > message->data.size())
        {
          ROS_ERROR_THROTTLE(2.0, "Input PointCloud2 data buffer is malformed");
          return;
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
          continue;
        }
        if (sensor_x * sensor_x + sensor_y * sensor_y + sensor_z * sensor_z >
            max_range_squared)
        {
          continue;
        }

        const tf2::Vector3 global_point = sensor_to_global *
          tf2::Vector3(sensor_x, sensor_y, sensor_z);
        const double x = global_point.x();
        const double y = global_point.y();
        const double z = global_point.z();
        if (z < min_z_ || z > max_z_)
        {
          continue;
        }

        std::uint32_t label = kInvalidSemanticLabel;
        const FieldView& label_field = semantic.valid ? semantic : fallback_label;
        const bool packed_float = semantic.valid && semantic_field_ == "semantic_color";
        if (!readSemanticLabel(point, label_field, packed_float, label))
        {
          continue;
        }

        double confidence_value = default_confidence_;
        if (confidence.valid)
        {
          readNumber(point, confidence, confidence_value);
        }
        const float point_confidence = std::isfinite(confidence_value) ?
          clampUnit(static_cast<float>(confidence_value)) : default_confidence_;
        if (point_confidence <= 0.0f)
        {
          continue;
        }
        ++valid_points;

        ScanVoxel& aggregated = scan_voxels[map_->worldToKey(x, y, z)];
        aggregated.label_weights[label] += point_confidence;
        aggregated.sample_count += 1.0f;
        double point_cost = 0.0;
        if (cost.valid && readNumber(point, cost, point_cost) && std::isfinite(point_cost))
        {
          aggregated.cost_sum += clampUnit(static_cast<float>(point_cost)) * point_confidence;
          aggregated.cost_weight += point_confidence;
        }
      }
    }

    for (const auto& item : scan_voxels)
    {
      auto winning = std::max_element(
        item.second.label_weights.begin(), item.second.label_weights.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
      if (winning == item.second.label_weights.end())
      {
        continue;
      }
      VoxelObservation observation;
      observation.label = winning->first;
      // This includes both sensor confidence and within-voxel class consensus.
      observation.semantic_confidence = clampUnit(
        winning->second / std::max(1.0f, item.second.sample_count));
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
      have_sensor_origin_ = true;
    }
    ++timing_frame_count_;
    timing_elapsed_sum_ms_ += elapsed_ms;
    timing_elapsed_min_ms_ = std::min(timing_elapsed_min_ms_, elapsed_ms);
    timing_elapsed_max_ms_ = std::max(timing_elapsed_max_ms_, elapsed_ms);
    if (timing_report_frames_ > 0 &&
        timing_frame_count_ >= static_cast<std::size_t>(timing_report_frames_))
    {
      ROS_INFO(
        "Map update timing over %zu depth frames: avg=%.2f ms, min=%.2f ms, "
        "max=%.2f ms; latest sampled=%zu/%zu, valid=%zu, scan_voxels=%zu, "
        "expired=%zu, map_voxels=%zu",
        timing_frame_count_, timing_elapsed_sum_ms_ / timing_frame_count_,
        timing_elapsed_min_ms_, timing_elapsed_max_ms_, sampled_points,
        point_count, valid_points, scan_voxels.size(), temporally_removed,
        map_->size());
      timing_frame_count_ = 0u;
      timing_elapsed_sum_ms_ = 0.0;
      timing_elapsed_min_ms_ = std::numeric_limits<double>::max();
      timing_elapsed_max_ms_ = 0.0;
    }
  }

  void timerCallback(const ros::TimerEvent&)
  {
    const ros::Time now = ros::Time::now();
    if (local_radius_ > 0.0)
    {
      std::lock_guard<std::mutex> lock(sensor_origin_mutex_);
      if (have_sensor_origin_)
      {
        map_->pruneOutside(latest_sensor_x_, latest_sensor_y_, latest_sensor_z_,
                           local_radius_);
      }
    }
    publish(now);
  }

  void publish(const ros::Time& stamp)
  {
    const std::vector<VoxelSnapshot> voxels = map_->snapshot();
    visualization_msgs::Marker semantic_marker;
    semantic_marker.header.frame_id = global_frame_;
    semantic_marker.header.stamp = stamp;
    semantic_marker.ns = "semantic_voxels";
    semantic_marker.id = 0;
    semantic_marker.type = visualization_msgs::Marker::CUBE_LIST;
    semantic_marker.action = visualization_msgs::Marker::ADD;
    semantic_marker.pose.orientation.w = 1.0;
    semantic_marker.scale.x = map_->voxelSize();
    semantic_marker.scale.y = map_->voxelSize();
    semantic_marker.scale.z = map_->voxelSize();

    visualization_msgs::Marker cost_marker = semantic_marker;
    cost_marker.ns = "traversability_voxels";
    semantic_marker.points.reserve(voxels.size());
    semantic_marker.colors.reserve(voxels.size());
    cost_marker.points.reserve(voxels.size());
    cost_marker.colors.reserve(voxels.size());

    sensor_msgs::PointCloud2 cloud;
    cloud.header.frame_id = global_frame_;
    cloud.header.stamp = stamp;
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2Fields(
      9,
      "x", 1, sensor_msgs::PointField::FLOAT32,
      "y", 1, sensor_msgs::PointField::FLOAT32,
      "z", 1, sensor_msgs::PointField::FLOAT32,
      "rgb", 1, sensor_msgs::PointField::FLOAT32,
      "label", 1, sensor_msgs::PointField::UINT32,
      "semantic_confidence", 1, sensor_msgs::PointField::FLOAT32,
      "traversability", 1, sensor_msgs::PointField::FLOAT32,
      "intensity", 1, sensor_msgs::PointField::FLOAT32,
      "observations", 1, sensor_msgs::PointField::UINT32);
    modifier.resize(voxels.size());
    sensor_msgs::PointCloud2Iterator<float> x_iterator(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> y_iterator(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> z_iterator(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> rgb_iterator(cloud, "rgb");
    sensor_msgs::PointCloud2Iterator<std::uint32_t> label_iterator(cloud, "label");
    sensor_msgs::PointCloud2Iterator<float> confidence_iterator(cloud, "semantic_confidence");
    sensor_msgs::PointCloud2Iterator<float> cost_iterator(cloud, "traversability");
    sensor_msgs::PointCloud2Iterator<float> intensity_iterator(cloud, "intensity");
    sensor_msgs::PointCloud2Iterator<std::uint32_t> observations_iterator(cloud, "observations");

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
      *cost_iterator = voxel.traversability_cost;
      *intensity_iterator = voxel.traversability_cost;
      *observations_iterator = voxel.observation_count;
      ++x_iterator; ++y_iterator; ++z_iterator; ++rgb_iterator; ++label_iterator;
      ++confidence_iterator; ++cost_iterator; ++intensity_iterator;
      ++observations_iterator;
    }
    cloud.is_dense = true;
    semantic_marker_pub_.publish(semantic_marker);
    cost_marker_pub_.publish(cost_marker);
    cloud_pub_.publish(cloud);
  }

  bool resetCallback(std_srvs::Empty::Request&, std_srvs::Empty::Response&)
  {
    map_->clear();
    publish(ros::Time::now());
    ROS_INFO("Semantic voxel map reset");
    return true;
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
    if (response.success)
    {
      publish(ros::Time::now());
    }
    return true;
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::unique_ptr<SemanticVoxelMap> map_;
  ros::Subscriber cloud_sub_;
  ros::Publisher semantic_marker_pub_;
  ros::Publisher cost_marker_pub_;
  ros::Publisher cloud_pub_;
  ros::ServiceServer reset_service_;
  ros::ServiceServer save_service_;
  ros::ServiceServer load_service_;
  ros::Timer publish_timer_;

  std::string input_topic_;
  std::string global_frame_;
  std::string semantic_field_;
  std::string confidence_field_;
  std::string cost_field_;
  std::string map_file_;
  float default_confidence_ = 1.0f;
  double max_range_ = 15.0;
  double min_z_ = -std::numeric_limits<double>::max();
  double max_z_ = std::numeric_limits<double>::max();
  double local_radius_ = -1.0;
  double transform_timeout_ = 0.2;
  double publish_rate_ = 2.0;
  float marker_alpha_ = 0.85f;
  int input_image_width_ = 640;
  int input_image_height_ = 480;
  int pixel_stride_x_ = 4;
  int pixel_stride_y_ = 4;
  int pixel_offset_x_ = 2;
  int pixel_offset_y_ = 2;
  int timing_report_frames_ = 50;
  ros::Time latest_processed_frame_stamp_;
  std::size_t timing_frame_count_ = 0u;
  double timing_elapsed_sum_ms_ = 0.0;
  double timing_elapsed_min_ms_ = std::numeric_limits<double>::max();
  double timing_elapsed_max_ms_ = 0.0;

  std::mutex sensor_origin_mutex_;
  bool have_sensor_origin_ = false;
  double latest_sensor_x_ = 0.0;
  double latest_sensor_y_ = 0.0;
  double latest_sensor_z_ = 0.0;
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
