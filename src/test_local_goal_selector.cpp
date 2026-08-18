#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/Bool.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/Marker.h>

namespace
{

constexpr double kPi = 3.14159265358979323846;

struct Sample
{
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float cost = 0.0f;
};

struct Candidate
{
  Sample sample;
  double score = 0.0;
};

struct CellKey
{
  int x = 0;
  int y = 0;

  bool operator==(const CellKey& other) const
  {
    return x == other.x && y == other.y;
  }
};

struct CellKeyHash
{
  std::size_t operator()(const CellKey& key) const
  {
    const std::uint64_t packed =
      (static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.x)) << 32u) |
      static_cast<std::uint32_t>(key.y);
    return std::hash<std::uint64_t>()(packed);
  }
};

bool hasField(const sensor_msgs::PointCloud2& cloud, const std::string& name)
{
  return std::any_of(cloud.fields.begin(), cloud.fields.end(),
                     [&name](const sensor_msgs::PointField& field) {
                       return field.name == name;
                     });
}

double clamp01(const double value)
{
  return std::max(0.0, std::min(1.0, value));
}

}  // namespace

class TestLocalGoalSelector
{
public:
  TestLocalGoalSelector()
    : private_nh_("~"), tf_listener_(tf_buffer_)
  {
    private_nh_.param<std::string>("cost_cloud_topic", cost_cloud_topic_,
      "/local_3d_semantic_voxel_map/traversability_cost_cloud_wuba");
    private_nh_.param<std::string>("goal_topic", goal_topic_, "/way_point");
    private_nh_.param<std::string>("world_frame", world_frame_, "world");
    private_nh_.param("grid_resolution", grid_resolution_, 0.10);
    private_nh_.param("min_forward", min_forward_, 1.0);
    private_nh_.param("max_forward", max_forward_, 3.0);
    private_nh_.param("max_lateral", max_lateral_, 1.5);
    private_nh_.param("max_heading_deg", max_heading_deg_, 60.0);
    private_nh_.param("goal_cost_threshold", goal_cost_threshold_, 0.30);
    private_nh_.param("hard_cost_threshold", hard_cost_threshold_, 0.75);
    private_nh_.param("safety_radius", safety_radius_, 0.40);
    private_nh_.param("coverage_radius", coverage_radius_, 0.18);
    private_nh_.param("clearance_search_radius", clearance_search_radius_, 1.0);
    private_nh_.param("path_step", path_step_, 0.10);
    private_nh_.param("selection_period", selection_period_, 2.0);
    private_nh_.param("top_k", top_k_, 5);
    private_nh_.param("max_candidates_evaluated", max_candidates_evaluated_, 1000);
    int random_seed = 42;
    private_nh_.param("random_seed", random_seed, 42);
    random_engine_.seed(static_cast<std::mt19937::result_type>(random_seed));

    if (grid_resolution_ <= 0.0 || max_forward_ <= min_forward_ ||
        path_step_ <= 0.0 || safety_radius_ < 0.0 || coverage_radius_ <= 0.0)
    {
      throw std::runtime_error("Invalid local goal selector geometry parameters");
    }

    goal_pub_ = nh_.advertise<geometry_msgs::PointStamped>(goal_topic_, 1, true);
    valid_pub_ = private_nh_.advertise<std_msgs::Bool>("valid", 1, true);
    candidate_marker_pub_ = private_nh_.advertise<visualization_msgs::Marker>(
      "candidate_marker", 1, true);
    goal_marker_pub_ = private_nh_.advertise<visualization_msgs::Marker>(
      "goal_marker", 1, true);
    cloud_sub_ = nh_.subscribe(cost_cloud_topic_, 1,
      &TestLocalGoalSelector::cloudCallback, this);

    ROS_INFO("Test local goal selector: cloud=%s goal=%s world=%s, "
             "forward=[%.2f, %.2f] lateral=+/-%.2f",
             cost_cloud_topic_.c_str(), goal_topic_.c_str(), world_frame_.c_str(),
             min_forward_, max_forward_, max_lateral_);
  }

private:
  using Grid = std::unordered_map<CellKey, std::vector<std::size_t>, CellKeyHash>;

  CellKey cellFor(const double x, const double y) const
  {
    return CellKey{static_cast<int>(std::floor(x / grid_resolution_)),
                   static_cast<int>(std::floor(y / grid_resolution_))};
  }

  template <typename Function>
  void forEachNear(const double x, const double y, const double radius,
                   const std::vector<Sample>& samples, const Grid& grid,
                   Function function) const
  {
    const CellKey center = cellFor(x, y);
    const int cell_radius = static_cast<int>(std::ceil(radius / grid_resolution_));
    const double radius_squared = radius * radius;
    for (int dx = -cell_radius; dx <= cell_radius; ++dx)
    {
      for (int dy = -cell_radius; dy <= cell_radius; ++dy)
      {
        const auto found = grid.find(CellKey{center.x + dx, center.y + dy});
        if (found == grid.end())
        {
          continue;
        }
        for (const std::size_t index : found->second)
        {
          const Sample& sample = samples[index];
          const double offset_x = static_cast<double>(sample.x) - x;
          const double offset_y = static_cast<double>(sample.y) - y;
          const double squared_distance = offset_x * offset_x + offset_y * offset_y;
          if (squared_distance <= radius_squared)
          {
            function(sample, squared_distance);
          }
        }
      }
    }
  }

  bool nearestCost(const double x, const double y, const std::vector<Sample>& samples,
                   const Grid& grid, double& cost) const
  {
    bool found_point = false;
    double nearest_squared = std::numeric_limits<double>::max();
    forEachNear(x, y, coverage_radius_, samples, grid,
      [&found_point, &nearest_squared, &cost](const Sample& sample,
                                              const double squared_distance) {
        if (!found_point || squared_distance < nearest_squared)
        {
          found_point = true;
          nearest_squared = squared_distance;
          cost = sample.cost;
        }
      });
    return found_point;
  }

  bool footprintIsSafe(const double x, const double y,
                       const std::vector<Sample>& samples, const Grid& grid) const
  {
    bool safe = true;
    forEachNear(x, y, safety_radius_, samples, grid,
      [this, &safe](const Sample& sample, const double) {
        if (sample.cost >= hard_cost_threshold_)
        {
          safe = false;
        }
      });
    return safe;
  }

  bool evaluateStraightPath(const Sample& goal, const std::vector<Sample>& samples,
                            const Grid& grid, double& average_cost) const
  {
    const double distance = std::hypot(goal.x, goal.y);
    const int steps = std::max(1, static_cast<int>(std::ceil(distance / path_step_)));
    double cost_sum = 0.0;
    for (int step = 1; step <= steps; ++step)
    {
      const double ratio = static_cast<double>(step) / steps;
      const double x = ratio * goal.x;
      const double y = ratio * goal.y;
      double center_cost = 0.0;
      // Unknown center-line space is rejected instead of interpreted as free.
      if (!nearestCost(x, y, samples, grid, center_cost) ||
          center_cost >= hard_cost_threshold_ ||
          !footprintIsSafe(x, y, samples, grid))
      {
        return false;
      }
      cost_sum += center_cost;
    }
    average_cost = cost_sum / steps;
    return true;
  }

  double clearanceScore(const Sample& goal, const std::vector<Sample>& samples,
                        const Grid& grid) const
  {
    double nearest_hard_squared = clearance_search_radius_ * clearance_search_radius_;
    forEachNear(goal.x, goal.y, clearance_search_radius_, samples, grid,
      [this, &nearest_hard_squared](const Sample& sample,
                                    const double squared_distance) {
        if (sample.cost >= hard_cost_threshold_ &&
            squared_distance < nearest_hard_squared)
        {
          nearest_hard_squared = squared_distance;
        }
      });
    return clamp01(std::sqrt(nearest_hard_squared) / clearance_search_radius_);
  }

  void publishValidity(const bool valid)
  {
    std_msgs::Bool message;
    message.data = valid;
    valid_pub_.publish(message);
  }

  void publishCandidateMarker(const std::vector<Candidate>& candidates,
                              const std_msgs::Header& header)
  {
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = "test_local_goal_candidates";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::SPHERE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.08;
    marker.scale.y = 0.08;
    marker.scale.z = 0.08;
    marker.color.r = 0.1f;
    marker.color.g = 0.9f;
    marker.color.b = 0.2f;
    marker.color.a = 0.75f;
    marker.points.reserve(candidates.size());
    for (const Candidate& candidate : candidates)
    {
      geometry_msgs::Point point;
      point.x = candidate.sample.x;
      point.y = candidate.sample.y;
      point.z = candidate.sample.z + 0.10;
      marker.points.push_back(point);
    }
    candidate_marker_pub_.publish(marker);
  }

  void publishGoalMarker(const Sample& local_goal, const std_msgs::Header& header)
  {
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = "test_local_goal";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = local_goal.x;
    marker.pose.position.y = local_goal.y;
    marker.pose.position.z = local_goal.z + 0.25;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.30;
    marker.scale.y = 0.30;
    marker.scale.z = 0.30;
    marker.color.r = 1.0f;
    marker.color.g = 0.15f;
    marker.color.b = 0.05f;
    marker.color.a = 1.0f;
    goal_marker_pub_.publish(marker);
  }

  void rejectCloud(const std::string& reason)
  {
    publishValidity(false);
    ROS_WARN_THROTTLE(1.0, "Test local goal selector: %s", reason.c_str());
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud)
  {
    if (selection_period_ > 0.0 && !last_selection_stamp_.isZero() &&
        cloud->header.stamp >= last_selection_stamp_ &&
        (cloud->header.stamp - last_selection_stamp_).toSec() < selection_period_)
    {
      return;
    }
    last_selection_stamp_ = cloud->header.stamp;

    if (cloud->header.frame_id.empty() || cloud->width * cloud->height == 0u)
    {
      rejectCloud("received empty cost cloud");
      return;
    }
    if (!hasField(*cloud, "x") || !hasField(*cloud, "y") ||
        !hasField(*cloud, "z") || !hasField(*cloud, "intensity"))
    {
      rejectCloud("cost cloud requires x/y/z/intensity fields");
      return;
    }

    std::vector<Sample> samples;
    samples.reserve(static_cast<std::size_t>(cloud->width) * cloud->height);
    sensor_msgs::PointCloud2ConstIterator<float> x_iterator(*cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> y_iterator(*cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> z_iterator(*cloud, "z");
    sensor_msgs::PointCloud2ConstIterator<float> cost_iterator(*cloud, "intensity");
    for (; x_iterator != x_iterator.end();
         ++x_iterator, ++y_iterator, ++z_iterator, ++cost_iterator)
    {
      if (!std::isfinite(*x_iterator) || !std::isfinite(*y_iterator) ||
          !std::isfinite(*z_iterator) || !std::isfinite(*cost_iterator))
      {
        continue;
      }
      samples.push_back(Sample{*x_iterator, *y_iterator, *z_iterator,
                               static_cast<float>(clamp01(*cost_iterator))});
    }
    if (samples.empty())
    {
      rejectCloud("cost cloud contains no finite points");
      return;
    }

    Grid grid;
    grid.reserve(samples.size());
    for (std::size_t index = 0; index < samples.size(); ++index)
    {
      grid[cellFor(samples[index].x, samples[index].y)].push_back(index);
    }

    std::vector<std::size_t> possible_indices;
    possible_indices.reserve(samples.size());
    const double maximum_heading = max_heading_deg_ * kPi / 180.0;
    for (std::size_t index = 0; index < samples.size(); ++index)
    {
      const Sample& point = samples[index];
      if (point.x < min_forward_ || point.x > max_forward_ ||
          std::abs(point.y) > max_lateral_ ||
          std::abs(std::atan2(point.y, point.x)) > maximum_heading ||
          point.cost > goal_cost_threshold_)
      {
        continue;
      }
      possible_indices.push_back(index);
    }
    std::shuffle(possible_indices.begin(), possible_indices.end(), random_engine_);
    if (max_candidates_evaluated_ > 0 &&
        possible_indices.size() > static_cast<std::size_t>(max_candidates_evaluated_))
    {
      possible_indices.resize(static_cast<std::size_t>(max_candidates_evaluated_));
    }

    std::vector<Candidate> candidates;
    for (const std::size_t index : possible_indices)
    {
      const Sample& point = samples[index];
      if (!footprintIsSafe(point.x, point.y, samples, grid))
      {
        continue;
      }
      double path_average_cost = 0.0;
      if (!evaluateStraightPath(point, samples, grid, path_average_cost))
      {
        continue;
      }

      const double distance_score = clamp01(
        (std::hypot(point.x, point.y) - min_forward_) /
        (max_forward_ - min_forward_));
      const double risk_score = 1.0 - clamp01(point.cost);
      const double path_risk_score = 1.0 - clamp01(path_average_cost);
      const double heading_score = 1.0 - clamp01(
        std::abs(std::atan2(point.y, point.x)) / maximum_heading);
      const double score = 0.25 * distance_score + 0.25 * risk_score +
                           0.25 * clearanceScore(point, samples, grid) +
                           0.15 * path_risk_score + 0.10 * heading_score;
      candidates.push_back(Candidate{point, score});
    }

    std::sort(candidates.begin(), candidates.end(),
      [](const Candidate& left, const Candidate& right) {
        return left.score > right.score;
      });
    publishCandidateMarker(candidates, cloud->header);
    if (candidates.empty())
    {
      rejectCloud("no covered, collision-free candidate has a safe straight path");
      return;
    }

    const std::size_t choice_count = std::min(
      candidates.size(), static_cast<std::size_t>(std::max(1, top_k_)));
    std::uniform_int_distribution<std::size_t> distribution(0u, choice_count - 1u);
    const Candidate& selected = candidates[distribution(random_engine_)];

    try
    {
      const geometry_msgs::TransformStamped transform = tf_buffer_.lookupTransform(
        world_frame_, cloud->header.frame_id, cloud->header.stamp,
        ros::Duration(0.2));
      const tf2::Quaternion rotation(
        transform.transform.rotation.x, transform.transform.rotation.y,
        transform.transform.rotation.z, transform.transform.rotation.w);
      const tf2::Vector3 translation(
        transform.transform.translation.x, transform.transform.translation.y,
        transform.transform.translation.z);
      const tf2::Vector3 local(selected.sample.x, selected.sample.y, selected.sample.z);
      const tf2::Vector3 world = tf2::Transform(rotation, translation) * local;

      geometry_msgs::PointStamped goal;
      goal.header.stamp = cloud->header.stamp;
      goal.header.frame_id = world_frame_;
      goal.point.x = world.x();
      goal.point.y = world.y();
      goal.point.z = world.z();
      goal_pub_.publish(goal);
      publishValidity(true);
      publishGoalMarker(selected.sample, cloud->header);
      ROS_INFO("Test local goal: local=(%.2f, %.2f), world=(%.2f, %.2f), "
               "cost=%.3f score=%.3f candidates=%zu",
               selected.sample.x, selected.sample.y, goal.point.x, goal.point.y,
               selected.sample.cost, selected.score, candidates.size());
    }
    catch (const tf2::TransformException& exception)
    {
      rejectCloud(std::string("goal TF failed: ") + exception.what());
    }
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  ros::Subscriber cloud_sub_;
  ros::Publisher goal_pub_;
  ros::Publisher valid_pub_;
  ros::Publisher candidate_marker_pub_;
  ros::Publisher goal_marker_pub_;

  std::string cost_cloud_topic_;
  std::string goal_topic_;
  std::string world_frame_;
  double grid_resolution_ = 0.10;
  double min_forward_ = 1.0;
  double max_forward_ = 3.0;
  double max_lateral_ = 1.5;
  double max_heading_deg_ = 60.0;
  double goal_cost_threshold_ = 0.30;
  double hard_cost_threshold_ = 0.75;
  double safety_radius_ = 0.40;
  double coverage_radius_ = 0.18;
  double clearance_search_radius_ = 1.0;
  double path_step_ = 0.10;
  double selection_period_ = 2.0;
  int top_k_ = 5;
  int max_candidates_evaluated_ = 1000;
  ros::Time last_selection_stamp_;
  std::mt19937 random_engine_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "test_local_goal_selector");
  try
  {
    TestLocalGoalSelector selector;
    ros::spin();
  }
  catch (const std::exception& exception)
  {
    ROS_FATAL("Test local goal selector failed: %s", exception.what());
    return 1;
  }
  return 0;
}
