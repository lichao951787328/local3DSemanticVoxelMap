#include "local3d_semantic_voxel_map/global_semantic_admission.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace local3d_semantic_voxel_map
{

namespace
{

struct FrameAggregate
{
  std::unordered_map<std::uint32_t, double> label_weights;
  double total_weight = 0.0;
  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_z = 0.0;
  double cost_sum = 0.0;
  double cost_weight = 0.0;
};

AdmissionPoint aggregatePoint(const FrameAggregate& aggregate,
                              const std::uint32_t label)
{
  AdmissionPoint point;
  const double weight = std::max(aggregate.total_weight, 1e-9);
  point.x = aggregate.sum_x / weight;
  point.y = aggregate.sum_y / weight;
  point.z = aggregate.sum_z / weight;
  point.traversability = aggregate.cost_weight > 0.0 ?
    static_cast<float>(aggregate.cost_sum / aggregate.cost_weight) : 0.5f;
  point.label = label;
  return point;
}

}  // namespace

GlobalSemanticAdmission::GlobalSemanticAdmission(const GlobalAdmissionConfig& config)
  : config_(config)
{
  if (config_.voxel_size <= 0.0 || config_.minimum_frames == 0u ||
      config_.minimum_duration < 0.0 || config_.minimum_pose_buckets == 0u ||
      config_.pose_bucket_size <= 0.0 || config_.minimum_robot_baseline < 0.0 ||
      config_.maximum_position_stddev < 0.0 || config_.candidate_timeout < 0.0)
  {
    throw std::invalid_argument("invalid global semantic admission configuration");
  }
}

std::size_t GlobalSemanticAdmission::PoseBucketHash::operator()(
  const PoseBucket& bucket) const
{
  const std::size_t hx = std::hash<std::int32_t>()(bucket.x);
  const std::size_t hy = std::hash<std::int32_t>()(bucket.y);
  return hx ^ (hy + 0x9e3779b9u + (hx << 6u) + (hx >> 2u));
}

VoxelKey GlobalSemanticAdmission::keyFor(
  const double x, const double y, const double z) const
{
  return VoxelKey{
    static_cast<std::int32_t>(std::floor(x / config_.voxel_size)),
    static_cast<std::int32_t>(std::floor(y / config_.voxel_size)),
    static_cast<std::int32_t>(std::floor(z / config_.voxel_size))};
}

GlobalSemanticAdmission::PoseBucket GlobalSemanticAdmission::poseBucketFor(
  const double x, const double y) const
{
  return PoseBucket{
    static_cast<std::int32_t>(std::floor(x / config_.pose_bucket_size)),
    static_cast<std::int32_t>(std::floor(y / config_.pose_bucket_size))};
}

bool GlobalSemanticAdmission::isTerrain(const std::uint32_t label)
{
  return label == 0u || label == 1u || label == 9u;
}

bool GlobalSemanticAdmission::isStatic(const std::uint32_t label)
{
  return label >= 2u && label <= 8u;
}

bool GlobalSemanticAdmission::isDynamic(const std::uint32_t label)
{
  return label >= 11u && label <= 18u;
}

AdmissionPoint GlobalSemanticAdmission::candidatePoint(
  const Candidate& candidate) const
{
  AdmissionPoint point;
  const double count = static_cast<double>(std::max<std::size_t>(1u,
                                                                 candidate.frame_count));
  point.x = candidate.sum_x / count;
  point.y = candidate.sum_y / count;
  point.z = candidate.sum_z / count;
  point.traversability = static_cast<float>(candidate.cost_sum / count);
  point.label = candidate.label;
  return point;
}

double GlobalSemanticAdmission::candidateStddev(const Candidate& candidate) const
{
  if (candidate.frame_count == 0u)
  {
    return std::numeric_limits<double>::infinity();
  }
  const double count = static_cast<double>(candidate.frame_count);
  const double mean_squared_norm =
    (candidate.sum_x * candidate.sum_x + candidate.sum_y * candidate.sum_y +
     candidate.sum_z * candidate.sum_z) / (count * count);
  return std::sqrt(std::max(0.0,
    candidate.sum_squared_norm / count - mean_squared_norm));
}

double GlobalSemanticAdmission::candidateBaseline(const Candidate& candidate) const
{
  double maximum_squared = 0.0;
  for (std::size_t first = 0u; first < candidate.robot_positions.size(); ++first)
  {
    for (std::size_t second = first + 1u;
         second < candidate.robot_positions.size(); ++second)
    {
      const double dx = candidate.robot_positions[first].first -
                        candidate.robot_positions[second].first;
      const double dy = candidate.robot_positions[first].second -
                        candidate.robot_positions[second].second;
      maximum_squared = std::max(maximum_squared, dx * dx + dy * dy);
    }
  }
  return std::sqrt(maximum_squared);
}

bool GlobalSemanticAdmission::ready(const Candidate& candidate) const
{
  return candidate.frame_count >= config_.minimum_frames &&
         (candidate.last_seen - candidate.first_seen).toSec() >=
           config_.minimum_duration &&
         candidate.pose_buckets.size() >= config_.minimum_pose_buckets &&
         candidateBaseline(candidate) >= config_.minimum_robot_baseline &&
         candidateStddev(candidate) <= config_.maximum_position_stddev;
}

AdmissionFrameResult GlobalSemanticAdmission::processFrame(
  const std::vector<AdmissionObservation>& observations,
  const double robot_x, const double robot_y, const ros::Time& stamp)
{
  AdmissionFrameResult result;
  std::unordered_map<VoxelKey, FrameAggregate, VoxelKeyHash> eligible;
  std::unordered_map<VoxelKey, FrameAggregate, VoxelKeyHash> rejected_dynamic;
  std::unordered_map<VoxelKey, FrameAggregate, VoxelKeyHash> rejected_unknown;
  std::unordered_map<VoxelKey, FrameAggregate, VoxelKeyHash> rejected_rear;

  auto add = [&](std::unordered_map<VoxelKey, FrameAggregate, VoxelKeyHash>& target,
                 const AdmissionObservation& observation)
  {
    FrameAggregate& aggregate = target[keyFor(observation.x, observation.y,
                                               observation.z)];
    const double weight = observation.has_semantic ?
      std::max(1e-3, static_cast<double>(observation.semantic_confidence)) : 1.0;
    aggregate.label_weights[observation.label] += weight;
    aggregate.total_weight += weight;
    aggregate.sum_x += weight * observation.x;
    aggregate.sum_y += weight * observation.y;
    aggregate.sum_z += weight * observation.z;
    if (observation.has_traversability)
    {
      aggregate.cost_sum += weight * observation.traversability;
      aggregate.cost_weight += weight;
    }
  };

  for (const AdmissionObservation& observation : observations)
  {
    if (observation.rear_excluded)
    {
      add(rejected_rear, observation);
    }
    else if (observation.has_semantic && isDynamic(observation.label))
    {
      add(rejected_dynamic, observation);
    }
    else if (observation.has_semantic &&
             (isTerrain(observation.label) || isStatic(observation.label)))
    {
      add(eligible, observation);
    }
    else
    {
      add(rejected_unknown, observation);
    }
  }

  auto appendAggregates = [](const auto& source, std::vector<AdmissionPoint>& output)
  {
    output.reserve(source.size());
    for (const auto& item : source)
    {
      const auto winning = std::max_element(
        item.second.label_weights.begin(), item.second.label_weights.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
      output.push_back(aggregatePoint(item.second, winning->first));
    }
  };
  appendAggregates(rejected_dynamic, result.rejected_dynamic);
  appendAggregates(rejected_unknown, result.rejected_unknown);
  appendAggregates(rejected_rear, result.rejected_rear);

  for (auto iterator = candidates_.begin(); iterator != candidates_.end();)
  {
    if ((stamp - iterator->second.last_seen).toSec() > config_.candidate_timeout)
    {
      iterator = candidates_.erase(iterator);
    }
    else
    {
      ++iterator;
    }
  }

  for (const auto& item : eligible)
  {
    const auto winning = std::max_element(
      item.second.label_weights.begin(), item.second.label_weights.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
    const AdmissionPoint frame_point = aggregatePoint(item.second, winning->first);
    if (isTerrain(frame_point.label))
    {
      const auto existing = admitted_.find(item.first);
      if (existing == admitted_.end() || !existing->second.stability_confirmed)
      {
        admitted_[item.first] = AdmittedVoxel{frame_point, false};
      }
      continue;
    }

    const auto admitted = admitted_.find(item.first);
    if (admitted != admitted_.end() && admitted->second.stability_confirmed)
    {
      continue;
    }

    Candidate& candidate = candidates_[item.first];
    if (candidate.frame_count != 0u && candidate.label != frame_point.label)
    {
      candidate = Candidate();
    }
    if (candidate.frame_count == 0u)
    {
      candidate.label = frame_point.label;
      candidate.first_seen = stamp;
    }
    else if (candidate.last_seen == stamp)
    {
      // A repeated delivery of one acquisition must never satisfy the
      // "different frames" requirement.
      continue;
    }
    candidate.last_seen = stamp;
    ++candidate.frame_count;
    candidate.sum_x += frame_point.x;
    candidate.sum_y += frame_point.y;
    candidate.sum_z += frame_point.z;
    candidate.sum_squared_norm += frame_point.x * frame_point.x +
                                  frame_point.y * frame_point.y +
                                  frame_point.z * frame_point.z;
    candidate.cost_sum += frame_point.traversability;
    candidate.pose_buckets.insert(poseBucketFor(robot_x, robot_y));
    candidate.robot_positions.emplace_back(robot_x, robot_y);

    if (ready(candidate))
    {
      admitted_[item.first] = AdmittedVoxel{candidatePoint(candidate), true};
      candidates_.erase(item.first);
    }
  }

  result.candidates.reserve(candidates_.size());
  for (const auto& item : candidates_)
  {
    result.candidates.push_back(candidatePoint(item.second));
  }
  result.confirmed = admitted();
  return result;
}

void GlobalSemanticAdmission::clear()
{
  candidates_.clear();
  admitted_.clear();
}

std::vector<AdmissionPoint> GlobalSemanticAdmission::admitted() const
{
  std::vector<AdmissionPoint> output;
  output.reserve(admitted_.size());
  for (const auto& item : admitted_)
  {
    output.push_back(item.second.point);
  }
  return output;
}

std::size_t GlobalSemanticAdmission::candidateCount() const
{
  return candidates_.size();
}

}  // namespace local3d_semantic_voxel_map
