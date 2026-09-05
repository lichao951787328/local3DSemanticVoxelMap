#include "local3d_semantic_voxel_map/obstacle_revocation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace local3d_semantic_voxel_map
{

ObstacleRevocationTracker::ObstacleRevocationTracker(
  const ObstacleRevocationConfig& config)
  : config_(config),
    terrain_labels_(config.terrain_labels.begin(), config.terrain_labels.end()),
    obstacle_labels_(config.obstacle_labels.begin(), config.obstacle_labels.end()),
    ambiguous_obstacle_labels_(config.ambiguous_obstacle_labels.begin(),
                               config.ambiguous_obstacle_labels.end()),
    dynamic_labels_(config.dynamic_labels.begin(), config.dynamic_labels.end())
{
  if (config_.minimum_free_evidence == 0.0)
  {
    config_.minimum_free_evidence =
      static_cast<double>(config_.minimum_free_frames);
  }
  if (config_.voxel_size <= 0.0 || config_.minimum_free_frames == 0u ||
      config_.minimum_free_duration < 0.0 ||
      config_.minimum_free_evidence < 0.0 ||
      config_.free_evidence_decay_per_second < 0.0 ||
      config_.free_max_traversability < 0.0f ||
      config_.free_max_traversability > 1.0f ||
      config_.obstacle_min_traversability < 0.0f ||
      config_.obstacle_min_traversability > 1.0f ||
      config_.ambiguous_obstacle_reset_min_traversability < 0.0f ||
      config_.ambiguous_obstacle_reset_min_traversability > 1.0f ||
      config_.minimum_semantic_confidence < 0.0f ||
      config_.minimum_semantic_confidence > 1.0f ||
      config_.ray_endpoint_margin < 0.0 ||
      config_.terrain_labels.empty() || config_.obstacle_labels.empty())
  {
    throw std::invalid_argument("invalid obstacle revocation configuration");
  }
  for (const std::uint32_t label : terrain_labels_)
  {
    if (obstacle_labels_.count(label) != 0u ||
        dynamic_labels_.count(label) != 0u)
    {
      throw std::invalid_argument("obstacle revocation semantic roles overlap");
    }
  }
  for (const std::uint32_t label : obstacle_labels_)
  {
    if (dynamic_labels_.count(label) != 0u)
    {
      throw std::invalid_argument("obstacle revocation semantic roles overlap");
    }
  }
}

VoxelKey ObstacleRevocationTracker::keyFor(
  const double x, const double y, const double z) const
{
  return VoxelKey{
    static_cast<std::int32_t>(std::floor(x / config_.voxel_size)),
    static_cast<std::int32_t>(std::floor(y / config_.voxel_size)),
    static_cast<std::int32_t>(std::floor(z / config_.voxel_size))};
}

bool ObstacleRevocationTracker::isTracked(const VoxelKey& key) const
{
  return tracked_.count(key) != 0u;
}

std::size_t ObstacleRevocationTracker::trackedCount() const
{
  return tracked_.size();
}

void ObstacleRevocationTracker::collectTrackedRayEvidence(
  const double origin_x, const double origin_y, const double origin_z,
  const double endpoint_x, const double endpoint_y, const double endpoint_z,
  std::unordered_set<VoxelKey, VoxelKeyHash>& evidence) const
{
  if (tracked_.empty())
  {
    return;
  }
  const double dx = endpoint_x - origin_x;
  const double dy = endpoint_y - origin_y;
  const double dz = endpoint_z - origin_z;
  const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
  const double free_distance = distance - config_.ray_endpoint_margin;
  if (!std::isfinite(distance) || free_distance <= 0.0)
  {
    return;
  }

  // Half-voxel sampling cannot jump over a voxel along any axis. Only keys
  // already sent as obstacles are retained, so memory is bounded by the
  // persistent-obstacle mirror rather than by the full observed ray volume.
  const double step = config_.voxel_size * 0.5;
  const std::size_t samples = static_cast<std::size_t>(
    std::ceil(free_distance / step));
  const double inverse_distance = 1.0 / distance;
  for (std::size_t index = 0u; index <= samples; ++index)
  {
    const double travelled = std::min(free_distance, index * step);
    const double ratio = travelled * inverse_distance;
    const VoxelKey key = keyFor(
      origin_x + ratio * dx,
      origin_y + ratio * dy,
      origin_z + ratio * dz);
    if (isTracked(key))
    {
      evidence.insert(key);
    }
  }
}

bool ObstacleRevocationTracker::isDynamic(const std::uint32_t label) const
{
  return dynamic_labels_.count(label) != 0u;
}

bool ObstacleRevocationTracker::isTerrain(const std::uint32_t label) const
{
  return terrain_labels_.count(label) != 0u;
}

bool ObstacleRevocationTracker::isSemanticObstacle(
  const std::uint32_t label) const
{
  return obstacle_labels_.count(label) != 0u;
}

bool ObstacleRevocationTracker::isAmbiguousObstacle(
  const std::uint32_t label) const
{
  return ambiguous_obstacle_labels_.count(label) != 0u;
}

bool ObstacleRevocationTracker::isObstacle(const VoxelSnapshot& voxel) const
{
  if (isDynamic(voxel.label))
  {
    return false;
  }
  return isSemanticObstacle(voxel.label) ||
    (std::isfinite(voxel.traversability_cost) &&
     voxel.traversability_cost >= config_.obstacle_min_traversability);
}

bool ObstacleRevocationTracker::isStrongObstacleContradiction(
  const VoxelSnapshot& voxel) const
{
  if (isDynamic(voxel.label))
  {
    return true;
  }
  if (!isObstacle(voxel))
  {
    return false;
  }
  if (!isAmbiguousObstacle(voxel.label))
  {
    return true;
  }
  // Missing or invalid measured geometry stays conservative. A low measured
  // cost on an ambiguous semantic label is neutral: it neither adds free
  // evidence nor destroys evidence obtained from a stronger geometric test.
  return !voxel.has_measured_traversability ||
    !std::isfinite(voxel.measured_traversability_cost) ||
    voxel.measured_traversability_cost >=
      config_.ambiguous_obstacle_reset_min_traversability;
}

ObstacleRevocationPoint ObstacleRevocationTracker::makePoint(
  const VoxelKey& key, const TrackedObstacle& obstacle) const
{
  ObstacleRevocationPoint point;
  point.key = key;
  point.x = obstacle.x;
  point.y = obstacle.y;
  point.z = obstacle.z;
  point.label = obstacle.label;
  point.traversability = obstacle.traversability;
  point.evidence_frames = obstacle.free_frames;
  point.evidence_score = obstacle.free_evidence;
  return point;
}

ObstacleRevocationResult ObstacleRevocationTracker::update(
  const std::vector<VoxelSnapshot>& current_voxels,
  const std::unordered_set<VoxelKey, VoxelKeyHash>& ray_free_evidence,
  const ros::Time& stamp,
  const std::unordered_set<VoxelKey, VoxelKeyHash>& reclassified_evidence)
{
  ObstacleRevocationResult result;
  if (stamp.isZero())
  {
    return result;
  }
  if (!latest_stamp_.isZero() && stamp <= latest_stamp_)
  {
    return result;
  }
  latest_stamp_ = stamp;
  struct CurrentState
  {
    bool obstacle = false;
    bool strong_obstacle = false;
    bool free_terrain = false;
    double free_evidence_weight = 0.0;
    const VoxelSnapshot* obstacle_voxel = nullptr;
  };
  std::unordered_map<VoxelKey, CurrentState, VoxelKeyHash> current;
  current.reserve(current_voxels.size());
  for (const VoxelSnapshot& voxel : current_voxels)
  {
    const VoxelKey key = keyFor(voxel.x, voxel.y, voxel.z);
    CurrentState& state = current[key];
    if (isObstacle(voxel))
    {
      state.obstacle = true;
      if (state.obstacle_voxel == nullptr ||
          voxel.traversability_cost >
            state.obstacle_voxel->traversability_cost)
      {
        state.obstacle_voxel = &voxel;
      }
    }
    if (isStrongObstacleContradiction(voxel))
    {
      state.strong_obstacle = true;
    }
    if (isTerrain(voxel.label) &&
        voxel.last_observed == stamp &&
        voxel.semantic_confidence >= config_.minimum_semantic_confidence &&
        std::isfinite(voxel.traversability_cost) &&
        voxel.traversability_cost <= config_.free_max_traversability)
    {
      state.free_terrain = true;
      state.free_evidence_weight = std::max(
        state.free_evidence_weight,
        static_cast<double>(std::max(0.0f,
          std::min(1.0f, voxel.semantic_confidence))));
    }
  }

  const auto reset_free_evidence = [](TrackedObstacle& tracked)
  {
    tracked.first_free_stamp = ros::Time();
    tracked.last_free_stamp = ros::Time();
    tracked.free_frames = 0u;
    tracked.free_evidence = 0.0;
  };

  // Decay accumulated evidence in acquisition time. Missing or semantically
  // flickering observations do not reset it abruptly.
  for (auto& item : tracked_)
  {
    TrackedObstacle& tracked = item.second;
    if (!tracked.last_evidence_update_stamp.isZero())
    {
      const double elapsed =
        (stamp - tracked.last_evidence_update_stamp).toSec();
      tracked.free_evidence = std::max(
        0.0, tracked.free_evidence -
          config_.free_evidence_decay_per_second * std::max(0.0, elapsed));
      if (tracked.free_evidence <= 1e-9)
      {
        reset_free_evidence(tracked);
      }
    }
    tracked.last_evidence_update_stamp = stamp;
  }

  // Only a positive obstacle/dynamic observation resets free evidence. A
  // low-cost ambiguous label is deliberately neutral.
  for (const auto& item : current)
  {
    if (!item.second.strong_obstacle)
    {
      continue;
    }
    const auto tracked = tracked_.find(item.first);
    if (tracked != tracked_.end())
    {
      reset_free_evidence(tracked->second);
    }
  }

  std::unordered_map<VoxelKey, double, VoxelKeyHash> evidence_weights;
  evidence_weights.reserve(ray_free_evidence.size() +
                           reclassified_evidence.size() + current.size());
  for (const VoxelKey& key : ray_free_evidence)
  {
    evidence_weights[key] = 1.0;
  }
  for (const VoxelKey& key : reclassified_evidence)
  {
    evidence_weights[key] = 1.0;
  }
  for (const auto& item : current)
  {
    if (item.second.free_terrain)
    {
      evidence_weights[item.first] = std::max(
        evidence_weights[item.first], item.second.free_evidence_weight);
    }
  }

  for (const auto& evidence : evidence_weights)
  {
    const VoxelKey& key = evidence.first;
    auto tracked_iterator = tracked_.find(key);
    if (tracked_iterator == tracked_.end())
    {
      continue;
    }
    const auto current_iterator = current.find(key);
    if (current_iterator != current.end() &&
        current_iterator->second.strong_obstacle)
    {
      continue;
    }

    TrackedObstacle& tracked = tracked_iterator->second;
    if (tracked.free_evidence <= 1e-9)
    {
      tracked.first_free_stamp = stamp;
      tracked.free_frames = 0u;
    }
    tracked.last_free_stamp = stamp;
    ++tracked.free_frames;
    tracked.free_evidence += std::max(0.0, std::min(1.0, evidence.second));
    result.candidates.push_back(makePoint(key, tracked));

    const double duration =
      (stamp - tracked.first_free_stamp).toSec();
    if (tracked.free_frames >= config_.minimum_free_frames &&
        tracked.free_evidence >= config_.minimum_free_evidence &&
        duration >= config_.minimum_free_duration)
    {
      const ObstacleRevocationPoint revoked =
        makePoint(key, tracked);
      if (reclassified_evidence.count(key) != 0u)
      {
        result.revoked_reclassified.push_back(revoked);
      }
      else
      {
        result.revoked_free.push_back(revoked);
      }
      result.revoked_keys.insert(key);
      tracked_.erase(tracked_iterator);
    }
  }

  // Register the obstacle-like subset that this frame will send to SSMI.
  // Existing entries keep their contradiction history only when no obstacle
  // was observed; a positive obstacle observation above reset it already.
  for (const auto& item : current)
  {
    if (!item.second.obstacle || item.second.obstacle_voxel == nullptr)
    {
      continue;
    }
    const VoxelSnapshot& voxel = *item.second.obstacle_voxel;
    TrackedObstacle& tracked = tracked_[item.first];
    tracked.x = voxel.x;
    tracked.y = voxel.y;
    tracked.z = voxel.z;
    tracked.label = voxel.label;
    tracked.traversability = voxel.traversability_cost;
    if (item.second.strong_obstacle)
    {
      reset_free_evidence(tracked);
    }
    if (tracked.last_evidence_update_stamp.isZero())
    {
      tracked.last_evidence_update_stamp = stamp;
    }
  }

  return result;
}

void ObstacleRevocationTracker::clear()
{
  tracked_.clear();
  latest_stamp_ = ros::Time();
}

}  // namespace local3d_semantic_voxel_map
