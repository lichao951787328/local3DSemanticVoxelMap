#include "local3d_semantic_voxel_map/obstacle_revocation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace local3d_semantic_voxel_map
{

ObstacleRevocationTracker::ObstacleRevocationTracker(
  const ObstacleRevocationConfig& config)
  : config_(config)
{
  if (config_.voxel_size <= 0.0 || config_.minimum_free_frames == 0u ||
      config_.minimum_free_duration < 0.0 ||
      config_.free_max_traversability < 0.0f ||
      config_.free_max_traversability > 1.0f ||
      config_.obstacle_min_traversability < 0.0f ||
      config_.obstacle_min_traversability > 1.0f ||
      config_.minimum_semantic_confidence < 0.0f ||
      config_.minimum_semantic_confidence > 1.0f ||
      config_.ray_endpoint_margin < 0.0)
  {
    throw std::invalid_argument("invalid obstacle revocation configuration");
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

bool ObstacleRevocationTracker::isDynamic(const std::uint32_t label)
{
  return label >= 11u && label <= 18u;
}

bool ObstacleRevocationTracker::isTerrain(const std::uint32_t label)
{
  return label == 0u || label == 1u || label == 9u;
}

bool ObstacleRevocationTracker::isObstacle(const VoxelSnapshot& voxel) const
{
  if (isDynamic(voxel.label))
  {
    return false;
  }
  const bool static_semantic = voxel.label >= 2u && voxel.label <= 8u;
  return static_semantic ||
    (std::isfinite(voxel.traversability_cost) &&
     voxel.traversability_cost >= config_.obstacle_min_traversability);
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
  return point;
}

ObstacleRevocationResult ObstacleRevocationTracker::update(
  const std::vector<VoxelSnapshot>& current_voxels,
  const std::unordered_set<VoxelKey, VoxelKeyHash>& ray_free_evidence,
  const ros::Time& stamp)
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
  ++frame_sequence_;

  struct CurrentState
  {
    bool obstacle = false;
    bool dynamic = false;
    bool free_terrain = false;
    const VoxelSnapshot* obstacle_voxel = nullptr;
  };
  std::unordered_map<VoxelKey, CurrentState, VoxelKeyHash> current;
  current.reserve(current_voxels.size());
  for (const VoxelSnapshot& voxel : current_voxels)
  {
    const VoxelKey key = keyFor(voxel.x, voxel.y, voxel.z);
    CurrentState& state = current[key];
    if (isDynamic(voxel.label))
    {
      state.dynamic = true;
    }
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
    if (isTerrain(voxel.label) &&
        voxel.last_observed == stamp &&
        voxel.semantic_confidence >= config_.minimum_semantic_confidence &&
        std::isfinite(voxel.traversability_cost) &&
        voxel.traversability_cost <= config_.free_max_traversability)
    {
      state.free_terrain = true;
    }
  }

  std::unordered_set<VoxelKey, VoxelKeyHash> evidence_keys = ray_free_evidence;
  for (const auto& item : current)
  {
    if (item.second.free_terrain)
    {
      evidence_keys.insert(item.first);
    }
  }

  for (const VoxelKey& key : evidence_keys)
  {
    auto tracked_iterator = tracked_.find(key);
    if (tracked_iterator == tracked_.end())
    {
      continue;
    }
    const auto current_iterator = current.find(key);
    if (current_iterator != current.end() &&
        (current_iterator->second.obstacle || current_iterator->second.dynamic))
    {
      tracked_iterator->second.first_free_stamp = ros::Time();
      tracked_iterator->second.last_free_stamp = ros::Time();
      tracked_iterator->second.free_frames = 0u;
      tracked_iterator->second.last_free_frame_sequence = 0u;
      continue;
    }

    TrackedObstacle& tracked = tracked_iterator->second;
    const bool consecutive = tracked.free_frames != 0u &&
      tracked.last_free_frame_sequence + 1u == frame_sequence_;
    if (!consecutive)
    {
      tracked.first_free_stamp = stamp;
      tracked.free_frames = 0u;
    }
    tracked.last_free_stamp = stamp;
    tracked.last_free_frame_sequence = frame_sequence_;
    ++tracked.free_frames;
    result.candidates.push_back(makePoint(key, tracked));

    const double duration =
      (stamp - tracked.first_free_stamp).toSec();
    if (tracked.free_frames >= config_.minimum_free_frames &&
        duration >= config_.minimum_free_duration)
    {
      const ObstacleRevocationPoint revoked =
        makePoint(key, tracked);
      result.revoked_free.push_back(revoked);
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
    tracked.first_free_stamp = ros::Time();
    tracked.last_free_stamp = ros::Time();
    tracked.free_frames = 0u;
    tracked.last_free_frame_sequence = 0u;
  }

  return result;
}

void ObstacleRevocationTracker::clear()
{
  tracked_.clear();
  latest_stamp_ = ros::Time();
  frame_sequence_ = 0u;
}

}  // namespace local3d_semantic_voxel_map
