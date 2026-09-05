#ifndef LOCAL3D_SEMANTIC_VOXEL_MAP_OBSTACLE_REVOCATION_HPP_
#define LOCAL3D_SEMANTIC_VOXEL_MAP_OBSTACLE_REVOCATION_HPP_

#include "local3d_semantic_voxel_map/semantic_voxel_map.hpp"

#include <ros/time.h>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace local3d_semantic_voxel_map
{

struct ObstacleRevocationConfig
{
  double voxel_size = 0.10;
  std::size_t minimum_free_frames = 5u;
  double minimum_free_duration = 0.5;
  // Qualifying free observations need not be consecutive. Each observation
  // adds confidence-weighted evidence and missing observations decay it. A
  // strong obstacle observation resets it immediately.
  // Zero inherits minimum_free_frames for backward-compatible callers.
  double minimum_free_evidence = 0.0;
  double free_evidence_decay_per_second = 0.0;
  float free_max_traversability = 0.45f;
  float obstacle_min_traversability = 0.75f;
  float minimum_semantic_confidence = 0.60f;
  double ray_endpoint_margin = 0.20;
  // Defaults preserve the original Cityscapes contract. Dataset-specific
  // pipelines may override these roles (for example obstacle 0 and terrain
  // 1/2/3/4 in the five-class bag).
  std::vector<std::uint32_t> terrain_labels{0u, 1u, 9u};
  std::vector<std::uint32_t> obstacle_labels{2u, 3u, 4u, 5u, 6u, 7u, 8u};
  // Ambiguous labels remain local safety obstacles, but a low measured cost
  // does not erase accumulated free evidence merely because semantic output
  // flickered back to the ambiguous label.
  std::vector<std::uint32_t> ambiguous_obstacle_labels;
  float ambiguous_obstacle_reset_min_traversability = 0.65f;
  std::vector<std::uint32_t> dynamic_labels{
    11u, 12u, 13u, 14u, 15u, 16u, 17u, 18u};
};

struct ObstacleRevocationPoint
{
  VoxelKey key;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  std::uint32_t label = kInvalidSemanticLabel;
  float traversability = 1.0f;
  std::size_t evidence_frames = 0u;
  double evidence_score = 0.0;
};

struct ObstacleRevocationResult
{
  std::vector<ObstacleRevocationPoint> candidates;
  std::vector<ObstacleRevocationPoint> revoked_free;
  std::vector<ObstacleRevocationPoint> revoked_reclassified;
  std::unordered_set<VoxelKey, VoxelKeyHash> revoked_keys;
};

// Mirrors only obstacle-like endpoints already sent to SSMI. A remembered
// endpoint is revoked only after direct low-cost terrain evidence or a sensor
// ray has traversed its voxel for several distinct acquisition frames. Mere
// absence from the rolling local map is never evidence.
class ObstacleRevocationTracker
{
public:
  explicit ObstacleRevocationTracker(
    const ObstacleRevocationConfig& config = ObstacleRevocationConfig());

  VoxelKey keyFor(double x, double y, double z) const;
  bool isTracked(const VoxelKey& key) const;
  std::size_t trackedCount() const;

  // Add tracked keys intersected before the measured endpoint. The endpoint
  // margin prevents a surface return from clearing itself or nearby voxels.
  void collectTrackedRayEvidence(
    double origin_x, double origin_y, double origin_z,
    double endpoint_x, double endpoint_y, double endpoint_z,
    std::unordered_set<VoxelKey, VoxelKeyHash>& evidence) const;

  ObstacleRevocationResult update(
    const std::vector<VoxelSnapshot>& current_voxels,
    const std::unordered_set<VoxelKey, VoxelKeyHash>& ray_free_evidence,
    const ros::Time& stamp,
    const std::unordered_set<VoxelKey, VoxelKeyHash>&
      reclassified_evidence =
        std::unordered_set<VoxelKey, VoxelKeyHash>());

  void clear();

private:
  struct TrackedObstacle
  {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    std::uint32_t label = kInvalidSemanticLabel;
    float traversability = 1.0f;
    ros::Time first_free_stamp;
    ros::Time last_free_stamp;
    std::size_t free_frames = 0u;
    double free_evidence = 0.0;
    ros::Time last_evidence_update_stamp;
  };

  bool isDynamic(std::uint32_t label) const;
  bool isTerrain(std::uint32_t label) const;
  bool isSemanticObstacle(std::uint32_t label) const;
  bool isAmbiguousObstacle(std::uint32_t label) const;
  bool isObstacle(const VoxelSnapshot& voxel) const;
  bool isStrongObstacleContradiction(const VoxelSnapshot& voxel) const;
  ObstacleRevocationPoint makePoint(
    const VoxelKey& key, const TrackedObstacle& obstacle) const;

  ObstacleRevocationConfig config_;
  std::unordered_set<std::uint32_t> terrain_labels_;
  std::unordered_set<std::uint32_t> obstacle_labels_;
  std::unordered_set<std::uint32_t> ambiguous_obstacle_labels_;
  std::unordered_set<std::uint32_t> dynamic_labels_;
  std::unordered_map<VoxelKey, TrackedObstacle, VoxelKeyHash> tracked_;
  ros::Time latest_stamp_;
};

}  // namespace local3d_semantic_voxel_map

#endif  // LOCAL3D_SEMANTIC_VOXEL_MAP_OBSTACLE_REVOCATION_HPP_
