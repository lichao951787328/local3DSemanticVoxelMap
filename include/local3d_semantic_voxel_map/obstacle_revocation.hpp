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
  float free_max_traversability = 0.45f;
  float obstacle_min_traversability = 0.75f;
  float minimum_semantic_confidence = 0.60f;
  double ray_endpoint_margin = 0.20;
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
};

struct ObstacleRevocationResult
{
  std::vector<ObstacleRevocationPoint> candidates;
  std::vector<ObstacleRevocationPoint> revoked_free;
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
    const ros::Time& stamp);

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
    std::size_t last_free_frame_sequence = 0u;
  };

  static bool isDynamic(std::uint32_t label);
  static bool isTerrain(std::uint32_t label);
  bool isObstacle(const VoxelSnapshot& voxel) const;
  ObstacleRevocationPoint makePoint(
    const VoxelKey& key, const TrackedObstacle& obstacle) const;

  ObstacleRevocationConfig config_;
  std::unordered_map<VoxelKey, TrackedObstacle, VoxelKeyHash> tracked_;
  ros::Time latest_stamp_;
  std::size_t frame_sequence_ = 0u;
};

}  // namespace local3d_semantic_voxel_map

#endif  // LOCAL3D_SEMANTIC_VOXEL_MAP_OBSTACLE_REVOCATION_HPP_
