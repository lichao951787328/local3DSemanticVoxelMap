#ifndef LOCAL3D_SEMANTIC_VOXEL_MAP_SEMANTIC_VOXEL_MAP_HPP_
#define LOCAL3D_SEMANTIC_VOXEL_MAP_SEMANTIC_VOXEL_MAP_HPP_

#include "local3d_semantic_voxel_map/semantic_fusion.hpp"

#include <ros/time.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace local3d_semantic_voxel_map
{

struct VoxelKey
{
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t z = 0;

  bool operator==(const VoxelKey& other) const
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash
{
  std::size_t operator()(const VoxelKey& key) const;
};

struct SemanticClass
{
  std::uint32_t label = 0;
  std::string name;
  std::uint8_t red = 127;
  std::uint8_t green = 127;
  std::uint8_t blue = 127;
  float traversability_cost = 0.5f;
};

struct SemanticVoxelMapConfig
{
  double voxel_size = 0.10;
  double decay_seconds = 0.5;
  std::size_t max_voxels = 500000;
  float unknown_cost = 0.5f;
  float semantic_cost_weight = 0.8f;
  float cost_rise_alpha = 0.65f;
  float cost_fall_alpha = 0.15f;
  SemanticFusionConfig semantic_fusion;
};

struct VoxelObservation
{
  std::uint32_t label = kInvalidSemanticLabel;
  float semantic_confidence = 1.0f;
  bool has_traversability_cost = false;
  float traversability_cost = 0.5f;
  ros::Time stamp;
};

struct SemanticVoxel
{
  explicit SemanticVoxel(const SemanticFusionConfig& config = SemanticFusionConfig())
    : semantics(config)
  {
  }

  SemanticEvidence semantics;
  float traversability_cost = 0.5f;
  std::uint32_t observation_count = 0;
  ros::Time last_observed;
};

struct VoxelSnapshot
{
  VoxelKey key;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  std::uint32_t label = kInvalidSemanticLabel;
  float semantic_confidence = 0.0f;
  float traversability_cost = 0.5f;
  std::uint32_t observation_count = 0;
  ros::Time last_observed;
};

class SemanticVoxelMap
{
public:
  explicit SemanticVoxelMap(const SemanticVoxelMapConfig& config);

  void setSemanticClasses(const std::vector<SemanticClass>& classes);
  VoxelKey worldToKey(double x, double y, double z) const;
  void keyToWorld(const VoxelKey& key, double& x, double& y, double& z) const;
  void integrate(const VoxelKey& key, const VoxelObservation& observation);
  void integrate(double x, double y, double z, const VoxelObservation& observation);

  std::size_t prune(const ros::Time& reference_stamp);
  std::size_t pruneOutside(double center_x, double center_y, double center_z,
                           double radius);
  void clear();
  std::size_t size() const;
  std::vector<VoxelSnapshot> snapshot() const;

  float classCost(std::uint32_t label) const;
  SemanticClass classDescription(std::uint32_t label) const;
  double voxelSize() const;

  bool saveCsv(const std::string& path, std::string& error) const;
  bool loadCsv(const std::string& path, std::string& error);

private:
  float expectedSemanticCost(const SemanticEvidence& semantics) const;
  void enforceCapacity();

  SemanticVoxelMapConfig config_;
  std::unordered_map<std::uint32_t, SemanticClass> classes_;
  std::unordered_map<VoxelKey, SemanticVoxel, VoxelKeyHash> voxels_;
  mutable std::mutex mutex_;
};

}  // namespace local3d_semantic_voxel_map

#endif  // LOCAL3D_SEMANTIC_VOXEL_MAP_SEMANTIC_VOXEL_MAP_HPP_
