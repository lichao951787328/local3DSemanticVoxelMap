#ifndef LOCAL3D_SEMANTIC_VOXEL_MAP_TERRAIN_BOUNDARY_FILTER_HPP_
#define LOCAL3D_SEMANTIC_VOXEL_MAP_TERRAIN_BOUNDARY_FILTER_HPP_

#include "local3d_semantic_voxel_map/semantic_voxel_map.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace local3d_semantic_voxel_map
{

// Removes thin obstacle-label seams created between two or more traversable
// semantic classes. The original terrain labels remain distinct; they are
// treated as one super-class only while deciding whether an obstacle voxel is
// a class-boundary artifact.
struct TerrainBoundaryFilterConfig
{
  bool enabled = false;
  std::uint32_t obstacle_label = 0u;
  std::vector<std::uint32_t> terrain_labels{1u, 2u, 3u, 4u};
  // Binary opening radius on the native XY voxel grid. One cell is a 3x3
  // opening and removes one-cell-wide seams while retaining dense 3x3 blocks.
  std::size_t opening_radius_cells = 1u;
  double neighborhood_radius = 0.30;
  double vertical_tolerance = 0.15;
  std::size_t minimum_terrain_neighbors = 6u;
  std::size_t minimum_distinct_terrain_labels = 2u;
  double minimum_terrain_ratio = 0.70;
};

struct TerrainBoundaryRelabel
{
  VoxelKey key;
  std::uint32_t original_label = kInvalidSemanticLabel;
  std::uint32_t replacement_label = kInvalidSemanticLabel;
};

struct TerrainBoundaryFilterResult
{
  std::vector<VoxelSnapshot> voxels;
  std::vector<TerrainBoundaryRelabel> relabeled;
};

class TerrainBoundaryFilter
{
public:
  explicit TerrainBoundaryFilter(
    const TerrainBoundaryFilterConfig& config = TerrainBoundaryFilterConfig());

  TerrainBoundaryFilterResult filter(
    const std::vector<VoxelSnapshot>& voxels,
    double voxel_size_xy, double voxel_size_z) const;

private:
  bool isTerrain(std::uint32_t label) const;

  TerrainBoundaryFilterConfig config_;
  std::unordered_set<std::uint32_t> terrain_labels_;
};

}  // namespace local3d_semantic_voxel_map

#endif  // LOCAL3D_SEMANTIC_VOXEL_MAP_TERRAIN_BOUNDARY_FILTER_HPP_
