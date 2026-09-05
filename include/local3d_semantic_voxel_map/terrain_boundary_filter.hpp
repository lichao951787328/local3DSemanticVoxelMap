#ifndef LOCAL3D_SEMANTIC_VOXEL_MAP_TERRAIN_BOUNDARY_FILTER_HPP_
#define LOCAL3D_SEMANTIC_VOXEL_MAP_TERRAIN_BOUNDARY_FILTER_HPP_

#include "local3d_semantic_voxel_map/semantic_voxel_map.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace local3d_semantic_voxel_map
{

// Recovers observed static-obstacle voxels supported by the morphologically
// closed projection of the traversable surface. This includes both small
// filled holes/gaps and static voxels sharing an XY column with original
// terrain. A single relative-height gate is the only geometric acceptance
// test. Dynamic and ignored roles are explicitly excluded by configuration.
struct TerrainBoundaryFilterConfig
{
  bool enabled = false;
  std::vector<std::uint32_t> terrain_labels{1u};
  std::vector<std::uint32_t> recoverable_labels{0u};
  std::vector<std::uint32_t> excluded_labels;
  // Explicit dilation followed by erosion on the 2D union of terrain labels.
  // Neither operation creates a voxel: only an already-observed recoverable
  // voxel inside the closed terrain support can be relabelled.
  double closing_radius = 0.15;
  double maximum_height_difference = 0.10;

  // Debugging is observational and never changes the relabeling result.
  bool debug_enabled = false;
};

enum class TerrainBoundaryDecisionReason : std::uint8_t
{
  OutsideClosedTerrainSupport = 0u,
  NoTerrainReference,
  HeightDifferenceTooHigh,
  Recovered,
  Count
};

constexpr std::size_t kTerrainBoundaryDecisionReasonCount =
  static_cast<std::size_t>(TerrainBoundaryDecisionReason::Count);

const char* terrainBoundaryDecisionReasonName(
  TerrainBoundaryDecisionReason reason);

std::uint32_t terrainBoundaryDecisionReasonBit(
  TerrainBoundaryDecisionReason reason);

struct TerrainBoundaryDebugRecord
{
  VoxelKey key;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  TerrainBoundaryDecisionReason first_failure =
    TerrainBoundaryDecisionReason::Recovered;
  std::uint32_t failure_mask = 0u;
  bool closed_terrain = false;
  bool proposed = false;
  bool reference_found = false;
  VoxelKey reference_key;
  double reference_x = std::numeric_limits<double>::quiet_NaN();
  double reference_y = std::numeric_limits<double>::quiet_NaN();
  double reference_z = std::numeric_limits<double>::quiet_NaN();
  std::uint32_t reference_label = kInvalidSemanticLabel;
  double reference_distance_xy = std::numeric_limits<double>::quiet_NaN();
  double height_difference = std::numeric_limits<double>::quiet_NaN();
  std::uint32_t original_label = kInvalidSemanticLabel;
  std::uint32_t replacement_label = kInvalidSemanticLabel;
};

// Flattened masks for the one projected XY grid. They are populated only when
// debug_enabled=true, so the normal navigation path has no image copies.
struct TerrainBoundaryLayerDebug
{
  std::int32_t minimum_x = 0;
  std::int32_t minimum_y = 0;
  int rows = 0;
  int columns = 0;
  std::vector<std::uint8_t> terrain_original;
  std::vector<std::uint8_t> terrain_after_dilation;
  std::vector<std::uint8_t> terrain_after_erosion;
  std::vector<std::uint8_t> recoverable_static;
  std::vector<std::uint8_t> excluded;
  std::vector<std::uint8_t> newly_filled;
  std::vector<std::uint8_t> proposed;
  std::vector<std::uint8_t> reference_found;
  std::vector<std::uint8_t> recovered;
};

struct TerrainBoundaryDebugStatistics
{
  std::size_t terrain_voxels = 0u;
  std::size_t recoverable_static_voxels = 0u;
  std::size_t excluded_voxels = 0u;
  std::size_t opencv_newly_filled_cells = 0u;
  std::size_t opencv_proposed_cells = 0u;
  std::size_t opencv_proposed_voxels = 0u;
  std::size_t reference_found_voxels = 0u;
  std::array<std::size_t, kTerrainBoundaryDecisionReasonCount>
    first_decision_counts{{}};
  int closing_radius_cells = 0;
  int maximum_height_difference_cells = 0;
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
  TerrainBoundaryDebugStatistics debug_statistics;
  std::vector<TerrainBoundaryDebugRecord> debug_records;
  std::vector<TerrainBoundaryLayerDebug> debug_layers;
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
  bool isRecoverable(std::uint32_t label) const;

  TerrainBoundaryFilterConfig config_;
  std::unordered_set<std::uint32_t> terrain_labels_;
  std::unordered_set<std::uint32_t> recoverable_labels_;
  std::unordered_set<std::uint32_t> excluded_labels_;
};

}  // namespace local3d_semantic_voxel_map

#endif  // LOCAL3D_SEMANTIC_VOXEL_MAP_TERRAIN_BOUNDARY_FILTER_HPP_
