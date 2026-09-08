#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace local3d_semantic_voxel_map
{

struct TwoLayerClearingPoint
{
  double local_x = 0.0;
  double local_y = 0.0;
  double map_z = 0.0;
};

struct TwoLayerClearingConfig
{
  double octomap_resolution = 0.40;
  double angular_resolution_deg = 5.0;
  double local_grid_resolution = 0.10;
  double local_box_min_x = -2.0;
  double local_box_max_x = 8.0;
  double local_box_min_y = -5.0;
  double local_box_max_y = 5.0;
};

struct TwoLayerClearingRay
{
  double origin_x = 0.0;
  double origin_y = 0.0;
  double origin_z = 0.0;
  double end_x = 0.0;
  double end_y = 0.0;
  double end_z = 0.0;
  bool hit = false;
  std::uint8_t layer = 0u;
  std::uint16_t angle_bin = 0u;
};

struct TwoLayerClearingResult
{
  std::vector<TwoLayerClearingRay> rays;
  double maximum_z = 0.0;
  double top_layer_z = 0.0;
  std::size_t hit_rays = 0u;
  std::size_t no_hit_rays = 0u;
  std::size_t visited_local_cells = 0u;
};

/// Build two robot-centred horizontal 360-degree clearing sweeps. Input points
/// are current-acquisition returns, expressed as robot-aligned local x/y and
/// gravity-aligned map z. Historical fused voxels must not be passed here.
TwoLayerClearingResult generateTwoLayerAngularClearingRays(
  const std::vector<TwoLayerClearingPoint>& current_points,
  double map_origin_x,
  double map_origin_y,
  double map_yaw,
  const TwoLayerClearingConfig& config);

}  // namespace local3d_semantic_voxel_map
