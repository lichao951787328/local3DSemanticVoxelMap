#include "local3d_semantic_voxel_map/two_layer_angular_clearing.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace local3d_semantic_voxel_map
{
namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kDirectionEpsilon = 1e-12;

std::int64_t layerIndex(const double z, const double resolution)
{
  return static_cast<std::int64_t>(std::floor(z / resolution));
}

double layerCenter(const std::int64_t index, const double resolution)
{
  return (static_cast<double>(index) + 0.5) * resolution;
}

std::uint64_t packCell(const int x, const int y)
{
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) << 32u) |
         static_cast<std::uint32_t>(y);
}

int cellIndex(const double value, const double minimum, const double resolution)
{
  return static_cast<int>(std::floor((value - minimum) / resolution));
}

double distanceToBoxExit(const double dx, const double dy,
                         const TwoLayerClearingConfig& config)
{
  const double tx = dx > kDirectionEpsilon ? config.local_box_max_x / dx :
                    dx < -kDirectionEpsilon ? config.local_box_min_x / dx :
                    std::numeric_limits<double>::infinity();
  const double ty = dy > kDirectionEpsilon ? config.local_box_max_y / dy :
                    dy < -kDirectionEpsilon ? config.local_box_min_y / dy :
                    std::numeric_limits<double>::infinity();
  return std::min(tx, ty);
}

struct LocalRayResult
{
  double distance = 0.0;
  bool hit = false;
  std::size_t visited_cells = 0u;
};

LocalRayResult traceLocalGrid(
  const double dx, const double dy,
  const std::unordered_set<std::uint64_t>& occupied,
  const TwoLayerClearingConfig& config)
{
  LocalRayResult result;
  const double exit_distance = distanceToBoxExit(dx, dy, config);
  result.distance = std::max(0.0, exit_distance);
  if (!std::isfinite(exit_distance) || exit_distance <= 0.0)
  {
    return result;
  }

  int cell_x = cellIndex(0.0, config.local_box_min_x,
                         config.local_grid_resolution);
  int cell_y = cellIndex(0.0, config.local_box_min_y,
                         config.local_grid_resolution);
  const int width = static_cast<int>(std::ceil(
    (config.local_box_max_x - config.local_box_min_x) /
    config.local_grid_resolution));
  const int height = static_cast<int>(std::ceil(
    (config.local_box_max_y - config.local_box_min_y) /
    config.local_grid_resolution));

  const int step_x = dx > kDirectionEpsilon ? 1 :
                     dx < -kDirectionEpsilon ? -1 : 0;
  const int step_y = dy > kDirectionEpsilon ? 1 :
                     dy < -kDirectionEpsilon ? -1 : 0;
  const double delta_x = step_x == 0 ?
    std::numeric_limits<double>::infinity() :
    config.local_grid_resolution / std::abs(dx);
  const double delta_y = step_y == 0 ?
    std::numeric_limits<double>::infinity() :
    config.local_grid_resolution / std::abs(dy);

  const double next_x = step_x > 0 ?
    config.local_box_min_x + (cell_x + 1) * config.local_grid_resolution :
    config.local_box_min_x + cell_x * config.local_grid_resolution;
  const double next_y = step_y > 0 ?
    config.local_box_min_y + (cell_y + 1) * config.local_grid_resolution :
    config.local_box_min_y + cell_y * config.local_grid_resolution;
  double max_x = step_x == 0 ? std::numeric_limits<double>::infinity() :
                 next_x / dx;
  double max_y = step_y == 0 ? std::numeric_limits<double>::infinity() :
                 next_y / dy;
  double entry_distance = 0.0;
  const int origin_cell_x = cell_x;
  const int origin_cell_y = cell_y;

  while (cell_x >= 0 && cell_x < width && cell_y >= 0 && cell_y < height &&
         entry_distance <= exit_distance + 1e-9)
  {
    ++result.visited_cells;
    // Ignore only the cell containing the ray origin. All later current-frame
    // returns, irrespective of semantics, stop the clearing ray.
    if ((cell_x != origin_cell_x || cell_y != origin_cell_y) &&
        occupied.count(packCell(cell_x, cell_y)) != 0u)
    {
      result.hit = true;
      // Place the endpoint just inside the hit cell. The consumer also receives
      // hit=true and protects the complete endpoint OctoMap key.
      result.distance = std::min(
        exit_distance,
        entry_distance + 1e-4 * config.local_grid_resolution);
      return result;
    }

    if (max_x < max_y)
    {
      entry_distance = max_x;
      max_x += delta_x;
      cell_x += step_x;
    }
    else if (max_y < max_x)
    {
      entry_distance = max_y;
      max_y += delta_y;
      cell_y += step_y;
    }
    else
    {
      entry_distance = max_x;
      max_x += delta_x;
      max_y += delta_y;
      cell_x += step_x;
      cell_y += step_y;
    }
  }
  return result;
}

void validateConfig(const TwoLayerClearingConfig& config)
{
  if (!std::isfinite(config.octomap_resolution) ||
      config.octomap_resolution <= 0.0)
    throw std::invalid_argument("octomap_resolution must be positive");
  if (!std::isfinite(config.angular_resolution_deg) ||
      config.angular_resolution_deg <= 0.0 ||
      config.angular_resolution_deg > 360.0)
    throw std::invalid_argument(
      "angular_resolution_deg must be in (0, 360]");
  if (std::abs(360.0 / config.angular_resolution_deg -
               std::round(360.0 / config.angular_resolution_deg)) > 1e-9)
    throw std::invalid_argument(
      "angular_resolution_deg must divide 360 exactly");
  if (!std::isfinite(config.local_grid_resolution) ||
      config.local_grid_resolution <= 0.0)
    throw std::invalid_argument("local_grid_resolution must be positive");
  if (!(config.local_box_min_x < 0.0 && config.local_box_max_x > 0.0 &&
        config.local_box_min_y < 0.0 && config.local_box_max_y > 0.0))
    throw std::invalid_argument(
      "local box must contain the robot origin in x/y");
}

}  // namespace

TwoLayerClearingResult generateTwoLayerAngularClearingRays(
  const std::vector<TwoLayerClearingPoint>& current_points,
  const double map_origin_x,
  const double map_origin_y,
  const double map_yaw,
  const TwoLayerClearingConfig& config)
{
  validateConfig(config);
  TwoLayerClearingResult result;
  if (current_points.empty())
    return result;

  double maximum_z = -std::numeric_limits<double>::infinity();
  for (const TwoLayerClearingPoint& point : current_points)
  {
    if (std::isfinite(point.local_x) && std::isfinite(point.local_y) &&
        std::isfinite(point.map_z))
      maximum_z = std::max(maximum_z, point.map_z);
  }
  if (!std::isfinite(maximum_z))
    return result;

  result.maximum_z = maximum_z;
  const std::int64_t top_layer_index = layerIndex(
    maximum_z, config.octomap_resolution);
  result.top_layer_z = layerCenter(
    top_layer_index, config.octomap_resolution);

  std::array<std::unordered_set<std::uint64_t>, 2> occupied;
  for (const TwoLayerClearingPoint& point : current_points)
  {
    if (!std::isfinite(point.local_x) || !std::isfinite(point.local_y) ||
        !std::isfinite(point.map_z))
      continue;
    const std::int64_t point_layer = layerIndex(
      point.map_z, config.octomap_resolution);
    const std::int64_t layer_offset = top_layer_index - point_layer;
    if (layer_offset < 0 || layer_offset > 1)
      continue;
    const int cell_x = cellIndex(
      point.local_x, config.local_box_min_x,
      config.local_grid_resolution);
    const int cell_y = cellIndex(
      point.local_y, config.local_box_min_y,
      config.local_grid_resolution);
    const int width = static_cast<int>(std::ceil(
      (config.local_box_max_x - config.local_box_min_x) /
      config.local_grid_resolution));
    const int height = static_cast<int>(std::ceil(
      (config.local_box_max_y - config.local_box_min_y) /
      config.local_grid_resolution));
    if (cell_x < 0 || cell_x >= width || cell_y < 0 || cell_y >= height)
      continue;
    occupied[static_cast<std::size_t>(layer_offset)].insert(
      packCell(cell_x, cell_y));
  }

  const std::size_t angular_bins = std::max<std::size_t>(
    1u, static_cast<std::size_t>(
      std::llround(360.0 / config.angular_resolution_deg)));
  result.rays.reserve(2u * angular_bins);
  for (std::size_t layer = 0u; layer < 2u; ++layer)
  {
    const double layer_z = layerCenter(
      top_layer_index - static_cast<std::int64_t>(layer),
      config.octomap_resolution);
    for (std::size_t angle_bin = 0u; angle_bin < angular_bins; ++angle_bin)
    {
      const double angle = 2.0 * kPi *
        static_cast<double>(angle_bin) / static_cast<double>(angular_bins);
      const double local_dx = std::cos(angle);
      const double local_dy = std::sin(angle);
      const LocalRayResult local_ray = traceLocalGrid(
        local_dx, local_dy, occupied[layer], config);
      result.visited_local_cells += local_ray.visited_cells;
      if (local_ray.hit)
        ++result.hit_rays;
      else
        ++result.no_hit_rays;

      const double map_angle = map_yaw + angle;
      TwoLayerClearingRay ray;
      ray.origin_x = map_origin_x;
      ray.origin_y = map_origin_y;
      ray.origin_z = layer_z;
      ray.end_x = map_origin_x + std::cos(map_angle) * local_ray.distance;
      ray.end_y = map_origin_y + std::sin(map_angle) * local_ray.distance;
      ray.end_z = layer_z;
      ray.hit = local_ray.hit;
      ray.layer = static_cast<std::uint8_t>(layer);
      ray.angle_bin = static_cast<std::uint16_t>(angle_bin);
      result.rays.push_back(ray);
    }
  }
  return result;
}

}  // namespace local3d_semantic_voxel_map
