#include "local3d_semantic_voxel_map/terrain_boundary_filter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace local3d_semantic_voxel_map
{

namespace
{

bool containsNearbyKey(
  const std::unordered_set<VoxelKey, VoxelKeyHash>& keys,
  const VoxelKey& center, const int radius_xy, const int radius_z)
{
  for (int dz = -radius_z; dz <= radius_z; ++dz)
  {
    for (int dx = -radius_xy; dx <= radius_xy; ++dx)
    {
      for (int dy = -radius_xy; dy <= radius_xy; ++dy)
      {
        if (keys.count(VoxelKey{center.x + dx, center.y + dy,
                                center.z + dz}) != 0u)
        {
          return true;
        }
      }
    }
  }
  return false;
}

}  // namespace

TerrainBoundaryFilter::TerrainBoundaryFilter(
  const TerrainBoundaryFilterConfig& config)
  : config_(config), terrain_labels_(config.terrain_labels.begin(),
                                    config.terrain_labels.end())
{
  if (config_.terrain_labels.empty() ||
      terrain_labels_.count(config_.obstacle_label) != 0u ||
      config_.neighborhood_radius < 0.0 ||
      config_.vertical_tolerance < 0.0 ||
      config_.minimum_terrain_neighbors == 0u ||
      config_.minimum_distinct_terrain_labels == 0u ||
      config_.minimum_terrain_ratio < 0.0 ||
      config_.minimum_terrain_ratio > 1.0)
  {
    throw std::invalid_argument("invalid terrain boundary filter configuration");
  }
}

bool TerrainBoundaryFilter::isTerrain(const std::uint32_t label) const
{
  return terrain_labels_.count(label) != 0u;
}

TerrainBoundaryFilterResult TerrainBoundaryFilter::filter(
  const std::vector<VoxelSnapshot>& voxels,
  const double voxel_size_xy, const double voxel_size_z) const
{
  TerrainBoundaryFilterResult result;
  result.voxels = voxels;
  if (!config_.enabled || voxels.empty())
  {
    return result;
  }
  if (!std::isfinite(voxel_size_xy) || voxel_size_xy <= 0.0 ||
      !std::isfinite(voxel_size_z) || voxel_size_z <= 0.0)
  {
    throw std::invalid_argument("terrain boundary filter voxel sizes must be positive");
  }

  std::unordered_map<VoxelKey, std::size_t, VoxelKeyHash> voxel_indices;
  voxel_indices.reserve(voxels.size());
  std::unordered_set<VoxelKey, VoxelKeyHash> obstacle_keys;
  obstacle_keys.reserve(voxels.size() / 8u + 1u);
  for (std::size_t index = 0u; index < voxels.size(); ++index)
  {
    voxel_indices[voxels[index].key] = index;
    if (voxels[index].label == config_.obstacle_label)
    {
      obstacle_keys.insert(voxels[index].key);
    }
  }
  if (obstacle_keys.empty())
  {
    return result;
  }

  // Perform a binary XY opening without introducing an OpenCV dependency into
  // the semantic map library. A surface may move by vertical_tolerance across
  // the kernel, so each required XY cell may match a nearby Z key.
  const int opening_radius = static_cast<int>(config_.opening_radius_cells);
  const int opening_vertical_radius = static_cast<int>(std::ceil(
    config_.vertical_tolerance / voxel_size_z));
  std::unordered_set<VoxelKey, VoxelKeyHash> eroded;
  eroded.reserve(obstacle_keys.size());
  for (const VoxelKey& key : obstacle_keys)
  {
    bool complete_kernel = true;
    for (int dx = -opening_radius; dx <= opening_radius && complete_kernel; ++dx)
    {
      for (int dy = -opening_radius; dy <= opening_radius; ++dy)
      {
        const VoxelKey required{key.x + dx, key.y + dy, key.z};
        bool occupied = false;
        for (int dz = -opening_vertical_radius;
             dz <= opening_vertical_radius; ++dz)
        {
          if (obstacle_keys.count(
                VoxelKey{required.x, required.y, required.z + dz}) != 0u)
          {
            occupied = true;
            break;
          }
        }
        if (!occupied)
        {
          complete_kernel = false;
          break;
        }
      }
    }
    if (complete_kernel)
    {
      eroded.insert(key);
    }
  }

  const int neighborhood_xy = static_cast<int>(std::ceil(
    config_.neighborhood_radius / voxel_size_xy));
  const int neighborhood_z = static_cast<int>(std::ceil(
    config_.vertical_tolerance / voxel_size_z));
  result.relabeled.reserve(obstacle_keys.size() / 4u + 1u);
  for (std::size_t index = 0u; index < voxels.size(); ++index)
  {
    const VoxelSnapshot& source = voxels[index];
    if (source.label != config_.obstacle_label)
    {
      continue;
    }
    // Dilation of the eroded mask restores dense obstacle bodies. Only pixels
    // removed by the complete opening remain eligible for boundary cleanup.
    if (containsNearbyKey(eroded, source.key, opening_radius,
                          opening_vertical_radius))
    {
      continue;
    }

    std::unordered_map<std::uint32_t, double> label_weights;
    std::size_t terrain_count = 0u;
    std::size_t obstacle_count = 0u;
    for (int dx = -neighborhood_xy; dx <= neighborhood_xy; ++dx)
    {
      for (int dy = -neighborhood_xy; dy <= neighborhood_xy; ++dy)
      {
        for (int dz = -neighborhood_z; dz <= neighborhood_z; ++dz)
        {
          const auto neighbor_iterator = voxel_indices.find(VoxelKey{
            source.key.x + dx, source.key.y + dy, source.key.z + dz});
          if (neighbor_iterator == voxel_indices.end() ||
              neighbor_iterator->second == index)
          {
            continue;
          }
          const VoxelSnapshot& neighbor = voxels[neighbor_iterator->second];
          if (isTerrain(neighbor.label))
          {
            ++terrain_count;
            label_weights[neighbor.label] += std::max(
              1e-3, static_cast<double>(neighbor.semantic_confidence));
          }
          else if (neighbor.label == config_.obstacle_label)
          {
            ++obstacle_count;
          }
        }
      }
    }
    const std::size_t classified_count = terrain_count + obstacle_count;
    const double terrain_ratio = classified_count == 0u ? 0.0 :
      static_cast<double>(terrain_count) /
      static_cast<double>(classified_count);
    if (terrain_count < config_.minimum_terrain_neighbors ||
        label_weights.size() < config_.minimum_distinct_terrain_labels ||
        terrain_ratio < config_.minimum_terrain_ratio)
    {
      continue;
    }

    const auto winning = std::max_element(
      label_weights.begin(), label_weights.end(),
      [](const auto& lhs, const auto& rhs)
      {
        if (lhs.second != rhs.second)
        {
          return lhs.second < rhs.second;
        }
        return lhs.first > rhs.first;
      });
    const std::uint32_t replacement_label = winning->first;
    double confidence_sum = 0.0;
    double semantic_cost_sum = 0.0;
    double traversability_sum = 0.0;
    double measured_sum = 0.0;
    std::size_t replacement_count = 0u;
    std::size_t measured_count = 0u;
    for (int dx = -neighborhood_xy; dx <= neighborhood_xy; ++dx)
    {
      for (int dy = -neighborhood_xy; dy <= neighborhood_xy; ++dy)
      {
        for (int dz = -neighborhood_z; dz <= neighborhood_z; ++dz)
        {
          const auto neighbor_iterator = voxel_indices.find(VoxelKey{
            source.key.x + dx, source.key.y + dy, source.key.z + dz});
          if (neighbor_iterator == voxel_indices.end())
          {
            continue;
          }
          const VoxelSnapshot& neighbor = voxels[neighbor_iterator->second];
          if (neighbor.label != replacement_label)
          {
            continue;
          }
          ++replacement_count;
          confidence_sum += neighbor.semantic_confidence;
          semantic_cost_sum += neighbor.semantic_cost;
          traversability_sum += neighbor.traversability_cost;
          if (neighbor.has_measured_traversability)
          {
            measured_sum += neighbor.measured_traversability_cost;
            ++measured_count;
          }
        }
      }
    }
    if (replacement_count == 0u)
    {
      continue;
    }

    VoxelSnapshot& replacement = result.voxels[index];
    replacement.label = replacement_label;
    replacement.semantic_confidence = static_cast<float>(
      confidence_sum / static_cast<double>(replacement_count));
    replacement.semantic_cost = static_cast<float>(
      semantic_cost_sum / static_cast<double>(replacement_count));
    replacement.traversability_cost = static_cast<float>(
      traversability_sum / static_cast<double>(replacement_count));
    replacement.has_measured_traversability = measured_count != 0u;
    if (replacement.has_measured_traversability)
    {
      replacement.measured_traversability_cost = static_cast<float>(
        measured_sum / static_cast<double>(measured_count));
    }
    result.relabeled.push_back(TerrainBoundaryRelabel{
      source.key, source.label, replacement_label});
  }
  return result;
}

}  // namespace local3d_semantic_voxel_map
