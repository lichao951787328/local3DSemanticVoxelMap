#include "local3d_semantic_voxel_map/terrain_boundary_filter.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace local3d_semantic_voxel_map
{

namespace
{

struct RasterBounds
{
  std::int32_t minimum_x = 0;
  std::int32_t minimum_y = 0;
  int padding = 0;
  int rows = 0;
  int columns = 0;

  int row(const VoxelKey& key) const
  {
    return static_cast<int>(
      static_cast<std::int64_t>(key.y) - minimum_y + padding);
  }

  int column(const VoxelKey& key) const
  {
    return static_cast<int>(
      static_cast<std::int64_t>(key.x) - minimum_x + padding);
  }
};

cv::Mat ellipseKernel(const int radius)
{
  if (radius <= 0)
  {
    return cv::Mat();
  }
  return cv::getStructuringElement(
    cv::MORPH_ELLIPSE, cv::Size(2 * radius + 1, 2 * radius + 1));
}

std::vector<std::uint8_t> flattenMask(const cv::Mat& mask)
{
  std::vector<std::uint8_t> output(
    static_cast<std::size_t>(mask.rows) * static_cast<std::size_t>(mask.cols));
  for (int row = 0; row < mask.rows; ++row)
  {
    const std::uint8_t* source = mask.ptr<std::uint8_t>(row);
    std::copy(source, source + mask.cols,
              output.begin() + static_cast<std::size_t>(row) * mask.cols);
  }
  return output;
}

bool isBetterReference(
  const VoxelSnapshot& candidate, const VoxelSnapshot& reference,
  const VoxelSnapshot* current)
{
  if (current == nullptr)
  {
    return true;
  }
  const double distance = std::hypot(
    reference.x - candidate.x, reference.y - candidate.y);
  const double current_distance = std::hypot(
    current->x - candidate.x, current->y - candidate.y);
  constexpr double kEpsilon = 1e-9;
  if (distance + kEpsilon < current_distance)
  {
    return true;
  }
  if (std::fabs(distance - current_distance) > kEpsilon)
  {
    return false;
  }
  const double height = std::fabs(reference.z - candidate.z);
  const double current_height = std::fabs(current->z - candidate.z);
  if (height + kEpsilon < current_height)
  {
    return true;
  }
  if (std::fabs(height - current_height) > kEpsilon)
  {
    return false;
  }
  if (reference.semantic_confidence != current->semantic_confidence)
  {
    return reference.semantic_confidence > current->semantic_confidence;
  }
  return reference.label < current->label;
}

}  // namespace

const char* terrainBoundaryDecisionReasonName(
  const TerrainBoundaryDecisionReason reason)
{
  switch (reason)
  {
    case TerrainBoundaryDecisionReason::OutsideClosedTerrainSupport:
      return "outside_closed_terrain_support";
    case TerrainBoundaryDecisionReason::NoTerrainReference:
      return "no_terrain_reference";
    case TerrainBoundaryDecisionReason::HeightDifferenceTooHigh:
      return "height_difference_too_high";
    case TerrainBoundaryDecisionReason::Recovered:
      return "recovered";
    case TerrainBoundaryDecisionReason::Count:
      return "invalid_reason";
  }
  return "invalid_reason";
}

std::uint32_t terrainBoundaryDecisionReasonBit(
  const TerrainBoundaryDecisionReason reason)
{
  const std::size_t index = static_cast<std::size_t>(reason);
  if (reason == TerrainBoundaryDecisionReason::Recovered ||
      index >= static_cast<std::size_t>(TerrainBoundaryDecisionReason::Count) ||
      index >= 32u)
  {
    return 0u;
  }
  return std::uint32_t{1u} << index;
}

TerrainBoundaryFilter::TerrainBoundaryFilter(
  const TerrainBoundaryFilterConfig& config)
  : config_(config),
    terrain_labels_(config.terrain_labels.begin(), config.terrain_labels.end()),
    recoverable_labels_(config.recoverable_labels.begin(),
                        config.recoverable_labels.end()),
    excluded_labels_(config.excluded_labels.begin(), config.excluded_labels.end())
{
  bool overlapping_roles = false;
  for (const std::uint32_t label : terrain_labels_)
  {
    overlapping_roles = overlapping_roles ||
      recoverable_labels_.count(label) != 0u ||
      excluded_labels_.count(label) != 0u;
  }
  for (const std::uint32_t label : recoverable_labels_)
  {
    overlapping_roles = overlapping_roles || excluded_labels_.count(label) != 0u;
  }
  if (config_.terrain_labels.empty() || config_.recoverable_labels.empty() ||
      overlapping_roles || config_.closing_radius < 0.0 ||
      !std::isfinite(config_.closing_radius) ||
      config_.maximum_height_difference < 0.0 ||
      !std::isfinite(config_.maximum_height_difference))
  {
    throw std::invalid_argument("invalid terrain boundary filter configuration");
  }
}

bool TerrainBoundaryFilter::isTerrain(const std::uint32_t label) const
{
  return terrain_labels_.count(label) != 0u;
}

bool TerrainBoundaryFilter::isRecoverable(const std::uint32_t label) const
{
  return recoverable_labels_.count(label) != 0u;
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
    throw std::invalid_argument(
      "terrain boundary filter voxel sizes must be positive");
  }

  const double radius_value = std::ceil(config_.closing_radius / voxel_size_xy);
  const double height_cells_value = std::ceil(
    config_.maximum_height_difference / voxel_size_z);
  const double safe_radius = static_cast<double>(
    (std::numeric_limits<int>::max() - 1) / 2);
  if (radius_value > safe_radius || height_cells_value > safe_radius)
  {
    throw std::invalid_argument("terrain boundary raster kernel is too large");
  }
  const int radius_cells = static_cast<int>(radius_value);
  result.debug_statistics.closing_radius_cells = radius_cells;
  result.debug_statistics.maximum_height_difference_cells =
    static_cast<int>(height_cells_value);

  std::int32_t minimum_x = voxels.front().key.x;
  std::int32_t maximum_x = voxels.front().key.x;
  std::int32_t minimum_y = voxels.front().key.y;
  std::int32_t maximum_y = voxels.front().key.y;
  std::vector<std::size_t> terrain_indices;
  std::vector<std::size_t> recoverable_indices;
  terrain_indices.reserve(voxels.size());
  recoverable_indices.reserve(voxels.size());
  std::unordered_map<VoxelKey, std::vector<std::size_t>, VoxelKeyHash>
    terrain_columns;
  for (std::size_t index = 0u; index < voxels.size(); ++index)
  {
    const VoxelSnapshot& voxel = voxels[index];
    minimum_x = std::min(minimum_x, voxel.key.x);
    maximum_x = std::max(maximum_x, voxel.key.x);
    minimum_y = std::min(minimum_y, voxel.key.y);
    maximum_y = std::max(maximum_y, voxel.key.y);
    if (isTerrain(voxel.label))
    {
      terrain_indices.push_back(index);
      terrain_columns[VoxelKey{voxel.key.x, voxel.key.y, 0}].push_back(index);
      ++result.debug_statistics.terrain_voxels;
    }
    else if (isRecoverable(voxel.label))
    {
      recoverable_indices.push_back(index);
      ++result.debug_statistics.recoverable_static_voxels;
    }
    else if (excluded_labels_.count(voxel.label) != 0u)
    {
      ++result.debug_statistics.excluded_voxels;
    }
  }
  if (terrain_indices.empty() || recoverable_indices.empty())
  {
    return result;
  }

  const int padding = radius_cells + 1;
  const std::int64_t columns =
    static_cast<std::int64_t>(maximum_x) - minimum_x + 1 + 2 * padding;
  const std::int64_t rows =
    static_cast<std::int64_t>(maximum_y) - minimum_y + 1 + 2 * padding;
  if (columns <= 0 || rows <= 0 ||
      columns > std::numeric_limits<int>::max() ||
      rows > std::numeric_limits<int>::max())
  {
    throw std::invalid_argument("terrain boundary raster bounds are too large");
  }
  const RasterBounds bounds{minimum_x, minimum_y, padding,
                            static_cast<int>(rows),
                            static_cast<int>(columns)};

  cv::Mat terrain = cv::Mat::zeros(bounds.rows, bounds.columns, CV_8UC1);
  cv::Mat recoverable = cv::Mat::zeros(bounds.rows, bounds.columns, CV_8UC1);
  cv::Mat excluded = cv::Mat::zeros(bounds.rows, bounds.columns, CV_8UC1);
  for (const std::size_t index : terrain_indices)
  {
    const VoxelSnapshot& voxel = voxels[index];
    terrain.at<std::uint8_t>(bounds.row(voxel.key), bounds.column(voxel.key)) =
      255u;
  }
  for (const std::size_t index : recoverable_indices)
  {
    const VoxelSnapshot& voxel = voxels[index];
    recoverable.at<std::uint8_t>(bounds.row(voxel.key),
                                 bounds.column(voxel.key)) = 255u;
  }
  for (const VoxelSnapshot& voxel : voxels)
  {
    if (excluded_labels_.count(voxel.label) != 0u)
    {
      excluded.at<std::uint8_t>(bounds.row(voxel.key),
                                bounds.column(voxel.key)) = 255u;
    }
  }

  cv::Mat dilated;
  cv::Mat closed;
  if (radius_cells > 0)
  {
    const cv::Mat kernel = ellipseKernel(radius_cells);
    cv::dilate(terrain, dilated, kernel, cv::Point(-1, -1), 1,
               cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::erode(dilated, closed, kernel, cv::Point(-1, -1), 1,
              cv::BORDER_CONSTANT, cv::Scalar(0));
  }
  else
  {
    dilated = terrain.clone();
    closed = terrain.clone();
  }
  cv::Mat inverse_terrain;
  cv::Mat newly_filled;
  cv::Mat proposed;
  cv::bitwise_not(terrain, inverse_terrain);
  cv::bitwise_and(closed, inverse_terrain, newly_filled);
  // Use the complete morphologically supported terrain region, not only its
  // newly-filled subset. A recoverable static voxel may share an XY column
  // with an original terrain voxel at another Z; the full-height terrain
  // projection is already white in that case, so restricting proposals to
  // newly_filled would skip the voxel before the relative-height gate runs.
  // The recoverable mask still guarantees that no unobserved voxel is created,
  // and dynamic/ignored labels never enter this mask.
  cv::bitwise_and(closed, recoverable, proposed);
  result.debug_statistics.opencv_newly_filled_cells =
    static_cast<std::size_t>(cv::countNonZero(newly_filled));
  result.debug_statistics.opencv_proposed_cells =
    static_cast<std::size_t>(cv::countNonZero(proposed));

  cv::Mat reference_found = cv::Mat::zeros(
    bounds.rows, bounds.columns, CV_8UC1);
  cv::Mat recovered = cv::Mat::zeros(
    bounds.rows, bounds.columns, CV_8UC1);
  result.relabeled.reserve(recoverable_indices.size());
  if (config_.debug_enabled)
  {
    result.debug_records.reserve(recoverable_indices.size());
  }

  for (const std::size_t source_index : recoverable_indices)
  {
    const VoxelSnapshot& source = voxels[source_index];
    TerrainBoundaryDebugRecord record;
    record.key = source.key;
    record.x = source.x;
    record.y = source.y;
    record.z = source.z;
    record.original_label = source.label;
    const int row = bounds.row(source.key);
    const int column = bounds.column(source.key);
    record.closed_terrain = closed.at<std::uint8_t>(row, column) != 0u;
    record.proposed = proposed.at<std::uint8_t>(row, column) != 0u;
    if (record.proposed)
    {
      ++result.debug_statistics.opencv_proposed_voxels;
    }

    if (!record.proposed)
    {
      record.first_failure =
        TerrainBoundaryDecisionReason::OutsideClosedTerrainSupport;
      record.failure_mask = terrainBoundaryDecisionReasonBit(
        record.first_failure);
    }
    else
    {
      const VoxelSnapshot* nearest = nullptr;
      const VoxelSnapshot* nearest_height_valid = nullptr;
      for (int dx = -radius_cells; dx <= radius_cells; ++dx)
      {
        for (int dy = -radius_cells; dy <= radius_cells; ++dy)
        {
          const std::int64_t dx64 = dx;
          const std::int64_t dy64 = dy;
          const std::int64_t radius64 = radius_cells;
          if (dx64 * dx64 + dy64 * dy64 > radius64 * radius64)
          {
            continue;
          }
          const auto column_iterator = terrain_columns.find(VoxelKey{
            source.key.x + dx, source.key.y + dy, 0});
          if (column_iterator == terrain_columns.end())
          {
            continue;
          }
          for (const std::size_t reference_index : column_iterator->second)
          {
            const VoxelSnapshot& candidate_reference = voxels[reference_index];
            if (isBetterReference(source, candidate_reference, nearest))
            {
              nearest = &candidate_reference;
            }
            if (std::fabs(candidate_reference.z - source.z) <=
                  config_.maximum_height_difference + 1e-9 &&
                isBetterReference(
                  source, candidate_reference, nearest_height_valid))
            {
              nearest_height_valid = &candidate_reference;
            }
          }
        }
      }

      const VoxelSnapshot* diagnostic_reference =
        nearest_height_valid != nullptr ? nearest_height_valid : nearest;
      if (diagnostic_reference != nullptr)
      {
        record.reference_found = true;
        record.reference_key = diagnostic_reference->key;
        record.reference_x = diagnostic_reference->x;
        record.reference_y = diagnostic_reference->y;
        record.reference_z = diagnostic_reference->z;
        record.reference_label = diagnostic_reference->label;
        record.reference_distance_xy = std::hypot(
          diagnostic_reference->x - source.x,
          diagnostic_reference->y - source.y);
        record.height_difference = std::fabs(
          diagnostic_reference->z - source.z);
        reference_found.at<std::uint8_t>(row, column) = 255u;
        ++result.debug_statistics.reference_found_voxels;
      }

      if (nearest == nullptr)
      {
        record.first_failure = TerrainBoundaryDecisionReason::NoTerrainReference;
        record.failure_mask = terrainBoundaryDecisionReasonBit(
          record.first_failure);
      }
      else if (nearest_height_valid == nullptr)
      {
        record.first_failure =
          TerrainBoundaryDecisionReason::HeightDifferenceTooHigh;
        record.failure_mask = terrainBoundaryDecisionReasonBit(
          record.first_failure);
      }
      else
      {
        record.first_failure = TerrainBoundaryDecisionReason::Recovered;
        record.replacement_label = nearest_height_valid->label;
        VoxelSnapshot& replacement = result.voxels[source_index];
        replacement.label = nearest_height_valid->label;
        replacement.semantic_confidence =
          nearest_height_valid->semantic_confidence;
        replacement.semantic_cost = nearest_height_valid->semantic_cost;
        replacement.measured_traversability_cost =
          nearest_height_valid->measured_traversability_cost;
        replacement.has_measured_traversability =
          nearest_height_valid->has_measured_traversability;
        replacement.traversability_cost =
          nearest_height_valid->traversability_cost;
        recovered.at<std::uint8_t>(row, column) = 255u;
        result.relabeled.push_back(TerrainBoundaryRelabel{
          source.key, source.label, replacement.label});
      }
    }

    const std::size_t reason_index = static_cast<std::size_t>(
      record.first_failure);
    if (reason_index < result.debug_statistics.first_decision_counts.size())
    {
      ++result.debug_statistics.first_decision_counts[reason_index];
    }
    if (config_.debug_enabled)
    {
      result.debug_records.push_back(record);
    }
  }

  if (config_.debug_enabled)
  {
    TerrainBoundaryLayerDebug debug;
    debug.minimum_x = bounds.minimum_x - bounds.padding;
    debug.minimum_y = bounds.minimum_y - bounds.padding;
    debug.rows = bounds.rows;
    debug.columns = bounds.columns;
    debug.terrain_original = flattenMask(terrain);
    debug.terrain_after_dilation = flattenMask(dilated);
    debug.terrain_after_erosion = flattenMask(closed);
    debug.recoverable_static = flattenMask(recoverable);
    debug.excluded = flattenMask(excluded);
    debug.newly_filled = flattenMask(newly_filled);
    debug.proposed = flattenMask(proposed);
    debug.reference_found = flattenMask(reference_found);
    debug.recovered = flattenMask(recovered);
    result.debug_layers.push_back(std::move(debug));
  }
  return result;
}

}  // namespace local3d_semantic_voxel_map
