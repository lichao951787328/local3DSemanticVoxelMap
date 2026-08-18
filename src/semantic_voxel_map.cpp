#include "local3d_semantic_voxel_map/semantic_voxel_map.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace local3d_semantic_voxel_map
{

namespace
{
float clampUnit(const float value)
{
  return std::max(0.0f, std::min(1.0f, value));
}
}  // namespace

std::size_t VoxelKeyHash::operator()(const VoxelKey& key) const
{
  std::size_t seed = std::hash<std::int32_t>()(key.x);
  seed ^= std::hash<std::int32_t>()(key.y) + 0x9e3779b9u + (seed << 6u) + (seed >> 2u);
  seed ^= std::hash<std::int32_t>()(key.z) + 0x9e3779b9u + (seed << 6u) + (seed >> 2u);
  return seed;
}

SemanticVoxelMap::SemanticVoxelMap(const SemanticVoxelMapConfig& config)
  : config_(config)
{
  if (config_.voxel_size <= 0.0)
  {
    config_.voxel_size = 0.10;
  }
  config_.unknown_cost = clampUnit(config_.unknown_cost);
  config_.semantic_cost_weight = clampUnit(config_.semantic_cost_weight);
  config_.semantic_risk_alpha = clampUnit(config_.semantic_risk_alpha);
  config_.cost_rise_alpha = clampUnit(config_.cost_rise_alpha);
  config_.cost_fall_alpha = clampUnit(config_.cost_fall_alpha);
}

void SemanticVoxelMap::setSemanticClasses(const std::vector<SemanticClass>& classes)
{
  std::lock_guard<std::mutex> lock(mutex_);
  classes_.clear();
  for (auto semantic_class : classes)
  {
    semantic_class.traversability_cost = clampUnit(semantic_class.traversability_cost);
    classes_[semantic_class.label] = semantic_class;
  }
}

VoxelKey SemanticVoxelMap::worldToKey(const double x, const double y, const double z) const
{
  VoxelKey key;
  key.x = static_cast<std::int32_t>(std::floor(x / config_.voxel_size));
  key.y = static_cast<std::int32_t>(std::floor(y / config_.voxel_size));
  key.z = static_cast<std::int32_t>(std::floor(z / config_.voxel_size));
  return key;
}

void SemanticVoxelMap::keyToWorld(const VoxelKey& key,
                                  double& x, double& y, double& z) const
{
  const double half = config_.voxel_size * 0.5;
  x = static_cast<double>(key.x) * config_.voxel_size + half;
  y = static_cast<double>(key.y) * config_.voxel_size + half;
  z = static_cast<double>(key.z) * config_.voxel_size + half;
}

void SemanticVoxelMap::integrate(const VoxelKey& key,
                                 const VoxelObservation& observation)
{
  const bool has_semantic = observation.label != kInvalidSemanticLabel &&
                            observation.semantic_confidence > 0.0f;
  if (!has_semantic && !observation.has_traversability_cost)
  {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  auto found = voxels_.find(key);
  if (found == voxels_.end())
  {
    found = voxels_.emplace(key, SemanticVoxel(config_.semantic_fusion)).first;
    found->second.semantic_cost = config_.unknown_cost;
    found->second.measured_traversability_cost = config_.unknown_cost;
    found->second.traversability_cost = config_.unknown_cost;
  }

  SemanticVoxel& voxel = found->second;
  const float semantic_confidence = clampUnit(observation.semantic_confidence);
  if (has_semantic)
  {
    voxel.semantics.fuse(observation.label, semantic_confidence);
    voxel.semantic_cost = expectedSemanticCost(voxel.semantics);
    ++voxel.semantic_observation_count;
  }

  if (observation.has_traversability_cost)
  {
    const float measured_cost = clampUnit(observation.traversability_cost);
    if (!voxel.has_measured_traversability)
    {
      voxel.measured_traversability_cost = measured_cost;
      voxel.has_measured_traversability = true;
    }
    else
    {
      const float alpha = measured_cost >= voxel.measured_traversability_cost ?
        config_.cost_rise_alpha : config_.cost_fall_alpha;
      voxel.measured_traversability_cost = clampUnit(
        voxel.measured_traversability_cost +
        alpha * (measured_cost - voxel.measured_traversability_cost));
    }
    ++voxel.traversability_observation_count;
  }

  voxel.traversability_cost = combinedTraversabilityCost(voxel);
  voxel.last_observed = observation.stamp.isZero() ? ros::Time::now() : observation.stamp;
  ++voxel.observation_count;

  // Evict in batches instead of sorting the whole map for every insertion once
  // capacity is reached. A small temporary overshoot keeps high-rate clouds
  // from turning capacity enforcement into an O(N) operation per point.
  const std::size_t eviction_batch = std::max<std::size_t>(
    1024u, config_.max_voxels / 100u);
  if (config_.max_voxels > 0 &&
      voxels_.size() > config_.max_voxels + eviction_batch)
  {
    enforceCapacity();
  }
}

void SemanticVoxelMap::integrate(const double x, const double y, const double z,
                                 const VoxelObservation& observation)
{
  integrate(worldToKey(x, y, z), observation);
}

std::size_t SemanticVoxelMap::prune(const ros::Time& reference_stamp)
{
  if (config_.decay_seconds < 0.0)
  {
    return 0;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  std::size_t removed = 0;
  for (auto iterator = voxels_.begin(); iterator != voxels_.end();)
  {
    if ((reference_stamp - iterator->second.last_observed).toSec() >
        config_.decay_seconds)
    {
      iterator = voxels_.erase(iterator);
      ++removed;
    }
    else
    {
      ++iterator;
    }
  }
  return removed;
}

std::size_t SemanticVoxelMap::pruneOutside(const double center_x,
                                           const double center_y,
                                           const double center_z,
                                           const double radius)
{
  if (radius <= 0.0)
  {
    return 0;
  }

  const double radius_squared = radius * radius;
  std::lock_guard<std::mutex> lock(mutex_);
  std::size_t removed = 0;
  for (auto iterator = voxels_.begin(); iterator != voxels_.end();)
  {
    double x, y, z;
    keyToWorld(iterator->first, x, y, z);
    const double dx = x - center_x;
    const double dy = y - center_y;
    const double dz = z - center_z;
    if (dx * dx + dy * dy + dz * dz > radius_squared)
    {
      iterator = voxels_.erase(iterator);
      ++removed;
    }
    else
    {
      ++iterator;
    }
  }
  return removed;
}

std::size_t SemanticVoxelMap::pruneOutsideBox(
  const double center_x, const double center_y, const double center_z,
  const std::array<double, 4>& box_to_world_quaternion,
  const std::array<double, 3>& minimum,
  const std::array<double, 3>& maximum)
{
  if (minimum[0] >= maximum[0] || minimum[1] >= maximum[1] ||
      minimum[2] >= maximum[2])
  {
    return 0;
  }

  double qx = box_to_world_quaternion[0];
  double qy = box_to_world_quaternion[1];
  double qz = box_to_world_quaternion[2];
  double qw = box_to_world_quaternion[3];
  const double norm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  if (norm > 1e-12)
  {
    qx /= norm;
    qy /= norm;
    qz /= norm;
    qw /= norm;
  }
  else
  {
    qx = qy = qz = 0.0;
    qw = 1.0;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  std::size_t removed = 0;
  for (auto iterator = voxels_.begin(); iterator != voxels_.end();)
  {
    double x, y, z;
    keyToWorld(iterator->first, x, y, z);
    const double dx = x - center_x;
    const double dy = y - center_y;
    const double dz = z - center_z;

    // Rotate map-frame displacement by the inverse box orientation.  The
    // resulting coordinates are relative to the current input sensor frame.
    const double tx = 2.0 * (-qy * dz + qz * dy);
    const double ty = 2.0 * (-qz * dx + qx * dz);
    const double tz = 2.0 * (-qx * dy + qy * dx);
    const double local_x = dx + qw * tx + (-qy * tz + qz * ty);
    const double local_y = dy + qw * ty + (-qz * tx + qx * tz);
    const double local_z = dz + qw * tz + (-qx * ty + qy * tx);

    if (local_x < minimum[0] || local_x > maximum[0] ||
        local_y < minimum[1] || local_y > maximum[1] ||
        local_z < minimum[2] || local_z > maximum[2])
    {
      iterator = voxels_.erase(iterator);
      ++removed;
    }
    else
    {
      ++iterator;
    }
  }
  return removed;
}

void SemanticVoxelMap::clear()
{
  std::lock_guard<std::mutex> lock(mutex_);
  voxels_.clear();
}

std::size_t SemanticVoxelMap::size() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return voxels_.size();
}

std::vector<VoxelSnapshot> SemanticVoxelMap::snapshot() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<VoxelSnapshot> output;
  output.reserve(voxels_.size());
  for (const auto& entry : voxels_)
  {
    VoxelSnapshot item;
    item.key = entry.first;
    keyToWorld(entry.first, item.x, item.y, item.z);
    item.label = entry.second.semantics.dominantLabel();
    item.semantic_confidence = entry.second.semantics.dominantConfidence();
    item.semantic_cost = entry.second.semantic_cost;
    item.measured_traversability_cost = entry.second.measured_traversability_cost;
    item.has_measured_traversability = entry.second.has_measured_traversability;
    item.traversability_cost = entry.second.traversability_cost;
    item.observation_count = entry.second.observation_count;
    item.semantic_observation_count = entry.second.semantic_observation_count;
    item.traversability_observation_count =
      entry.second.traversability_observation_count;
    item.last_observed = entry.second.last_observed;
    output.push_back(item);
  }
  return output;
}

std::vector<TraversabilityColumnSnapshot>
SemanticVoxelMap::traversabilityColumns() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  std::unordered_map<std::uint64_t, TraversabilityColumnSnapshot> columns;
  columns.reserve(voxels_.size());
  const double half = config_.voxel_size * 0.5;
  for (const auto& entry : voxels_)
  {
    const std::uint64_t key =
      (static_cast<std::uint64_t>(static_cast<std::uint32_t>(entry.first.x)) << 32u) |
      static_cast<std::uint32_t>(entry.first.y);
    const double z = static_cast<double>(entry.first.z) * config_.voxel_size + half;
    const auto found = columns.find(key);
    if (found == columns.end())
    {
      TraversabilityColumnSnapshot column;
      column.x_index = entry.first.x;
      column.y_index = entry.first.y;
      column.x = static_cast<double>(entry.first.x) * config_.voxel_size + half;
      column.y = static_cast<double>(entry.first.y) * config_.voxel_size + half;
      column.z = z;
      column.traversability_cost = entry.second.traversability_cost;
      columns.emplace(key, column);
    }
    else if (entry.second.traversability_cost > found->second.traversability_cost ||
             (entry.second.traversability_cost == found->second.traversability_cost &&
              z < found->second.z))
    {
      found->second.z = z;
      found->second.traversability_cost = entry.second.traversability_cost;
    }
  }

  std::vector<TraversabilityColumnSnapshot> output;
  output.reserve(columns.size());
  for (const auto& entry : columns)
  {
    output.push_back(entry.second);
  }
  return output;
}

std::size_t applyTerrainHeightDiscontinuityCost(
  std::vector<VoxelSnapshot>& voxels, const double voxel_size,
  const TerrainHeightCostConfig& config)
{
  if (!config.enabled || voxel_size <= 0.0 ||
      config.height_difference_threshold < 0.0 ||
      config.neighborhood_radius < 0.0)
  {
    return 0u;
  }

  struct TerrainColumn
  {
    std::int32_t x = 0;
    std::int32_t y = 0;
    double minimum_z = std::numeric_limits<double>::max();
    double maximum_z = -std::numeric_limits<double>::max();
    std::vector<std::size_t> voxel_indices;
  };

  const auto column_key = [](const std::int32_t x, const std::int32_t y)
  {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) << 32u) |
           static_cast<std::uint32_t>(y);
  };
  const auto is_terrain = [](const std::uint32_t label)
  {
    return label == 0u || label == 1u || label == 9u;
  };

  std::unordered_map<std::uint64_t, TerrainColumn> columns;
  columns.reserve(voxels.size());
  for (std::size_t index = 0u; index < voxels.size(); ++index)
  {
    const VoxelSnapshot& voxel = voxels[index];
    if (!is_terrain(voxel.label) ||
        (config.only_without_measured_traversability &&
         voxel.has_measured_traversability))
    {
      continue;
    }
    TerrainColumn& column = columns[column_key(voxel.key.x, voxel.key.y)];
    column.x = voxel.key.x;
    column.y = voxel.key.y;
    column.minimum_z = std::min(column.minimum_z, voxel.z);
    column.maximum_z = std::max(column.maximum_z, voxel.z);
    column.voxel_indices.push_back(index);
  }

  std::vector<bool> discontinuity(voxels.size(), false);
  const auto mark_column = [&discontinuity](const TerrainColumn& column)
  {
    for (const std::size_t index : column.voxel_indices)
    {
      discontinuity[index] = true;
    }
  };
  const double threshold = config.height_difference_threshold;
  for (const auto& item : columns)
  {
    const TerrainColumn& column = item.second;
    if (column.maximum_z - column.minimum_z > threshold)
    {
      mark_column(column);
    }
  }

  const int cell_radius = static_cast<int>(
    std::ceil(config.neighborhood_radius / voxel_size));
  const double radius_squared =
    config.neighborhood_radius * config.neighborhood_radius + 1e-12;
  for (const auto& item : columns)
  {
    const TerrainColumn& column = item.second;
    for (int delta_x = -cell_radius; delta_x <= cell_radius; ++delta_x)
    {
      for (int delta_y = -cell_radius; delta_y <= cell_radius; ++delta_y)
      {
        // Compare every pair once and never compare a column with itself.
        if (delta_x < 0 || (delta_x == 0 && delta_y <= 0))
        {
          continue;
        }
        const double planar_distance_squared = voxel_size * voxel_size *
          static_cast<double>(delta_x * delta_x + delta_y * delta_y);
        if (planar_distance_squared > radius_squared)
        {
          continue;
        }
        const auto neighbor = columns.find(column_key(
          column.x + delta_x, column.y + delta_y));
        if (neighbor == columns.end())
        {
          continue;
        }
        const TerrainColumn& other = neighbor->second;
        const double maximum_separation = std::max(
          std::abs(column.maximum_z - other.minimum_z),
          std::abs(other.maximum_z - column.minimum_z));
        if (maximum_separation > threshold)
        {
          mark_column(column);
          mark_column(other);
        }
      }
    }
  }

  std::size_t changed = 0u;
  for (std::size_t index = 0u; index < voxels.size(); ++index)
  {
    if (discontinuity[index] &&
        voxels[index].traversability_cost < config.obstacle_cost)
    {
      voxels[index].traversability_cost = config.obstacle_cost;
      ++changed;
    }
  }
  return changed;
}

std::vector<TraversabilityColumnSnapshot> projectTraversabilityColumns(
  const std::vector<VoxelSnapshot>& voxels)
{
  std::unordered_map<std::uint64_t, TraversabilityColumnSnapshot> columns;
  columns.reserve(voxels.size());
  for (const VoxelSnapshot& voxel : voxels)
  {
    const std::uint64_t key =
      (static_cast<std::uint64_t>(static_cast<std::uint32_t>(voxel.key.x)) << 32u) |
      static_cast<std::uint32_t>(voxel.key.y);
    const auto found = columns.find(key);
    if (found == columns.end())
    {
      TraversabilityColumnSnapshot column;
      column.x_index = voxel.key.x;
      column.y_index = voxel.key.y;
      column.x = voxel.x;
      column.y = voxel.y;
      column.z = voxel.z;
      column.traversability_cost = voxel.traversability_cost;
      columns.emplace(key, column);
    }
    else if (voxel.traversability_cost > found->second.traversability_cost ||
             (voxel.traversability_cost == found->second.traversability_cost &&
              voxel.z < found->second.z))
    {
      found->second.z = voxel.z;
      found->second.traversability_cost = voxel.traversability_cost;
    }
  }

  std::vector<TraversabilityColumnSnapshot> output;
  output.reserve(columns.size());
  for (const auto& item : columns)
  {
    output.push_back(item.second);
  }
  return output;
}

float SemanticVoxelMap::classCost(const std::uint32_t label) const
{
  std::lock_guard<std::mutex> lock(mutex_);
  const auto found = classes_.find(label);
  return found == classes_.end() ? config_.unknown_cost : found->second.traversability_cost;
}

SemanticClass SemanticVoxelMap::classDescription(const std::uint32_t label) const
{
  std::lock_guard<std::mutex> lock(mutex_);
  const auto found = classes_.find(label);
  if (found != classes_.end())
  {
    return found->second;
  }

  SemanticClass fallback;
  fallback.label = label;
  fallback.name = "unconfigured";
  fallback.traversability_cost = config_.unknown_cost;
  // SSMI semantic labels are packed RGB values. Use them directly when no
  // explicit palette entry exists.
  fallback.red = static_cast<std::uint8_t>((label >> 16u) & 0xffu);
  fallback.green = static_cast<std::uint8_t>((label >> 8u) & 0xffu);
  fallback.blue = static_cast<std::uint8_t>(label & 0xffu);
  return fallback;
}

double SemanticVoxelMap::voxelSize() const
{
  return config_.voxel_size;
}

bool SemanticVoxelMap::saveCsv(const std::string& path, std::string& error) const
{
  std::lock_guard<std::mutex> lock(mutex_);
  std::ofstream stream(path.c_str());
  if (!stream)
  {
    error = "cannot open map file for writing: " + path;
    return false;
  }

  stream << "# local3d_semantic_voxel_map v2 voxel_size=" << config_.voxel_size << '\n';
  stream << "x,y,z,label0,loge0,label1,loge1,label2,loge2,others,"
            "semantic_cost,measured_cost,has_measured,final_cost,observations,"
            "semantic_observations,traversability_observations,last_observed\n";
  stream << std::setprecision(10);
  for (const auto& entry : voxels_)
  {
    const auto& hypotheses = entry.second.semantics.hypotheses();
    stream << entry.first.x << ',' << entry.first.y << ',' << entry.first.z;
    for (const auto& hypothesis : hypotheses)
    {
      stream << ',' << hypothesis.label << ',' << hypothesis.log_evidence;
    }
    stream << ',' << entry.second.semantics.othersLogEvidence()
           << ',' << entry.second.semantic_cost
           << ',' << entry.second.measured_traversability_cost
           << ',' << (entry.second.has_measured_traversability ? 1 : 0)
           << ',' << entry.second.traversability_cost
           << ',' << entry.second.observation_count
           << ',' << entry.second.semantic_observation_count
           << ',' << entry.second.traversability_observation_count
           << ',' << entry.second.last_observed.toSec() << '\n';
  }
  if (!stream.good())
  {
    error = "failed while writing map file: " + path;
    return false;
  }
  return true;
}

bool SemanticVoxelMap::loadCsv(const std::string& path, std::string& error)
{
  std::ifstream stream(path.c_str());
  if (!stream)
  {
    error = "cannot open map file for reading: " + path;
    return false;
  }

  std::unordered_map<VoxelKey, SemanticVoxel, VoxelKeyHash> loaded;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(stream, line))
  {
    ++line_number;
    if (line.empty() || line[0] == '#' || line.compare(0, 5, "x,y,z") == 0)
    {
      continue;
    }
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream fields(line);
    VoxelKey key;
    std::array<SemanticHypothesis, kSemanticTopK> hypotheses;
    float others = 0.0f;
    if (!(fields >> key.x >> key.y >> key.z))
    {
      error = "invalid voxel key at line " + std::to_string(line_number);
      return false;
    }
    for (auto& hypothesis : hypotheses)
    {
      if (!(fields >> hypothesis.label >> hypothesis.log_evidence))
      {
        error = "invalid semantic evidence at line " + std::to_string(line_number);
        return false;
      }
    }
    if (!(fields >> others))
    {
      error = "invalid voxel state at line " + std::to_string(line_number);
      return false;
    }

    std::vector<double> state;
    double value = 0.0;
    while (fields >> value)
    {
      state.push_back(value);
    }
    if (state.size() != 3u && state.size() != 8u)
    {
      error = "invalid voxel state field count at line " +
              std::to_string(line_number);
      return false;
    }

    SemanticVoxel voxel(config_.semantic_fusion);
    voxel.semantics.restore(hypotheses, others);
    if (state.size() == 8u)
    {
      voxel.semantic_cost = clampUnit(static_cast<float>(state[0]));
      voxel.measured_traversability_cost = clampUnit(static_cast<float>(state[1]));
      voxel.has_measured_traversability = state[2] != 0.0;
      voxel.traversability_cost = clampUnit(static_cast<float>(state[3]));
      voxel.observation_count = static_cast<std::uint32_t>(state[4]);
      voxel.semantic_observation_count = static_cast<std::uint32_t>(state[5]);
      voxel.traversability_observation_count = static_cast<std::uint32_t>(state[6]);
      voxel.last_observed.fromSec(state[7]);
    }
    else
    {
      // Version 1 stored only the final cost. Treat it as a measured value so
      // loading an old map preserves its navigation behavior.
      voxel.semantic_cost = expectedSemanticCost(voxel.semantics);
      voxel.measured_traversability_cost = clampUnit(static_cast<float>(state[0]));
      voxel.has_measured_traversability = true;
      voxel.traversability_cost = voxel.measured_traversability_cost;
      voxel.observation_count = static_cast<std::uint32_t>(state[1]);
      voxel.semantic_observation_count = voxel.observation_count;
      voxel.traversability_observation_count = voxel.observation_count;
      voxel.last_observed.fromSec(state[2]);
    }
    loaded.emplace(key, voxel);
  }

  std::lock_guard<std::mutex> lock(mutex_);
  voxels_.swap(loaded);
  enforceCapacity();
  return true;
}

float SemanticVoxelMap::expectedSemanticCost(const SemanticEvidence& semantics) const
{
  float expected = semantics.otherProbability() * config_.unknown_cost;
  for (const auto& item : semantics.probabilities())
  {
    const auto found = classes_.find(item.first);
    const float cost = found == classes_.end() ?
      config_.unknown_cost : found->second.traversability_cost;
    expected += item.second * cost;
  }
  return clampUnit(expected);
}

float SemanticVoxelMap::combinedTraversabilityCost(const SemanticVoxel& voxel) const
{
  const bool has_semantic = !voxel.semantics.empty();
  if (!has_semantic)
  {
    return voxel.has_measured_traversability ?
      voxel.measured_traversability_cost : config_.unknown_cost;
  }
  if (!voxel.has_measured_traversability)
  {
    return voxel.semantic_cost;
  }
  if (config_.traversability_fusion_method == TraversabilityFusionMethod::Maximum)
  {
    return std::max(voxel.semantic_cost, voxel.measured_traversability_cost);
  }
  if (config_.traversability_fusion_method ==
      TraversabilityFusionMethod::ConfidenceWeightedRaise)
  {
    const float semantic_confidence = voxel.semantics.dominantConfidence();
    const float risk_gap = std::max(
      0.0f, voxel.semantic_cost - voxel.measured_traversability_cost);
    return clampUnit(voxel.measured_traversability_cost +
      config_.semantic_risk_alpha * semantic_confidence * risk_gap);
  }
  return clampUnit(config_.semantic_cost_weight * voxel.semantic_cost +
    (1.0f - config_.semantic_cost_weight) * voxel.measured_traversability_cost);
}

void SemanticVoxelMap::enforceCapacity()
{
  if (config_.max_voxels == 0 || voxels_.size() <= config_.max_voxels)
  {
    return;
  }

  std::vector<std::pair<ros::Time, VoxelKey>> by_age;
  by_age.reserve(voxels_.size());
  for (const auto& entry : voxels_)
  {
    by_age.emplace_back(entry.second.last_observed, entry.first);
  }
  const std::size_t remove_count = voxels_.size() - config_.max_voxels;
  std::nth_element(by_age.begin(), by_age.begin() + remove_count, by_age.end(),
                   [](const auto& lhs, const auto& rhs) {
                     return lhs.first < rhs.first;
                   });
  for (std::size_t index = 0; index < remove_count; ++index)
  {
    voxels_.erase(by_age[index].second);
  }
}

}  // namespace local3d_semantic_voxel_map
