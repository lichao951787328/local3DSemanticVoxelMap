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
  if (observation.label == kInvalidSemanticLabel)
  {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  auto found = voxels_.find(key);
  if (found == voxels_.end())
  {
    found = voxels_.emplace(key, SemanticVoxel(config_.semantic_fusion)).first;
    found->second.traversability_cost = config_.unknown_cost;
  }

  SemanticVoxel& voxel = found->second;
  const float confidence = clampUnit(observation.semantic_confidence);
  voxel.semantics.fuse(observation.label, confidence);

  const float semantic_cost = expectedSemanticCost(voxel.semantics);
  float target_cost = semantic_cost;
  if (observation.has_traversability_cost)
  {
    const float measured_cost = clampUnit(observation.traversability_cost);
    target_cost = config_.semantic_cost_weight * semantic_cost +
                  (1.0f - config_.semantic_cost_weight) * measured_cost;
  }

  const float alpha = target_cost >= voxel.traversability_cost ?
    config_.cost_rise_alpha : config_.cost_fall_alpha;
  voxel.traversability_cost = clampUnit(voxel.traversability_cost +
    alpha * confidence * (target_cost - voxel.traversability_cost));
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
    item.traversability_cost = entry.second.traversability_cost;
    item.observation_count = entry.second.observation_count;
    item.last_observed = entry.second.last_observed;
    output.push_back(item);
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

  stream << "# local3d_semantic_voxel_map v1 voxel_size=" << config_.voxel_size << '\n';
  stream << "x,y,z,label0,loge0,label1,loge1,label2,loge2,others,cost,observations,last_observed\n";
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
           << ',' << entry.second.traversability_cost
           << ',' << entry.second.observation_count
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
    float cost = config_.unknown_cost;
    std::uint32_t observations = 0;
    double stamp = 0.0;
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
    if (!(fields >> others >> cost >> observations >> stamp))
    {
      error = "invalid voxel state at line " + std::to_string(line_number);
      return false;
    }

    SemanticVoxel voxel(config_.semantic_fusion);
    voxel.semantics.restore(hypotheses, others);
    voxel.traversability_cost = clampUnit(cost);
    voxel.observation_count = observations;
    voxel.last_observed.fromSec(stamp);
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
