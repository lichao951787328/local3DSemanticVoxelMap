#include "local3d_semantic_voxel_map/semantic_fusion.hpp"
#include "local3d_semantic_voxel_map/semantic_voxel_map.hpp"
#include "local3d_semantic_voxel_map/global_semantic_admission.hpp"
#include "local3d_semantic_voxel_map/obstacle_revocation.hpp"
#include "local3d_semantic_voxel_map/ssmi_semantic_encoding.hpp"
#include "local3d_semantic_voxel_map/terrain_boundary_filter.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <set>
#include <string>
#include <unistd.h>

namespace map = local3d_semantic_voxel_map;

namespace
{

map::AdmissionObservation admissionObservation(
  const double x, const double y, const double z, const std::uint32_t label,
  const float cost = 0.8f)
{
  map::AdmissionObservation observation;
  observation.x = x;
  observation.y = y;
  observation.z = z;
  observation.label = label;
  observation.has_semantic = label != map::kInvalidSemanticLabel;
  observation.semantic_confidence = observation.has_semantic ? 1.0f : 0.0f;
  observation.traversability = cost;
  observation.has_traversability = true;
  return observation;
}

bool containsAdmissionLabel(const std::vector<map::AdmissionPoint>& points,
                            const std::uint32_t label)
{
  return std::any_of(points.begin(), points.end(),
    [label](const map::AdmissionPoint& point) { return point.label == label; });
}

map::VoxelSnapshot voxelSnapshot(
  const std::int32_t x, const std::int32_t y, const std::int32_t z,
  const std::uint32_t label, const float cost = 0.05f,
  const bool has_measured = false)
{
  map::VoxelSnapshot voxel;
  voxel.key = map::VoxelKey{x, y, z};
  voxel.x = 0.1 * static_cast<double>(x) + 0.05;
  voxel.y = 0.1 * static_cast<double>(y) + 0.05;
  voxel.z = 0.1 * static_cast<double>(z) + 0.05;
  voxel.label = label;
  voxel.semantic_confidence = 1.0f;
  voxel.traversability_cost = cost;
  voxel.has_measured_traversability = has_measured;
  voxel.measured_traversability_cost = cost;
  return voxel;
}

ros::Time stamp(const double seconds)
{
  ros::Time output;
  output.fromSec(seconds);
  return output;
}

}  // namespace

TEST(SsmiSemanticEncoding, UsesCanonicalLabelColorAfterSourceNormalization)
{
  const map::SemanticRgb road = map::ssmiSemanticColor(0u, 0.10f, 0.75f);
  const map::SemanticRgb vehicle = map::ssmiSemanticColor(13u, 0.10f, 0.75f);

  EXPECT_NE(map::packSemanticRgb(road), map::packSemanticRgb(vehicle));
  EXPECT_EQ(0x00804080u, map::packSemanticRgb(road));
}

TEST(SsmiSemanticEncoding, CanonicalLabelsHaveDistinctOnWireEncodings)
{
  std::set<std::uint32_t> encodings;
  for (std::uint32_t label = 0u; label <= 18u; ++label)
  {
    encodings.insert(map::packSemanticRgb(
      map::ssmiSemanticColor(label, 0.10f, 0.75f)));
  }
  EXPECT_EQ(19u, encodings.size());
}

TEST(SsmiSemanticEncoding, HighCostOverridesTerrainWithObstacleEncoding)
{
  const map::SemanticRgb high_cost =
    map::ssmiSemanticColor(9u, 0.75f, 0.75f);
  const map::SemanticRgb wall = map::ssmiSemanticColor(3u, 0.10f, 0.75f);
  const map::SemanticRgb terrain = map::ssmiSemanticColor(9u, 0.74f, 0.75f);

  EXPECT_EQ(map::packSemanticRgb(wall), map::packSemanticRgb(high_cost));
  EXPECT_NE(map::packSemanticRgb(terrain), map::packSemanticRgb(high_cost));
}

TEST(SsmiSemanticEncoding, UnknownLowCostLabelUsesNeutralEncoding)
{
  const map::SemanticRgb unknown =
    map::ssmiSemanticColor(map::kInvalidSemanticLabel, 0.10f, 0.75f);
  EXPECT_EQ(0x007f7f7fu, map::packSemanticRgb(unknown));
}

TEST(ObstacleRevocation, AbsenceAloneNeverRevokesRememberedObstacle)
{
  map::ObstacleRevocationConfig config;
  config.minimum_free_frames = 2u;
  config.minimum_free_duration = 0.1;
  map::ObstacleRevocationTracker tracker(config);
  const std::vector<map::VoxelSnapshot> obstacle{
    voxelSnapshot(10, 0, 0, 2u, 0.95f)};
  const std::unordered_set<map::VoxelKey, map::VoxelKeyHash> no_rays;

  tracker.update(obstacle, no_rays, stamp(1.0));
  EXPECT_EQ(1u, tracker.trackedCount());
  EXPECT_TRUE(tracker.update({}, no_rays, stamp(1.1)).revoked_free.empty());
  EXPECT_TRUE(tracker.update({}, no_rays, stamp(1.2)).revoked_free.empty());
  EXPECT_EQ(1u, tracker.trackedCount());
}

TEST(ObstacleRevocation, ConsecutiveLowCostTerrainRevokesOldObstacle)
{
  map::ObstacleRevocationConfig config;
  config.minimum_free_frames = 3u;
  config.minimum_free_duration = 0.2;
  map::ObstacleRevocationTracker tracker(config);
  const std::unordered_set<map::VoxelKey, map::VoxelKeyHash> no_rays;
  tracker.update({voxelSnapshot(10, 0, 0, 2u, 0.95f)}, no_rays,
                 stamp(1.0));
  std::vector<map::VoxelSnapshot> terrain{
    voxelSnapshot(10, 0, 0, 0u, 0.05f)};

  terrain.front().last_observed = stamp(1.1);
  EXPECT_TRUE(tracker.update(terrain, no_rays, stamp(1.1)).revoked_free.empty());
  terrain.front().last_observed = stamp(1.2);
  EXPECT_TRUE(tracker.update(terrain, no_rays, stamp(1.2)).revoked_free.empty());
  terrain.front().last_observed = stamp(1.3);
  const map::ObstacleRevocationResult result =
    tracker.update(terrain, no_rays, stamp(1.3));
  ASSERT_EQ(1u, result.revoked_free.size());
  EXPECT_EQ(0u, tracker.trackedCount());
  EXPECT_EQ(tracker.keyFor(1.05, 0.05, 0.05),
            result.revoked_free.front().key);
}

TEST(ObstacleRevocation, DynamicOcclusionBreaksFreeContradiction)
{
  map::ObstacleRevocationConfig config;
  config.minimum_free_frames = 2u;
  config.minimum_free_duration = 0.1;
  map::ObstacleRevocationTracker tracker(config);
  const std::unordered_set<map::VoxelKey, map::VoxelKeyHash> no_rays;
  tracker.update({voxelSnapshot(10, 0, 0, 2u, 0.95f)}, no_rays,
                 stamp(1.0));
  std::vector<map::VoxelSnapshot> terrain{
    voxelSnapshot(10, 0, 0, 0u, 0.05f)};
  const std::vector<map::VoxelSnapshot> dynamic{
    voxelSnapshot(10, 0, 0, 11u, 1.0f)};

  terrain.front().last_observed = stamp(1.1);
  tracker.update(terrain, no_rays, stamp(1.1));
  tracker.update(dynamic, no_rays, stamp(1.2));
  terrain.front().last_observed = stamp(1.3);
  EXPECT_TRUE(tracker.update(terrain, no_rays, stamp(1.3)).revoked_free.empty());
  terrain.front().last_observed = stamp(1.4);
  EXPECT_EQ(1u,
    tracker.update(terrain, no_rays, stamp(1.4)).revoked_free.size());
}

TEST(ObstacleRevocation, CachedTerrainIsNotRepeatedFreeEvidence)
{
  map::ObstacleRevocationConfig config;
  config.minimum_free_frames = 2u;
  config.minimum_free_duration = 0.1;
  map::ObstacleRevocationTracker tracker(config);
  const std::unordered_set<map::VoxelKey, map::VoxelKeyHash> no_rays;
  tracker.update({voxelSnapshot(10, 0, 0, 2u, 0.95f)}, no_rays,
                 stamp(1.0));
  std::vector<map::VoxelSnapshot> cached_terrain{
    voxelSnapshot(10, 0, 0, 0u, 0.05f)};
  cached_terrain.front().last_observed = stamp(1.1);

  tracker.update(cached_terrain, no_rays, stamp(1.1));
  EXPECT_TRUE(
    tracker.update(cached_terrain, no_rays, stamp(1.2)).revoked_free.empty());
  EXPECT_EQ(1u, tracker.trackedCount());
}

TEST(ObstacleRevocation, RayTraversalClearsFalseStaticTrailButNotEndpoint)
{
  map::ObstacleRevocationConfig config;
  config.minimum_free_frames = 1u;
  config.minimum_free_duration = 0.0;
  config.ray_endpoint_margin = 0.20;
  map::ObstacleRevocationTracker tracker(config);
  const std::unordered_set<map::VoxelKey, map::VoxelKeyHash> no_rays;
  tracker.update({voxelSnapshot(10, 0, 0, 2u, 0.95f)}, no_rays,
                 stamp(1.0));

  std::unordered_set<map::VoxelKey, map::VoxelKeyHash> ray_evidence;
  tracker.collectTrackedRayEvidence(
    0.05, 0.05, 0.05, 3.05, 0.05, 0.05, ray_evidence);
  ASSERT_EQ(1u, ray_evidence.size());
  EXPECT_EQ(1u,
    tracker.update({}, ray_evidence, stamp(1.1)).revoked_free.size());

  map::ObstacleRevocationTracker endpoint_tracker(config);
  endpoint_tracker.update(
    {voxelSnapshot(30, 0, 0, 2u, 0.95f)}, no_rays, stamp(1.0));
  ray_evidence.clear();
  endpoint_tracker.collectTrackedRayEvidence(
    0.05, 0.05, 0.05, 3.05, 0.05, 0.05, ray_evidence);
  EXPECT_TRUE(ray_evidence.empty());
}

TEST(ObstacleRevocation, FiveClassRolesRevokeBackgroundAfterTerrainRelabel)
{
  map::ObstacleRevocationConfig config;
  config.minimum_free_frames = 2u;
  config.minimum_free_duration = 0.1;
  config.obstacle_labels = {0u};
  config.terrain_labels = {1u, 2u, 3u, 4u};
  config.dynamic_labels = {11u};
  map::ObstacleRevocationTracker tracker(config);
  const std::unordered_set<map::VoxelKey, map::VoxelKeyHash> no_rays;
  tracker.update({voxelSnapshot(10, 0, 0, 0u, 1.0f)}, no_rays,
                 stamp(1.0));
  ASSERT_EQ(1u, tracker.trackedCount());

  std::vector<map::VoxelSnapshot> grass{
    voxelSnapshot(10, 0, 0, 2u, 0.25f)};
  grass.front().last_observed = stamp(1.1);
  EXPECT_TRUE(tracker.update(grass, no_rays, stamp(1.1)).revoked_free.empty());
  grass.front().last_observed = stamp(1.2);
  EXPECT_EQ(1u, tracker.update(grass, no_rays, stamp(1.2)).revoked_free.size());
  EXPECT_EQ(0u, tracker.trackedCount());
}

TEST(ObstacleRevocation, ExplicitBoundaryRelabelRevokesHigherCostManhole)
{
  map::ObstacleRevocationConfig config;
  config.minimum_free_frames = 2u;
  config.minimum_free_duration = 0.1;
  config.free_max_traversability = 0.45f;
  config.obstacle_labels = {0u};
  config.terrain_labels = {1u, 2u, 3u, 4u};
  config.dynamic_labels = {11u};
  map::ObstacleRevocationTracker tracker(config);
  const std::unordered_set<map::VoxelKey, map::VoxelKeyHash> no_rays;
  const map::VoxelSnapshot obstacle = voxelSnapshot(10, 0, 0, 0u, 1.0f);
  tracker.update({obstacle}, no_rays, stamp(1.0));
  std::vector<map::VoxelSnapshot> manhole{
    voxelSnapshot(10, 0, 0, 4u, 0.50f)};
  const std::unordered_set<map::VoxelKey, map::VoxelKeyHash> reclassified{
    tracker.keyFor(manhole.front().x, manhole.front().y, manhole.front().z)};

  manhole.front().last_observed = stamp(1.1);
  EXPECT_TRUE(tracker.update(manhole, no_rays, stamp(1.1), reclassified)
                .revoked_reclassified.empty());
  manhole.front().last_observed = stamp(1.2);
  const map::ObstacleRevocationResult result =
    tracker.update(manhole, no_rays, stamp(1.2), reclassified);
  EXPECT_TRUE(result.revoked_free.empty());
  EXPECT_EQ(1u, result.revoked_reclassified.size());
  EXPECT_EQ(0u, tracker.trackedCount());
}

TEST(TerrainBoundaryFilter, RelabelsThinSeamBetweenAnyTerrainClasses)
{
  for (const auto& labels : std::vector<std::pair<std::uint32_t, std::uint32_t>>{
         {1u, 2u}, {1u, 3u}, {2u, 4u}, {3u, 4u}})
  {
    map::TerrainBoundaryFilterConfig config;
    config.enabled = true;
    config.opening_radius_cells = 1u;
    config.neighborhood_radius = 0.20;
    config.vertical_tolerance = 0.0;
    config.minimum_terrain_neighbors = 4u;
    config.minimum_distinct_terrain_labels = 2u;
    config.minimum_terrain_ratio = 0.70;
    map::TerrainBoundaryFilter filter(config);
    std::vector<map::VoxelSnapshot> voxels;
    for (std::int32_t x = 0; x < 5; ++x)
    {
      for (std::int32_t y = 0; y < 5; ++y)
      {
        const std::uint32_t label = x < 2 ? labels.first :
          (x == 2 ? 0u : labels.second);
        const float cost = label == 0u ? 1.0f :
          (label == labels.first ? 0.10f : 0.30f);
        voxels.push_back(voxelSnapshot(x, y, 0, label, cost, true));
      }
    }

    const map::TerrainBoundaryFilterResult result =
      filter.filter(voxels, 0.10, 0.10);
    ASSERT_EQ(5u, result.relabeled.size());
    EXPECT_TRUE(std::none_of(result.voxels.begin(), result.voxels.end(),
      [](const map::VoxelSnapshot& voxel) { return voxel.label == 0u; }));
    EXPECT_TRUE(std::all_of(result.voxels.begin(), result.voxels.end(),
      [](const map::VoxelSnapshot& voxel)
      {
        return voxel.label != 0u || voxel.traversability_cost < 0.60f;
      }));
  }
}

TEST(TerrainBoundaryFilter, KeepsDenseObstacleSurvivingOpening)
{
  map::TerrainBoundaryFilterConfig config;
  config.enabled = true;
  config.neighborhood_radius = 0.30;
  config.vertical_tolerance = 0.0;
  config.minimum_terrain_neighbors = 4u;
  config.minimum_terrain_ratio = 0.50;
  map::TerrainBoundaryFilter filter(config);
  std::vector<map::VoxelSnapshot> voxels;
  for (std::int32_t x = 0; x < 7; ++x)
  {
    for (std::int32_t y = 0; y < 7; ++y)
    {
      const bool obstacle = x >= 2 && x <= 4 && y >= 2 && y <= 4;
      const std::uint32_t label = obstacle ? 0u : (x < 3 ? 1u : 2u);
      voxels.push_back(voxelSnapshot(
        x, y, 0, label, obstacle ? 1.0f : 0.20f, true));
    }
  }

  const map::TerrainBoundaryFilterResult result =
    filter.filter(voxels, 0.10, 0.10);
  EXPECT_TRUE(result.relabeled.empty());
  EXPECT_EQ(9u, std::count_if(result.voxels.begin(), result.voxels.end(),
    [](const map::VoxelSnapshot& voxel) { return voxel.label == 0u; }));
}

TEST(TerrainBoundaryFilter, KeepsSpotInsideOneTerrainClassAndUnknown)
{
  map::TerrainBoundaryFilterConfig config;
  config.enabled = true;
  config.neighborhood_radius = 0.20;
  config.vertical_tolerance = 0.0;
  config.minimum_terrain_neighbors = 4u;
  map::TerrainBoundaryFilter filter(config);
  std::vector<map::VoxelSnapshot> voxels;
  for (std::int32_t x = 0; x < 5; ++x)
  {
    for (std::int32_t y = 0; y < 5; ++y)
    {
      const std::uint32_t label = (x == 2 && y == 2) ? 0u : 1u;
      voxels.push_back(voxelSnapshot(x, y, 0, label,
                                    label == 0u ? 1.0f : 0.05f, true));
    }
  }
  voxels.push_back(voxelSnapshot(6, 2, 0, map::kInvalidSemanticLabel));

  const map::TerrainBoundaryFilterResult result =
    filter.filter(voxels, 0.10, 0.10);
  EXPECT_TRUE(result.relabeled.empty());
  EXPECT_EQ(1u, std::count_if(result.voxels.begin(), result.voxels.end(),
    [](const map::VoxelSnapshot& voxel) { return voxel.label == 0u; }));
  EXPECT_EQ(1u, std::count_if(result.voxels.begin(), result.voxels.end(),
    [](const map::VoxelSnapshot& voxel)
    {
      return voxel.label == map::kInvalidSemanticLabel;
    }));
}

TEST(TerrainHeightCost, RaisesBothSidesOfMissingCostTerrainStep)
{
  map::TerrainHeightCostConfig config;
  config.enabled = true;
  config.height_difference_threshold = 0.15;
  config.neighborhood_radius = 0.20;
  config.obstacle_cost = 1.0f;
  std::vector<map::VoxelSnapshot> voxels{
    voxelSnapshot(0, 0, 0, 0u),
    voxelSnapshot(2, 0, 2, 9u)};

  EXPECT_EQ(2u, map::applyTerrainHeightDiscontinuityCost(voxels, 0.10, config));
  EXPECT_FLOAT_EQ(1.0f, voxels[0].traversability_cost);
  EXPECT_FLOAT_EQ(1.0f, voxels[1].traversability_cost);

  const auto columns = map::projectTraversabilityColumns(voxels);
  ASSERT_EQ(2u, columns.size());
  EXPECT_TRUE(std::all_of(columns.begin(), columns.end(),
    [](const map::TraversabilityColumnSnapshot& column)
    {
      return column.traversability_cost == 1.0f;
    }));
}

TEST(TerrainHeightCost, EightNeighborRadiusAllowsCumulativeStairRise)
{
  map::TerrainHeightCostConfig config;
  config.enabled = true;
  config.height_difference_threshold = 0.15;
  config.neighborhood_radius = 0.15;
  config.obstacle_cost = 1.0f;
  std::vector<map::VoxelSnapshot> voxels{
    voxelSnapshot(0, 0, 0, 0u),
    voxelSnapshot(1, 0, 1, 9u),
    voxelSnapshot(2, 0, 2, 9u)};

  EXPECT_EQ(0u, map::applyTerrainHeightDiscontinuityCost(voxels, 0.10, config));
  EXPECT_TRUE(std::all_of(voxels.begin(), voxels.end(),
    [](const map::VoxelSnapshot& voxel)
    {
      return voxel.traversability_cost < 1.0f;
    }));
}

TEST(TerrainHeightCost, EightNeighborRadiusStillRejectsAdjacentCliff)
{
  map::TerrainHeightCostConfig config;
  config.enabled = true;
  config.height_difference_threshold = 0.15;
  config.neighborhood_radius = 0.15;
  config.obstacle_cost = 1.0f;
  std::vector<map::VoxelSnapshot> voxels{
    voxelSnapshot(0, 0, 0, 0u),
    voxelSnapshot(1, 0, 2, 9u)};

  EXPECT_EQ(2u, map::applyTerrainHeightDiscontinuityCost(voxels, 0.10, config));
  EXPECT_TRUE(std::all_of(voxels.begin(), voxels.end(),
    [](const map::VoxelSnapshot& voxel)
    {
      return voxel.traversability_cost == 1.0f;
    }));
}

TEST(TerrainHeightCost, VerticalHalfResolutionDoesNotInflateOneStepIntoObstacle)
{
  map::SemanticVoxelMapConfig map_config;
  map_config.voxel_size_xy = 0.10;
  map_config.voxel_size_z = 0.05;
  map_config.decay_seconds = -1.0;
  map::SemanticVoxelMap voxel_map(map_config);

  map::VoxelObservation observation;
  observation.label = 9u;
  observation.stamp.fromSec(1.0);
  // The two raw heights straddle 0.05 m bin boundaries. Their true separation
  // is about 0.10 m, while their 0.05 m voxel centers are 0.15 m apart.
  voxel_map.integrate(0.01, 0.01, 0.499999, observation);
  voxel_map.integrate(0.01, 0.01, 0.600001, observation);

  std::vector<map::VoxelSnapshot> voxels = voxel_map.snapshot();
  ASSERT_EQ(2u, voxels.size());
  const auto minimum_maximum = std::minmax_element(
    voxels.begin(), voxels.end(),
    [](const map::VoxelSnapshot& lhs, const map::VoxelSnapshot& rhs)
    {
      return lhs.z < rhs.z;
    });
  EXPECT_NEAR(0.15,
              minimum_maximum.second->z - minimum_maximum.first->z, 1e-9);

  map::TerrainHeightCostConfig height_config;
  height_config.enabled = true;
  height_config.height_difference_threshold = 0.15;
  height_config.comparison_epsilon = 1e-6;
  EXPECT_EQ(0u, map::applyTerrainHeightDiscontinuityCost(
    voxels, voxel_map.voxelSizeXY(), height_config));
  EXPECT_TRUE(std::all_of(voxels.begin(), voxels.end(),
    [](const map::VoxelSnapshot& voxel)
    {
      return voxel.traversability_cost < 1.0f;
    }));
}

TEST(InputPointFilter, PlanarRobotBodyExclusionIncludesConfiguredBoundary)
{
  EXPECT_TRUE(map::isInsidePlanarExclusion(
    -0.5, -0.3, -0.5, 0.3, -0.3, 0.3));
  EXPECT_TRUE(map::isInsidePlanarExclusion(
    0.3, 0.3, -0.5, 0.3, -0.3, 0.3));
  EXPECT_TRUE(map::isInsidePlanarExclusion(
    0.0, 0.0, -0.5, 0.3, -0.3, 0.3));
}

TEST(InputPointFilter, PlanarRobotBodyExclusionKeepsOutsidePoints)
{
  EXPECT_FALSE(map::isInsidePlanarExclusion(
    -0.5001, 0.0, -0.5, 0.3, -0.3, 0.3));
  EXPECT_FALSE(map::isInsidePlanarExclusion(
    0.3001, 0.0, -0.5, 0.3, -0.3, 0.3));
  EXPECT_FALSE(map::isInsidePlanarExclusion(
    0.0, 0.3001, -0.5, 0.3, -0.3, 0.3));
}

TEST(TerrainHeightCost, LeavesFlatMeasuredAndNonTerrainVoxelsUnchanged)
{
  map::TerrainHeightCostConfig config;
  config.enabled = true;
  std::vector<map::VoxelSnapshot> flat{
    voxelSnapshot(0, 0, 0, 0u),
    voxelSnapshot(1, 0, 1, 9u)};
  EXPECT_EQ(0u, map::applyTerrainHeightDiscontinuityCost(flat, 0.10, config));

  std::vector<map::VoxelSnapshot> measured{
    voxelSnapshot(0, 0, 0, 0u, 0.20f, true),
    voxelSnapshot(1, 0, 2, 9u)};
  EXPECT_EQ(0u, map::applyTerrainHeightDiscontinuityCost(measured, 0.10, config));
  EXPECT_FLOAT_EQ(0.20f, measured[0].traversability_cost);
  EXPECT_FLOAT_EQ(0.05f, measured[1].traversability_cost);

  std::vector<map::VoxelSnapshot> obstacles{
    voxelSnapshot(0, 0, 0, 2u),
    voxelSnapshot(1, 0, 3, 3u)};
  EXPECT_EQ(0u, map::applyTerrainHeightDiscontinuityCost(obstacles, 0.10, config));
}

TEST(LocalAdmissionFilter, RetainsAllNonDynamicVoxels)
{
  using map::LocalAdmissionDecision;
  EXPECT_EQ(LocalAdmissionDecision::Admitted,
            map::classifyLocalAdmissionVoxel(0u, true));
  EXPECT_EQ(LocalAdmissionDecision::Admitted,
            map::classifyLocalAdmissionVoxel(9u, true));
  EXPECT_EQ(LocalAdmissionDecision::Admitted,
            map::classifyLocalAdmissionVoxel(2u, true));
  EXPECT_EQ(LocalAdmissionDecision::Admitted,
            map::classifyLocalAdmissionVoxel(map::kInvalidSemanticLabel, true));
  EXPECT_EQ(LocalAdmissionDecision::RejectedDynamic,
            map::classifyLocalAdmissionVoxel(11u, true));
  EXPECT_EQ(LocalAdmissionDecision::RejectedDynamic,
            map::classifyLocalAdmissionVoxel(18u, true));
}

TEST(LocalAdmissionFilter, DynamicExclusionCanBeDisabled)
{
  using map::LocalAdmissionDecision;
  EXPECT_EQ(LocalAdmissionDecision::Admitted,
            map::classifyLocalAdmissionVoxel(11u, false));
}

TEST(GlobalSemanticAdmission, OnlyTerrainBypassesStabilityAndRejectsUnsafeClasses)
{
  map::GlobalSemanticAdmission admission;
  std::vector<map::AdmissionObservation> observations;
  observations.push_back(admissionObservation(0.1, 0.1, 0.1, 0u));
  observations.push_back(admissionObservation(1.1, 0.1, 0.1, 11u));
  observations.push_back(admissionObservation(2.1, 0.1, 0.1,
                                               map::kInvalidSemanticLabel, 1.0f));
  map::AdmissionObservation rear = admissionObservation(-1.0, 0.0, 0.0, 2u);
  rear.rear_excluded = true;
  observations.push_back(rear);

  ros::Time stamp;
  stamp.fromSec(10.0);
  const map::AdmissionFrameResult result =
    admission.processFrame(observations, 0.0, 0.0, stamp);

  ASSERT_EQ(1u, result.confirmed.size());
  EXPECT_EQ(0u, result.confirmed.front().label);
  ASSERT_EQ(1u, result.rejected_dynamic.size());
  EXPECT_EQ(11u, result.rejected_dynamic.front().label);
  ASSERT_EQ(1u, result.rejected_unknown.size());
  EXPECT_EQ(map::kInvalidSemanticLabel, result.rejected_unknown.front().label);
  ASSERT_EQ(1u, result.rejected_rear.size());
  EXPECT_EQ(2u, result.rejected_rear.front().label);
  EXPECT_EQ(0u, admission.candidateCount());
}

TEST(GlobalSemanticAdmission, StaticVoxelRequiresAllStabilityGates)
{
  map::GlobalAdmissionConfig config;
  config.minimum_frames = 8u;
  config.minimum_duration = 1.0;
  config.minimum_pose_buckets = 3u;
  config.pose_bucket_size = 0.25;
  config.minimum_robot_baseline = 0.5;
  config.maximum_position_stddev = 0.18;
  map::GlobalSemanticAdmission admission(config);

  map::AdmissionFrameResult result;
  for (std::size_t frame = 0u; frame < 8u; ++frame)
  {
    ros::Time stamp;
    stamp.fromSec(20.0 + 0.2 * static_cast<double>(frame));
    // Multiple hits in one 0.40 m voxel still count as only one frame.
    std::vector<map::AdmissionObservation> observations;
    observations.push_back(admissionObservation(1.01, 2.01, 0.01, 4u));
    observations.push_back(admissionObservation(1.03, 2.02, 0.02, 4u));
    const double robot_x = frame < 3u ? 0.0 : (frame < 6u ? 0.3 : 0.6);
    result = admission.processFrame(observations, robot_x, 0.0, stamp);
    if (frame < 7u)
    {
      EXPECT_FALSE(containsAdmissionLabel(result.confirmed, 4u));
    }
  }

  EXPECT_TRUE(containsAdmissionLabel(result.confirmed, 4u));
  EXPECT_EQ(0u, admission.candidateCount());
}

TEST(GlobalSemanticAdmission, DuplicateTimestampDoesNotCountAsAnotherFrame)
{
  map::GlobalAdmissionConfig config;
  config.minimum_frames = 2u;
  config.minimum_duration = 0.0;
  config.minimum_pose_buckets = 1u;
  config.minimum_robot_baseline = 0.0;
  map::GlobalSemanticAdmission admission(config);
  ros::Time stamp;
  stamp.fromSec(30.0);
  const std::vector<map::AdmissionObservation> observations{
    admissionObservation(0.1, 0.1, 0.1, 2u)};

  admission.processFrame(observations, 0.0, 0.0, stamp);
  const map::AdmissionFrameResult duplicate =
    admission.processFrame(observations, 0.0, 0.0, stamp);

  EXPECT_TRUE(duplicate.confirmed.empty());
  EXPECT_EQ(1u, admission.candidateCount());
}

TEST(GlobalSemanticAdmission, CandidateTimeoutAndClearDiscardState)
{
  map::GlobalAdmissionConfig config;
  config.candidate_timeout = 2.0;
  map::GlobalSemanticAdmission admission(config);
  ros::Time first_stamp;
  first_stamp.fromSec(40.0);
  admission.processFrame({admissionObservation(0.1, 0.1, 0.1, 3u)},
                         0.0, 0.0, first_stamp);
  EXPECT_EQ(1u, admission.candidateCount());

  ros::Time expired_stamp;
  expired_stamp.fromSec(42.01);
  admission.processFrame({}, 0.0, 0.0, expired_stamp);
  EXPECT_EQ(0u, admission.candidateCount());

  admission.processFrame({admissionObservation(0.1, 0.1, 0.1, 0u)},
                         0.0, 0.0, expired_stamp);
  ASSERT_EQ(1u, admission.admitted().size());
  admission.clear();
  EXPECT_TRUE(admission.admitted().empty());
}

TEST(GlobalSemanticAdmission, StableDynamicReclassificationRevokesConfirmedStatic)
{
  map::GlobalAdmissionConfig config;
  config.minimum_frames = 1u;
  config.minimum_duration = 0.0;
  config.minimum_pose_buckets = 1u;
  config.minimum_robot_baseline = 0.0;
  config.revocation_minimum_frames = 3u;
  config.revocation_minimum_duration = 0.2;
  map::GlobalSemanticAdmission admission(config);

  ros::Time stamp;
  stamp.fromSec(50.0);
  map::AdmissionFrameResult result = admission.processFrame(
    {admissionObservation(1.0, 0.0, 0.0, 2u)}, 0.0, 0.0, stamp);
  ASSERT_TRUE(containsAdmissionLabel(result.confirmed, 2u));

  for (int frame = 1; frame <= 3; ++frame)
  {
    stamp.fromSec(50.0 + 0.1 * frame);
    result = admission.processFrame(
      {admissionObservation(1.0, 0.0, 0.0, 13u)}, 0.0, 0.0, stamp);
    if (frame == 1)
    {
      const map::AdmissionFrameResult duplicate = admission.processFrame(
        {admissionObservation(1.0, 0.0, 0.0, 13u)}, 0.0, 0.0, stamp);
      EXPECT_TRUE(duplicate.revoked_reclassified.empty());
      ASSERT_EQ(1u, duplicate.revocation_candidates.size());
    }
    if (frame < 3)
    {
      EXPECT_TRUE(containsAdmissionLabel(result.confirmed, 2u));
      EXPECT_TRUE(result.revoked_reclassified.empty());
      ASSERT_EQ(1u, result.revocation_candidates.size());
    }
  }

  EXPECT_FALSE(containsAdmissionLabel(result.confirmed, 2u));
  ASSERT_EQ(1u, result.revoked_reclassified.size());
  EXPECT_EQ(13u, result.revoked_reclassified.front().label);
}

TEST(GlobalSemanticAdmission, MissingFrameBreaksRevocationButNeverMeansFree)
{
  map::GlobalAdmissionConfig config;
  config.minimum_frames = 1u;
  config.minimum_duration = 0.0;
  config.minimum_pose_buckets = 1u;
  config.minimum_robot_baseline = 0.0;
  config.revocation_minimum_frames = 2u;
  config.revocation_minimum_duration = 0.0;
  map::GlobalSemanticAdmission admission(config);

  ros::Time stamp;
  stamp.fromSec(60.0);
  admission.processFrame({admissionObservation(1.0, 0.0, 0.0, 2u)},
                         0.0, 0.0, stamp);
  stamp.fromSec(60.1);
  admission.processFrame({admissionObservation(1.0, 0.0, 0.0, 13u)},
                         0.0, 0.0, stamp);
  stamp.fromSec(60.2);
  map::AdmissionFrameResult result = admission.processFrame({}, 0.0, 0.0, stamp);
  EXPECT_TRUE(result.revocation_candidates.empty());
  EXPECT_TRUE(containsAdmissionLabel(result.confirmed, 2u));

  stamp.fromSec(60.3);
  result = admission.processFrame(
    {admissionObservation(1.0, 0.0, 0.0, 13u)}, 0.0, 0.0, stamp);
  EXPECT_TRUE(result.revoked_reclassified.empty());
  EXPECT_TRUE(containsAdmissionLabel(result.confirmed, 2u));
}

TEST(GlobalSemanticAdmission, OnlyLowCostTerrainCanRevokeStaticAsFree)
{
  map::GlobalAdmissionConfig config;
  config.minimum_frames = 1u;
  config.minimum_duration = 0.0;
  config.minimum_pose_buckets = 1u;
  config.minimum_robot_baseline = 0.0;
  config.revocation_minimum_frames = 2u;
  config.revocation_minimum_duration = 0.1;
  config.revocation_free_max_traversability = 0.45f;
  map::GlobalSemanticAdmission admission(config);

  ros::Time stamp;
  stamp.fromSec(70.0);
  admission.processFrame({admissionObservation(1.0, 0.0, 0.0, 2u)},
                         0.0, 0.0, stamp);
  for (int frame = 1; frame <= 2; ++frame)
  {
    stamp.fromSec(70.0 + 0.1 * frame);
    admission.processFrame({admissionObservation(1.0, 0.0, 0.0, 0u, 0.9f)},
                           0.0, 0.0, stamp);
  }
  EXPECT_TRUE(containsAdmissionLabel(admission.admitted(), 2u));

  map::AdmissionFrameResult result;
  for (int frame = 3; frame <= 4; ++frame)
  {
    stamp.fromSec(70.0 + 0.1 * frame);
    result = admission.processFrame(
      {admissionObservation(1.0, 0.0, 0.0, 0u, 0.1f)}, 0.0, 0.0, stamp);
  }
  ASSERT_EQ(1u, result.revoked_free.size());
  EXPECT_EQ(0u, result.revoked_free.front().label);
  EXPECT_FALSE(containsAdmissionLabel(result.confirmed, 2u));
  EXPECT_TRUE(containsAdmissionLabel(result.confirmed, 0u));
}

TEST(GlobalSemanticAdmission, RevokedStaticMustPassFullAdmissionAgain)
{
  map::GlobalAdmissionConfig config;
  config.minimum_frames = 2u;
  config.minimum_duration = 0.0;
  config.minimum_pose_buckets = 1u;
  config.minimum_robot_baseline = 0.0;
  config.revocation_minimum_frames = 2u;
  config.revocation_minimum_duration = 0.0;
  map::GlobalSemanticAdmission admission(config);
  ros::Time stamp;

  for (int frame = 0; frame < 2; ++frame)
  {
    stamp.fromSec(80.0 + 0.1 * frame);
    admission.processFrame({admissionObservation(1.0, 0.0, 0.0, 2u)},
                           0.0, 0.0, stamp);
  }
  ASSERT_TRUE(containsAdmissionLabel(admission.admitted(), 2u));
  for (int frame = 2; frame < 4; ++frame)
  {
    stamp.fromSec(80.0 + 0.1 * frame);
    admission.processFrame({admissionObservation(1.0, 0.0, 0.0, 13u)},
                           0.0, 0.0, stamp);
  }
  ASSERT_FALSE(containsAdmissionLabel(admission.admitted(), 2u));

  stamp.fromSec(80.4);
  map::AdmissionFrameResult result = admission.processFrame(
    {admissionObservation(1.0, 0.0, 0.0, 2u)}, 0.0, 0.0, stamp);
  EXPECT_FALSE(containsAdmissionLabel(result.confirmed, 2u));
  EXPECT_EQ(1u, admission.candidateCount());
  stamp.fromSec(80.5);
  result = admission.processFrame(
    {admissionObservation(1.0, 0.0, 0.0, 2u)}, 0.0, 0.0, stamp);
  EXPECT_TRUE(containsAdmissionLabel(result.confirmed, 2u));
}

TEST(SemanticEvidence, RepeatedObservationsWin)
{
  map::SemanticEvidence evidence;
  evidence.fuse(10u, 1.0f);
  evidence.fuse(20u, 1.0f);
  evidence.fuse(10u, 1.0f);
  evidence.fuse(10u, 1.0f);

  EXPECT_EQ(10u, evidence.dominantLabel());
  EXPECT_GT(evidence.dominantConfidence(), 0.5f);
  EXPECT_LE(evidence.probabilities().size(), map::kSemanticTopK);
}

TEST(SemanticEvidence, KeepsOnlyTopThreeClasses)
{
  map::SemanticEvidence evidence;
  for (std::uint32_t label = 1; label <= 5; ++label)
  {
    evidence.fuse(label, 1.0f);
  }
  EXPECT_EQ(map::kSemanticTopK, evidence.probabilities().size());
  EXPECT_GT(evidence.otherProbability(), 0.0f);
}

TEST(SemanticVoxelMap, QuantizesNegativeCoordinatesWithFloor)
{
  map::SemanticVoxelMapConfig config;
  config.voxel_size = 0.1;
  map::SemanticVoxelMap voxel_map(config);

  const map::VoxelKey key = voxel_map.worldToKey(-0.01, -0.11, 0.09);
  EXPECT_EQ(-1, key.x);
  EXPECT_EQ(-2, key.y);
  EXPECT_EQ(0, key.z);
}

TEST(SemanticVoxelMap, UsesIndependentPlanarAndVerticalResolution)
{
  map::SemanticVoxelMapConfig config;
  config.voxel_size_xy = 0.10;
  config.voxel_size_z = 0.05;
  map::SemanticVoxelMap voxel_map(config);

  const map::VoxelKey key = voxel_map.worldToKey(0.09, 0.11, 0.09);
  EXPECT_EQ(0, key.x);
  EXPECT_EQ(1, key.y);
  EXPECT_EQ(1, key.z);
  EXPECT_DOUBLE_EQ(0.10, voxel_map.voxelSizeXY());
  EXPECT_DOUBLE_EQ(0.05, voxel_map.voxelSizeZ());

  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  voxel_map.keyToWorld(key, x, y, z);
  EXPECT_NEAR(0.05, x, 1e-12);
  EXPECT_NEAR(0.15, y, 1e-12);
  EXPECT_NEAR(0.075, z, 1e-12);
}

TEST(SemanticVoxelMap, FusesSemanticCostAndPersistsState)
{
  map::SemanticVoxelMapConfig config;
  config.voxel_size = 0.2;
  config.decay_seconds = -1.0;
  config.unknown_cost = 0.5f;
  config.cost_rise_alpha = 1.0f;
  map::SemanticVoxelMap voxel_map(config);

  map::SemanticClass obstacle;
  obstacle.label = 7u;
  obstacle.traversability_cost = 1.0f;
  voxel_map.setSemanticClasses({obstacle});

  map::VoxelObservation observation;
  observation.label = 7u;
  observation.semantic_confidence = 1.0f;
  observation.stamp.fromSec(10.0);
  voxel_map.integrate(0.1, 0.1, 0.1, observation);

  const auto voxels = voxel_map.snapshot();
  ASSERT_EQ(1u, voxels.size());
  EXPECT_EQ(7u, voxels.front().label);
  EXPECT_GT(voxels.front().traversability_cost, 0.5f);
  EXPECT_EQ(1u, voxels.front().observation_count);
}

TEST(SemanticVoxelMap, DecaysInSuppliedDepthFrameTimeDomain)
{
  map::SemanticVoxelMapConfig config;
  config.decay_seconds = 0.5;
  map::SemanticVoxelMap voxel_map(config);

  map::VoxelObservation observation;
  observation.label = 7u;
  observation.stamp.fromSec(10.0);
  voxel_map.integrate(0.0, 0.0, 0.0, observation);

  ros::Time next_frame_stamp;
  next_frame_stamp.fromSec(10.49);
  EXPECT_EQ(0u, voxel_map.prune(next_frame_stamp));
  EXPECT_EQ(1u, voxel_map.size());

  next_frame_stamp.fromSec(10.50);
  EXPECT_EQ(0u, voxel_map.prune(next_frame_stamp));
  EXPECT_EQ(1u, voxel_map.size());

  next_frame_stamp.fromSec(10.51);
  EXPECT_EQ(1u, voxel_map.prune(next_frame_stamp));
  EXPECT_EQ(0u, voxel_map.size());
}

TEST(SemanticVoxelMap, PrunesOutsideAsymmetricLocalBox)
{
  map::SemanticVoxelMapConfig config;
  config.voxel_size = 0.1;
  config.decay_seconds = -1.0;
  map::SemanticVoxelMap voxel_map(config);

  map::VoxelObservation observation;
  observation.has_traversability_cost = true;
  observation.stamp.fromSec(11.0);
  voxel_map.integrate(1.0, 0.0, 0.0, observation);
  voxel_map.integrate(-1.0, 0.0, 0.0, observation);
  voxel_map.integrate(0.0, 2.0, 0.0, observation);

  const std::array<double, 4> identity{{0.0, 0.0, 0.0, 1.0}};
  EXPECT_EQ(2u, voxel_map.pruneOutsideBox(
    0.0, 0.0, 0.0, identity,
    {{-0.5, -1.0, -1.0}}, {{1.5, 1.0, 1.0}}));
  ASSERT_EQ(1u, voxel_map.size());
  EXPECT_NEAR(1.05, voxel_map.snapshot().front().x, 1e-9);
}

TEST(SemanticVoxelMap, LocalBoxFollowsSensorYaw)
{
  map::SemanticVoxelMapConfig config;
  config.voxel_size = 0.1;
  config.decay_seconds = -1.0;
  map::SemanticVoxelMap voxel_map(config);

  map::VoxelObservation observation;
  observation.has_traversability_cost = true;
  observation.stamp.fromSec(12.0);
  // With a +90 degree sensor yaw, map +Y is sensor +X and must be retained.
  voxel_map.integrate(0.0, 1.0, 0.0, observation);
  voxel_map.integrate(1.0, 0.0, 0.0, observation);

  const double half_sqrt_two = std::sqrt(0.5);
  const std::array<double, 4> yaw_90{{0.0, 0.0, half_sqrt_two,
                                      half_sqrt_two}};
  EXPECT_EQ(1u, voxel_map.pruneOutsideBox(
    0.0, 0.0, 0.0, yaw_90,
    {{0.5, -0.25, -1.0}}, {{1.5, 0.25, 1.0}}));
  ASSERT_EQ(1u, voxel_map.size());
  EXPECT_NEAR(1.05, voxel_map.snapshot().front().y, 1e-9);
}

TEST(SemanticVoxelMap, CostOnlyObservationCreatesVoxel)
{
  map::SemanticVoxelMapConfig config;
  config.decay_seconds = -1.0;
  map::SemanticVoxelMap voxel_map(config);

  map::VoxelObservation observation;
  observation.has_traversability_cost = true;
  observation.traversability_cost = 0.8f;
  observation.stamp.fromSec(20.0);
  voxel_map.integrate(1.0, 2.0, 3.0, observation);

  const auto voxels = voxel_map.snapshot();
  ASSERT_EQ(1u, voxels.size());
  EXPECT_EQ(map::kInvalidSemanticLabel, voxels.front().label);
  EXPECT_TRUE(voxels.front().has_measured_traversability);
  EXPECT_FLOAT_EQ(0.8f, voxels.front().measured_traversability_cost);
  EXPECT_FLOAT_EQ(0.8f, voxels.front().traversability_cost);
  EXPECT_EQ(0u, voxels.front().semantic_observation_count);
  EXPECT_EQ(1u, voxels.front().traversability_observation_count);
}

TEST(SemanticVoxelMap, MaximumFusionKeepsSaferCost)
{
  map::SemanticVoxelMapConfig config;
  config.decay_seconds = -1.0;
  config.traversability_fusion_method = map::TraversabilityFusionMethod::Maximum;
  map::SemanticVoxelMap voxel_map(config);

  map::SemanticClass road;
  road.label = 0u;
  road.traversability_cost = 0.05f;
  voxel_map.setSemanticClasses({road});

  map::VoxelObservation observation;
  observation.label = 0u;
  observation.semantic_confidence = 1.0f;
  observation.has_traversability_cost = true;
  observation.traversability_cost = 0.8f;
  observation.stamp.fromSec(30.0);
  voxel_map.integrate(0.0, 0.0, 0.0, observation);

  const auto voxels = voxel_map.snapshot();
  ASSERT_EQ(1u, voxels.size());
  EXPECT_GT(voxels.front().measured_traversability_cost,
            voxels.front().semantic_cost);
  EXPECT_FLOAT_EQ(voxels.front().measured_traversability_cost,
                  voxels.front().traversability_cost);
}

TEST(SemanticVoxelMap, SemanticRiskRaisesMeasuredCostByConfidence)
{
  map::SemanticVoxelMapConfig config;
  config.decay_seconds = -1.0;
  config.semantic_risk_alpha = 0.8f;
  config.traversability_fusion_method =
    map::TraversabilityFusionMethod::ConfidenceWeightedRaise;
  map::SemanticVoxelMap voxel_map(config);

  map::SemanticClass obstacle;
  obstacle.label = 2u;
  obstacle.traversability_cost = 1.0f;
  voxel_map.setSemanticClasses({obstacle});

  map::VoxelObservation observation;
  observation.label = 2u;
  observation.semantic_confidence = 0.6f;
  observation.has_traversability_cost = true;
  observation.traversability_cost = 0.2f;
  observation.stamp.fromSec(32.0);
  voxel_map.integrate(0.0, 0.0, 0.0, observation);

  const auto voxels = voxel_map.snapshot();
  ASSERT_EQ(1u, voxels.size());
  const float expected = voxels.front().measured_traversability_cost +
    config.semantic_risk_alpha * voxels.front().semantic_confidence *
    std::max(0.0f, voxels.front().semantic_cost -
                    voxels.front().measured_traversability_cost);
  EXPECT_NEAR(expected, voxels.front().traversability_cost, 1e-6f);
  EXPECT_GT(voxels.front().traversability_cost,
            voxels.front().measured_traversability_cost);
  EXPECT_LT(voxels.front().traversability_cost,
            voxels.front().semantic_cost);
}

TEST(SemanticVoxelMap, SemanticRiskNeverLowersMeasuredCost)
{
  map::SemanticVoxelMapConfig config;
  config.decay_seconds = -1.0;
  config.semantic_risk_alpha = 0.8f;
  config.traversability_fusion_method =
    map::TraversabilityFusionMethod::ConfidenceWeightedRaise;
  map::SemanticVoxelMap voxel_map(config);

  map::SemanticClass road;
  road.label = 0u;
  road.traversability_cost = 0.05f;
  voxel_map.setSemanticClasses({road});

  map::VoxelObservation observation;
  observation.label = 0u;
  observation.has_traversability_cost = true;
  observation.traversability_cost = 0.9f;
  observation.stamp.fromSec(33.0);
  voxel_map.integrate(0.0, 0.0, 0.0, observation);

  const auto voxels = voxel_map.snapshot();
  ASSERT_EQ(1u, voxels.size());
  EXPECT_FLOAT_EQ(voxels.front().measured_traversability_cost,
                  voxels.front().traversability_cost);
}

TEST(SemanticVoxelMap, ProjectsMaximumCostAlongEachVerticalColumn)
{
  map::SemanticVoxelMapConfig config;
  config.voxel_size = 0.1;
  config.decay_seconds = -1.0;
  map::SemanticVoxelMap voxel_map(config);

  map::VoxelObservation observation;
  observation.has_traversability_cost = true;
  observation.stamp.fromSec(34.0);
  observation.traversability_cost = 0.2f;
  voxel_map.integrate(0.01, 0.01, 0.01, observation);
  observation.traversability_cost = 0.9f;
  voxel_map.integrate(0.01, 0.01, 0.31, observation);
  observation.traversability_cost = 0.4f;
  voxel_map.integrate(0.11, 0.01, 0.01, observation);

  const auto columns = voxel_map.traversabilityColumns();
  ASSERT_EQ(2u, columns.size());
  for (const auto& column : columns)
  {
    if (column.x_index == 0)
    {
      EXPECT_EQ(0, column.y_index);
      EXPECT_FLOAT_EQ(0.9f, column.traversability_cost);
      EXPECT_NEAR(0.35, column.z, 1e-9);
    }
    else
    {
      EXPECT_EQ(1, column.x_index);
      EXPECT_FLOAT_EQ(0.4f, column.traversability_cost);
    }
  }
}

TEST(SemanticVoxelMap, MeasuredCostUsesAsymmetricTemporalFilter)
{
  map::SemanticVoxelMapConfig config;
  config.decay_seconds = -1.0;
  config.cost_rise_alpha = 0.5f;
  config.cost_fall_alpha = 0.1f;
  map::SemanticVoxelMap voxel_map(config);

  map::VoxelObservation observation;
  observation.has_traversability_cost = true;
  observation.traversability_cost = 0.2f;
  observation.stamp.fromSec(35.0);
  voxel_map.integrate(0.0, 0.0, 0.0, observation);
  observation.traversability_cost = 0.8f;
  voxel_map.integrate(0.0, 0.0, 0.0, observation);
  observation.traversability_cost = 0.1f;
  voxel_map.integrate(0.0, 0.0, 0.0, observation);

  const auto voxels = voxel_map.snapshot();
  ASSERT_EQ(1u, voxels.size());
  // 0.2 -> 0.5 with rise alpha, then 0.5 -> 0.46 with fall alpha.
  EXPECT_NEAR(0.46f, voxels.front().measured_traversability_cost, 1e-6f);
}

TEST(SemanticVoxelMap, VersionTwoPersistencePreservesSeparatedCosts)
{
  map::SemanticVoxelMapConfig config;
  config.decay_seconds = -1.0;
  config.traversability_fusion_method = map::TraversabilityFusionMethod::Maximum;
  map::SemanticVoxelMap source(config);

  map::SemanticClass obstacle;
  obstacle.label = 2u;
  obstacle.traversability_cost = 1.0f;
  source.setSemanticClasses({obstacle});

  map::VoxelObservation observation;
  observation.label = 2u;
  observation.has_traversability_cost = true;
  observation.traversability_cost = 0.3f;
  observation.stamp.fromSec(40.0);
  source.integrate(0.1, 0.2, 0.3, observation);

  const std::string path = "/tmp/local3d_semantic_voxel_map_test_" +
                           std::to_string(static_cast<long long>(getpid())) + ".csv";
  std::string error;
  ASSERT_TRUE(source.saveCsv(path, error)) << error;

  map::SemanticVoxelMap restored(config);
  restored.setSemanticClasses({obstacle});
  ASSERT_TRUE(restored.loadCsv(path, error)) << error;
  std::remove(path.c_str());

  const auto voxels = restored.snapshot();
  ASSERT_EQ(1u, voxels.size());
  EXPECT_EQ(2u, voxels.front().label);
  EXPECT_FLOAT_EQ(0.3f, voxels.front().measured_traversability_cost);
  EXPECT_GT(voxels.front().semantic_cost,
            voxels.front().measured_traversability_cost);
  EXPECT_FLOAT_EQ(voxels.front().semantic_cost,
                  voxels.front().traversability_cost);
  EXPECT_EQ(1u, voxels.front().semantic_observation_count);
  EXPECT_EQ(1u, voxels.front().traversability_observation_count);
}

TEST(SemanticVoxelMap, LoadsVersionOnePersistenceAsMeasuredCost)
{
  map::SemanticVoxelMapConfig config;
  config.decay_seconds = -1.0;
  map::SemanticVoxelMap voxel_map(config);
  const std::string path = "/tmp/local3d_semantic_voxel_map_v1_test_" +
                           std::to_string(static_cast<long long>(getpid())) + ".csv";
  {
    std::ofstream stream(path);
    ASSERT_TRUE(stream.good());
    stream << "# local3d_semantic_voxel_map v1 voxel_size=0.1\n"
              "x,y,z,label0,loge0,label1,loge1,label2,loge2,others,cost,observations,last_observed\n"
              "1,2,3,7,1,4294967295,-0.1,4294967295,-0.1,-0.2,0.7,4,50\n";
  }

  std::string error;
  ASSERT_TRUE(voxel_map.loadCsv(path, error)) << error;
  std::remove(path.c_str());

  const auto voxels = voxel_map.snapshot();
  ASSERT_EQ(1u, voxels.size());
  EXPECT_EQ(7u, voxels.front().label);
  EXPECT_TRUE(voxels.front().has_measured_traversability);
  EXPECT_FLOAT_EQ(0.7f, voxels.front().measured_traversability_cost);
  EXPECT_FLOAT_EQ(0.7f, voxels.front().traversability_cost);
  EXPECT_EQ(4u, voxels.front().semantic_observation_count);
  EXPECT_EQ(4u, voxels.front().traversability_observation_count);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
