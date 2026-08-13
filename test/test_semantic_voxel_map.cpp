#include "local3d_semantic_voxel_map/semantic_fusion.hpp"
#include "local3d_semantic_voxel_map/semantic_voxel_map.hpp"

#include <gtest/gtest.h>

namespace map = local3d_semantic_voxel_map;

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

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
