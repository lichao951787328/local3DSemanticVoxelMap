#include "local3d_semantic_voxel_map/two_layer_angular_clearing.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <vector>

namespace
{
using local3d_semantic_voxel_map::TwoLayerClearingConfig;
using local3d_semantic_voxel_map::TwoLayerClearingPoint;
using local3d_semantic_voxel_map::TwoLayerClearingResult;
using local3d_semantic_voxel_map::generateTwoLayerAngularClearingRays;

TwoLayerClearingConfig config()
{
  TwoLayerClearingConfig value;
  value.octomap_resolution = 0.4;
  value.angular_resolution_deg = 5.0;
  value.local_grid_resolution = 0.1;
  value.local_box_min_x = -2.0;
  value.local_box_max_x = 8.0;
  value.local_box_min_y = -5.0;
  value.local_box_max_y = 5.0;
  return value;
}
}  // namespace

TEST(TwoLayerAngularClearing, EmptyCurrentFrameProducesNoBlindClearing)
{
  const TwoLayerClearingResult result =
    generateTwoLayerAngularClearingRays({}, 0.0, 0.0, 0.0, config());
  EXPECT_TRUE(result.rays.empty());
}

TEST(TwoLayerAngularClearing, FiveDegreesProducesExactlyTwoTimesSeventyTwoRays)
{
  const std::vector<TwoLayerClearingPoint> points{{1.0, 1.0, 1.01}};
  const TwoLayerClearingResult result =
    generateTwoLayerAngularClearingRays(points, 4.0, -3.0, 0.0, config());
  ASSERT_EQ(144u, result.rays.size());
  EXPECT_NEAR(1.0, result.top_layer_z, 1e-9);
  EXPECT_EQ(144u, result.hit_rays + result.no_hit_rays);
}

TEST(TwoLayerAngularClearing, NoHitRaysStopAtAsymmetricLocalBox)
{
  const std::vector<TwoLayerClearingPoint> points{{1.0, 1.0, 1.01}};
  const TwoLayerClearingResult result =
    generateTwoLayerAngularClearingRays(points, 0.0, 0.0, 0.0, config());
  ASSERT_EQ(144u, result.rays.size());

  const auto& forward_lower = result.rays[72u];
  const auto& left_lower = result.rays[72u + 18u];
  const auto& rear_lower = result.rays[72u + 36u];
  EXPECT_FALSE(forward_lower.hit);
  EXPECT_NEAR(8.0, forward_lower.end_x, 1e-6);
  EXPECT_NEAR(0.0, forward_lower.end_y, 1e-6);
  EXPECT_NEAR(0.0, left_lower.end_x, 1e-6);
  EXPECT_NEAR(5.0, left_lower.end_y, 1e-6);
  EXPECT_NEAR(-2.0, rear_lower.end_x, 1e-6);
  EXPECT_NEAR(0.0, rear_lower.end_y, 1e-6);
}

TEST(TwoLayerAngularClearing, FirstCurrentFrameGridCellStopsHorizontalRay)
{
  const std::vector<TwoLayerClearingPoint> points{
    {2.05, 0.02, 1.01}, {4.05, 0.02, 1.01}};
  const TwoLayerClearingResult result =
    generateTwoLayerAngularClearingRays(points, 0.0, 0.0, 0.0, config());
  ASSERT_EQ(144u, result.rays.size());
  const auto& top_forward = result.rays[0u];
  EXPECT_TRUE(top_forward.hit);
  EXPECT_GT(top_forward.end_x, 2.0);
  EXPECT_LT(top_forward.end_x, 2.01);
  EXPECT_NEAR(result.top_layer_z, top_forward.end_z, 1e-9);
}

TEST(TwoLayerAngularClearing, HitInTopLayerDoesNotStopLowerLayer)
{
  const std::vector<TwoLayerClearingPoint> points{{2.05, 0.02, 1.01}};
  const TwoLayerClearingResult result =
    generateTwoLayerAngularClearingRays(points, 0.0, 0.0, 0.0, config());
  ASSERT_EQ(144u, result.rays.size());
  EXPECT_TRUE(result.rays[0u].hit);
  EXPECT_FALSE(result.rays[72u].hit);
  EXPECT_NEAR(8.0, result.rays[72u].end_x, 1e-6);
}

TEST(TwoLayerAngularClearing, GenerationCostIsReportedWithoutWallTimeAssertion)
{
  std::vector<TwoLayerClearingPoint> points;
  points.reserve(50000u);
  for (std::size_t index = 0u; index < 50000u; ++index)
  {
    const double angle = static_cast<double>(index % 3600u) *
                         3.14159265358979323846 / 1800.0;
    const double radius = 1.0 + static_cast<double>(index % 80u) * 0.1;
    points.push_back({radius * std::cos(angle), radius * std::sin(angle),
                      1.01 - 0.4 * static_cast<double>(index % 2u)});
  }
  const auto start = std::chrono::steady_clock::now();
  const TwoLayerClearingResult result =
    generateTwoLayerAngularClearingRays(points, 0.0, 0.0, 0.0, config());
  const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::steady_clock::now() - start).count();
  ASSERT_EQ(144u, result.rays.size());
  RecordProperty("points", points.size());
  RecordProperty("elapsed_us", elapsed_us);
  RecordProperty("visited_cells", result.visited_local_cells);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
