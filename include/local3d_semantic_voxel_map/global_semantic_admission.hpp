#ifndef LOCAL3D_SEMANTIC_VOXEL_MAP_GLOBAL_SEMANTIC_ADMISSION_HPP_
#define LOCAL3D_SEMANTIC_VOXEL_MAP_GLOBAL_SEMANTIC_ADMISSION_HPP_

#include "local3d_semantic_voxel_map/semantic_voxel_map.hpp"

#include <ros/time.h>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace local3d_semantic_voxel_map
{

struct GlobalAdmissionConfig
{
  double voxel_size = 0.40;
  std::size_t minimum_frames = 8u;
  double minimum_duration = 1.0;
  std::size_t minimum_pose_buckets = 3u;
  double pose_bucket_size = 0.25;
  double minimum_robot_baseline = 0.5;
  double maximum_position_stddev = 0.18;
  double candidate_timeout = 2.0;
  std::size_t revocation_minimum_frames = 8u;
  double revocation_minimum_duration = 1.0;
  float revocation_free_max_traversability = 0.45f;
};

struct AdmissionObservation
{
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  std::uint32_t label = kInvalidSemanticLabel;
  bool has_semantic = false;
  float semantic_confidence = 0.0f;
  float traversability = 0.5f;
  bool has_traversability = false;
  bool rear_excluded = false;
};

struct AdmissionPoint
{
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  float traversability = 0.5f;
  std::uint32_t label = kInvalidSemanticLabel;
};

struct AdmissionFrameResult
{
  std::vector<AdmissionPoint> candidates;
  std::vector<AdmissionPoint> confirmed;
  std::vector<AdmissionPoint> rejected_dynamic;
  std::vector<AdmissionPoint> rejected_unknown;
  std::vector<AdmissionPoint> rejected_rear;
  std::vector<AdmissionPoint> revocation_candidates;
  std::vector<AdmissionPoint> revoked_free;
  std::vector<AdmissionPoint> revoked_reclassified;
};

enum class LocalAdmissionDecision
{
  Admitted,
  RejectedDynamic
};

LocalAdmissionDecision classifyLocalAdmissionVoxel(
  std::uint32_t label, bool exclude_dynamic);

class GlobalSemanticAdmission
{
public:
  explicit GlobalSemanticAdmission(
    const GlobalAdmissionConfig& config = GlobalAdmissionConfig());

  AdmissionFrameResult processFrame(
    const std::vector<AdmissionObservation>& observations,
    double robot_x, double robot_y, const ros::Time& stamp);
  void clear();
  std::vector<AdmissionPoint> admitted() const;
  std::size_t candidateCount() const;

private:
  struct PoseBucket
  {
    std::int32_t x = 0;
    std::int32_t y = 0;

    bool operator==(const PoseBucket& other) const
    {
      return x == other.x && y == other.y;
    }
  };

  struct PoseBucketHash
  {
    std::size_t operator()(const PoseBucket& bucket) const;
  };

  struct Candidate
  {
    std::uint32_t label = kInvalidSemanticLabel;
    ros::Time first_seen;
    ros::Time last_seen;
    std::size_t frame_count = 0u;
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_z = 0.0;
    double sum_squared_norm = 0.0;
    double cost_sum = 0.0;
    std::unordered_set<PoseBucket, PoseBucketHash> pose_buckets;
    std::vector<std::pair<double, double>> robot_positions;
  };

  struct AdmittedVoxel
  {
    AdmissionPoint point;
    bool stability_confirmed = false;
  };

  enum class RevocationReason
  {
    Free,
    Reclassified
  };

  struct RevocationCandidate
  {
    RevocationReason reason = RevocationReason::Free;
    std::uint32_t evidence_label = kInvalidSemanticLabel;
    ros::Time first_seen;
    ros::Time last_seen;
    std::size_t frame_count = 0u;
    std::size_t last_frame_sequence = 0u;
    float evidence_traversability = 0.5f;
  };

  VoxelKey keyFor(double x, double y, double z) const;
  PoseBucket poseBucketFor(double x, double y) const;
  AdmissionPoint candidatePoint(const Candidate& candidate) const;
  double candidateStddev(const Candidate& candidate) const;
  double candidateBaseline(const Candidate& candidate) const;
  bool ready(const Candidate& candidate) const;
  bool revocationReady(const RevocationCandidate& candidate) const;
  static bool isTerrain(std::uint32_t label);
  static bool isStatic(std::uint32_t label);
  static bool isDynamic(std::uint32_t label);

  GlobalAdmissionConfig config_;
  std::unordered_map<VoxelKey, Candidate, VoxelKeyHash> candidates_;
  std::unordered_map<VoxelKey, AdmittedVoxel, VoxelKeyHash> admitted_;
  std::unordered_map<VoxelKey, RevocationCandidate, VoxelKeyHash>
    revocation_candidates_;
  std::size_t frame_sequence_ = 0u;
  ros::Time latest_frame_stamp_;
};

}  // namespace local3d_semantic_voxel_map

#endif  // LOCAL3D_SEMANTIC_VOXEL_MAP_GLOBAL_SEMANTIC_ADMISSION_HPP_
