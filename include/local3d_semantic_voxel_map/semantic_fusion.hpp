#ifndef LOCAL3D_SEMANTIC_VOXEL_MAP_SEMANTIC_FUSION_HPP_
#define LOCAL3D_SEMANTIC_VOXEL_MAP_SEMANTIC_FUSION_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace local3d_semantic_voxel_map
{

constexpr std::size_t kSemanticTopK = 3;
constexpr std::uint32_t kInvalidSemanticLabel = 0xffffffffu;

struct SemanticFusionConfig
{
  float positive_update = 1.0f;
  float negative_update = -0.1f;
  float min_log_evidence = -4.59512f;
  float max_log_evidence = 4.59512f;
  float initial_log_evidence = -0.1f;
  float new_class_prior = 0.8f;
};

struct SemanticHypothesis
{
  std::uint32_t label = kInvalidSemanticLabel;
  float log_evidence = -0.1f;
};

// A compact semantic distribution inspired by SSMI: retain the three most
// likely classes and merge all remaining probability mass into `others`.
class SemanticEvidence
{
public:
  explicit SemanticEvidence(const SemanticFusionConfig& config = SemanticFusionConfig());

  void fuse(std::uint32_t observed_label, float observation_confidence = 1.0f);

  bool empty() const;
  std::uint32_t dominantLabel() const;
  float dominantConfidence() const;
  float otherProbability() const;
  std::vector<std::pair<std::uint32_t, float>> probabilities() const;

  const std::array<SemanticHypothesis, kSemanticTopK>& hypotheses() const;
  float othersLogEvidence() const;
  void restore(const std::array<SemanticHypothesis, kSemanticTopK>& hypotheses,
               float others_log_evidence);

private:
  float clamp(float value) const;
  void sortAndTruncate(std::vector<SemanticHypothesis>& candidates);
  static float logAddExp(float lhs, float rhs);

  SemanticFusionConfig config_;
  std::array<SemanticHypothesis, kSemanticTopK> hypotheses_;
  float others_log_evidence_;
};

}  // namespace local3d_semantic_voxel_map

#endif  // LOCAL3D_SEMANTIC_VOXEL_MAP_SEMANTIC_FUSION_HPP_
