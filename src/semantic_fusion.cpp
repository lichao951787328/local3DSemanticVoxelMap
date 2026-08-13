#include "local3d_semantic_voxel_map/semantic_fusion.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace local3d_semantic_voxel_map
{

SemanticEvidence::SemanticEvidence(const SemanticFusionConfig& config)
  : config_(config), others_log_evidence_(config.initial_log_evidence)
{
  for (auto& hypothesis : hypotheses_)
  {
    hypothesis.label = kInvalidSemanticLabel;
    hypothesis.log_evidence = config_.initial_log_evidence;
  }
}

void SemanticEvidence::fuse(const std::uint32_t observed_label,
                            const float observation_confidence)
{
  if (observed_label == kInvalidSemanticLabel)
  {
    return;
  }

  const float weight = std::max(0.0f, std::min(1.0f, observation_confidence));
  if (weight <= 0.0f)
  {
    return;
  }

  if (empty())
  {
    hypotheses_[0].label = observed_label;
    hypotheses_[0].log_evidence = clamp(config_.initial_log_evidence +
                                         weight * config_.positive_update);
    others_log_evidence_ = clamp(config_.initial_log_evidence +
                                 weight * config_.negative_update);
    return;
  }

  for (auto& hypothesis : hypotheses_)
  {
    if (hypothesis.label == observed_label)
    {
      for (auto& updated : hypotheses_)
      {
        if (updated.label == kInvalidSemanticLabel)
        {
          continue;
        }
        const float delta = updated.label == observed_label ?
          config_.positive_update : config_.negative_update;
        updated.log_evidence = clamp(updated.log_evidence + weight * delta);
      }
      others_log_evidence_ = clamp(others_log_evidence_ +
                                   weight * config_.negative_update);

      std::sort(hypotheses_.begin(), hypotheses_.end(),
                [](const SemanticHypothesis& lhs, const SemanticHypothesis& rhs) {
                  if (lhs.label == kInvalidSemanticLabel) return false;
                  if (rhs.label == kInvalidSemanticLabel) return true;
                  return lhs.log_evidence > rhs.log_evidence;
                });
      return;
    }
  }

  // SSMI introduces a previously unseen class from the `others` bucket. The
  // alpha split avoids giving a new, possibly noisy class all unknown mass.
  const float alpha = std::max(1e-4f, std::min(1.0f - 1e-4f,
                                              config_.new_class_prior));
  std::vector<SemanticHypothesis> candidates;
  candidates.reserve(kSemanticTopK + 1);
  for (auto hypothesis : hypotheses_)
  {
    if (hypothesis.label != kInvalidSemanticLabel)
    {
      hypothesis.log_evidence = clamp(hypothesis.log_evidence +
                                      weight * config_.negative_update);
      candidates.push_back(hypothesis);
    }
  }

  SemanticHypothesis incoming;
  incoming.label = observed_label;
  incoming.log_evidence = clamp(others_log_evidence_ + std::log(alpha) +
                                weight * config_.positive_update);
  candidates.push_back(incoming);
  others_log_evidence_ = clamp(others_log_evidence_ + std::log(1.0f - alpha) +
                               weight * config_.negative_update);
  sortAndTruncate(candidates);
}

bool SemanticEvidence::empty() const
{
  return hypotheses_[0].label == kInvalidSemanticLabel;
}

std::uint32_t SemanticEvidence::dominantLabel() const
{
  return empty() ? kInvalidSemanticLabel : hypotheses_[0].label;
}

std::vector<std::pair<std::uint32_t, float>> SemanticEvidence::probabilities() const
{
  std::vector<std::pair<std::uint32_t, float>> output;
  if (empty())
  {
    return output;
  }

  float maximum = others_log_evidence_;
  for (const auto& hypothesis : hypotheses_)
  {
    if (hypothesis.label != kInvalidSemanticLabel)
    {
      maximum = std::max(maximum, hypothesis.log_evidence);
    }
  }

  double denominator = std::exp(static_cast<double>(others_log_evidence_ - maximum));
  for (const auto& hypothesis : hypotheses_)
  {
    if (hypothesis.label != kInvalidSemanticLabel)
    {
      denominator += std::exp(static_cast<double>(hypothesis.log_evidence - maximum));
    }
  }

  output.reserve(kSemanticTopK);
  for (const auto& hypothesis : hypotheses_)
  {
    if (hypothesis.label != kInvalidSemanticLabel)
    {
      const float probability = static_cast<float>(
        std::exp(static_cast<double>(hypothesis.log_evidence - maximum)) / denominator);
      output.emplace_back(hypothesis.label, probability);
    }
  }
  return output;
}

float SemanticEvidence::dominantConfidence() const
{
  const auto distribution = probabilities();
  return distribution.empty() ? 0.0f : distribution.front().second;
}

float SemanticEvidence::otherProbability() const
{
  if (empty())
  {
    return 1.0f;
  }

  float maximum = others_log_evidence_;
  for (const auto& hypothesis : hypotheses_)
  {
    if (hypothesis.label != kInvalidSemanticLabel)
    {
      maximum = std::max(maximum, hypothesis.log_evidence);
    }
  }

  double denominator = std::exp(static_cast<double>(others_log_evidence_ - maximum));
  for (const auto& hypothesis : hypotheses_)
  {
    if (hypothesis.label != kInvalidSemanticLabel)
    {
      denominator += std::exp(static_cast<double>(hypothesis.log_evidence - maximum));
    }
  }
  return static_cast<float>(
    std::exp(static_cast<double>(others_log_evidence_ - maximum)) / denominator);
}

const std::array<SemanticHypothesis, kSemanticTopK>& SemanticEvidence::hypotheses() const
{
  return hypotheses_;
}

float SemanticEvidence::othersLogEvidence() const
{
  return others_log_evidence_;
}

void SemanticEvidence::restore(
  const std::array<SemanticHypothesis, kSemanticTopK>& hypotheses,
  const float others_log_evidence)
{
  hypotheses_ = hypotheses;
  others_log_evidence_ = clamp(others_log_evidence);
}

float SemanticEvidence::clamp(const float value) const
{
  return std::max(config_.min_log_evidence,
                  std::min(config_.max_log_evidence, value));
}

void SemanticEvidence::sortAndTruncate(std::vector<SemanticHypothesis>& candidates)
{
  std::sort(candidates.begin(), candidates.end(),
            [](const SemanticHypothesis& lhs, const SemanticHypothesis& rhs) {
              return lhs.log_evidence > rhs.log_evidence;
            });

  while (candidates.size() > kSemanticTopK)
  {
    others_log_evidence_ = clamp(logAddExp(others_log_evidence_,
                                           candidates.back().log_evidence));
    candidates.pop_back();
  }

  for (std::size_t index = 0; index < kSemanticTopK; ++index)
  {
    if (index < candidates.size())
    {
      hypotheses_[index] = candidates[index];
    }
    else
    {
      hypotheses_[index].label = kInvalidSemanticLabel;
      hypotheses_[index].log_evidence = config_.initial_log_evidence;
    }
  }
}

float SemanticEvidence::logAddExp(const float lhs, const float rhs)
{
  const float maximum = std::max(lhs, rhs);
  if (!std::isfinite(maximum))
  {
    return maximum;
  }
  return maximum + std::log(std::exp(lhs - maximum) + std::exp(rhs - maximum));
}

}  // namespace local3d_semantic_voxel_map
