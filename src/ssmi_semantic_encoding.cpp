#include "local3d_semantic_voxel_map/ssmi_semantic_encoding.hpp"

#include <array>
#include <cmath>

namespace local3d_semantic_voxel_map
{

namespace
{

const std::array<SemanticRgb, 19>& canonicalPalette()
{
  static const std::array<SemanticRgb, 19> palette = {{
    {{128, 64, 128}}, {{244, 35, 232}}, {{70, 70, 70}},
    {{102, 102, 156}}, {{190, 153, 153}}, {{153, 153, 153}},
    {{250, 170, 30}}, {{220, 220, 0}}, {{107, 142, 35}},
    {{152, 251, 152}}, {{70, 130, 180}}, {{220, 20, 60}},
    {{255, 0, 0}}, {{0, 0, 142}}, {{0, 0, 70}},
    {{0, 60, 100}}, {{0, 80, 100}}, {{0, 0, 230}},
    {{119, 11, 32}}
  }};
  return palette;
}

}  // namespace

SemanticRgb ssmiSemanticColor(const std::uint32_t canonical_label,
                              const float traversability,
                              const float obstacle_threshold)
{
  if (std::isfinite(traversability) && traversability >= obstacle_threshold)
  {
    // Canonical wall encoding. SSMI includes this color in its configured
    // obstacle-semantic set, so high-cost terrain edges become obstacles.
    return canonicalPalette()[3u];
  }
  if (canonical_label < canonicalPalette().size())
  {
    return canonicalPalette()[canonical_label];
  }
  return SemanticRgb{{127, 127, 127}};
}

std::uint32_t packSemanticRgb(const SemanticRgb& color)
{
  return (static_cast<std::uint32_t>(color[0]) << 16u) |
         (static_cast<std::uint32_t>(color[1]) << 8u) |
         static_cast<std::uint32_t>(color[2]);
}

}  // namespace local3d_semantic_voxel_map
