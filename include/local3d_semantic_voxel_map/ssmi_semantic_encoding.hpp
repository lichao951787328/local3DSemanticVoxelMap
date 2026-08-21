#ifndef LOCAL3D_SEMANTIC_VOXEL_MAP_SSMI_SEMANTIC_ENCODING_HPP_
#define LOCAL3D_SEMANTIC_VOXEL_MAP_SSMI_SEMANTIC_ENCODING_HPP_

#include <array>
#include <cstdint>

namespace local3d_semantic_voxel_map
{

using SemanticRgb = std::array<std::uint8_t, 3>;

// Return the canonical on-wire semantic color consumed by SSMI. Input-source
// palettes are deliberately handled before this function, when the source
// value is normalized to a canonical label.
SemanticRgb ssmiSemanticColor(std::uint32_t canonical_label,
                              float traversability,
                              float obstacle_threshold);

std::uint32_t packSemanticRgb(const SemanticRgb& color);

}  // namespace local3d_semantic_voxel_map

#endif  // LOCAL3D_SEMANTIC_VOXEL_MAP_SSMI_SEMANTIC_ENCODING_HPP_
