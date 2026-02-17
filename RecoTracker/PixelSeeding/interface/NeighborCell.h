#ifndef RecoTracker_PixelSeeding_interface_NeighborCell_h
#define RecoTracker_PixelSeeding_interface_NeighborCell_h

#include <cstdint>

namespace caStructures {
  // index-float pair for cell-to-cell association
  struct NeighborCell {
    uint32_t index;   // index of the neighboring cell
    float curvature;  // curvature of the triplet formed together with the neighboring cell
  };
}  // namespace caStructures

#endif  // RecoTracker_PixelSeeding_interface_NeighborCell_h
