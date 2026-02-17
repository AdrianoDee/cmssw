#ifndef RecoTracker_PixelSeeding_interface_CAPairSoA_h
#define RecoTracker_PixelSeeding_interface_CAPairSoA_h

#include <Eigen/Core>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "NeighborCell.h"

namespace caStructures {

  // pair of indices
  GENERATE_SOA_LAYOUT(CAPairLayout, SOA_COLUMN(uint32_t, inner), SOA_COLUMN(uint32_t, outer))

  using CAPairSoA = CAPairLayout<>;
  using CAPairSoAView = CAPairSoA::View;
  using CAPairSoAConstView = CAPairSoA::ConstView;

  // pair of index and NeighborCell
  GENERATE_SOA_LAYOUT(CACellPairLayout,
                      SOA_COLUMN(uint32_t, inner),
                      SOA_COLUMN(NeighborCell, outer),
                      SOA_COLUMN(int16_t, phiResid))  // quantized phi residual at middle hit [rad]

  using CACellPairSoA = CACellPairLayout<>;
  using CACellPairSoAView = CACellPairSoA::View;
  using CACellPairSoAConstView = CACellPairSoA::ConstView;

}  // namespace caStructures

#endif  // RecoTracker_PixelSeeding_interface_CAPairSoA_h
