#ifndef RecoTracker_PixelSeeding_interface_TripletDumpSoA_h
#define RecoTracker_PixelSeeding_interface_TripletDumpSoA_h

// Per-built-triplet training-dataset row, filled by Kernel_connect in CA_TRIPLET_DUMP builds only.
// The first 18 columns are the base triplet-DNN features in input order; the derived features are
// recomputed offline from these and lay1/2/3.

#include <alpaka/alpaka.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace caStructures {

  GENERATE_SOA_LAYOUT(TripletDumpLayout,
                      // 18 base features, in DNN input order.
                      SOA_COLUMN(float, absCurvature),
                      SOA_COLUMN(float, tipTimesCurvature),
                      SOA_COLUMN(float, dca),
                      SOA_COLUMN(float, curvatureStubs),
                      SOA_COLUMN(float, curvatureStubsErrSquared),
                      SOA_COLUMN(float, curvature13),
                      SOA_COLUMN(float, dPhi12),
                      SOA_COLUMN(float, dPhi13),
                      SOA_COLUMN(float, dPhi23),
                      SOA_COLUMN(float, dr12),
                      SOA_COLUMN(float, dr13),
                      SOA_COLUMN(float, r1),
                      SOA_COLUMN(float, r2),
                      SOA_COLUMN(float, r3),
                      SOA_COLUMN(float, z1),
                      SOA_COLUMN(float, z2),
                      SOA_COLUMN(float, z3),
                      SOA_COLUMN(float, nStubs),
                      // Signed curvature: the offline derived features need the sign that
                      // absCurvature loses.
                      SOA_COLUMN(float, curvature),
                      SOA_COLUMN(int32_t, lay1),
                      SOA_COLUMN(int32_t, lay2),
                      SOA_COLUMN(int32_t, lay3),
                      // Merged-hit indices, the join key against the truth.
                      SOA_COLUMN(uint32_t, h1),
                      SOA_COLUMN(uint32_t, h2),
                      SOA_COLUMN(uint32_t, h3),
                      SOA_COLUMN(int32_t, iter),
                      // In-kernel DNN score for this triplet, -1 if not evaluated.
                      SOA_COLUMN(float, inKernelScore),
                      // The collection is allocated at full capacity; only rows [0, nValid) are written.
                      SOA_SCALAR(uint32_t, nValid))

  using TripletDumpSoA = TripletDumpLayout<>;
  using TripletDumpSoAView = TripletDumpSoA::View;
  using TripletDumpSoAConstView = TripletDumpSoA::ConstView;

}  // namespace caStructures

#endif  // RecoTracker_PixelSeeding_interface_TripletDumpSoA_h
