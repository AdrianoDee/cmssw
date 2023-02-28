#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HelixFitOnGPU.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  template <typename TrackerTraits>
  void HelixFitOnGPU<TrackerTraits>::allocateOnGPU(TupleMultiplicity const *tupleMultiplicity,
                                                   OutputSoAView &helix_fit_results) {
    tuples_ = &helix_fit_results.hitIndices();
    tupleMultiplicity_ = tupleMultiplicity;
    outputSoa_ = helix_fit_results;

    ALPAKA_ASSERT_OFFLOAD(tuples_);
    ALPAKA_ASSERT_OFFLOAD(tupleMultiplicity_);
  }

  template <typename TrackerTraits>
  void HelixFitOnGPU<TrackerTraits>::deallocateOnGPU() {}

  template class HelixFitOnGPU<pixelTopology::Phase1>;
  template class HelixFitOnGPU<pixelTopology::Phase2>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
