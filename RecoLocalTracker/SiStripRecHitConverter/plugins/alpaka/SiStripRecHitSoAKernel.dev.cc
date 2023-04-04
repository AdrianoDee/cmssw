// C++ headers
#include <algorithm>
#include <numeric>

// Alpaka headers
#include <alpaka/alpaka.hpp>

// CMSSW headers
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "SiStripRecHitSoAKernel.h"

//#define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;

  namespace hitkernels {

    template <typename TrackerTraits>
    TrackingRecHitAlpakaCollection<TrackerTraits> SiStripRecHitSoAKernel<TrackerTraits>::fillHitsAsync(
        StripHitsHost const& hits_h,
        Queue queue) const {
          
      TrackingRecHitAlpakaCollection<TrackerTraits> hits_d(hits_h.nHits(),queue);
      alpaka::memcpy(queue, hits_d.buffer(), hits_h.buffer());

      // assuming full warp of threads is better than a smaller number...
      if (hits_h.nHits()) {
        
        constexpr auto nLayers = TrackerTraits::numberOfLayers;
        cms::alpakatools::fillManyFromVector<Acc1D>(&(hits_d.view().phiBinner()),
                                                    nLayers,
                                                    hits_d.view().iphi(),
                                                    hits_d.view().hitsLayerStart().data(),
                                                    hits_h.nHits(),
                                                    (uint32_t)256,
                                                    queue);

#ifdef GPU_DEBUG
          alpaka::wait(queue);
#endif
        }

#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "SiStripRecHitSoAKernel -> DONE!" << std::endl;
#endif

      return hits_d;
    }

    template class SiStripRecHitSoAKernel<pixelTopology::Phase1Strip>;

  }  // namespace hitkernels
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
