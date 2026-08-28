// C++ headers
#include <cassert>
#include <cstdint>
#include <type_traits>

// Alpaka headers
#include <alpaka/alpaka.hpp>

// CMSSW headers
#include "DataFormats/BeamSpot/interface/BeamSpotPOD.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsMaskingSoACollection.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"

// local headers
#include "PixelRecHitMaskingKernel.h"
#include "PixelRecHits.h"

//#define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace cms::alpakatools;

  using namespace ALPAKA_ACCELERATOR_NAMESPACE::reco;

  namespace pixelgpudetails {

    TrackingRecHitsMaskingSoACollection makeHitsMaskingAsync(uint32_t const nHits,
                                                                                    Queue& queue) {
      
      TrackingRecHitsMaskingSoACollection mask_d(queue, nHits);

      auto deviceMask = cms::alpakatools::make_device_view<uint32_t>(queue, mask_d.view().recHitMask(), nHits);
    
      alpaka::memset(queue,deviceMask,0x0);

      return mask_d;
    }
  }  // namespace pixelgpudetails
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
