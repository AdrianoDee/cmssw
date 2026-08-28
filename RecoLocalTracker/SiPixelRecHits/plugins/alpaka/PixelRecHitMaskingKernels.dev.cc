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

    TrackingRecHitsMaskingSoACollection PixelRecHitMaskingKernel::makeHitsMaskingAsync(uint32_t const nHits,
                                                                                    Queue queue) const {
      // using namespace pixelRecHits;

//       TrackingRecHitsMaskingSoACollection mask_d(queue, nHits);

//       int threadsPerBlock = 128;
//       int blocks = cms::alpakatools::divide_up_by(nHits, threadsPerBlock);
//       const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

// #ifdef GPU_DEBUG
//       std::cout << "launching LaunchZerosPixelMask kernel on " << alpaka::core::demangled<Acc1D> << " with " << blocks
//                 << " blocks" << std::endl;
// #endif
//       alpaka::exec<Acc1D>(queue, workDiv1D, LaunchZerosPixelMask{}, mask_d.view());

// #ifdef GPU_DEBUG
//       alpaka::wait(queue);
//       std::cout << "makeHitsMaskingAsync -> DONE!" << std::endl;
// #endif
      
      TrackingRecHitsMaskingSoACollection mask_d(queue, nHits);

      alpaka::memset(queue,cms::alpakatools::make_device_view(queue, mask_d.view().recHitMask(), nHits),0);
      return mask_d;
    }
  }  // namespace pixelgpudetails
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
