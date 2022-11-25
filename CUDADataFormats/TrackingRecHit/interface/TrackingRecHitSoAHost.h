#ifndef CUDADataFormats_RecHits_TrackingRecHitsHost_h
#define CUDADataFormats_RecHits_TrackingRecHitsHost_h

#include <cstdint>

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitsUtilities.h"
#include "CUDADataFormats/Common/interface/PortableHostCollection.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

namespace trackingRecHit {

  class TrackingRecHitSoAHost : public cms::cuda::PortableHostCollection<TrackingRecHitSoALayout<>> {
  public:
    TrackingRecHitSoAHost() = default;

    // This SoA Host is used basically only for DQM
    // so we  just need a slim constructor
    explicit TrackingRecHitSoAHost(uint32_t nHits, cudaStream_t stream)
        : PortableHostCollection<TrackingRecHitSoALayout<>>(nHits, stream) {}

    explicit TrackingRecHitSoAHost(uint32_t nHits,
                                   bool isPhase2,
                                   int32_t offsetBPIX2,
                                   pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                   uint32_t const* hitsModuleStart,
                                   cudaStream_t stream)
        : PortableHostCollection<TrackingRecHitSoALayout<>>(nHits, stream),
          nHits_(nHits),
          cpeParams_(cpeParams),
          offsetBPIX2_(offsetBPIX2) {
      nModules_ = isPhase2 ? phase2PixelTopology::numberOfModules : phase1PixelTopology::numberOfModules;
      std::cout << "PORCA MADONNA!!!!!!!!!!!!!!!!!" << std::endl;
      view().nHits() = nHits;
      view().nMaxModules() = nModules_;
      std::copy(hitsModuleStart, hitsModuleStart + nModules_ + 1, view().hitsModuleStart().begin());
      memcpy(&(view().cpeParams()), cpeParams, sizeof(pixelCPEforGPU::ParamsOnGPU));
      view().offsetBPIX2() = offsetBPIX2;
    }

    uint32_t nHits() const { return nHits_; }
    uint32_t nModules() const { return nModules_; }
    uint32_t offsetBPIX2() const { return offsetBPIX2_; }
    auto phiBinnerStorage() { return phiBinnerStorage_; }

  private:
    uint32_t nHits_;  //Needed for the host SoA size
    pixelCPEforGPU::ParamsOnGPU const* cpeParams_;
    uint32_t offsetBPIX2_;

    uint32_t nModules_;
    trackingRecHitSoA::PhiBinnerStorageType* phiBinnerStorage_;
  };

}  // namespace trackingRecHit

#endif  // CUDADataFormats_Track_TrackHeterogeneousT_H
