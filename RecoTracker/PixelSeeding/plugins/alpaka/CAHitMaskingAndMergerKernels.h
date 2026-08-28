#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernels_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernels_h

// #define GPU_DEBUG
// #define DUMP_GPU_TK_TUPLES

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsMaskingSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AtomicPairCounter.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/CAPairSoACollection.h"

#include "CACell.h"
#include "CAPixelDoublets.h"
#include "CAStructures.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class CAHitMaskingAndMergerKernels {
  public:
    CAHitMaskingAndMergerKernels() = default;
    ~CAHitMaskingAndMergerKernels() = default;

    CAHitMaskingAndMergerKernels(const CAHitMaskingAndMergerKernels&) = delete;
    CAHitMaskingAndMergerKernels(CAHitMaskingAndMergerKernels&&) = delete;
    CAHitMaskingAndMergerKernels& operator=(const CAHitMaskingAndMergerKernels&) = delete;
    CAHitMaskingAndMergerKernels& operator=(CAHitMaskingAndMergerKernels&&) = delete;

    void updateMasking(Queue& queue,
                       ::reco::TrackingRecHitsMaskingView& mask_view,
                       const ::reco::TrackSoAConstView& trackd_view,
                       const ::reco::TrackHitSoAConstView& trackhitd_view,
                       pixelTrack::Quality minQuality,
                       uint32_t iterationIndex);

    void updateHitOffsets(Queue& queue, int tksBeg, int tksEnd, int nHits, ::reco::TrackSoAView& trackd_view);

    void filterTracks(Queue& queue,
                      ::reco::TrackSoAView& track_view,
                      ::reco::TrackHitSoAView& trackHit_view,
                      const ::reco::TrackSoAConstView& inpTrack_view,
                      const ::reco::TrackHitSoAConstView& inpTrackHit_view,
                      pixelTrack::Quality minQuality,
                      double matchFraction);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernels_h
