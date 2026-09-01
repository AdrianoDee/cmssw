#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernels_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernels_h

// #define GPU_DEBUG
// #define DUMP_GPU_TK_TUPLES

#include <cstdint>
#include <utility>

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
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"

namespace reco {
  constexpr uint32_t maxTrackSoACollections = 50;

  struct InputTracks {
    reco::TrackSoAConstView views[maxTrackSoACollections];
    reco::TrackHitSoAConstView hitViews[maxTrackSoACollections];
    int nInputs;
  };
}  // namespace reco

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

    CAHitMaskingAndMergerKernels(uint32_t maxTracks, Queue& queue);

    void updateMasking(Queue& queue,
                       ::reco::TrackingRecHitsMaskingView& mask_view,
                       const ::reco::TrackSoAConstView& trackd_view,
                       const ::reco::TrackHitSoAConstView& trackhitd_view,
                       pixelTrack::Quality minQuality,
                       uint32_t iterationIndex);

    void countGoodTracks(Queue& queue,
                                        ::reco::InputTracks const& allTracks,
                                        pixelTrack::Quality minQuality);
    void fillGoodTracks(Queue& queue,
                                    ::reco::InputTracks const& allTracks);

    void updateHitOffsets(Queue& queue, int tksBeg, int tksEnd, int nHits, ::reco::TrackSoAView& trackd_view);

    void filterTracks(Queue& queue,
                      ::reco::TrackSoAView& track_view,
                      ::reco::TrackHitSoAView& trackHit_view,
                      // const ::reco::TrackSoAConstView& inpTrack_view,
                      // const ::reco::TrackHitSoAConstView& inpTrackHit_view,
                      pixelTrack::Quality minQuality,
                      double matchFraction);
  private:

    std::optional<reco::TracksSoACollection> tracks_d_;
    std::optional<CAPairSoACollection> iterGood_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t []>> totCounters_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> iterGoodHits_;

    std::optional<cms::alpakatools::device_view<Device, uint32_t>> totTracks_;
    std::optional<cms::alpakatools::device_view<Device, uint32_t>> totHits_;
      
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernels_h
