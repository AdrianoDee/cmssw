// #define GPU_DEBUG
// #define DUMP_GPU_TK_TUPLES

#include <array>
#include <cassert>
#include <functional>
#include <vector>

#include <alpaka/alpaka.hpp>


// #include "FWCore/ParameterSet/interface/ParameterSet.h"
// #include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
// #include "FWCore/Utilities/interface/Exception.h"
// #include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "CAHitMaskingAndMerger.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    using MapToHit = reco::TrackingRecHitsMaskingSoACollection;
    using TkSoADevice = reco::TracksSoACollection;


    MapToHit CAHitMaskingAndMerger::makeMaskingAsync(Queue& queue,
                                                   MapToHit const& mask_d,
                                                   TkSoADevice const& tracks_d,
                                                   pixelTrack::Quality minQuality,
                                                   uint32_t iterationIndex) const {
    const int nHits = mask_d.view().metadata().size();

    MapToHit mask(queue, static_cast<uint32_t>(nHits));

    // alpaka::memcpy(queue,
    //                cms::alpakatools::make_device_view(queue, mask.view().recHitMask(), nHits),
    //                cms::alpakatools::make_device_view(queue, mask_d.view().recHitMask(), nHits));
    alpaka::memcpy(queue, mask.buffer(), mask_d.buffer());

    CAHitMaskingAndMergerKernels kernels;

    auto tracksd_view = tracks_d.view().tracks();
    auto tracks_hitsd_view = tracks_d.view().trackHits();

    kernels.updateMasking(queue, mask.view(), tracksd_view, tracks_hitsd_view, minQuality, iterationIndex);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "finished updating pixel masking on GPU" << std::endl;
#endif

    return mask;
  }

  void CAHitMaskingAndMerger::updateHitOffsets(
      Queue& queue, int tksBeg, int tksEnd, int nHits, TkSoADevice& tracks_d) const {
    CAHitMaskingAndMergerKernels kernels;

    auto tracksd_view = tracks_d.view().tracks();

    kernels.updateHitOffsets(queue, tksBeg, tksEnd, nHits, tracksd_view);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "finished updating track SoAs hit offsets on GPU" << std::endl;
#endif

    return;
  }

  TkSoADevice CAHitMaskingAndMerger::makeFilteredTracks(Queue& queue,
                                                        int nTracks,
                                                        int nHits,
                                                        TkSoADevice const& inpTracks,
                                                        pixelTrack::Quality minQuality,
                                                        double matchFraction) const {
    CAHitMaskingAndMergerKernels kernels;

    TkSoADevice tracks(queue, nTracks, nHits);

    auto tracksd_view = tracks.view().tracks();
    auto tracks_hitsd_view = tracks.view().trackHits();
    auto inptracksd_view = inpTracks.view().tracks();
    auto inptracks_hitsd_view = inpTracks.view().trackHits();

    kernels.filterTracks(
        queue, tracksd_view, tracks_hitsd_view, inptracksd_view, inptracks_hitsd_view, minQuality, matchFraction);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "finished filtering track SoAs on GPU" << std::endl;
#endif

    return tracks;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
