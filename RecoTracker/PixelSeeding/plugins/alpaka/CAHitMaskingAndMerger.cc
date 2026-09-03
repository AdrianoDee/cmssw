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

    CAHitMaskingAndMerger::TkSoADevice CAHitMaskingAndMerger::makeMergedTracks(Queue& queue,
                                                                                ::reco::InputTracks const& allTracks,
                                                                                int maxTracks,
                                                                                float matchFraction,
                                                                                int minHitsForDuplicate,
                                                                                pixelTrack::Quality minQuality) const {
	    CAHitMaskingAndMergerKernels kernels (maxTracks, queue);

	    kernels.countGoodTracks(queue, allTracks, minQuality);
	    kernels.fillGoodTracks(queue, allTracks);
	    kernels.filterTracks(queue, matchFraction, minHitsForDuplicate);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
	    std::cout << "finished filtering track SoAs on GPU" << std::endl;
	#endif

    return kernels.getTracks();
	  }


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


}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
