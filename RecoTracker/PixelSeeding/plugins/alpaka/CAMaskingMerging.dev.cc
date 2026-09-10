#include <array>
#include <cassert>
#include <functional>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "CAMaskingMerging.h"

// #define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE::caMasking {


    class Kernel_updateMasking {
    public:
      ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                    ::reco::TrackingRecHitsMaskingView mask_view,
                                    ::reco::TrackSoAConstView const& trackd_view,
                                    ::reco::TrackHitSoAConstView const& trackhitd_view,
                                    pixelTrack::Quality minQuality,
                                    uint32_t iterationIndex) const {
#ifdef GPU_DEBUG
        if (cms::alpakatools::once_per_grid(acc)) {
          printf("Kernel_updateMasking: nTracks: %u\n", trackd_view.metadata().size());
        }
#endif

        // note to self: this is launched with metadata.size, but here we loop on nTracks
        // would be better for this loop to go over metadata.size?
        for (uint32_t j : cms::alpakatools::uniform_elements_x(acc, trackd_view.nTracks())) {
          if (trackd_view[j].quality() < minQuality)
            continue;

          uint32_t const start = (j == 0) ? 0 : trackd_view[j - 1].hitOffsets();
          uint32_t const end = trackd_view[j].hitOffsets();
          uint32_t const nTrackHits = end - start;

          for (uint32_t k : cms::alpakatools::uniform_elements_y(acc, nTrackHits)) {
            mask_view[trackhitd_view[start + k].id()].recHitMask() = iterationIndex;
          }
        }
      }
    };


    void makeMaskingAsync(Queue& queue,
                                                    MapToHit& outMask,
                                                    MapToHit const& inMask,
                                                    TkSoADevice const& inTracks,
                                                    pixelTrack::Quality minQuality,
                                                    uint32_t iterationIndex) {

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting makeMaskingAsync::updateMasking" << std::endl;
#endif

    const int nHits = inMask.view().metadata().size();

    ALPAKA_ASSERT_ACC(nHits == outMask.view().metadata().size());

    auto tracks = inTracks.view().tracks();
    auto hits = inTracks.view().trackHits();
    auto mask = outMask.view();

    constexpr uint32_t hitBatch = 8;
    constexpr uint32_t tracksPerBlock  = 128 / hitBatch;

    Vec2D const blocks{cms::alpakatools::divide_up_by(tracks.metadata().size(), tracksPerBlock), 1u};
    Vec2D const threads{tracksPerBlock, hitBatch};

    auto const workDiv2D = cms::alpakatools::make_workdiv<Acc2D>(blocks, threads);
#ifdef GPU_DEBUG
    std::cout << "Kernel_updateMasking: workDiv2D: " << workDiv2D << std::endl;
    std::cout << "Kernel_updateMasking: nTracks: " << tracks.metadata().size() << std::endl;
    std::cout << "Kernel_updateMasking: nHits: " << hits.metadata().size() << std::endl;
#endif

    alpaka::exec<Acc2D>(
        queue, workDiv2D, Kernel_updateMasking{}, mask, tracks, hits, minQuality, iterationIndex);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_updateMasking -> done!" << std::endl;
    std::cout << "finished updating pixel masking on GPU" << std::endl;
#endif

  }


}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
