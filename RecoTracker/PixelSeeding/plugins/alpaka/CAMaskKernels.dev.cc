#include <cstdint>
#include <iostream>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "CAMaskKernels.h"

// #define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE::caMasking {

  struct Kernel_updateMasking {
  public:
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  ::reco::TrackingRecHitsMaskingView mask_view,
                                  ::reco::TrackSoAConstView const& track_view,
                                  ::reco::TrackHitSoAConstView const& trackhit_view,
                                  pixelTrack::Quality minQuality) const {
#ifdef GPU_DEBUG
      if (cms::alpakatools::once_per_grid(acc)) {
        printf("Kernel_updateMasking: nTracks: %u\n", track_view.metadata().size());
      }
#endif

      // tracks are distributed along y (one track per thread row), hits along x (within a block)
      for (uint32_t j : cms::alpakatools::uniform_elements_y(acc, track_view.nTracks())) {
        if (track_view[j].quality() < minQuality)
          continue;

        auto const maskValue = static_cast<uint32_t>(track_view[j].iteration()) + 1;  // first iteration has index 0

        uint32_t const start = (j == 0) ? 0 : track_view[j - 1].hitOffsets();
        uint32_t const end = track_view[j].hitOffsets();
        uint32_t const nTrackHits = end - start;

        for (uint32_t k : cms::alpakatools::uniform_elements_x(acc, nTrackHits)) {
          ALPAKA_ASSERT_ACC(static_cast<int>(trackhit_view[start + k].id()) < mask_view.metadata().size());
          // ^ if not it will crash below, but at least the assert is easier to catch
          mask_view[trackhit_view[start + k].id()].recHitMask() = maskValue;
        }
      }
    }
  };

  // As Kernel_updateMasking, but for tracks with at least params.minHits hits only the hits compatible with the
  // fitted helix are masked: the incompatible ones (wrong attachments) stay available to the next iteration.
  struct Kernel_updateMaskingCompatibleHits {
  public:
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  ::reco::TrackingRecHitsMaskingView mask_view,
                                  ::reco::TrackSoAConstView const& track_view,
                                  ::reco::TrackHitSoAConstView const& trackhit_view,
                                  HitsMultiView hits,
                                  CompatibleHitsParams params,
                                  pixelTrack::Quality minQuality) const {
      namespace chc = ::caHitCompatibility;

      // tracks are distributed along y (one track per thread row), hits along x (within a block)
      for (uint32_t j : cms::alpakatools::uniform_elements_y(acc, track_view.nTracks())) {
        if (track_view[j].quality() < minQuality)
          continue;

        auto const maskValue = static_cast<uint32_t>(track_view[j].iteration()) + 1;  // first iteration has index 0

        uint32_t const start = (j == 0) ? 0 : track_view[j - 1].hitOffsets();
        uint32_t const end = track_view[j].hitOffsets();
        uint32_t const nTrackHits = end - start;

        // the fit of a triplet passes through its hits: the residuals carry no information, mask all the hits
        bool const checkResiduals = nTrackHits >= params.minHits;
        chc::TrackHelix helix{};
        if (checkResiduals) {
          auto const state = track_view[j].state();
          helix = chc::makeTrackHelix(state(0), state(1), state(2), state(3), state(4), params.bField);
        }

        for (uint32_t k : cms::alpakatools::uniform_elements_x(acc, nTrackHits)) {
          uint32_t const hitId = trackhit_view[start + k].id();
          ALPAKA_ASSERT_ACC(static_cast<int>(hitId) < mask_view.metadata().size());
          ALPAKA_ASSERT_ACC(static_cast<int>(hitId) < hits.size());
          if (checkResiduals) {
            auto const hit = hits[hitId];
            auto const region =
                chc::hitRegion(hitId, params.nPixelHits, hit.detectorIndex(), params.nPixelBarrelModules);
            auto const res =
                chc::computeResiduals(helix, hit.xGlobal(), hit.yGlobal(), hit.zGlobal(), chc::isBarrelLike(region));
            if (not chc::isCompatible(params.cuts, region, res))
              continue;
          }
          mask_view[hitId].recHitMask() = maskValue;
        }
      }
    }
  };

  void makeMaskingAsync(Queue& queue, MapToHit& outMask, TkSoADevice const& inTracks, pixelTrack::Quality minQuality) {
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting makeMaskingAsync::updateMasking" << std::endl;
#endif

    auto tracks = inTracks.view().tracks();
    auto hits = inTracks.view().trackHits();
    auto mask = outMask.view();

    constexpr uint32_t hitBatch = 8;
    constexpr uint32_t tracksPerBlock = 128 / hitBatch;

    Vec2D const blocks{cms::alpakatools::divide_up_by(tracks.metadata().size(), tracksPerBlock), 1u};
    Vec2D const threads{tracksPerBlock, hitBatch};

    auto const workDiv2D = cms::alpakatools::make_workdiv<Acc2D>(blocks, threads);
#ifdef GPU_DEBUG
    std::cout << "Kernel_updateMasking: workDiv2D: " << workDiv2D << std::endl;
    std::cout << "Kernel_updateMasking: nTracks: " << tracks.metadata().size() << std::endl;
    std::cout << "Kernel_updateMasking: nHits: " << hits.metadata().size() << std::endl;
#endif

    alpaka::exec<Acc2D>(queue, workDiv2D, Kernel_updateMasking{}, mask, tracks, hits, minQuality);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_updateMasking -> done!" << std::endl;
    std::cout << "finished updating pixel masking on GPU" << std::endl;
#endif
  }

  void makeCompatibleHitsMaskingAsync(Queue& queue,
                                      MapToHit& outMask,
                                      TkSoADevice const& inTracks,
                                      pixelTrack::Quality minQuality,
                                      HitsMultiView const& hits,
                                      CompatibleHitsParams const& params) {
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting makeCompatibleHitsMaskingAsync::updateMaskingCompatibleHits" << std::endl;
#endif

    auto tracks = inTracks.view().tracks();
    auto trackHits = inTracks.view().trackHits();
    auto mask = outMask.view();

    // same work division as makeMaskingAsync
    constexpr uint32_t hitBatch = 8;
    constexpr uint32_t tracksPerBlock = 128 / hitBatch;

    Vec2D const blocks{cms::alpakatools::divide_up_by(tracks.metadata().size(), tracksPerBlock), 1u};
    Vec2D const threads{tracksPerBlock, hitBatch};

    auto const workDiv2D = cms::alpakatools::make_workdiv<Acc2D>(blocks, threads);

    alpaka::exec<Acc2D>(
        queue, workDiv2D, Kernel_updateMaskingCompatibleHits{}, mask, tracks, trackHits, hits, params, minQuality);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_updateMaskingCompatibleHits -> done!" << std::endl;
#endif
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caMasking
