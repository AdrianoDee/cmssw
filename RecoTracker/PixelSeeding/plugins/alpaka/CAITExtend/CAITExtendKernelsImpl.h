#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAITExtend_CAITExtendKernelsImpl_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAITExtend_CAITExtendKernelsImpl_h

// Device-side kernel functors for the IT inward-extension refit pipeline.
// These are template functors with ALPAKA_FN_ACC operator() bodies.  Safe to
// define in a header AS LONG AS this header is only included from .dev.cc
// translation units (the host-only static_assert in alpaka fires when these
// are instantiated for an actual accelerator from a non-device-compiled .cc).

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/OneToManyAssoc.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "../CAStructures.h"
#include "CAITExtendDoublets.h"  // ExtendedChain

namespace ALPAKA_ACCELERATOR_NAMESPACE::caITExtend {

  // Compute per-track new size and write into off[i+1]. off[0] = 0 must be
  // pre-zeroed (the caller does a memset before this kernel).
  template <typename TrackerTraits>
  struct kernelComputeNewSizes {
    using TkSoAConstView = ::reco::TrackSoAConstView;

    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  TkSoAConstView tracks,
                                  ExtendedChain const* chains,
                                  caStructures::SequentialContainerOffsets* off,
                                  uint16_t refitMinNewHits,
                                  ::pixelTrack::Iteration sourceIter,
                                  uint32_t /*nTracksCapacity*/) const {
      // tracks.nTracks() is the actual filled count (set by Kernel_fillHitDetIndices).
      // Iterating past it would read uninitialised hitOffsets and produce a
      // garbage (unsigned-underflowed) nOld -- then the bucket key in the
      // multiplicity count phase would exceed nOnes() and assert.
      const uint32_t actualN = tracks.nTracks();
      for (auto trackIdx : cms::alpakatools::uniform_elements(acc, actualN)) {
        const auto origStart = (trackIdx == 0) ? 0u : tracks[trackIdx - 1].hitOffsets();
        const auto origEnd = tracks[trackIdx].hitOffsets();
        const uint32_t nOld = (origEnd >= origStart) ? (origEnd - origStart) : 0u;
        const bool sourceTagged = (tracks[trackIdx].iteration() == sourceIter);
        const uint32_t nNew =
            (sourceTagged && chains[trackIdx].nHits >= refitMinNewHits) ? chains[trackIdx].nHits : 0u;
        off[trackIdx + 1] = nOld + nNew;
      }
      // off[i+1] for trackIdx >= actualN stays at 0 (from launchZero); after
      // prefix scan those bucket sizes are zero and downstream kernels skip
      // them via the (n == 0) guard.
    }
  };

  // Write [new IT hits inner->outer] + [original hits] into the new content array.
  // Requires `off` to be post-prefix-scan (cumulative offsets).
  template <typename TrackerTraits>
  struct kernelWriteExtendedContent {
    using TkSoAConstView = ::reco::TrackSoAConstView;
    using TkHitSoAConstView = ::reco::TrackHitSoAConstView;

    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  TkSoAConstView tracks,
                                  TkHitSoAConstView origTrackHits,
                                  ExtendedChain const* chains,
                                  caStructures::SequentialContainerOffsets const* off,
                                  caStructures::hindex_type* content,
                                  uint16_t refitMinNewHits,
                                  ::pixelTrack::Iteration sourceIter,
                                  uint32_t /*nTracksCapacity*/) const {
      const uint32_t actualN = tracks.nTracks();
      for (auto trackIdx : cms::alpakatools::uniform_elements(acc, actualN)) {
        const auto origStart = (trackIdx == 0) ? 0u : tracks[trackIdx - 1].hitOffsets();
        const auto origEnd = tracks[trackIdx].hitOffsets();
        const uint32_t nOld = (origEnd >= origStart) ? (origEnd - origStart) : 0u;
        const bool sourceTagged = (tracks[trackIdx].iteration() == sourceIter);
        const uint32_t nNew =
            (sourceTagged && chains[trackIdx].nHits >= refitMinNewHits) ? chains[trackIdx].nHits : 0u;

        const auto dstStart = off[trackIdx];
        for (uint32_t k = 0; k < nNew; ++k)
          content[dstStart + k] = chains[trackIdx].hitIds[k];
        for (uint32_t k = 0; k < nOld; ++k)
          content[dstStart + nNew + k] = origTrackHits[origStart + k].id();
      }
    }
  };

  // Count phase of the new TupleMultiplicity.
  template <typename TrackerTraits>
  struct kernelCountMultiplicity {
    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  caStructures::SequentialContainerOffsets const* tuplesOff,
                                  caStructures::GenericContainer* multiplicity,
                                  uint32_t nTracks) const {
      for (auto trackIdx : cms::alpakatools::uniform_elements(acc, nTracks)) {
        const uint32_t n = tuplesOff[trackIdx + 1] - tuplesOff[trackIdx];
        if (n == 0)
          continue;
        multiplicity->count(acc, n);
      }
    }
  };

  // Fill phase of the new TupleMultiplicity.
  template <typename TrackerTraits>
  struct kernelFillMultiplicity {
    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  caStructures::SequentialContainerOffsets const* tuplesOff,
                                  caStructures::GenericContainer* multiplicity,
                                  uint32_t nTracks) const {
      for (auto trackIdx : cms::alpakatools::uniform_elements(acc, nTracks)) {
        const uint32_t n = tuplesOff[trackIdx + 1] - tuplesOff[trackIdx];
        if (n == 0)
          continue;
        multiplicity->fill(acc, n, trackIdx);
      }
    }
  };

  // After the refit, rebuild the TrackHitSoA from the extended Tuples and
  // update tracks[i].hitOffsets to point to the new end-of-track position.
  // Also stamps the iteration tag.
  template <typename TrackerTraits>
  struct kernelFillExtendedTrackHits {
    using TkSoAView = ::reco::TrackSoAView;
    using TkHitSoAView = ::reco::TrackHitSoAView;

    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  TkSoAView tracks,
                                  TkHitSoAView trackHits,
                                  ExtendedChain const* chains,
                                  caStructures::SequentialContainerOffsets const* newOff,
                                  caStructures::hindex_type const* newContent,
                                  ::reco::TrackingRecHitConstView hits,
                                  ::pixelTrack::Iteration sourceIter,
                                  ::pixelTrack::Iteration extendedIter,
                                  uint16_t refitMinNewHits,
                                  uint32_t /*nTracksCapacity*/,
                                  uint32_t* __restrict__ extCounters) const {
      const uint32_t actualN = tracks.nTracks();
      // Sentinel: thread 0 of block 0 stamps counters[15] = nTotal so we can
      // confirm host-side that the kernel ran AND that newOff is sized as
      // expected.  Use atomicMax in case other diagnostic kernels also write.
      if (cms::alpakatools::once_per_grid(acc) && extCounters) {
        extCounters[15] = newOff[actualN];
      }
      for (auto trackIdx : cms::alpakatools::uniform_elements(acc, actualN)) {
        tracks[trackIdx].hitOffsets() = newOff[trackIdx + 1];
        if (tracks[trackIdx].iteration() == sourceIter && chains[trackIdx].nHits >= refitMinNewHits) {
          tracks[trackIdx].iteration() = extendedIter;
          if (extCounters)
            alpaka::atomicAdd(acc,
                              &extCounters[16],
                              uint32_t(chains[trackIdx].nHits),
                              alpaka::hierarchy::Blocks{});
        }
      }
      const auto nTotal = newOff[actualN];
      for (auto idx : cms::alpakatools::uniform_elements(acc, nTotal)) {
        const auto hi = newContent[idx];
        trackHits[idx].id() = hi;
        trackHits[idx].detId() = hits[hi].detectorIndex();
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caITExtend

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAITExtend_CAITExtendKernelsImpl_h
