#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAITExtend_CAITCandidateCollector_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAITExtend_CAITCandidateCollector_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Math/interface/approx_atan2.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "../CAStructures.h"
#include "CAITExtendLayout.h"
#include "HelixPropagator.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::caITExtend {

  using HitsConstView = ::reco::TrackingRecHitConstView;
  using TkSoAConstView = ::reco::TrackSoAConstView;

  // Per-(track, slot) candidate-hit bucket.  Fixed-cap atomic-append container
  // -- same VecArray<> pattern CA already uses for CellNeighborsT / CellTracksT
  // (see CAStructures.h:97-101).  Compile-time cap matches
  // Layout<>::kMaxCandidatesPerTrackLayer so memory is statically bounded.
  using CandidateSlot =
      cms::alpakatools::VecArray<caStructures::hindex_type,
                                 ::caITExtend::Layout<::caITExtend::Phase2OTStubsTag>::kMaxCandidatesPerTrackLayer>;

  // Kernel-launch-time parameter POD.  All fields are filled host-side and
  // passed BY VALUE to the kernel (Alpaka marshals it into kernel parameter
  // memory -- no device allocation, no memcpy).  Mix of Python-config-derived
  // and per-event runtime values:
  //   - bField, offsetStubs : per-event runtime, read from the magnetic-field
  //                            record and the hit SoA cached host-side scalar
  //                            (see HelixFit.h:82-84 for the offsetStubs
  //                            convention -- never dereference the device view
  //                            from host).
  //   - filterIter          : per-iteration constant from the producer.
  //   - everything else     : from Python (inwardExtension PSet).
  //
  // Note: the per-slot cap is compile-time (CandidateSlot::capacity()), not a
  // runtime param -- same convention as CellNeighborsT::maxCellNeighbors.
  struct CollectorParams {
    float nSigmaPhi;
    float nSigmaZ;
    float floorDPhi;
    float floorDZ;
    float maxDPhi;
    float maxDZ;
    float bField;          // runtime (per event)
    float materialDensity;
    uint16_t maxLayersPerTrack;
    uint32_t offsetStubs;  // runtime: first OT-stub index in unified hit collection
    ::pixelTrack::Iteration filterIter;
    // Quality threshold: tracks with quality < minQuality are skipped.
    // Quality enum order: bad(0) < edup(1) < dup(2) < loose(3) < strict(4) <
    // tight(5) < highPurity(6).  Recommended default: edup, which skips ONLY
    // bad tracks (NaN fits / sub-doublets).  Note: setting minQuality = bad
    // would accept everything including NaN, which is almost certainly not
    // what you want.
    // The OT-only displaced iteration may carry valid tracks at quality below
    // `strict` (e.g. when the downstream DNN discriminator is bypassed with
    // score = 0); use `edup` or `dup` to let them through.
    ::pixelTrack::Quality minQuality;
  };

  // Layout of the per-track candidate buffer:
  //   slots    [trackIdx * maxLayersPerTrack + slot]  -> VecArray of hit ids (cap = CandidateSlot::capacity)
  //   slotLayer[trackIdx * maxLayersPerTrack + slot]  -> CA layer idx for that slot (0xff = unused)
  // Slot size is read via slots[k].size(); membership via slots[k][i].
  struct CollectorBuffers {
    CandidateSlot* slots;
    uint8_t* slotLayer;
  };

  // kernelCollectITCandidates: per-track helix propagation + PhiBinner window query.
  // Work distribution: 1 block per input track, BlockSize threads sharing the prediction.
  template <typename TrackerTraits, typename LayoutTraits>
  struct kernelCollectITCandidates {
    using PhiBinner = caStructures::PhiBinnerT<TrackerTraits>;

    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  TkSoAConstView tracks,
                                  HitsConstView hits,
                                  PhiBinner const* phiBinner,
                                  CollectorParams const params,
                                  CollectorBuffers buffers,
                                  uint32_t nTracks,
                                  uint32_t* __restrict__ extCounters) const {
      // 1 block per track in the grid Y dimension; X is the thread axis used
      // for cooperative bin scanning.
      using namespace ::caITExtend;
      using namespace caStructures;
      using Layout = ::caITExtend::Layout<LayoutTraits>;

      const auto blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];
      // First gate: skip uninitialised SoA slots past the actual track count.
      // (extCounters[10..12] are funnel counters that should only reflect
      //  *real* tracks, not capacity overflow.)
      if (int(blockIdx) >= tracks.nTracks())
        return;
      if (blockIdx >= nTracks)
        return;
      const tindex_type trackIdx = blockIdx;
      const auto leader = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];

      // Filter: only tracks tagged for inward extension.
      if (tracks[trackIdx].iteration() != params.filterIter) {
        if (extCounters && leader == 0)
          alpaka::atomicAdd(acc, &extCounters[10], 1u, alpaka::hierarchy::Blocks{});
        return;
      }
      // Skip tracks below the configured minimum quality.  See the comment in
      // CollectorParams::minQuality for the enum order and recommended values.
      // The recommended default is `edup`: skips ONLY NaN-fit / sub-doublet
      // tracks (which stay at `bad`), letting everything else through --
      // including edup/dup/loose -- since the OT-only displaced iteration may
      // hold valid tracks at lower quality values.
      if (tracks[trackIdx].quality() < params.minQuality) {
        if (extCounters && leader == 0)
          alpaka::atomicAdd(acc, &extCounters[11], 1u, alpaka::hierarchy::Blocks{});
        return;
      }
      // Passed both filters.
      if (extCounters && leader == 0)
        alpaka::atomicAdd(acc, &extCounters[12], 1u, alpaka::hierarchy::Blocks{});

      const ::reco::Vector5f state = tracks[trackIdx].state();
      const ::reco::Vector15f covPacked = tracks[trackIdx].covariance();
      const float cotTheta = state(3);

      // Determine which disk side this track explores (sign of cotTheta).
      const bool forwardSide = cotTheta > 0.f;

      // Slot index in the per-track output (slot 0 = outermost reached layer)
      // The walk is ordered outermost -> innermost (matches Layout::barrel/forward/backward).
      uint16_t slot = 0;

      // ---- Disk side (outermost-z first) -----------------------------------
      const int nDisks = forwardSide ? Layout::kNForward : Layout::kNBackward;
      bool anyDiskCrossed = false;
      for (int d = 0; d < nDisks; ++d) {
        if (slot >= params.maxLayersPerTrack)
          break;
        const auto& L = forwardSide ? Layout::forward()[d] : Layout::backward()[d];
        const float targetZ = 0.5f * (L.zMin + L.zMax);
        const auto crossing =
            helixCrossingToZ(state, covPacked, targetZ, params.bField, params.materialDensity);
        if (!crossing.valid)
          continue;
        // Box test on r at the disk plane.
        if (crossing.r < L.rMin || crossing.r > L.rMax)
          continue;
        anyDiskCrossed = true;
        scanLayer<TAcc>(acc, hits, phiBinner, params, buffers, trackIdx, slot, L.caLayerIdx,
                        crossing, /*isBarrel=*/false, L.rMin, L.rMax, extCounters);
        ++slot;
      }
      if (anyDiskCrossed && extCounters && leader == 0)
        alpaka::atomicAdd(acc, &extCounters[13], 1u, alpaka::hierarchy::Blocks{});

      // ---- Barrel layers, outermost-inward --------------------------------
      bool anyBarrelCrossed = false;
      for (int b = 0; b < Layout::kNBarrel; ++b) {
        if (slot >= params.maxLayersPerTrack)
          break;
        const auto& L = Layout::barrel()[b];
        const auto crossing =
            helixCrossingToR(state, covPacked, L.r, params.bField, params.materialDensity);
        if (!crossing.valid)
          continue;
        if (crossing.z < L.zMin || crossing.z > L.zMax)
          continue;
        anyBarrelCrossed = true;
        scanLayer<TAcc>(acc, hits, phiBinner, params, buffers, trackIdx, slot, L.caLayerIdx,
                        crossing, /*isBarrel=*/true, L.zMin, L.zMax, extCounters);
        ++slot;
      }
      if (anyBarrelCrossed && extCounters && leader == 0)
        alpaka::atomicAdd(acc, &extCounters[14], 1u, alpaka::hierarchy::Blocks{});

      // Mark remaining slots as unused.  `leader` is already defined at the
      // top of this function -- reuse it.
      if (leader == 0) {
        for (uint16_t s = slot; s < params.maxLayersPerTrack; ++s) {
          const auto idx = trackIdx * params.maxLayersPerTrack + s;
          buffers.slotLayer[idx] = 0xff;
          buffers.slots[idx].reset();
        }
      }
    }

   private:
    // Cooperative phi-bin scan: all threads in the block walk the bins in
    // (iphi_pred +/- iphiHalfWindow); each thread visits a stride of hit indices
    // inside the bins and atomically appends survivors to the per-slot list.
    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC void scanLayer(TAcc const& acc,
                                 HitsConstView hits,
                                 PhiBinner const* phiBinner,
                                 CollectorParams const& params,
                                 CollectorBuffers buffers,
                                 caStructures::tindex_type trackIdx,
                                 uint16_t slot,
                                 uint8_t caLayer,
                                 ::caITExtend::HelixCrossing const& crossing,
                                 bool isBarrel,
                                 float lowerExtent,
                                 float upperExtent,
                                 uint32_t* __restrict__ extCounters) const {
      using PhiBinner = caStructures::PhiBinnerT<TrackerTraits>;
      using namespace ::caITExtend;

      const auto leader = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];

      const float sigmaPhi = std::sqrt(crossing.sigma2Phi);
      const float sigmaZ = std::sqrt(crossing.sigma2Z);
      float windowPhi = params.nSigmaPhi * sigmaPhi;
      if (windowPhi < params.floorDPhi)
        windowPhi = params.floorDPhi;
      if (windowPhi > params.maxDPhi)
        windowPhi = params.maxDPhi;
      float windowSecondary = params.nSigmaZ * sigmaZ;
      if (windowSecondary < params.floorDZ)
        windowSecondary = params.floorDZ;
      if (windowSecondary > params.maxDZ)
        windowSecondary = params.maxDZ;

      const int16_t iphiPred = phi2short(crossing.phi);
      const int16_t iphiHalf = phi2short(windowPhi);  // half-width in iphi units

      const auto hoff = PhiBinner::histOff(caLayer);
      const auto klo = PhiBinner::bin(int16_t(iphiPred - iphiHalf));
      const auto khi = PhiBinner::bin(int16_t(iphiPred + iphiHalf));
      auto incr = [](auto& k) { k = (k + 1) % PhiBinner::nbins(); };

      const auto slotIdx = trackIdx * params.maxLayersPerTrack + slot;
      if (leader == 0) {
        buffers.slots[slotIdx].reset();
        buffers.slotLayer[slotIdx] = caLayer;
      }
      alpaka::syncBlockThreads(acc);

      // Iterate phi bins in [klo, khi] inclusive (wrapping).
      auto khh = khi;
      incr(khh);
      const auto nThreads = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0];
      for (auto kk = klo; kk != khh; incr(kk)) {
        auto const* __restrict__ pBeg = phiBinner->begin(kk + hoff);
        auto const* __restrict__ pEnd = phiBinner->end(kk + hoff);
        const auto nInBin = static_cast<int>(pEnd - pBeg);
        // Distribute hits in bin across threads.
        for (int j = leader; j < nInBin; j += nThreads) {
          const auto hi = pBeg[j];
          // Defensively skip OT stub indices (PhiBinner is shared across all
          // layers; a hit with an OT layer in iphi space should never appear in
          // an IT layer bin, but the check is cheap).
          if (hi >= params.offsetStubs)
            continue;

          // Box test on the secondary coordinate (z for barrel, r for disk).
          const float zo = hits[hi].zGlobal();
          const float ro = hits[hi].rGlobal();
          if (isBarrel) {
            if (std::abs(zo - crossing.z) > windowSecondary)
              continue;
          } else {
            if (std::abs(ro - crossing.r) > windowSecondary)
              continue;
            // also enforce the disk-plane z slab in case hits at this CA index
            // span more than one physical disk
            if (zo < lowerExtent || zo > upperExtent)
              continue;
          }

          // Atomic append.  push_back returns -1 if the slot is already at its
          // compile-time capacity (CandidateSlot::capacity()).
          const int pos = buffers.slots[slotIdx].push_back(acc, hi);
          if (extCounters) {
            if (pos >= 0)
              alpaka::atomicAdd(acc, &extCounters[0], 1u, alpaka::hierarchy::Blocks{});
            else
              alpaka::atomicAdd(acc, &extCounters[1], 1u, alpaka::hierarchy::Blocks{});
          }
        }
      }
      alpaka::syncBlockThreads(acc);
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caITExtend

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAITExtend_CAITCandidateCollector_h
