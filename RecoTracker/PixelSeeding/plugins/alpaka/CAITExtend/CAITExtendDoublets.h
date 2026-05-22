#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAITExtend_CAITExtendDoublets_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAITExtend_CAITExtendDoublets_h

#include <algorithm>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Math/interface/approx_atan2.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "../CAStructures.h"
#include "CAITCandidateCollector.h"
#include "CAITExtendLayout.h"
#include "HelixPropagator.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::caITExtend {

  using HitsConstView = ::reco::TrackingRecHitConstView;
  using TkSoAConstView = ::reco::TrackSoAConstView;

  // Top-K-per-layer + global enumeration chain selector ("Level 2"):
  //   1. Per IT layer crossed by the helix, keep the top kTopK candidates by
  //      per-hit chi2 to the helix prediction.
  //   2. Enumerate kTopK^L combinations across the L populated slots (capped
  //      at maxNewLayers slots).  For each combination compute the chain
  //      score: sum of per-hit chi2 + per-consecutive-pair kappa-consistency
  //      penalty.
  //   3. Write the lowest-score chain into ExtendedChain.
  //
  // Memory per track: kMaxLayersForEnum * kTopK * sizeof(float+id) bytes in
  // registers/local memory.  With kTopK=2 and kMaxLayersForEnum=4 this is
  // 4 * 2 * 8 = 64 B per track -- trivial.

  static constexpr int kTopK = 2;
  static constexpr int kMaxLayersForEnum = 4;  // matches default maxNewLayers

  struct ExtendedChain {
    caStructures::hindex_type hitIds[kMaxLayersForEnum];  // outer-most -> inner-most
    uint8_t nHits;
  };

  struct ChainWalkParams {
    float bField;
    float materialDensity;
    float kappaSigmaCut;   // doublet kappa-consistency cut, in sigma units
    float scoreFloor2Phi;  // floor on (sigma_phi)^2
    float scoreFloor2Z;
    uint16_t maxLayersPerTrack;
    uint16_t maxNewLayers;
    uint16_t refitMinNewHits;
    ::pixelTrack::Iteration filterIter;
    // Same semantics as CollectorParams::minQuality -- must match the value
    // passed to kernelCollectITCandidates so the same set of tracks is walked.
    ::pixelTrack::Quality minQuality;
  };

  template <typename TrackerTraits, typename LayoutTraits>
  struct kernelITBuildChain {
    template <alpaka::concepts::Acc TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  TkSoAConstView tracks,
                                  HitsConstView hits,
                                  CollectorBuffers collector,
                                  ChainWalkParams const params,
                                  ExtendedChain* chains,
                                  uint32_t nTracks,
                                  uint32_t* __restrict__ extCounters) const {
      using namespace ::caITExtend;
      using Layout = ::caITExtend::Layout<LayoutTraits>;

      // Bound the iteration by the actual filled-track count (set by
      // Kernel_fillHitDetIndices).  Without this, the chain walker would
      // process up to maxNumberOfTuples (~500 k) ghost slots whose `iteration`
      // got stamped to filterIter by Kernel_assignIteration but whose
      // `quality` is undefined (PortableDeviceCollection does NOT zero-init).
      // Those ghosts then pass the filters and bump the `fallback` counter
      // and write garbage into chains[].
      const auto actualN = static_cast<uint32_t>(std::max(0, tracks.nTracks()));
      const auto iterBound = std::min(nTracks, actualN);
      for (auto trackIdx : cms::alpakatools::uniform_elements(acc, iterBound)) {
        chains[trackIdx].nHits = 0;
        for (int k = 0; k < kMaxLayersForEnum; ++k)
          chains[trackIdx].hitIds[k] = 0;

        if (tracks[trackIdx].iteration() != params.filterIter)
          continue;
        // Must match the threshold used by kernelCollectITCandidates -- see the
        // comment on CollectorParams::minQuality.
        if (tracks[trackIdx].quality() < params.minQuality)
          continue;

        const ::reco::Vector5f state = tracks[trackIdx].state();
        const ::reco::Vector15f cov = tracks[trackIdx].covariance();
        const float kappaTrack = state(2);
        const float sigKappa = std::sqrt(cov(9) + 1.e-12f);

        // ----- Phase 1: per crossed layer, accumulate top-K candidates ------
        struct SlotTopK {
          caStructures::hindex_type id[kTopK];
          float chi2[kTopK];
          float hitPhi[kTopK];
          float hitR[kTopK];
          float predPhi;
          float predR;
          uint8_t n;
          bool isBarrel;
        };
        SlotTopK slots[kMaxLayersForEnum] = {};
        uint8_t nFilled = 0;

        const uint16_t scanLimit = std::min<uint16_t>(params.maxLayersPerTrack, 64);
        for (uint16_t s = 0; s < scanLimit && nFilled < kMaxLayersForEnum; ++s) {
          const auto slotIdx = trackIdx * params.maxLayersPerTrack + s;
          const uint8_t caLayer = collector.slotLayer[slotIdx];
          if (caLayer == 0xff)
            break;
          auto const& slotCands = collector.slots[slotIdx];
          const uint16_t nCand = slotCands.size();
          if (nCand == 0)
            continue;

          const bool isBarrel = caLayer <= 3;  // Phase2 IT BPIX = CA 0..3
          HelixCrossing cross{};
          if (isBarrel) {
            const float targetR = nominalRForBarrelLayer<LayoutTraits>(caLayer);
            cross = helixCrossingToR(state, cov, targetR, params.bField, params.materialDensity);
          } else {
            const float targetZ = nominalZForDiskLayer<LayoutTraits>(caLayer);
            cross = helixCrossingToZ(state, cov, targetZ, params.bField, params.materialDensity);
          }
          if (!cross.valid)
            continue;

          const float sigma2Phi = cross.sigma2Phi + params.scoreFloor2Phi;
          const float sigma2Z = cross.sigma2Z + params.scoreFloor2Z;

          SlotTopK& slot = slots[nFilled];
          slot.predPhi = cross.phi;
          slot.predR = cross.r;
          slot.isBarrel = isBarrel;
          slot.n = 0;
          for (uint16_t c = 0; c < nCand; ++c) {
            const auto hi = slotCands[c];
            const float zHit = hits[hi].zGlobal();
            const float rHit = hits[hi].rGlobal();
            const float phiHit = short2phi(hits[hi].iphi());
            const float dPhi = ::reco::deltaPhi(phiHit, cross.phi);
            const float dSec = isBarrel ? (zHit - cross.z) : (rHit - cross.r);
            const float chi2 = (dPhi * dPhi) / sigma2Phi + (dSec * dSec) / sigma2Z;
            // Insertion into top-K
            if (slot.n < kTopK) {
              slot.id[slot.n] = hi;
              slot.chi2[slot.n] = chi2;
              slot.hitPhi[slot.n] = phiHit;
              slot.hitR[slot.n] = rHit;
              slot.n++;
              // bubble up
              for (int j = int(slot.n) - 1; j > 0; --j) {
                if (slot.chi2[j] < slot.chi2[j - 1]) {
                  std::swap(slot.id[j], slot.id[j - 1]);
                  std::swap(slot.chi2[j], slot.chi2[j - 1]);
                  std::swap(slot.hitPhi[j], slot.hitPhi[j - 1]);
                  std::swap(slot.hitR[j], slot.hitR[j - 1]);
                }
              }
            } else if (chi2 < slot.chi2[kTopK - 1]) {
              slot.id[kTopK - 1] = hi;
              slot.chi2[kTopK - 1] = chi2;
              slot.hitPhi[kTopK - 1] = phiHit;
              slot.hitR[kTopK - 1] = rHit;
              for (int j = kTopK - 1; j > 0; --j) {
                if (slot.chi2[j] < slot.chi2[j - 1]) {
                  std::swap(slot.id[j], slot.id[j - 1]);
                  std::swap(slot.chi2[j], slot.chi2[j - 1]);
                  std::swap(slot.hitPhi[j], slot.hitPhi[j - 1]);
                  std::swap(slot.hitR[j], slot.hitR[j - 1]);
                }
              }
            }
          }
          ++nFilled;
        }

        if (nFilled == 0) {
          chains[trackIdx].nHits = 0;
          if (extCounters)
            alpaka::atomicAdd(acc, &extCounters[4], 1u, alpaka::hierarchy::Blocks{});
          continue;
        }

        // ----- Phase 2: enumerate kTopK^nFilled combinations -----------------
        // bit i of combo selects slot[i].id[(combo>>i)&1].  Since kTopK=2 this
        // is a simple bitmask scan.  Slots with slot.n == 1 only contribute
        // pick=0 (we mark pick=1 invalid).
        const uint32_t totalCombos = 1u << nFilled;
        float bestScore = 1.e30f;
        uint32_t bestCombo = 0;

        for (uint32_t combo = 0; combo < totalCombos; ++combo) {
          float score = 0.f;
          bool valid = true;
          float prevPhi = 0.f, prevR = 0.f;
          bool havePrev = false;
          for (uint8_t l = 0; l < nFilled; ++l) {
            const uint8_t pick = (combo >> l) & 0x1;
            if (pick >= slots[l].n) {
              valid = false;
              break;
            }
            score += slots[l].chi2[pick];
            if (havePrev) {
              const float dPhi = ::reco::deltaPhi(slots[l].hitPhi[pick], prevPhi);
              const float dR = slots[l].hitR[pick] - prevR;
              if (std::abs(dR) > 1.e-3f) {
                const float kappaImplied = 2.f * dPhi / dR;
                const float kResid = kappaImplied - kappaTrack;
                const float sig = sigKappa * params.kappaSigmaCut;
                if (sig > 0.f) {
                  score += (kResid * kResid) / (sig * sig);
                }
              }
            }
            prevPhi = slots[l].hitPhi[pick];
            prevR = slots[l].hitR[pick];
            havePrev = true;
          }
          if (!valid)
            continue;
          if (score < bestScore) {
            bestScore = score;
            bestCombo = combo;
          }
        }

        // ----- Phase 3: write the best chain ----------------------------------
        // Slots are populated outer->inner; refit / merge consumers want
        // inner-most first.  Reverse the order here.
        const uint8_t out = nFilled;
        for (uint8_t l = 0; l < out; ++l) {
          const uint8_t pick = (bestCombo >> l) & 0x1;
          chains[trackIdx].hitIds[out - 1 - l] = slots[l].id[pick];
        }
        chains[trackIdx].nHits = out;

        if (extCounters) {
          // counter[3]   = tracks extended (>=1 IT hit)
          // counter[5+k] = tracks extended with exactly (k+1) IT hits  (k = 0..3)
          // counter[9]   = total IT hits attached (sum)
          alpaka::atomicAdd(acc, &extCounters[3], 1u, alpaka::hierarchy::Blocks{});
          alpaka::atomicAdd(acc, &extCounters[9], uint32_t(out), alpaka::hierarchy::Blocks{});
          if (out >= 1 && out <= 4)
            alpaka::atomicAdd(acc, &extCounters[4 + out], 1u, alpaka::hierarchy::Blocks{});
        }
      }
    }

   private:
    template <typename LT>
    ALPAKA_FN_ACC static float nominalRForBarrelLayer(uint8_t caLayer) {
      using L = ::caITExtend::Layout<LT>;
      auto const* arr = L::barrel();
      for (int i = 0; i < L::kNBarrel; ++i)
        if (arr[i].caLayerIdx == caLayer)
          return arr[i].r;
      return 0.f;
    }
    template <typename LT>
    ALPAKA_FN_ACC static float nominalZForDiskLayer(uint8_t caLayer) {
      using L = ::caITExtend::Layout<LT>;
      auto const* fwd = L::forward();
      for (int i = 0; i < L::kNForward; ++i)
        if (fwd[i].caLayerIdx == caLayer)
          return 0.5f * (fwd[i].zMin + fwd[i].zMax);
      auto const* bwd = L::backward();
      for (int i = 0; i < L::kNBackward; ++i)
        if (bwd[i].caLayerIdx == caLayer)
          return 0.5f * (bwd[i].zMin + bwd[i].zMax);
      return 0.f;
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caITExtend

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAITExtend_CAITExtendDoublets_h
