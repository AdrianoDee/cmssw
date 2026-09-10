#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h

// #define GPU_DEBUG
// #define NTUPLE_DEBUG
// #define CA_DEBUG
// #define CA_WARNINGS
// Per-track printf of the fitted chi2 and its inputs; keep off in timed or high-occupancy runs.
// #define CA_CHI2_DUMP

// C++ includes
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>

// Alpaka includes
#include <alpaka/alpaka.hpp>

// CMSSW includes
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AtomicPairCounter.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "RecoTracker/PixelSeeding/interface/CAPairSoA.h"
// Type defined unconditionally (zero memory); the kernel argument and the writes are #ifdef'd.
#include "RecoTracker/PixelSeeding/interface/TripletDumpSoA.h"
#include "RecoTracker/PixelSeeding/interface/CircleEq.h"
#include "RecoTracker/PixelSeeding/interface/CATrackFeatures.h"
#include "CAFitHitSelection.h"

// local includes
#include "CACell.h"
#include "CAExtensionKernels.h"  // caExtension::isOTId / otIdx (OT-tagged hit branches below)
#include "CAHitNtupletGeneratorKernels.h"
#include "CAStructures.h"
#include "CADnnBank.h"
#include "CATrackDNN.h"
#include "CATripletCuts.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::caHitNtupletGeneratorKernels {

  using namespace ::caStructures;

  constexpr uint32_t tkNotFound = std::numeric_limits<uint32_t>::max();
  constexpr float maxScore = std::numeric_limits<float>::max();
  // Gate width of the two-parameter (1/pT, cot(theta)) compatibility check used by the Phase-1
  // specializations of the duplicate removers.
  constexpr float nSigma2Phase1 = 25.f;
  // The five-parameter gate width is the runtime cfi parameter fastDupNSigma2.
  constexpr int nTrackParameters = 5;
  // Per-track hit capacity of the twin-merge union step; matches the in-fit extension's merged-hit cap.
  constexpr int kTwinMaxMergedHits = 32;
  // pi constants (M_PI is not guaranteed on all alpaka device backends)
  constexpr float kTwinPi = 3.14159265358979323846f;
  constexpr float kTwinTwoPi = 2.f * kTwinPi;
  // map: index of a track parameter -> index of its covariance
  HOST_DEVICE_CONSTANT std::array<uint8_t, nTrackParameters> iParam2iCov = {0u, 5u, 9u, 12u, 14u};

  // Phi bins of the twinFindBest pre-filter: 2pi/128 ~ 0.049 rad >= the 0.03 twin dPhi window, so a
  // centered bin sweep visits a superset of the pairs the |dPhi| gate can accept.
  constexpr int kTwinPhiBins = 128;
  // eta-phi binner of the 0-shared forward fallback; candidate generation only, the cov gate is the physics gate.
  constexpr int kDedupFbPhiBins = 128;
  constexpr int kDedupFbEtaSlabs = 50;
  constexpr float kDedupFbEtaMax = 4.0f;
  // Fallback drop authority is bounded to |eta| <= this: beyond it the covariance is at its widest, so
  // dropping costs efficiency. Candidates beyond the bound are counted in the diagnostics, never dropped.
  constexpr float kDedupFbDropAbsEtaMax = 2.5f;
  // Post-refit cov-dedup nSigma^2 gate; acts on the refitted covariance, unlike the twin gate.
  constexpr float kDedupNSigma2Default = 25.f;
  // |eta| boundary of the central/forward dedup diagnostics; not a physics gate.
  constexpr float kDedupFwdEta = 1.3f;
  // Capacity of the merge-or-keep-both contested-pair list: at most one pair per loser track. Pairs
  // beyond the cap are counted and kept both.
  constexpr uint32_t kDedupConfirmMaxPairs = 1024u;

  // Bin a track by (eta, phi). nEtaSlabs <= 1 gives pure phi binning; otherwise the key is
  // etaSlab * nPhiBins + phiBin. Shared by the fill and the mark/scan kernels.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int trackBinKey(float eta, float phi, int nPhiBins, int nEtaSlabs, float etaMax) {
    float f = (phi + kTwinPi) * (float(nPhiBins) / kTwinTwoPi);
    int pb = int(f);
    if (pb < 0)
      pb = 0;
    if (pb >= nPhiBins)
      pb = nPhiBins - 1;
    if (nEtaSlabs <= 1)
      return pb;
    float g = (eta + etaMax) * (float(nEtaSlabs) / (2.f * etaMax));
    int eb = int(g);
    if (eb < 0)
      eb = 0;
    if (eb >= nEtaSlabs)
      eb = nEtaSlabs - 1;
    return eb * nPhiBins + pb;
  }

  // all of these below are mostly to avoid carrying around the relative namespace

  using Quality = ::pixelTrack::Quality;
  using TkSoAView = ::reco::TrackSoAView;
  using TkHitSoAView = ::reco::TrackHitSoAView;

  template <typename TrackerTraits>
  using QualityCuts = ::pixelTrack::QualityCutsT<TrackerTraits>;

  using Counters = caHitNtupletGenerator::Counters;
  using HitToTuple = caStructures::GenericContainer;
  using HitContainer = caStructures::SequentialContainer;
  using TupleMultiplicity = caStructures::GenericContainer;
  using HitToCell = caStructures::GenericContainer;
  using CellToCell = caStructures::GenericContainer;
  using CellToTrack = caStructures::GenericContainer;

  using namespace cms::alpakatools;

  class SetHitsLayerStart {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const reco::HitModuleSoAConstView &mm,
                                  const reco::CALayersSoAConstView &ll,
                                  uint32_t *__restrict__ hitsLayerStart) const {
      ALPAKA_ASSERT_ACC(0 == mm.moduleStart()[0]);

      for (int32_t i : cms::alpakatools::uniform_elements(acc, ll.metadata().size())) {
        hitsLayerStart[i] = mm.moduleStart()[ll.layerStarts()[i]];
#ifdef GPU_DEBUG
        int old = i == 0 ? 0 : mm.moduleStart()[ll.layerStarts()[i - 1]];
        printf("LayerStart %d/%d at module %d: %d - %d\n",
               i,
               ll.metadata().size() - 1,
               ll.layerStarts()[i],
               hitsLayerStart[i],
               hitsLayerStart[i] - old);
#endif
      }
    }
  };

  class Kernel_printSizes {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsConstView hh,
                                  TkSoAView tt,
                                  uint32_t const *__restrict__ nCells,
                                  uint32_t const *__restrict__ nTrips,
                                  uint32_t const *__restrict__ nCellTracks) const {
      if (cms::alpakatools::once_per_grid(acc))
        printf(
            "nSizes: hh.metadata().size() %d; hh.metadata().size() - hh.offsetBPIX2() %d; nCells %d; nTrips %d; "
            "nCellTracks %d; nTracks %d; tt.metadata().size() %d\n",
            hh.metadata().size(),
            hh.metadata().size() - hh.offsetBPIX2(),
            *nCells,
            *nTrips,
            *nCellTracks,
            tt.nTracks(),
            tt.metadata().size());
    }
  };

  template <typename TrackerTraits>
  class Kernel_checkOverflows {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  TupleMultiplicity const *tupleMultiplicity,
                                  HitToTuple const *hitToTuple,
                                  cms::alpakatools::AtomicPairCounter *apc,
                                  CACell<TrackerTraits> const *__restrict__ cells,
                                  uint32_t const *__restrict__ nCells,
                                  uint32_t const *__restrict__ nTrips,
                                  uint32_t const *__restrict__ nCellTracks,
                                  caStructures::CAPairSoAConstView cellCell,
                                  caStructures::CAPairSoAConstView cellTrack,
                                  int32_t nHits,
                                  uint32_t maxNumberOfDoublets,
                                  AlgoParams const &params,
                                  Counters *counters) const {
      auto &c = *counters;
      // counters once per event
      if (cms::alpakatools::once_per_grid(acc)) {
        alpaka::atomicAdd(acc, &c.nEvents, 1ull, alpaka::hierarchy::Blocks{});
        alpaka::atomicAdd(acc, &c.nHits, static_cast<unsigned long long>(nHits), alpaka::hierarchy::Blocks{});
        alpaka::atomicAdd(acc, &c.nCells, static_cast<unsigned long long>(*nCells), alpaka::hierarchy::Blocks{});
        alpaka::atomicAdd(
            acc, &c.nTuples, static_cast<unsigned long long>(apc->get().first), alpaka::hierarchy::Blocks{});
        alpaka::atomicAdd(acc,
                          &c.nFitTracks,
                          static_cast<unsigned long long>(tupleMultiplicity->size()),
                          alpaka::hierarchy::Blocks{});
      }

#ifdef NTUPLE_DEBUGS
      if (cms::alpakatools::once_per_grid(acc)) {
        printf("number of found cells %d \n found tuples %d with total hits %d out of %d\n",
               *nCells,
               apc->get().first,
               apc->get().second,
               nHits);
        if (apc->get().first < tracks_view.metadata().size()) {
          ALPAKA_ASSERT_ACC(foundNtuplets->size(apc->get().first) == 0);
          ALPAKA_ASSERT_ACC(foundNtuplets->size() == apc->get().second);
        }
      }

      for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(idx) > TrackerTraits::maxHitsOnTrack)  // current real limit
          printf("ERROR %d, %d\n", idx, foundNtuplets->size(idx));
        ALPAKA_ASSERT_ACC(foundNtuplets->size(idx) <= TrackerTraits::maxHitsOnTrack);
        for (auto ih = foundNtuplets->begin(idx); ih != foundNtuplets->end(idx); ++ih)
          ALPAKA_ASSERT_ACC(int(*ih) < nHits);
      }
#endif

      if (cms::alpakatools::once_per_grid(acc)) {
        // Non-corrupting: the build kernels already clamp; these counters surface the magnitude.
        if (apc->get().first >= uint32_t(tracks_view.metadata().size())) {
          printf("Tuples overflow\n");
          alpaka::atomicAdd(acc, &c.nTupleOverflow, 1ull, alpaka::hierarchy::Blocks{});
        }
        if (*nCells >= maxNumberOfDoublets) {
          printf("Cells overflow\n");
          alpaka::atomicAdd(acc, &c.nCellOverflow, 1ull, alpaka::hierarchy::Blocks{});
        }
        if (*nTrips >= uint32_t(cellCell.metadata().size())) {
          printf("Triplets overflow\n");
          alpaka::atomicAdd(acc, &c.nTripletOverflow, 1ull, alpaka::hierarchy::Blocks{});
        }
        if (*nCellTracks >= uint32_t(cellTrack.metadata().size())) {
          printf("TracksToCell overflow\n");
          alpaka::atomicAdd(acc, &c.nCellTrackOverflow, 1ull, alpaka::hierarchy::Blocks{});
        }
      }

      for (auto idx : cms::alpakatools::uniform_elements(acc, *nCells)) {
        auto const &thisCell = cells[idx];
        if (thisCell.hasFishbone() && !thisCell.isKilled())
          alpaka::atomicAdd(acc, &c.nFishCells, 1ull, alpaka::hierarchy::Blocks{});
        if (thisCell.isKilled())
          alpaka::atomicAdd(acc, &c.nKilledCells, 1ull, alpaka::hierarchy::Blocks{});
        if (!thisCell.unused())
          alpaka::atomicAdd(acc, &c.nEmptyCells, 1ull, alpaka::hierarchy::Blocks{});
        if ((0 == hitToTuple->size(thisCell.inner_hit_id())) && (0 == hitToTuple->size(thisCell.outer_hit_id())))
          alpaka::atomicAdd(acc, &c.nZeroTrackCells, 1ull, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // Always-on overflow sentinel: the capacity guards in the build kernels truncate silently, so this kernel
  // tests the same conditions once per event into a per-stream 8-word buffer reported at endStream.
  //   accum[0] = tuple-count overflow          (apc.first   >= tracks capacity)
  //   accum[1] = doublet/cell overflow         (nCells      >= maxNumberOfDoublets)
  //   accum[2] = cellToCell overflow           (nTriplets   >= cellCell capacity)
  //   accum[3] = cellToTrack overflow          (nCellTracks >= cellTrack capacity)
  //   accum[4] = hitContainer content overflow (apc.second  >  content slots)
  //   accum[5] = hitToTuple content overflow   (apc.second  >  its storage extent; UINT32_MAX disables it)
  // The cell counters saturate, so ">= cap" is the only reachable signature; apc is pure demand, hence an
  // upper bound for hitToTuple, whose check the caller disables when that storage is sized from readback.
  class Kernel_overflowSentinel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  cms::alpakatools::AtomicPairCounter const *apc,
                                  uint32_t const *__restrict__ nCells,
                                  uint32_t const *__restrict__ nTrips,
                                  uint32_t const *__restrict__ nCellTracks,
                                  uint32_t tracksCap,
                                  uint32_t maxNumberOfDoublets,
                                  uint32_t cellCellCap,
                                  uint32_t cellTrackCap,
                                  uint32_t hitContentCap,
                                  uint32_t hitToTupleContentCap,
                                  uint32_t *__restrict__ accum) const {
      if (cms::alpakatools::once_per_grid(acc)) {
        if (apc->get().first >= tracksCap)
          alpaka::atomicAdd(acc, &accum[0], 1u, alpaka::hierarchy::Blocks{});
        if (*nCells >= maxNumberOfDoublets)
          alpaka::atomicAdd(acc, &accum[1], 1u, alpaka::hierarchy::Blocks{});
        if (*nTrips >= cellCellCap)
          alpaka::atomicAdd(acc, &accum[2], 1u, alpaka::hierarchy::Blocks{});
        if (*nCellTracks >= cellTrackCap)
          alpaka::atomicAdd(acc, &accum[3], 1u, alpaka::hierarchy::Blocks{});
        if (apc->get().second > hitContentCap)
          alpaka::atomicAdd(acc, &accum[4], 1u, alpaka::hierarchy::Blocks{});
        if (hitToTupleContentCap != 0xFFFFFFFFu && apc->get().second > hitToTupleContentCap)
          alpaka::atomicAdd(acc, &accum[5], 1u, alpaka::hierarchy::Blocks{});
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_fishboneCleaner {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<TrackerTraits> const *cells,
                                  uint32_t const *__restrict__ nCells,
                                  CellToTrack const *__restrict__ cellTracksHisto,
                                  TkSoAView tracks_view) const {
      constexpr auto reject = Quality::dup;

      for (auto idx : cms::alpakatools::uniform_elements(acc, *nCells)) {
        auto const &thisCell = cells[idx];
        if (!thisCell.isKilled())
          continue;

        auto const *__restrict__ tracksOfCell = cellTracksHisto->begin(idx);
        for (auto i = 0u; i < cellTracksHisto->size(idx); i++)
          tracks_view[tracksOfCell[i]].quality() = reject;
      }
    }
  };

  // remove shorter tracks if sharing a cell
  // It does not seem to affect efficiency in any way!
  // Work division: Acc2D with Y indexing cells and X indexing warp lanes
  // (warpSize threads per cell). All lanes of a warp cooperate on a single cell
  template <typename TrackerTraits>
  class Kernel_earlyDuplicateRemover {
  public:
    ALPAKA_FN_ACC void operator()(Acc2D const &acc,
                                  CACell<TrackerTraits> const *cells,
                                  uint32_t const *__restrict__ nCells,
                                  CellToTrack const *__restrict__ cellTracksHisto,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough) const {
      // quality to mark rejected
      constexpr auto reject = Quality::edup;  /// cannot be loose
      ALPAKA_ASSERT_ACC(nCells);

      const int32_t warpSize = alpaka::warp::getSize(acc);
      const int32_t laneId = static_cast<int32_t>(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[1u]);

      for (uint32_t idx : cms::alpakatools::uniform_elements_y(acc, *nCells)) {
#ifdef CA_SIZES
        if (laneId == 0)
          printf("cellTracksSizes;%d;%d;%d\n", idx, cT.size(), cT.capacity());
#endif
        const int ntr = static_cast<int>(cellTracksHisto->size(idx));
        if (ntr < 2)
          continue;

        auto const *__restrict__ tracksOfCell = cellTracksHisto->begin(idx);

        // Warp-reduce maxNl over the cell's tracks.
        // Lanes scan a strided subset of the cell's tracks and hold a local maxNl in register
        int32_t localMax = 0;
        for (int k = laneId; k < ntr; k += warpSize) {
          const int32_t nl = tracks_view[tracksOfCell[k]].nLayers();
          if (nl > localMax)
            localMax = nl;
        }
        // Warp-reduce to find the maxNl across all lanes. The result is uniform across the warp.
        // Idle lanes start with 0 and do not influence the result.
        // All lanes must be active for the shuffle to work: no branching or return early here.
        for (int32_t off = 1; off < warpSize; off <<= 1) {
          const int32_t y = alpaka::warp::shfl_xor(acc, localMax, off);
          if (y > localMax)
            localMax = y;
        }
        const int32_t maxNl = localMax;

        // Process tracks sequentially using warps
        for (int i = 0; i < ntr; ++i) {
          const auto it = tracksOfCell[i];
          const int32_t nli = tracks_view[it].nLayers();
          // Same nli and maxNl across lanes, so uniform check and no early return here to keep all lanes active.
          if (nli >= maxNl) {
            continue;
          }

          // Look for compatible tracks in the same cell with fewer layers and similar curvature
          // tracks_view[].pt() holds the PRE-FIT CURVATURE here, not a pT, and the demotion is terminal
          // (Kernel_fillMultiplicity skips Quality::edup), so the window is topology dependent: Phase2OTStubs
          // uses a window RELATIVE to the compared curvatures, since an absolute |dcurv| window accepts every
          // pair at high pT and would delete the shorter track of every collimated pair; the other topologies
          // keep the absolute window.
          constexpr bool kHasStubs = std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>;
          constexpr float kEarlyDupRelCurv = 0.05f;       // relative window, stub topology
          constexpr float kEarlyDupAbsCurv2 = 0.000001f;  // absolute |dcurv|^2 window, upstream
          // Same value as CACell<TrackerTraits>::kUninitializeCurvature for every topology.
          constexpr float kUninitCurv = std::numeric_limits<float>::max();
          const float curvi = tracks_view[it].pt();
          bool foundCompatible = false;
          // Parallelize inner loop across lanes
          for (int j = laneId; j < ntr; j += warpSize) {
            const auto jt = tracksOfCell[j];
            if (tracks_view[jt].nLayers() <= nli)
              continue;  // need a strictly longer companion
            const float curvj = tracks_view[jt].pt();
            const float dcurv = curvi - curvj;
            if constexpr (kHasStubs) {
              // An uninitialised pre-fit curvature must never be compatible with anything.
              if (curvi == kUninitCurv || curvj == kUninitCurv)
                continue;
              const float thr = kEarlyDupRelCurv * (std::abs(curvi) + std::abs(curvj));
              if (dcurv * dcurv <= thr * thr) {
                foundCompatible = true;
                break;
              }
            } else {
              if (dcurv * dcurv <= kEarlyDupAbsCurv2) {
                foundCompatible = true;
                break;
              }
            }
          }
          // All lanes converge here to check if any foundCompatible is true, and if so, mark track as duplicate.
          if (alpaka::warp::any(acc, static_cast<int32_t>(foundCompatible))) {
            // One thread assigns warp-wide decision
            if (laneId == 0) {
              tracks_view[it].quality() = reject;
            }
          }
        }
      }
    }
  };

  // Specialization for Phase-1 to keep the same behavior as before.
  // remove shorter tracks if sharing a cell
  // It does not seem to affect efficiency in any way!
  class Kernel_earlyDuplicateRemoverPhase1 {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<pixelTopology::Phase1> const *cells,
                                  uint32_t const *__restrict__ nCells,
                                  CellToTrack const *__restrict__ cellTracksHisto,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough) const {
      // quality to mark rejected
      constexpr auto reject = Quality::edup;  /// cannot be loose
      ALPAKA_ASSERT_ACC(nCells);
      for (auto idx : cms::alpakatools::uniform_elements(acc, *nCells)) {
#ifdef CA_SIZES
        printf("cellTracksSizes;%d;%d;%d\n", idx, cT.size(), cT.capacity());
#endif
        if (cellTracksHisto->size(idx) < 2)
          continue;

        int8_t maxNl = 0;
        auto const *__restrict__ tracksOfCell = cellTracksHisto->begin(idx);

        // find maxNl
        for (auto i = 0u; i < cellTracksHisto->size(idx); i++) {
          if (int(tracksOfCell[i]) > tracks_view.metadata().size())
            printf(">WARNING: %d %d %d %d\n", idx, i, int(tracksOfCell[i]), tracks_view.metadata().size());
          auto nl = tracks_view[tracksOfCell[i]].nLayers();
          maxNl = std::max(nl, maxNl);
        }

        // if (maxNl<4) continue;
        // quad pass through (leave it here for tests)
        //  maxNl = std::min(4, maxNl);

        for (auto i = 0u; i < cellTracksHisto->size(idx); i++) {
          auto it = tracksOfCell[i];

          if (int(it) > tracks_view.metadata().size())
            printf(">WARNING: %d %d %d\n", i, it, tracks_view.metadata().size());
          if (tracks_view[it].nLayers() < maxNl)
            tracks_view[it].quality() = reject;  // no race: simple assignment of the same constant
        }
      }
    }
  };

  // Order/backend-independent duplicate removal: a track's final quality must not depend on the order
  // concurrent threads run in. Two disciplines share the int32 quality scratch (device_qualityScratch_)
  // and the two helpers below:
  //   - Kernel_fastDuplicateRemover is cell-parallel (several threads may demote the same shared track):
  //     it reads quality(), accumulates demotions into the scratch via atomicMin, and Kernel_applyQuality
  //     copies the scratch back into quality()
  //   - The hit-based removers (rejectDuplicate, sharedHitCleaner, triplet/simpleTripletCleaner) are
  //     track-parallel single-writers: Kernel_snapshotQuality freezes quality() into the scratch, then
  //     each thread reads that snapshot and writes only its own track's quality() (no atomics, no copy-back)
  class Kernel_snapshotQuality {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  int32_t *__restrict__ qualityScratch) const {
      for (auto i : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes()))
        qualityScratch[i] = static_cast<int32_t>(tracks_view[i].quality());
    }
  };

  class Kernel_applyQuality {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  int32_t const *__restrict__ qualityScratch) const {
      for (auto i : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes()))
        tracks_view[i].quality() = static_cast<Quality>(qualityScratch[i]);
    }
  };

  // Two-tier work division for Kernel_fastDuplicateRemover: the kernel does O(ntr^2) work per cell
  // (ntr = tracks through the cell) and ntr has a long tail, so tier 1 (ntr <= kDupCoopMinTracks) gives
  // one thread to a cell while tier 2 gives the whole warp to one heavy cell at a time, the lanes
  // discovering each other's heavy cells with one ballot per grid step. Both tiers run in the same
  // grid-stride loop.
  //
  // The file is built with -Ofast, so the two tiers stay bit-identical only if they evaluate the same
  // floating-point expressions in the same order and positive form: fastDupRemoverCell re-reads score(it)
  // inside the loop, tests compatibility before ordering, and runs the maxQual / min-chi2 passes
  // redundantly on every lane so that no reduction or shuffle touches a float.
  //
  // The two integer collectives must be convergent: the block size is a multiple of the warp size
  // (enforced in the launcher) and the grid-stride loop runs up to round_up_by(*nCells, warpSize), so all
  // lanes of a warp share their trip count. On the CPU backends the warp size is 1.
  inline constexpr int kDupCoopMinTracks = 64;

  // Per-cell body of Kernel_fastDuplicateRemover.
  //   Coop == false: (iFirst, iStep) = (0, 1)              -> serial loops, one thread per cell
  //   Coop == true : (iFirst, iStep) = (laneId, warpSize)  -> one warp per cell
  // Redistributing the i's cannot change the result: the kernel only reads tracks_view (the values frozen
  // by Kernel_snapshotQuality) and its only writes are atomicMin on qualityScratch, commutative and
  // idempotent; the `break` after a demotion is an early exit only.
  template <bool Coop>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void fastDupRemoverCell(Acc1D const &acc,
                                                         CellToTrack const *__restrict__ cellTracksHisto,
                                                         TkSoAView tracks_view,
                                                         int32_t *__restrict__ qualityScratch,
                                                         uint32_t cellIdx,
                                                         int ntr,
                                                         int lane,
                                                         int stride,
                                                         Quality reject,
                                                         float fastDupNSigma2) {
    constexpr auto loose = Quality::loose;

    auto score = [&](uint32_t it) { return tracks_view[it].chi2(); };
    auto demote = [&](uint32_t it, Quality q) {
      alpaka::atomicMin(acc, &qualityScratch[it], static_cast<int32_t>(q), alpaka::hierarchy::Blocks{});
    };

    auto const *__restrict__ thisCellTracks = cellTracksHisto->begin(cellIdx);

    // The i's this thread owns; compile-time constants for tier 1.
    const int iFirst = Coop ? lane : 0;
    const int iStep = Coop ? stride : 1;

    // Demote any track dominated by a compatible, strictly better one (higher quality, or equal
    // quality and lower chi2); each track tests all others and exact ties keep both
    for (int i = iFirst; i < ntr; i += iStep) {
      auto it = thisCellTracks[i];
      auto qi = tracks_view[it].quality();
      if (qi <= reject)
        continue;

      // get track parameters and covariances
      float iParams[nTrackParameters];
      float iCovs[nTrackParameters];
      for (int p{0}; p < nTrackParameters; ++p) {
        iParams[p] = tracks_view[it].state()(p);
        iCovs[p] = tracks_view[it].covariance()(iParam2iCov[p]);
      }
      // function that compares the five track parameters of tracks it and jt
      auto incompatibleTrackParams = [&](uint32_t jt) -> bool {
        // comparing phi, tip, 1/pT, cotan(theta) and zip
        for (int p{0}; p < nTrackParameters; ++p) {
          const auto dpij = iParams[p] - tracks_view[jt].state()(p);
          const auto e2dpij = fastDupNSigma2 * (iCovs[p] + tracks_view[jt].covariance()(iParam2iCov[p]));
          if (dpij * dpij > e2dpij)
            return true;  // incompatible param found
        }
        return false;  // all params compatible
      };

      for (int j = 0; j < ntr; ++j) {
        if (j == i)
          continue;
        auto jt = thisCellTracks[j];
        auto qj = tracks_view[jt].quality();
        if (qj <= reject)
          continue;
        if (incompatibleTrackParams(jt))
          continue;
        if ((qj > qi) || (qj == qi && score(jt) < score(it))) {
          demote(it, reject);
          break;
        }
      }
    }

    // Run whole by every lane: no reduction, so maxQual comes out of the same code on every lane.
    auto maxQual = reject;  // no duplicate!
    for (int i = 0; i < ntr; i++) {
      auto q = tracks_view[thisCellTracks[i]].quality();
      if (q > maxQual)
        maxQual = q;
    }

    if (maxQual <= loose)
      return;  // warp-uniform when Coop: every lane ran the same loop over the same data

    // min chi2 among the best-quality tracks (read from the unmodified quality, which the dup-marking
    // above does not affect for the max-quality min-chi2 track)
    // run whole by every lane, so mc is bit-for-bit the same on every lane
    float mc = maxScore;
    for (int i = 0; i < ntr; i++) {
      auto it = thisCellTracks[i];
      if (tracks_view[it].quality() == maxQual && score(it) < mc)
        mc = score(it);
    }

    // mark all other duplicates (keep them loose); same test on every lane, only the writes are distributed
    for (int i = iFirst; i < ntr; i += iStep) {
      auto it = thisCellTracks[i];
      if (tracks_view[it].quality() > loose && score(it) > mc)
        demote(it, loose);
    }
  }

  // assume the above (so, short tracks already removed)
  // Work division: one cell per thread, with the whole warp ganging up on the cells whose track list is
  // longer than kDupCoopMinTracks.
  template <typename TrackerTraits>
  class Kernel_fastDuplicateRemover {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<TrackerTraits> const *__restrict__ cells,
                                  uint32_t const *__restrict__ nCells,
                                  CellToTrack const *__restrict__ cellTracksHisto,
                                  TkSoAView tracks_view,
                                  int32_t *__restrict__ qualityScratch,
                                  bool dupPassThrough,
                                  float fastDupNSigma2) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;

      ALPAKA_ASSERT_ACC(nCells);
      const uint32_t ntNCells = (*nCells);

      const int warpSize = static_cast<int>(alpaka::warp::getSize(acc));
      const int laneId = static_cast<int>(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] % uint32_t(warpSize));
      // Invariant (a): the launcher must use a block size that is a multiple of the warp size.
      ALPAKA_ASSERT_ACC((0u == alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u] % uint32_t(warpSize)));
      // Invariant (b): lane-aligned extent, so a warp's lanes share their trip count.
      const uint32_t extent = cms::alpakatools::round_up_by(ntNCells, uint32_t(warpSize));

      for (auto idx : cms::alpakatools::uniform_elements(acc, extent)) {
        const bool inRange = (idx < ntNCells);
        const int ntr = inRange ? static_cast<int>(cellTracksHisto->size(idx)) : 0;

        // tier 2: hand the heavy cells of this warp to the whole warp, one at a time. The mask is
        // warp-uniform, so this loop and the collectives inside it are convergent.
        auto heavyMask = alpaka::warp::ballot(acc, (ntr > kDupCoopMinTracks) ? 1 : 0);
        using MaskT = decltype(heavyMask);
        if (heavyMask) {
          for (int l = 0; l < warpSize; ++l) {
            if (MaskT{0} == ((heavyMask >> l) & MaskT{1}))
              continue;
            const uint32_t cell = static_cast<uint32_t>(alpaka::warp::shfl(acc, static_cast<int32_t>(idx), l));
            const int n = alpaka::warp::shfl(acc, ntr, l);
            fastDupRemoverCell<true>(
                acc, cellTracksHisto, tracks_view, qualityScratch, cell, n, laneId, warpSize, reject, fastDupNSigma2);
          }
        }

        // tier 1: one cell per thread
        if (inRange && ntr >= 2 && ntr <= kDupCoopMinTracks)
          fastDupRemoverCell<false>(
              acc, cellTracksHisto, tracks_view, qualityScratch, idx, ntr, 0, 1, reject, fastDupNSigma2);
      }
    }
  };

  // Phase-1 specialization
  template <>
  class Kernel_fastDuplicateRemover<pixelTopology::Phase1> {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<pixelTopology::Phase1> const *__restrict__ cells,
                                  uint32_t const *__restrict__ nCells,
                                  CellToTrack const *__restrict__ cellTracksHisto,
                                  TkSoAView tracks_view,
                                  int32_t *__restrict__ qualityScratch,
                                  bool dupPassThrough,
                                  float fastDupNSigma2) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;
      constexpr auto loose = Quality::loose;

      ALPAKA_ASSERT_ACC(nCells);
      const auto ntNCells = (*nCells);

      auto score = [&](uint32_t it) { return std::abs(reco::tip(tracks_view, it)); };
      auto demote = [&](uint32_t it, Quality q) {
        alpaka::atomicMin(acc, &qualityScratch[it], static_cast<int32_t>(q), alpaka::hierarchy::Blocks{});
      };

      for (auto idx : cms::alpakatools::uniform_elements(acc, ntNCells)) {
        int ntr = cellTracksHisto->size(idx);
        if (ntr < 2)
          continue;

        auto const *__restrict__ thisCellTracks = cellTracksHisto->begin(idx);

        // Mark as duplicate any track dominated by a compatible, strictly better one
        // (order-independent; lower track index breaks exact ties)
        for (int i = 0; i < ntr; ++i) {
          auto it = thisCellTracks[i];
          auto qi = tracks_view[it].quality();
          if (qi <= reject)
            continue;
          auto opi = tracks_view[it].state()(2);
          auto e2opi = tracks_view[it].covariance()(9);
          auto cti = tracks_view[it].state()(3);
          auto e2cti = tracks_view[it].covariance()(12);
          for (int j = 0; j < ntr; ++j) {
            if (j == i)
              continue;
            auto jt = thisCellTracks[j];
            auto qj = tracks_view[jt].quality();
            if (qj <= reject)
              continue;
            auto opj = tracks_view[jt].state()(2);
            auto ctj = tracks_view[jt].state()(3);
            auto dct = nSigma2Phase1 * (tracks_view[jt].covariance()(12) + e2cti);
            if ((cti - ctj) * (cti - ctj) > dct)
              continue;
            auto dop = nSigma2Phase1 * (tracks_view[jt].covariance()(9) + e2opi);
            if ((opi - opj) * (opi - opj) > dop)
              continue;
            if ((qj > qi) || (qj == qi && (score(jt) < score(it) || (score(jt) == score(it) && jt < it)))) {
              demote(it, reject);
              break;
            }
          }
        }

        // find maxQual
        auto maxQual = reject;  // no duplicate!
        for (int i = 0; i < ntr; i++) {
          auto q = tracks_view[thisCellTracks[i]].quality();
          if (q > maxQual)
            maxQual = q;
        }

        if (maxQual <= loose)
          continue;

        // keep the single best-quality, min-score track (lower index breaks ties); demote the rest
        float mc = maxScore;
        uint32_t im = tkNotFound;
        for (int i = 0; i < ntr; i++) {
          auto it = thisCellTracks[i];
          if (tracks_view[it].quality() == maxQual) {
            auto s = score(it);
            if (s < mc || (s == mc && it < im)) {
              mc = s;
              im = it;
            }
          }
        }

        if (tkNotFound == im)
          continue;

        // mark all other duplicates (keep them loose)
        for (int i = 0; i < ntr; i++) {
          auto it = thisCellTracks[i];
          if (tracks_view[it].quality() > loose && it != im)
            demote(it, loose);
        }
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_connect {
  public:
    ALPAKA_FN_ACC void operator()(Acc2D const &acc,
                                  cms::alpakatools::AtomicPairCounter *apc,  // just to zero them
                                  HitsConstView hh,
                                  reco::CAGraphSoAConstView cc,
                                  reco::CATripletCutsSoAConstView tripletCuts,
                                  bool useTripletDNN,
                                  float tripletDNNThreshold,
                                  DnnBank tripletBank,  // per-iteration compile-time weight bank
#ifdef CA_TRIPLET_DUMP
                                  caStructures::TripletDumpSoAView tripletDump,  // per-triplet feature capture
                                  int dumpIteration,  // dump builds only; see CATripletCuts.h
#endif
                                  caStructures::CAPairSoAView cn,
                                  CACell<TrackerTraits> *cells,
                                  uint32_t const *nCells,
                                  uint32_t *nTrips,
                                  HitToCell const *__restrict__ outerHitHisto,
                                  CellToCell *cellNeighborsHisto,
                                  uint32_t *__restrict__ pipelineCounters) const {
      using Cell = CACell<TrackerTraits>;
      uint32_t maxTriplets = cn.metadata().size();

      if (cms::alpakatools::once_per_grid(acc)) {
        *apc = 0;
      }  // ready for next kernel

      // loop on outer cells
      for (uint32_t oCellIndex : cms::alpakatools::uniform_elements_y(acc, *nCells)) {
        auto &outerCell = cells[oCellIndex];
        auto middleHitId = outerCell.inner_hit_id() - hh.offsetBPIX2();

        if (int(middleHitId) < 0)
          continue;

        auto const *__restrict__ outerHitCells = outerHitHisto->begin(middleHitId);
        auto const numberOfPossibleNeighbors = outerHitHisto->size(middleHitId);

        // Per-layer-pair triplet cut rows: the RZ-alignment tolerance and the stub columns are anchored at
        // the outer pair (L2,L3), whose inner layer is the triplet's middle layer, while the beam-spot
        // (DCA, floorDCA) cut is anchored at the inner pair (L1,L2). See TripletCuts::accept.
        auto tripletVectorCutsCol = tripletCuts[outerCell.layerPairId()];
        auto skips = cc[outerCell.layerPairId()].skipsLayers();

#ifdef CA_DEBUG
        printf("numberOfPossibleFromHisto;%d;%d;%d;%d;%d\n",
               *nCells,
               middleHitId,
               oCellIndex,
               outerCell.innerLayer(cc),
               numberOfPossibleNeighbors);
#endif

        // loop on inner cells
        for (uint32_t j : cms::alpakatools::independent_group_elements_x(acc, numberOfPossibleNeighbors)) {
          auto iCellIndex = outerHitCells[j];
          auto &innerCell = cells[iCellIndex];
          float curvature = 0.f;

          // cc (the CA layer-pair graph) supplies the per-hit CA layer ids for the DNN layer-gap features
#ifdef CA_TRIPLET_DUMP
          float dumpFeat[18] =
              {};  // accept() fills 18 BASE DNN features; written to SoA below (zero-init defense-in-depth)
          float dumpScore = -1.f;  // accept() fills the in-kernel DNN score (consistency check)
#endif
          if (TripletCuts<TrackerTraits>::accept(acc,
                                                 innerCell,
                                                 outerCell,
                                                 curvature,
                                                 hh,
                                                 tripletCuts,
                                                 tripletVectorCutsCol,
                                                 tripletCuts[innerCell.layerPairId()],
                                                 cc,
                                                 useTripletDNN,
                                                 tripletDNNThreshold,
                                                 tripletBank,
#ifdef CA_TRIPLET_DUMP
                                                 dumpFeat,
                                                 &dumpScore,
                                                 dumpIteration,
#endif
                                                 pipelineCounters)) {
            auto t_ind = alpaka::atomicAdd(acc, nTrips, 1u, alpaka::hierarchy::Blocks{});

#ifdef CA_DEBUG
            printf("Triplet no. %d %.5f %.5f (%d %d) - %d %d -> (%d, %d, %d, %d) \n",
                   t_ind,
                   thetaCut,
                   dcaCut,
                   outerCell.layerPairId(),
                   innerCell.layerPairId(),
                   iCellIndex,
                   oCellIndex,
                   outerCell.inner_hit_id(),
                   outerCell.outer_hit_id(),
                   innerCell.inner_hit_id(),
                   innerCell.outer_hit_id());
            printf("filling cell no. %d %d: %d -> %d\n", t_ind, cellNeighborsHisto->size(), iCellIndex, oCellIndex);
#endif

            if (t_ind >= maxTriplets) {
#ifdef CA_WARNINGS
              printf("Warning!!!! Too many cell->cell (triplets) associations (limit = %d)!\n", cn.metadata().size());
#endif
              alpaka::atomicSub(acc, nTrips, 1u, alpaka::hierarchy::Blocks{});
              break;
            }

#ifdef CA_TRIPLET_DUMP
            // Per-built-triplet training row: 18 base features + the three merged-hit indices (truth join
            // key) + CA layers. t_ind < maxTriplets is guaranteed by the guard above.
            {
              auto row = tripletDump[t_ind];
              row.absCurvature() = dumpFeat[0];
              row.tipTimesCurvature() = dumpFeat[1];
              row.dca() = dumpFeat[2];
              row.curvatureStubs() = dumpFeat[3];
              row.curvatureStubsErrSquared() = dumpFeat[4];
              row.curvature13() = dumpFeat[5];
              row.dPhi12() = dumpFeat[6];
              row.dPhi13() = dumpFeat[7];
              row.dPhi23() = dumpFeat[8];
              row.dr12() = dumpFeat[9];
              row.dr13() = dumpFeat[10];
              row.r1() = dumpFeat[11];
              row.r2() = dumpFeat[12];
              row.r3() = dumpFeat[13];
              row.z1() = dumpFeat[14];
              row.z2() = dumpFeat[15];
              row.z3() = dumpFeat[16];
              row.nStubs() = dumpFeat[17];
              row.curvature() = curvature;  // SIGNED (Kernel_connect local, by-ref from accept); for derived feats
              row.lay1() = int32_t(innerCell.innerLayer(cc));
              row.lay2() = int32_t(outerCell.innerLayer(cc));
              row.lay3() = int32_t(outerCell.outerLayer(cc));
              row.h1() = uint32_t(innerCell.inner_hit_id());
              row.h2() = uint32_t(outerCell.inner_hit_id());
              row.h3() = uint32_t(outerCell.outer_hit_id());
              row.iter() = dumpIteration;
              row.inKernelScore() = dumpScore;
            }
#endif

            // One bin per cell (bin = iCellIndex). The non-layer-skipping vs
            // layer-skipping distinction is encoded in bit 31 of the stored
            // outer-cell index:
            //   bit 31 = 0 -> non-layer-skipping neighbor
            //   bit 31 = 1 -> layer-skipping neighbor
            // Key-range guard: iCellIndex is below the cell count the histogram was sized from, so a sizing
            // mismatch drops the association instead of writing outside off[].
            if (iCellIndex < cellNeighborsHisto->nOnes())
              cellNeighborsHisto->count(acc, iCellIndex);

            cn[t_ind].inner() = iCellIndex;
            cn[t_ind].outer() = oCellIndex | (skips ? caStructures::kSkipsLayerFlag : 0u);
            outerCell.setStatusBits(Cell::StatusBit::kUsed);
            innerCell.setStatusBits(Cell::StatusBit::kUsed);

            // Pipeline stage counters: classify triplet by hit types
            if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
              if (pipelineCounters) {
                using PC = caHitNtupletGenerator::PipelineCounter;
                alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsTotal], 1u, alpaka::hierarchy::Blocks{});
                auto hit1 = innerCell.inner_hit_id();
                auto hit2 = outerCell.inner_hit_id();
                auto hit3 = outerCell.outer_hit_id();
                int nStubs = (isStub(hh, hit1) ? 1 : 0) + (isStub(hh, hit2) ? 1 : 0) + (isStub(hh, hit3) ? 1 : 0);
                if (nStubs == 0)
                  alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsPixPixPix], 1u, alpaka::hierarchy::Blocks{});
                else if (nStubs == 1)
                  alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsPixPixOT], 1u, alpaka::hierarchy::Blocks{});
                else if (nStubs == 2)
                  alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsPixOTOT], 1u, alpaka::hierarchy::Blocks{});
                else {
                  alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsOTOTOT], 1u, alpaka::hierarchy::Blocks{});
                  auto layer1 = innerCell.innerLayer(cc);  // innermost
                  auto layer2 = outerCell.innerLayer(cc);  // middle
                  auto layer3 = outerCell.outerLayer(cc);  // outermost
                  bool l1Brl = (layer1 >= 28 && layer1 <= 33);
                  bool l2Brl = (layer2 >= 28 && layer2 <= 33);
                  bool l3Brl = (layer3 >= 28 && layer3 <= 33);
                  bool l1Fwd = (layer1 >= 34 && layer1 <= 43);  // disks at z > 0
                  bool l2Fwd = (layer2 >= 34 && layer2 <= 43);
                  bool l3Fwd = (layer3 >= 34 && layer3 <= 43);
                  bool l1Bwd = (layer1 >= 44 && layer1 <= 53);  // disks at z < 0
                  bool l2Bwd = (layer2 >= 44 && layer2 <= 53);
                  bool l3Bwd = (layer3 >= 44 && layer3 <= 53);
                  if (l1Brl && l2Brl && l3Brl)
                    alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsOOO_barrel], 1u, alpaka::hierarchy::Blocks{});
                  else if (l1Bwd && l2Bwd && l3Bwd)
                    alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsOOO_bwd], 1u, alpaka::hierarchy::Blocks{});
                  else if (l1Fwd && l2Fwd && l3Fwd)
                    alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsOOO_fwd], 1u, alpaka::hierarchy::Blocks{});
                  else if ((l1Brl || l2Brl) && (l2Bwd || l3Bwd))
                    alpaka::atomicAdd(
                        acc, &pipelineCounters[PC::kTripletsOOO_brlToBwd], 1u, alpaka::hierarchy::Blocks{});
                  else if ((l1Brl || l2Brl) && (l2Fwd || l3Fwd))
                    alpaka::atomicAdd(
                        acc, &pipelineCounters[PC::kTripletsOOO_brlToFwd], 1u, alpaka::hierarchy::Blocks{});
                  else
                    alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsOOO_other], 1u, alpaka::hierarchy::Blocks{});
                }
              }
            }
          }
        }  // loop on inner cells
      }  // loop on outer cells
    }
  };

  template <typename TrackerTraits>
  class FillDoubletsHisto {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<TrackerTraits> const *__restrict__ cells,
                                  uint32_t *nCells,
                                  uint32_t offsetBPIX2,
                                  HitToCell *outerHitHisto,
                                  Counters *counters) const {
      const auto nKeys = outerHitHisto->nOnes();
      for (auto cellIndex : cms::alpakatools::uniform_elements(acc, *nCells)) {
#ifdef DOUBLETS_DEBUG
        printf("outerHitHisto;%d;%d\n", cellIndex, cells[cellIndex].outer_hit_id());
#endif
        auto const key = cells[cellIndex].outer_hit_id() - offsetBPIX2;
        // Key-range guard: a key past nOnes means the hit->cell offsets were sized for a smaller hit count
        // than the cells reference, so drop the association instead of writing outside off[]. The matching
        // count pass (CAPixelDoubletsAlgos.h) skips exactly the same keys.
        if (key < nKeys)
          outerHitHisto->fill(acc, key, cellIndex);
        else
          alpaka::atomicAdd(acc, &counters->nHitToCellOverflow, 1ull, alpaka::hierarchy::Blocks{});
      }
    }
  };

  template <typename CAPairView, typename Container>
  class Kernel_fillGenericPair {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CAPairView cn,
                                  uint32_t const *nElements,
                                  Container *genericHisto) const {
      const auto nKeys = genericHisto->nOnes();
      for (uint32_t index : cms::alpakatools::uniform_elements(acc, *nElements)) {
        auto const key = cn[index].inner();
        // Key-range guard mirroring the count pass: the key is a cell index below the cell count the
        // histogram was sized from, so only a sizing mismatch can drop an entry here.
        if (key < nKeys)
          genericHisto->fill(acc, key, cn[index].outer());
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_find_ntuplets {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsConstView hh,
                                  const ::reco::CAGraphSoAConstView &cc,
                                  const ::reco::CANtupletCutsSoAConstView &ntupletCuts,
                                  TkSoAView tracks_view,
                                  HitContainer *foundNtuplets,
                                  CellToCell const *__restrict__ cellNeighborsHisto,
                                  CellToTrack *cellTracksHisto,
                                  caStructures::CAPairSoAView ct,
                                  CACell<TrackerTraits> *__restrict__ cells,
                                  uint32_t *nCellTracks,
                                  uint32_t const *nTriplets,
                                  uint32_t const *nCells,
                                  cms::alpakatools::AtomicPairCounter *apc,
                                  AlgoParams const &params) const {
      using Cell = CACell<TrackerTraits>;

#ifdef GPU_DEBUG
      if (cms::alpakatools::once_per_grid(acc))
        printf("starting producing ntuplets from %d cells and %d triplets \n", *nCells, *nTriplets);
#endif

      for (auto idx : cms::alpakatools::uniform_elements(acc, (*nCells))) {
        auto const &thisCell = cells[idx];

        // cut by earlyFishbone
        if (thisCell.isKilled())
          continue;

        // we require at least three hits
        if (cellNeighborsHisto->size(idx) == 0)
          continue;

        // check if the layer pair of the cell is among the set of starting pairs
        auto pid = thisCell.layerPairId();
        bool doit = cc[pid].startingPair();

        // check if the most inner hit does not fulfill the starting requirement
        auto lid = thisCell.innerLayer(cc);
        if (thisCell.inner_r(hh) > ntupletCuts[lid].startMaxInnerR())
          doit = false;

        constexpr uint32_t maxDepth = TrackerTraits::maxLayersPerTrack - 1;
#ifdef CA_DEBUG
        printf(
            "LayerPairId %d and inner layer %d doit ? %d From cell %d with nNeighbors = %d and innerR=%f < "
            "maxInnerR=%f ?\n",
            pid,
            lid,
            doit,
            idx,
            cellNeighborsHisto->size(idx),
            thisCell.inner_r(hh),
            ntupletCuts[lid].startMaxInnerR());
#endif

        if (doit) {
          typename Cell::TmpTuple stack;
          // Per-thread buffer that find_ntuplets fills when it saves an ntuplet. Declared here, not inside the
          // recursive (inlined) find_ntuplets, so the stack holds one copy per thread and not one per depth.
          typename Cell::hindex_type hits[TrackerTraits::maxHitsOnTrack];

          stack.reset();
          thisCell.template find_ntuplets<maxDepth>(acc,
                                                    hh,
                                                    ntupletCuts,
                                                    cc,
                                                    cells,
                                                    *foundNtuplets,
                                                    cellNeighborsHisto,
                                                    cellTracksHisto,
                                                    nCellTracks,
                                                    ct,
                                                    *apc,
                                                    tracks_view.quality().data(),
                                                    tracks_view.nLayers().data(),
                                                    tracks_view.pt().data(),
                                                    stack,
                                                    hits,
                                                    params.minHitsPerNtuplet_);
          ALPAKA_ASSERT_ACC(stack.empty());
        }
      }
    }
  };
#ifdef CA_PIPELINE_COUNTERS
  // Pipeline counter: classify n-tuplets by OT hit content
  template <typename TrackerTraits>
  class Kernel_pipelineNtupletCount {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsConstView hh,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  cms::alpakatools::AtomicPairCounter const *apc,
                                  uint32_t maxTuples,
                                  uint32_t *__restrict__ pipelineCounters) const {
      if (!pipelineCounters)
        return;
      using PC = caHitNtupletGenerator::PipelineCounter;
      // Clamp to container capacity -- apc may exceed maxTuples on overflow
      auto ntracks = std::min<uint32_t>(apc->get().first, maxTuples);
      for (auto idx : cms::alpakatools::uniform_elements(acc, ntracks)) {
        auto nh = foundNtuplets->size(idx);
        if (nh < 3)
          continue;
        alpaka::atomicAdd(acc, &pipelineCounters[PC::kNtupletsTotal], 1u, alpaka::hierarchy::Blocks{});
        if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
          auto nHits = hh.metadata().size();
          int nOT = 0;
          for (auto h = foundNtuplets->begin(idx); h != foundNtuplets->end(idx); ++h) {
            if (*h >= static_cast<unsigned int>(nHits))
              break;  // content buffer corruption from overflow
            if (isStub(hh, *h))
              ++nOT;
          }
          if (nOT >= 1)
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kNtupletsWithOT], 1u, alpaka::hierarchy::Blocks{});
          if (nOT >= 3)
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kNtupletsOT3Plus], 1u, alpaka::hierarchy::Blocks{});
        }
      }
    }
  };

  // Count cell status after all kill phases (reachability + fishbone)
  template <typename TrackerTraits>
  class Kernel_pipelineCellStatus {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<TrackerTraits> const *__restrict__ cells,
                                  uint32_t const *nCells,
                                  uint32_t *__restrict__ pipelineCounters) const {
      if (!pipelineCounters)
        return;
      using PC = ::caHitNtupletGenerator::PipelineCounter;
      for (auto idx : cms::alpakatools::uniform_elements(acc, *nCells)) {
        auto const &cell = cells[idx];
        if (!cell.unused())  // kUsed bit is set
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kCellsUsedInTriplet], 1u, alpaka::hierarchy::Blocks{});
        if (cell.isKilled())
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kCellsKilledTotal], 1u, alpaka::hierarchy::Blocks{});
        else
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kCellsAlive], 1u, alpaka::hierarchy::Blocks{});
      }
    }
  };

  class Kernel_pipelineCopyCellTrackCount {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  uint32_t const *nCellTracks,
                                  uint32_t *__restrict__ pipelineCounters) const {
      if (!pipelineCounters)
        return;
      if (cms::alpakatools::once_per_grid(acc))
        pipelineCounters[::caHitNtupletGenerator::kCellTrackPairs] = *nCellTracks;
    }
  };
#endif  // CA_PIPELINE_COUNTERS

  template <typename TrackerTraits>
  class Kernel_mark_used {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<TrackerTraits> *__restrict__ cells,
                                  CellToTrack const *__restrict__ cellTracksHisto,
                                  uint32_t const *nCells) const {
      using Cell = CACell<TrackerTraits>;
      for (auto idx : cms::alpakatools::uniform_elements(acc, (*nCells))) {
        auto &thisCell = cells[idx];
        if (cellTracksHisto->size(idx) > 0)
          thisCell.setStatusBits(Cell::StatusBit::kInTrack);
      }
    }
  };

  // Count the hits the fit will actually use, given the FitHitSelection mode (== nhits in the default All
  // mode); kept consistent with the fit's own selection in BrokenLineFit.dev.cc.
  template <typename TrackerTraits>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t nSelectedHits(HitContainer const *__restrict__ foundNtuplets,
                                                        uint32_t it,
                                                        HitsConstView hh) {
    // hasStubs enables the OT-stub hit treatment (kMode filtering and the same-layer pixel overlap merge).
    // It must match the fit, which keys off the runtime offsetStubs: a Phase2OTStubs collection carrying no
    // stubs sets offsetStubs to the unsigned sentinel and every hit is then a plain pixel hit.
    const bool hasStubs =
        std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits> && (static_cast<int32_t>(hh.offsetStubs()) >= 0);
    return caFitHitSel::dedupWalk(foundNtuplets, it, hh, hasStubs, /*k=*/-1);
  }

  template <typename TrackerTraits>
  class Kernel_countMultiplicity {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsConstView hh,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  TupleMultiplicity *tupleMultiplicity) const {
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        auto nhits = foundNtuplets->size(it);
        // printf("it: %d nhits: %d \n",it,nhits);
        if (nhits < 3)
          continue;
        if (tracks_view[it].quality() == Quality::edup)
          continue;
        // On hitContainer overflow bulkFill returns kOverflow and the quality stamp below is skipped,
        // so the slot keeps its pre-init value (0, or garbage from a caching allocator). Skip such slots:
        // the tuple was already dropped by the lossy truncation.
        if (tracks_view[it].quality() != Quality::bad)
          continue;
        // On content-buffer overflow the offset is plugged (the size is correct) but the content is
        // unwritten, so nhits can read garbage; more than maxHitsOnTrack hits is an overflow artifact.
        if (nhits > TrackerTraits::maxHitsOnTrack)
          continue;
        auto const nsel = nSelectedHits<TrackerTraits>(foundNtuplets, it, hh);
        if (nsel < 3)
          continue;  // too few selected hits to fit
        tupleMultiplicity->count(acc, nsel);
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_fillMultiplicity {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsConstView hh,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  TupleMultiplicity *tupleMultiplicity) const {
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        auto nhits = foundNtuplets->size(it);

        if (nhits < 3)
          continue;
        if (tracks_view[it].quality() == Quality::edup)
          continue;
        // Skip overflow tuples (see Kernel_countMultiplicity for rationale).
        if (tracks_view[it].quality() != Quality::bad)
          continue;
        if (nhits > TrackerTraits::maxHitsOnTrack)
          continue;
        auto const nsel = nSelectedHits<TrackerTraits>(foundNtuplets, it, hh);
        if (nsel < 3)
          continue;
        tupleMultiplicity->fill(acc, nsel, it);
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_classifyTracks {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  HitsConstView hh,
                                  QualityCuts<TrackerTraits> cuts,
                                  bool useTrackDNN,
                                  float trackDNNThreshold,
                                  DnnBank trackBank,  // per-iteration compile-time weight bank
                                  // Raw OT-rechit view for the feature walk. nOTHits == 0 means merged
                                  // hits only, and the view is then unused.
                                  ::reco::OTRecHitsConstView otHits,
                                  uint32_t nOTHits) const {
#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
      uint32_t nTracks = 0;
      uint32_t nFitted = 0;
      uint32_t nNaN = 0;
      uint32_t nDoublets = 0;
      uint32_t nDuplicates = 0;
#endif

      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        auto nhits = foundNtuplets->size(it);
        if (nhits == 0)
          break;  // guard

#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
        nTracks++;
#endif

        // if duplicate: not even fit
        if (tracks_view[it].quality() == Quality::edup) {
#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
          nDuplicates++;
#endif
          continue;
        }

        // Skip overflow tuples (see Kernel_countMultiplicity for rationale).
        if (tracks_view[it].quality() != Quality::bad)
          continue;

        // mark doublets as bad
        if (nhits < 3) {
#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
          nDoublets++;
#endif
          continue;
        }

        // if the fit has any invalid parameters, mark it as bad
        bool isNaN = false;
        for (int i = 0; i < 5; ++i) {
          isNaN |= edm::isNotFinite(tracks_view[it].state()(i));
        }
        // A non-finite chi2 is a failed fit and must never be promoted. The test is explicit because the
        // promotion gates downstream are written in the rejecting sense (strictCut returns chi2 >= maxChi2,
        // the stub-curvature walk tests chi2Stub > cut) and a comparison with a NaN operand is false.
        // edm::isNotFinite is a bit-pattern test, so it keeps working under -ffinite-math-only.
        isNaN |= edm::isNotFinite(tracks_view[it].chi2());
        // state(2) is the (finite) inverse pt: an exactly-zero value from a straight-line or
        // numerically-degenerate fit maps to an infinite momentum in the host local-to-global
        // transform, so treat it as bad here too and never promote such a track
        isNaN |= (tracks_view[it].state()(2) == 0.f);
        if (isNaN) {
#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
          nNaN++;
          printf("FIT_DEBUG: Track %d has NaN - nhits=%d chi2=%f pt=%f eta=%f\n",
                 it,
                 nhits,
                 tracks_view[it].chi2(),
                 tracks_view[it].pt(),
                 tracks_view[it].eta());
#endif
          continue;
        }

#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
        nFitted++;
        if (nFitted <= 10) {
          printf("FIT_DEBUG: Track %d FITTED - nhits=%d pt=%.3f eta=%.3f phi=%.3f chi2=%.3f tip=%.4f zip=%.4f\n",
                 it,
                 nhits,
                 tracks_view[it].pt(),
                 tracks_view[it].eta(),
                 tracks_view[it].state()(0),  // phi is state[0]
                 tracks_view[it].chi2(),
                 tracks_view[it].state()(1),   // tip is state[1]
                 tracks_view[it].state()(4));  // zip is state[4]
        }
#endif

        tracks_view[it].quality() = Quality::strict;

        bool failChi2 = cuts.strictCut(tracks_view, nhits, it);
        if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
          const auto nHitsTot = hh.metadata().size();

          // When enabled, the MLP score replaces the chi2-based strict->tight decision (both the strictCut
          // fit-chi2 gate and the stub-consistency demotion below), while the fit chi2 and chi2Stub stay
          // inputs of the network (feat[0], feat[8]). Feature order is documented in CATrackDNNWeights.h.
          bool dnnHandled = false;
          if (useTrackDNN) {
            // Single-source feature fill (RecoTracker/PixelSeeding/interface/CATrackFeatures.h), producing
            // values identical to the host-side nano table producer's. fill() returns false on a corrupt or
            // short hit list -> fall through to the chi2-based path.
            float feat[caTrackFeatures::kNFeat];
            static_assert(caTrackFeatures::kNFeat == caTrackDNN::kNFeat, "feature ABI mismatch");
            // Tagged extras resolve their global position through the OT view (nullptr when no tagged ids
            // can be present).
            const ::reco::OTRecHitsConstView *otViewPtr = (nOTHits > 0u) ? &otHits : nullptr;
            const bool featOk = caTrackFeatures::fill(foundNtuplets->begin(it),
                                                      foundNtuplets->end(it),
                                                      hh,
                                                      nHitsTot,
                                                      float(tracks_view[it].nLayers()),
                                                      tracks_view[it].chi2(),
                                                      feat,
                                                      /*rzKappaOut=*/nullptr,
                                                      otViewPtr);
            // The finiteness of the network inputs is established before the network is evaluated: a
            // non-finite input propagates through the MLP and the resulting score, compared the wrong way
            // round, would promote the track. A track with any non-finite feature stays Quality::bad
            // (quality() was optimistically set to strict above, so it is written back explicitly).
            bool featFinite = featOk;
            for (int k = 0; featFinite && k < int(caTrackFeatures::kNFeat); ++k)
              featFinite = !edm::isNotFinite(feat[k]);
            if (featOk && !featFinite) {
#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
              nNaN++;
#endif
              tracks_view[it].quality() = Quality::bad;
              continue;
            }
            if (featOk) {
              // Stage-1 high-recall loose->tight selector: a single threshold. Dedicated displaced fake
              // rejection is the post-reco displaced high-purity selector.
              const float defThr = (trackBank == DnnBank::kPrompt) ? caTrackDNN_prompt::kDefaultThreshold
                                                                   : caTrackDNN_displaced::kDefaultThreshold;
              const float dnnThr = (trackDNNThreshold < 0.f) ? defThr : trackDNNThreshold;
              const float dnnScore = (trackBank == DnnBank::kPrompt)
                                         ? caTrackDNN_eval::score<DnnBank::kPrompt>(feat)
                                         : caTrackDNN_eval::score<DnnBank::kDisplaced>(feat);
              // Promoting form on purpose: `score >= threshold` is the decision to promote and the rejection
              // is its negation. Under -ffinite-math-only a rejecting predicate may be rewritten into its
              // finite-arithmetic complement, which would let a NaN score take the promoting branch; in this
              // form anything the comparison cannot decide stays rejected.
              const bool dnnPromote = (dnnScore >= dnnThr);
              failChi2 = !dnnPromote;
              dnnHandled = true;
            }
          }

          // Ntuplet-wide stub-curvature consistency: inverse-variance-weighted reduced chi2 of all per-stub
          // curvatures on the track around their common weighted mean (same kappa as CATripletCuts). A
          // combinatorial chain admitted by the relaxed displaced DCA fails it and is demoted below `tight`.
          // Skipped when the DNN already decided, since its score subsumes it.
#ifdef CA_CHI2_DUMP
          const bool computeStubChi2 = true;
#else
          const bool computeStubChi2 = !dnnHandled;
#endif
          if (computeStubChi2) {
            int nStubK = 0;
            float sumW = 0.f, sumWK = 0.f, sumWK2 = 0.f;
            for (auto h = foundNtuplets->begin(it); h != foundNtuplets->end(it); ++h) {
              // A tagged raw-OT extra indexes the OT source, not hh, and is never a stub, so it contributes
              // nothing to the stub-curvature consistency.
              if (caExtension::isOTId(*h))
                continue;
              if (*h >= static_cast<unsigned int>(nHitsTot))
                break;  // content buffer corruption from overflow
              if (!isStub(hh, *h))
                continue;  // pixel hit
              const float s = hh[*h].dPhiDrError();
              if (s > 0.f) {
                const float d = hh[*h].dPhiDr();
                const float xg = hh[*h].xGlobal();
                const float yg = hh[*h].yGlobal();
                float den, w;  // same shared kappa formula as CATrackFeatures::fill
                caTrackFeatures::stubDenWeight(xg * xg + yg * yg, d, s, den, w);
                const float k = d / std::sqrt(den);  // stub curvature
                sumW += w;
                sumWK += w * k;
                sumWK2 += w * k * k;
                ++nStubK;
              }
            }
            // chi2Stub < 0 => not enough stubs to judge consistency.
            float chi2Stub = -1.f;
            if (nStubK >= 3 && sumW > 0.f) {
              chi2Stub = (sumWK2 - sumWK * sumWK / sumW) / float(nStubK - 1);
              if (!dnnHandled && cuts.maxNtupletStubChi2 >= 0.f) {
                // The keep decision is the positive comparison (chi2Stub <= cut), so a non-finite chi2Stub
                // from a degenerate stub set falls to the demoting side; the explicit isNotFinite keeps that
                // true under -Ofast.
                const bool stubConsistent = !edm::isNotFinite(chi2Stub) && (chi2Stub <= cuts.maxNtupletStubChi2);
                if (!stubConsistent)
                  failChi2 = true;
              }
            }
#ifdef CA_CHI2_DUMP
            // Per-track calibration dump: fit chi2 vs ntuplet-wide stub consistency.
            printf("[Chi2Dump] nhits=%d nStubK=%d chi2=%.4f chi2Stub=%.4f pt=%.4f eta=%.4f\n",
                   nhits,
                   nStubK,
                   tracks_view[it].chi2(),
                   chi2Stub,
                   tracks_view[it].pt(),
                   tracks_view[it].eta());
#endif
          }
        }
        if (failChi2)
          continue;

        tracks_view[it].quality() = Quality::tight;

        if (cuts.isHP(tracks_view, nhits, it))
          tracks_view[it].quality() = Quality::highPurity;
      }

#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
      if (cms::alpakatools::once_per_grid(acc)) {
        printf("FIT_DEBUG SUMMARY: total=%d fitted=%d NaN=%d doublets=%d duplicates=%d\n",
               nTracks,
               nFitted,
               nNaN,
               nDoublets,
               nDuplicates);
      }
#endif
    }
  };

  class Kernel_assignIteration {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  pixelTrack::Iteration iterationName) const {
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        tracks_view[it].iteration() = iterationName;
      }
      // Tail: the CA output SoA is allocated at capacity and only [0, nOnes) is filled.
      for (auto it : cms::alpakatools::uniform_elements(acc, uint32_t(tracks_view.metadata().size())))
        if (it >= foundNtuplets->nOnes())
          tracks_view[it].iteration() = pixelTrack::Iteration::notIteration;
    }
  };

  // updateMasking: one thread per track. Pure-write kernel: it never reads mask_view, so there is no
  // read-after-write dependency across tracks, and every write stores the same constant iterationIndex to
  // mask_view[hid].recHitMask(), so two kept tracks sharing a hit id store the identical value. The set of
  // (hid -> iterationIndex) writes is order-independent and needs no atomics.
  class Kernel_updateMaskingParallel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  ::reco::TrackingRecHitsMaskingView mask_view,
                                  const ::reco::TrackSoAConstView &trackd_view,
                                  const ::reco::TrackHitSoAConstView &trackhitd_view,
                                  const pixelTrack::Quality minQuality,
                                  uint32_t const &iterationIndex,
                                  bool maskAttachedHits) const {
      for (auto j : cms::alpakatools::uniform_elements(acc, uint32_t(trackd_view.nTracks()))) {
        if (trackd_view[j].quality() < minQuality)
          continue;
        const uint32_t end = trackd_view[j].hitOffsets();
        const uint32_t start = (j == 0u) ? 0u : trackd_view[j - 1].hitOffsets();
        for (uint32_t p = start; p < end; ++p) {
          // In-fit-extension attachments stay available to the next iteration unless maskAttachedHits;
          // tagged raw-OT extras index the OT source, not the merged mask domain, so they are skipped.
          if (!maskAttachedHits && trackhitd_view[p].attached() != 0)
            continue;
          const uint32_t hid = trackhitd_view[p].id();
          if (caExtension::isOTId(hid))
            continue;
          mask_view[hid].recHitMask() = iterationIndex;
        }
      }
    }
  };

  // Strict cross-arm twin merge (gated by PixelTracksSoAMerger twinMerge=true). The masking chain lets a
  // particle be reconstructed twice, as a pixel-rich prompt-arm track and an OT-rich displaced-arm track
  // built from the unmasked disk stubs; the two are largely disjoint, so the ordinary merger dedup never
  // pairs them. Twin-merge pairs them by trajectory and shared-hit evidence and unites their hit lists onto
  // the winner track.
  //
  // Kernel_twinFindBest: for every track, the single best opposite-arm partner passing the strict gate
  // (opposite arm, same charge, |dEta|/|dPhi| windows, >= minShared common hit ids), where best means most
  // shared hits, then smallest dR, then lowest index. Tier 2 (twinTier2) also accepts a pair that fails the
  // shared-hit evidence but sits inside the tighter windows; the ranking keeps tier 1 strictly ahead.
  class Kernel_twinFindBest {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ::reco::TrackSoAConstView inpTrack_view,
                                  const ::reco::TrackHitSoAConstView inpTrackHit_view,
                                  const int32_t *__restrict__ armOfTrack,
                                  const pixelTrack::Quality minQuality,
                                  const float twinDEta,
                                  const float twinDPhi,
                                  const int twinMinShared,
                                  const bool twinTier2,
                                  const float twinDEta2,
                                  const float twinDPhi2,
                                  const float twinNSigma2,
                                  const int twinMinSharedFwd,
                                  // phi->track OneToManyAssoc over this same collection, plus its bin count.
                                  // The whole-ring guard falls back to an exhaustive scan when the window
                                  // spans the ring, so either way the same pairs are visited.
                                  HitToTuple const *__restrict__ phiBinner,
                                  const int nPhiBins,
                                  int32_t *__restrict__ bestTwin) const {
      // Region boundary for the stricter forward same-particle discrimination (mirrors the merger's
      // finalDedup kDedupFwdEta): forward twins share fewer hits and start from worse parameters.
      constexpr float kTwinFwdEta = 1.3f;
      const int32_t nT = inpTrack_view.metadata().size();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nT)) {
        bestTwin[i] = -1;
        if (inpTrack_view[i].quality() < minQuality)
          continue;
        const int32_t armI = armOfTrack[i];
        const float etaI = inpTrack_view[i].eta();
        const float phiI = ::reco::phi(inpTrack_view, i);
        const float chgI = ::reco::charge(inpTrack_view, i);
        const uint32_t iBeg = (i == 0) ? 0u : inpTrack_view[i - 1].hitOffsets();
        const uint32_t iEnd = inpTrack_view[i].hitOffsets();

        int bestShared = 0;
        // Sentinel init so a tier-2 candidate (shared == bestShared == 0) can be recorded on the tie-break
        // arm; never consulted on the tier-1 path, where the first candidate wins via shared > bestShared.
        float bestDR2 = 1e30f;
        int32_t bestJ = -1;

        // Per-candidate evaluation, shared by the exhaustive scan and the phi-binned pre-filter sweep. The
        // winner update is a strict total order over (shared desc, dr2 asc, j asc), so bestJ does not depend
        // on the order in which j is visited and the binned iteration reproduces the exhaustive scan's bestJ.
        auto considerJ = [&](int32_t j) {
          if (j == i)
            return;
          if (armOfTrack[j] == armI)  // (a) opposite arm only
            return;
          if (inpTrack_view[j].quality() < minQuality)
            return;
          if (::reco::charge(inpTrack_view, j) != chgI)  // (b) same charge
            return;
          const float dEta = etaI - inpTrack_view[j].eta();
          if (dEta > twinDEta || dEta < -twinDEta)
            return;
          float dPhi = phiI - ::reco::phi(inpTrack_view, j);
          while (dPhi > kTwinPi)
            dPhi -= kTwinTwoPi;
          while (dPhi < -kTwinPi)
            dPhi += kTwinTwoPi;
          if (dPhi > twinDPhi || dPhi < -twinDPhi)
            return;
          // (b') Covariance-scaled arm-invariant compatibility gate; twinNSigma2 <= 0 skips the block. The
          // two halves must agree on the arm-invariant helix parameters phi (state 0), 1/pT (2) and cotTheta
          // (3); tip (1) and zip (4) are beamline-referenced and differ across arms for displaced tracks.
          // Reject the pair when any of the three has dp^2 > nSigma2 * (cov_i + cov_j), the post-GBL diagonal
          // covariance being read at the iParam2iCov offsets {0,9,12}.
          if (twinNSigma2 > 0.f) {
            bool incompatible = false;
            const int twParam[3] = {0, 2, 3};
            const int twCov[3] = {0, 9, 12};
            for (int t = 0; t < 3; ++t) {
              float dp = inpTrack_view[i].state()[twParam[t]] - inpTrack_view[j].state()[twParam[t]];
              if (t == 0) {  // phi: wrap the difference to [-pi,pi] (state 0 == reco::phi)
                while (dp > kTwinPi)
                  dp -= kTwinTwoPi;
                while (dp < -kTwinPi)
                  dp += kTwinTwoPi;
              }
              const float e2 =
                  twinNSigma2 * (inpTrack_view[i].covariance()[twCov[t]] + inpTrack_view[j].covariance()[twCov[t]]);
              if (dp * dp > e2) {
                incompatible = true;
                break;
              }
            }
            if (incompatible)
              return;
          }
          // (c) shared-hit evidence: count common hit ids between the two hit lists.
          const uint32_t jBeg = (j == 0) ? 0u : inpTrack_view[j - 1].hitOffsets();
          const uint32_t jEnd = inpTrack_view[j].hitOffsets();
          int shared = 0;
          for (uint32_t a = iBeg; a < iEnd; ++a) {
            const uint32_t ida = inpTrackHit_view[a].id();
            for (uint32_t b = jBeg; b < jEnd; ++b) {
              if (inpTrackHit_view[b].id() == ida) {
                ++shared;
                break;
              }
            }
          }
          // Region-aware shared-hit requirement: forward pairs (either member |eta| > kTwinFwdEta) may be
          // held to a stricter minimum (twinMinSharedFwd) against the forward over-merge of distinct
          // displaced tracks crossing the same OT sensor.
          const bool isFwd = (etaI > kTwinFwdEta || etaI < -kTwinFwdEta || inpTrack_view[j].eta() > kTwinFwdEta ||
                              inpTrack_view[j].eta() < -kTwinFwdEta);
          const bool fwdStricter = isFwd && twinMinSharedFwd > twinMinShared;
          const int reqShared = fwdStricter ? twinMinSharedFwd : twinMinShared;
          if (shared < reqShared) {
            // tier 1 failed: only a tier-2 pair (no shared-hit evidence, but inside the tighter trajectory
            // windows) may still qualify, and not in the forward region, where shared-hit evidence is required.
            if (!twinTier2)
              return;
            if (fwdStricter)
              return;
            if (dEta > twinDEta2 || dEta < -twinDEta2 || dPhi > twinDPhi2 || dPhi < -twinDPhi2)
              return;
          }
          const float dr2 = dEta * dEta + dPhi * dPhi;
          if (shared > bestShared || (shared == bestShared && (dr2 < bestDR2 || (dr2 == bestDR2 && j < bestJ)))) {
            bestShared = shared;
            bestDR2 = dr2;
            bestJ = j;
          }
        };

        // Phi pre-filter: sweep only the phi bins overlapping [phiI - twinDPhi, phiI + twinDPhi] (with
        // wraparound); half = floor(twinDPhi/binW) + 1 covers the window, plus one bin of floating-point
        // margin. If the window spans the whole ring, fall back to the exhaustive scan. Every j the |dPhi|
        // gate could accept lies in a swept bin and each track sits in exactly one bin, so bestJ is the same
        // as for an exhaustive scan.
        const float binW = kTwinTwoPi / float(nPhiBins);
        const int half = int(twinDPhi / binW) + 2;
        const int nVisit = 2 * half + 1;
        if (nVisit >= nPhiBins) {
          for (int32_t j = 0; j < nT; ++j)
            considerJ(j);
        } else {
          const int b0 = trackBinKey(0.f, phiI, nPhiBins, 1, 0.f);
          for (int d = -half; d <= half; ++d) {
            const int b = (b0 + d + nPhiBins) % nPhiBins;
            for (auto p = phiBinner->begin(b); p != phiBinner->end(b); ++p)
              considerJ(int32_t(*p));
          }
        }
        bestTwin[i] = bestJ;
      }
    }
  };

  // Kernel_twinConfirm: keep only mutual-best pairs (bestTwin[i]==j && bestTwin[j]==i), so every track takes
  // part in at most one merge with no atomics. The winner is chosen by the strict total ordering (max
  // nLayers -> max total hits -> max quality -> min chi2 -> min index) and records loserOf[winner] and
  // isLoser[loser]; isLoser must be zero-initialised by the caller.
  class Kernel_twinConfirm {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ::reco::TrackSoAConstView inpTrack_view,
                                  const int32_t *__restrict__ bestTwin,
                                  int32_t *__restrict__ loserOf,
                                  int32_t *__restrict__ isLoser) const {
      const int32_t nT = inpTrack_view.metadata().size();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nT)) {
        loserOf[i] = -1;  // single-writer (this thread owns slot i)
        const int32_t j = bestTwin[i];
        if (j < 0 || j == i)
          continue;
        if (bestTwin[j] != i)  // mutual-best match only
          continue;
        // Winner ordering (mirrors Kernel_rejectDuplicate).
        const int nli = inpTrack_view[i].nLayers();
        const int nlj = inpTrack_view[j].nLayers();
        const int nhi = ::reco::nHits(inpTrack_view, i);
        const int nhj = ::reco::nHits(inpTrack_view, j);
        const auto qi = inpTrack_view[i].quality();
        const auto qj = inpTrack_view[j].quality();
        const float ci = inpTrack_view[i].chi2();
        const float cj = inpTrack_view[j].chi2();
        bool iWins;
        if (nli != nlj)
          iWins = nli > nlj;
        else if (nhi != nhj)
          iWins = nhi > nhj;
        else if (qi != qj)
          iWins = qi > qj;
        else if (ci != cj)
          iWins = ci < cj;
        else
          iWins = i < j;
        if (iWins) {
          loserOf[i] = j;
          isLoser[j] = 1;  // unique writer: mutual match => only i claims j
        }
      }
    }
  };

  // One thread per input track. A track i is dropped iff quality(i) < minQuality, or isLoser[i]
  // (absorbed into its twin winner), or it has fewer than 3 hits. Every term is a pure function of the
  // read-only input SoAs and of the twinMerge outputs loserOf/isLoser, never of another track's decision,
  // so the marking is order-independent.
  //
  // Twin hit union: the winner's hit list is its own hits followed by the loser's non-duplicate hits (dedup
  // by id, capped at kTwinMaxMergedHits), reading only input hits and this winner's own output block, so it
  // is per-track-local. The only cross-track coupling is the compaction, recovered with an exclusive
  // placement derived from inclusive prefix sums of keep[] and the per-winner united-hit count, which
  // preserves the input order of the kept tracks.
  class Kernel_filterTracksMark {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ::reco::TrackSoAConstView inpTrack_view,
                                  const ::reco::TrackHitSoAConstView inpTrackHit_view,
                                  const pixelTrack::Quality minQuality,
                                  [[maybe_unused]] const double matchFraction,
                                  const int32_t *__restrict__ loserOf,
                                  const int32_t *__restrict__ isLoser,
                                  int32_t *__restrict__ keep,
                                  int32_t *__restrict__ outHitCnt) const {
      const int32_t nIn = int32_t(inpTrack_view.metadata().size());
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nIn)) {
        keep[i] = 0;
        outHitCnt[i] = 0;
        if (inpTrack_view[i].quality() < minQuality)
          continue;
        // twin-merge: this track was absorbed into its twin winner -> drop it.
        if (isLoser && isLoser[i])
          continue;

        const int32_t nhI = ::reco::nHits(inpTrack_view, i);
        // never forward a slot with fewer than 3 hits (a truncated/unfilled slot): downstream
        // (the SoA->legacy converter) asserts nHits >= 3 on every track it is handed.
        if (nhI < 3)
          continue;
        const uint32_t iBeg = (i == 0) ? 0u : inpTrack_view[i - 1].hitOffsets();

        // kept. Count the united hit block = own hits + twin-loser union (dedup by id, capped). Mirrors
        // Kernel_filterTracksScatter's write loop exactly, so the count matches what the scatter writes.
        keep[i] = 1;
        uint32_t nUnited = uint32_t(nhI);
        if (loserOf) {
          const int32_t loser = loserOf[i];
          if (loser >= 0 && nUnited < uint32_t(kTwinMaxMergedHits)) {
            uint32_t ids[kTwinMaxMergedHits];  // own count < cap here -> fits
            for (uint32_t k = 0; k < nUnited; ++k)
              ids[k] = inpTrackHit_view[iBeg + k].id();
            const uint32_t lBeg = (loser == 0) ? 0u : inpTrack_view[loser - 1].hitOffsets();
            const uint32_t lEnd = inpTrack_view[loser].hitOffsets();
            for (uint32_t k = lBeg; k < lEnd; ++k) {
              if (nUnited >= uint32_t(kTwinMaxMergedHits))
                break;
              const uint32_t lid = inpTrackHit_view[k].id();
              bool present = false;
              for (uint32_t m = 0; m < nUnited; ++m)
                if (ids[m] == lid) {
                  present = true;
                  break;
                }
              if (present)
                continue;
              ids[nUnited++] = lid;
            }
          }
        }
        outHitCnt[i] = int32_t(nUnited);
      }
    }
  };

  // Scatter phase: places each kept winner at its compacted output index (from the inclusive prefix
  // sums), writes its united hit block (own hits, then the twin loser's non-duplicate hits), then
  // applies the twin-merge refit ndof/chi2 recompute and the unitedMaskOut stamping.
  class Kernel_filterTracksScatter {
  public:
    static_assert(::reco::TrackSoA::Descriptor::num_cols == 11,
                  "reco::TrackLayout column count changed: update this compaction's column "
                  "enumeration (quality, chi2, nLayers, eta, pt, state[5], covariance[15], nTracks, "
                  "hitOffsets, iteration, ndof)");
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  ::reco::TrackSoAView track_view,
                                  ::reco::TrackHitSoAView trackHit_view,
                                  const ::reco::TrackSoAConstView inpTrack_view,
                                  const ::reco::TrackHitSoAConstView inpTrackHit_view,
                                  const int32_t *__restrict__ loserOf,
                                  const bool twinMergeRefit,
                                  const bool refitAllTracks,
                                  int32_t *__restrict__ unitedMaskOut,
                                  const int32_t *__restrict__ keep,
                                  const int32_t *__restrict__ outHitCnt,
                                  const int32_t *__restrict__ tkOff,   // inclusive scan of keep[]
                                  const int32_t *__restrict__ hitOff,  // inclusive scan of outHitCnt[]
                                  const int32_t nScanSize,
                                  // pocket gate: input-order arm in, output-order arm out; both null unless
                                  // the merger's arm-scoped pocket gate is on, and scattered like the tracks.
                                  const uint8_t *__restrict__ pocketArmIn,
                                  uint8_t *__restrict__ pocketArmIdOut) const {
      // authoritative output count = total kept = last inclusive-scan value.
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] == 0)
        track_view.nTracks() = tkOff[nScanSize - 1];

      const int32_t nIn = int32_t(inpTrack_view.metadata().size());
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nIn)) {
        if (!keep[i])
          continue;
        const uint32_t outTk = uint32_t(tkOff[i] - 1);                               // 0-based compacted track slot
        const uint32_t writtenBegin = uint32_t(hitOff[i]) - uint32_t(outHitCnt[i]);  // hit block base

        track_view[outTk].quality() = inpTrack_view[i].quality();
        track_view[outTk].chi2() = inpTrack_view[i].chi2();
        track_view[outTk].ndof() = inpTrack_view[i].ndof();
        // Provenance: carry the producing iteration through the compaction, alongside ndof.
        track_view[outTk].iteration() = inpTrack_view[i].iteration();
        track_view[outTk].nLayers() = inpTrack_view[i].nLayers();
        track_view[outTk].eta() = inpTrack_view[i].eta();
        track_view[outTk].pt() = inpTrack_view[i].pt();
        for (uint32_t k = 0; k < 5; ++k)
          track_view[outTk].state()[k] = inpTrack_view[i].state()[k];
        for (uint32_t k = 0; k < 15; ++k)
          track_view[outTk].covariance()[k] = inpTrack_view[i].covariance()[k];

        const uint32_t iBeg = (i == 0) ? 0u : inpTrack_view[i - 1].hitOffsets();
        const uint32_t iEnd = inpTrack_view[i].hitOffsets();
        uint32_t w = writtenBegin;
        for (uint32_t k = iBeg; k < iEnd; ++k) {
          trackHit_view[w].id() = inpTrackHit_view[k].id();
          trackHit_view[w].detId() = inpTrackHit_view[k].detId();
          trackHit_view[w].attached() = inpTrackHit_view[k].attached();
          ++w;
        }

        // twin-merge union (dedup by id against the already-written own+appended hits, capped); the
        // present-check reads only this track's output block, so it is per-track-local.
        if (loserOf) {
          const int32_t loser = loserOf[i];
          if (loser >= 0) {
            const uint32_t lBeg = (loser == 0) ? 0u : inpTrack_view[loser - 1].hitOffsets();
            const uint32_t lEnd = inpTrack_view[loser].hitOffsets();
            for (uint32_t k = lBeg; k < lEnd; ++k) {
              if ((w - writtenBegin) >= uint32_t(kTwinMaxMergedHits))
                break;
              const uint32_t lid = inpTrackHit_view[k].id();
              bool present = false;
              for (uint32_t m = writtenBegin; m < w; ++m) {
                if (trackHit_view[m].id() == lid) {
                  present = true;
                  break;
                }
              }
              if (present)
                continue;
              trackHit_view[w].id() = lid;
              trackHit_view[w].detId() = inpTrackHit_view[k].detId();
              trackHit_view[w].attached() = inpTrackHit_view[k].attached();
              ++w;
            }
          }
        }
        track_view[outTk].hitOffsets() = w;  // == hitOff[i] by construction

        if (twinMergeRefit && loserOf && loserOf[i] >= 0) {
          const uint32_t nUnited = w - writtenBegin;  // own + appended (CSR extent)
          constexpr uint32_t kTwinRefitNdofCap = 12;  // keep in sync with HelixFit::kRefitMaxN
          const uint32_t nFit = nUnited < kTwinRefitNdofCap ? nUnited : kTwinRefitNdofCap;
          int ndofUnited = 2 * int(nFit) - 5;
          const int ndofWinner = int(inpTrack_view[i].ndof());
          if (ndofUnited < ndofWinner)
            ndofUnited = ndofWinner;
          if (ndofUnited < 1)
            ndofUnited = 1;
          const float rawChi2 = inpTrack_view[i].chi2() * float(ndofWinner > 0 ? ndofWinner : 1);
          track_view[outTk].ndof() = int8_t(ndofUnited);
          track_view[outTk].chi2() = rawChi2 / float(ndofUnited);
          if (unitedMaskOut)
            unitedMaskOut[outTk] = int32_t(outTk);
        }
        if (refitAllTracks && unitedMaskOut)
          unitedMaskOut[outTk] = int32_t(outTk);
        // pocket gate: scatter the per-track arm to the same compacted slot the track went to (outTk), so
        // launchMergerAttach reads armId in the merged-SoA order. Null leaves it untouched.
        if (pocketArmIn && pocketArmIdOut)
          pocketArmIdOut[outTk] = pocketArmIn[i];
      }
      // Tail hygiene: stamp the unused output capacity so the iteration column is never allocator garbage.
      // Index-disjoint from the kept slots written above (outTk = tkOff[i]-1 < nOut), so no barrier is needed.
      const uint32_t nOut = uint32_t(tkOff[nScanSize - 1]);
      const uint32_t hitEndTail = uint32_t(hitOff[nScanSize - 1]);  // CSR end of the last kept track
      for (uint32_t k : cms::alpakatools::uniform_elements(acc, uint32_t(track_view.metadata().size())))
        if (k >= nOut) {
          track_view[k].iteration() = pixelTrack::Iteration::notIteration;
          track_view[k].hitOffsets() = hitEndTail;  // nHits() reads zero past the last track
        }
    }
  };

  // Final post-refit de-dup: compaction of the refined merged tracks into the output SoA, skipping the
  // tracks flagged by Kernel_dedupCovMark (parameters, covariance and the hit CSR copied verbatim, no union
  // and no re-fit; drop[i] == 0 means kept).
  //
  // keep[]/hitCnt[] are memset to 0 over the whole scan capacity by the launcher; the counts kernel fills
  // [0, nTracks) and the trailing zeros leave the inclusive scans constant past the last real track, so
  // tkOff[cap-1] is the total kept and the per-track placements are unchanged.
  // One-thread reporter for the two finalDedup diagnostic counters: consuming them on the device avoids a
  // D2H read of two words that would serialize the host against everything queued ahead of the copy.
  class Kernel_dedupOverflowReport {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const uint32_t *__restrict__ binnerOvf,
                                  const uint32_t *__restrict__ contestedOvf) const {
      if (cms::alpakatools::once_per_grid(acc)) {
        if (*binnerOvf != 0)
          printf("PixelTracksSoAMerger finalDedup: hit/track binner key overflow (clamped, not fatal): %u entries\n",
                 *binnerOvf);
        if (contestedOvf != nullptr && *contestedOvf != 0)
          printf(
              "PixelTracksSoAMerger finalDedup: contested-pair list overflow (kept both, not fatal; raise "
              "kDedupConfirmMaxPairs): %u pairs\n",
              *contestedOvf);
      }
    }
  };

  class Kernel_finalDedupCounts {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ::reco::TrackSoAConstView tracks_view,
                                  const uint8_t *__restrict__ drop,
                                  int32_t *__restrict__ keep,
                                  int32_t *__restrict__ hitCnt) const {
      const int32_t nT = tracks_view.nTracks();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nT)) {
        const int32_t k = drop[i] ? 0 : 1;
        keep[i] = k;
        const uint32_t beg = (i == 0) ? 0u : tracks_view[i - 1].hitOffsets();
        const uint32_t end = tracks_view[i].hitOffsets();
        hitCnt[i] = k ? int32_t(end - beg) : 0;
      }
    }
  };

  // Scatter: copies each kept track's fields and hit CSR verbatim (no union, no re-fit) to the
  // compacted indices from the inclusive prefix sums. Preserves the input order of the kept tracks.
  class Kernel_finalDedupScatter {
  public:
    static_assert(::reco::TrackSoA::Descriptor::num_cols == 11,
                  "reco::TrackLayout column count changed: update this compaction's column "
                  "enumeration (quality, chi2, nLayers, eta, pt, state[5], covariance[15], nTracks, "
                  "hitOffsets, iteration, ndof)");
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  ::reco::TrackSoAView out_view,
                                  ::reco::TrackHitSoAView outHit_view,
                                  const ::reco::TrackSoAConstView tracks_view,
                                  const ::reco::TrackHitSoAConstView trackHit_view,
                                  const int32_t *__restrict__ keep,
                                  const int32_t *__restrict__ tkOff,   // inclusive scan of keep[]
                                  const int32_t *__restrict__ hitOff,  // inclusive scan of hitCnt[]
                                  const int32_t nScanSize) const {
      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] == 0)
        out_view.nTracks() = tkOff[nScanSize - 1];

      const int32_t nT = tracks_view.nTracks();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nT)) {
        if (!keep[i])
          continue;
        const uint32_t outTk = uint32_t(tkOff[i] - 1);
        const uint32_t begin = (i == 0) ? 0u : tracks_view[i - 1].hitOffsets();
        const uint32_t end = tracks_view[i].hitOffsets();
        const uint32_t outHitEnd = uint32_t(hitOff[i]);
        uint32_t outHit = outHitEnd - (end - begin);

        out_view[outTk].quality() = tracks_view[i].quality();
        out_view[outTk].chi2() = tracks_view[i].chi2();
        out_view[outTk].ndof() = tracks_view[i].ndof();
        out_view[outTk].iteration() = tracks_view[i].iteration();
        out_view[outTk].nLayers() = tracks_view[i].nLayers();
        out_view[outTk].eta() = tracks_view[i].eta();
        out_view[outTk].pt() = tracks_view[i].pt();
        for (uint32_t k = 0; k < 5; ++k)
          out_view[outTk].state()[k] = tracks_view[i].state()[k];
        for (uint32_t k = 0; k < 15; ++k)
          out_view[outTk].covariance()[k] = tracks_view[i].covariance()[k];
        for (uint32_t k = begin; k < end; ++k) {
          outHit_view[outHit].id() = trackHit_view[k].id();
          outHit_view[outHit].detId() = trackHit_view[k].detId();
          outHit_view[outHit].attached() = trackHit_view[k].attached();
          ++outHit;
        }
        out_view[outTk].hitOffsets() = outHit;  // == outHitEnd by construction
      }
      // Tail hygiene (see Kernel_filterTracksScatter): unused output capacity stamped notIteration.
      const uint32_t nOutTk = uint32_t(tkOff[nScanSize - 1]);
      const uint32_t hitEndTail = uint32_t(hitOff[nScanSize - 1]);  // CSR end of the last kept track
      for (uint32_t k : cms::alpakatools::uniform_elements(acc, uint32_t(out_view.metadata().size())))
        if (k >= nOutTk) {
          out_view[k].iteration() = pixelTrack::Iteration::notIteration;
          out_view[k].hitOffsets() = hitEndTail;  // nHits() reads zero past the last track
        }
    }
  };

  template <typename TrackerTraits>
  class Kernel_doStatsForTracks {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  Counters *counters) const {
      for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(idx) == 0)
          break;  //guard
        if (tracks_view[idx].quality() < Quality::loose)
          continue;
        alpaka::atomicAdd(acc, &(counters->nLooseTracks), 1ull, alpaka::hierarchy::Blocks{});
        if (tracks_view[idx].quality() < Quality::strict)
          continue;
        alpaka::atomicAdd(acc, &(counters->nGoodTracks), 1ull, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // Final quality distribution counter: counts tracks at each quality level after classification, fishbone
  // and duplicate removal.
#ifdef CA_PIPELINE_COUNTERS
  template <typename TrackerTraits>
  class Kernel_countFinalQuality {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  HitsConstView hh,
                                  uint32_t *__restrict__ pipelineCounters) const {
      using Quality = pixelTrack::Quality;
      using PC = caHitNtupletGenerator::PipelineCounter;

      for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        auto nhits = foundNtuplets->size(idx);
        if (nhits == 0)
          break;  // guard

        alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualTotal], 1u, alpaka::hierarchy::Blocks{});

        auto q = tracks_view[idx].quality();
        if (q == Quality::bad) {
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualBad], 1u, alpaka::hierarchy::Blocks{});
        } else if (q == Quality::edup) {
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualEdup], 1u, alpaka::hierarchy::Blocks{});
        } else if (q == Quality::dup) {
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualDup], 1u, alpaka::hierarchy::Blocks{});
        } else if (q == Quality::loose) {
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualLoose], 1u, alpaka::hierarchy::Blocks{});
        } else {
          // strict, tight, or highPurity -- check OT once for all levels
          bool hasOT = false;
          if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
            auto nHits = hh.metadata().size();
            for (auto h = foundNtuplets->begin(idx); h != foundNtuplets->end(idx); ++h) {
              if (*h >= static_cast<unsigned int>(nHits))
                break;  // content buffer corruption from overflow
              if (isStub(hh, *h)) {
                hasOT = true;
                break;
              }
            }
          }
          alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualStrict], 1u, alpaka::hierarchy::Blocks{});
          if (hasOT)
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualStrictWithOT], 1u, alpaka::hierarchy::Blocks{});
          if (q >= Quality::tight) {
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualTight], 1u, alpaka::hierarchy::Blocks{});
            if (hasOT)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualTightWithOT], 1u, alpaka::hierarchy::Blocks{});
          }
          if (q >= Quality::highPurity) {
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualHP], 1u, alpaka::hierarchy::Blocks{});
            if (hasOT)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualHPWithOT], 1u, alpaka::hierarchy::Blocks{});
          }

          // Per-nhits quality breakdown
          float chi2 = tracks_view[idx].chi2();
          if (nhits <= 4) {
            if (q == Quality::strict)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualStrict34], 1u, alpaka::hierarchy::Blocks{});
            else if (q == Quality::tight)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualTight34], 1u, alpaka::hierarchy::Blocks{});
            else if (q >= Quality::highPurity)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualHP34], 1u, alpaka::hierarchy::Blocks{});
            if (chi2 >= 0.9f && chi2 < 1.1f)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kChi2Boundary34], 1u, alpaka::hierarchy::Blocks{});
          } else if (nhits == 5) {
            if (q == Quality::strict)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualStrict5], 1u, alpaka::hierarchy::Blocks{});
            else if (q == Quality::tight)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualTight5], 1u, alpaka::hierarchy::Blocks{});
            else if (q >= Quality::highPurity)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualHP5], 1u, alpaka::hierarchy::Blocks{});
            if (chi2 >= 2.7f && chi2 < 3.3f)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kChi2Boundary5], 1u, alpaka::hierarchy::Blocks{});
          } else {
            if (q == Quality::strict)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualStrict6p], 1u, alpaka::hierarchy::Blocks{});
            else if (q == Quality::tight)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualTight6p], 1u, alpaka::hierarchy::Blocks{});
            else if (q >= Quality::highPurity)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kQualHP6p], 1u, alpaka::hierarchy::Blocks{});
            if (chi2 >= 4.5f && chi2 < 5.5f)
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kChi2Boundary6p], 1u, alpaka::hierarchy::Blocks{});
          }

          // Fishbone-hit multiplicity per track. The hit container does not mark fishbone hits and
          // the cell count is not available here, so every track lands in the 0-fishbone bucket.
          uint32_t nFishbone = 0;
          nFishbone = 0;
          if (nFishbone == 0)
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kTracksFishbone0], 1u, alpaka::hierarchy::Blocks{});
          else if (nFishbone == 1)
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kTracksFishbone1], 1u, alpaka::hierarchy::Blocks{});
          else
            alpaka::atomicAdd(acc, &pipelineCounters[PC::kTracksFishbone2p], 1u, alpaka::hierarchy::Blocks{});
        }
      }
    }
  };
#endif  // CA_PIPELINE_COUNTERS

  // Map a track-hit id to its hit->tuple bin: merged hits key on the id directly; tagged OT extras key on
  // the extended domain slot nHits + otIdx (the container is sized nHits + nOTHits when the OT source is
  // active).
  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t hitToTupleKey(uint32_t id, uint32_t nHits) {
    return caExtension::isOTId(id) ? nHits + caExtension::otIdx(id) : id;
  }

  // The kernels below build the OneToManyAssoc candidate-generation structures behind the merger dedup,
  // following the CA's own count -> launchFinalize -> fill sequence. All fills are count-and-clamped: a key
  // outside [0, nKeys) is skipped and flagged in overflow[0] instead of written out of bounds, an unguarded
  // OneToManyAssoc overflow raising cudaErrorIllegalAddress at an allocation-dependent point.

  // Count pass for a track (eta,phi) binner (one thread per track). Used for the twinFindBest phi
  // pre-filter (nEtaSlabs=1 => pure phi) AND the 0-shared fallback eta-phi binner (nEtaSlabs>1).
  class Kernel_trackBinCount {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ::reco::TrackSoAConstView tracks_view,
                                  const int32_t nItems,  // consumer's iteration range; <0 => device nTracks()
                                  HitToTuple *__restrict__ assoc,
                                  const int nPhiBins,
                                  const int nEtaSlabs,
                                  const float etaMax,
                                  const uint32_t nKeys,
                                  uint32_t *__restrict__ overflow) const {
      const int32_t nT = (nItems < 0) ? tracks_view.nTracks() : nItems;
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nT)) {
        const uint32_t key =
            uint32_t(trackBinKey(tracks_view[i].eta(), ::reco::phi(tracks_view, i), nPhiBins, nEtaSlabs, etaMax));
        if (key < nKeys)
          assoc->count(acc, key);
        else
          alpaka::atomicAdd(acc, overflow, 1u, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // Fill pass for the (eta,phi) track binner (mirrors the count pass exactly).
  class Kernel_trackBinFill {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ::reco::TrackSoAConstView tracks_view,
                                  const int32_t nItems,
                                  HitToTuple *__restrict__ assoc,
                                  const int nPhiBins,
                                  const int nEtaSlabs,
                                  const float etaMax,
                                  const uint32_t nKeys,
                                  uint32_t *__restrict__ overflow) const {
      const int32_t nT = (nItems < 0) ? tracks_view.nTracks() : nItems;
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nT)) {
        const uint32_t key =
            uint32_t(trackBinKey(tracks_view[i].eta(), ::reco::phi(tracks_view, i), nPhiBins, nEtaSlabs, etaMax));
        if (key < nKeys)
          assoc->fill(acc, key, i);
        else
          alpaka::atomicAdd(acc, overflow, 1u, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // Count pass for the hit-id -> refined-track co-occurrence histogram (one thread per track, over the
  // track's CSR hit list). Merged pixel/strip ids bin on the id directly, bit30-tagged OT extras at
  // nHits+otIdx via hitToTupleKey, so the key space is [0, nHits + nOTHits) and the total fills, the sum of
  // the per-track hit counts, never exceed the trackHit CSR capacity the content is sized at.
  class Kernel_dedupHitCount {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ::reco::TrackSoAConstView tracks_view,
                                  const ::reco::TrackHitSoAConstView trackHit_view,
                                  HitToTuple *__restrict__ assoc,
                                  const uint32_t nHits,
                                  const uint32_t nKeys,
                                  uint32_t *__restrict__ overflow) const {
      const int32_t nT = tracks_view.nTracks();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nT)) {
        const uint32_t iBeg = (i == 0) ? 0u : tracks_view[i - 1].hitOffsets();
        const uint32_t iEnd = tracks_view[i].hitOffsets();
        for (uint32_t a = iBeg; a < iEnd; ++a) {
          const uint32_t key = hitToTupleKey(trackHit_view[a].id(), nHits);
          if (key < nKeys)
            assoc->count(acc, key);
          else
            alpaka::atomicAdd(acc, overflow, 1u, alpaka::hierarchy::Blocks{});
        }
      }
    }
  };

  // Fill pass for the hit-id co-occurrence histogram (mirrors the count pass exactly).
  class Kernel_dedupHitFill {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ::reco::TrackSoAConstView tracks_view,
                                  const ::reco::TrackHitSoAConstView trackHit_view,
                                  HitToTuple *__restrict__ assoc,
                                  const uint32_t nHits,
                                  const uint32_t nKeys,
                                  uint32_t *__restrict__ overflow) const {
      const int32_t nT = tracks_view.nTracks();
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nT)) {
        const uint32_t iBeg = (i == 0) ? 0u : tracks_view[i - 1].hitOffsets();
        const uint32_t iEnd = tracks_view[i].hitOffsets();
        for (uint32_t a = iBeg; a < iEnd; ++a) {
          const uint32_t key = hitToTupleKey(trackHit_view[a].id(), nHits);
          if (key < nKeys)
            assoc->fill(acc, key, i);
          else
            alpaka::atomicAdd(acc, overflow, 1u, alpaka::hierarchy::Blocks{});
        }
      }
    }
  };

  // Kernel_dedupCovMark: one thread per refined merged track i. i is the duplicate loser iff some better
  // track j (by the strict total order below) is a covariance-compatible duplicate of it. Candidate j's are
  // generated from the shared-hit co-occurrence histogram (hitAssoc): every j sharing a hit with i, with no
  // window, relPt, minShared or charge veto (the signed q/pT folds the charge in). Compatibility is
  // dp^2 <= kDedupNSigma2Default * (cov_i + cov_j) for all of {phi(0), q/pT(2), cotTheta(3)} at the cov
  // offsets {0,9,12}, phi wrapped. The ranking is a strict total order, so exactly one of a matched pair is
  // the loser and drop[loser]=1 is idempotent: no atomics, thread i being the sole writer of drop[i].
  //
  // 0-shared forward fallback: when etaPhiAssoc is non-null, tracks that are not already a shared-hit loser
  // are also checked against cov-compatible, genuinely 0-shared neighbours from an eta-phi track binner.
  // Drop authority is bounded to |eta| <= kDedupFbDropAbsEtaMax; out-of-bound candidates are only counted.
  // The two paths are mutually exclusive per track, so each track increments at most one diagnostic bucket.
  //
  // diag (optional, may be null; 18 uint32): [region*3 + bucket] for region {0=central,1=forward} (split at
  // kDedupFwdEta) x shared bucket {0=0-shared, 1, 2+}, [6+region] drop totals, [8+region] cov-gate-miss
  // survivors, [10]=contested captured, [11]=pre-gate rejected, [12]=confirmed drop, [13]=keep-both,
  // [14]=over-size union (nSel > kRefitMaxN), [15]=refit non-finite, [16]=finder-mode suppressed drops,
  // [17]=cross-arm corner guard fires. drop[] must be zero-initialised by the caller.
  class Kernel_dedupCovMark {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const &acc,
        const ::reco::TrackSoAConstView tracks_view,
        const ::reco::TrackHitSoAConstView trackHit_view,
        HitToTuple const *__restrict__ hitAssoc,
        HitToTuple const *__restrict__ etaPhiAssoc,
        const uint32_t nHits,
        const uint32_t nKeys,
        uint8_t *__restrict__ drop,
        uint32_t *__restrict__ diag,
        const float nSigma2,
        const int fbEtaReach,
        const int fbPhiReach,
        const float fbNSigma2,
        const float fbDropAbsEtaMax,
        const int fbEnable,
        // merge-or-keep-both confirm capture: with fbConfirm == 0 (or contestedPairs == nullptr) the
        // fallback branch drops its loser outright; when on, a droppable (i, fbPartner) pair is deferred to
        // the union-refit verdict and recorded into a capped device pair list by atomic append, subject to
        // the pre-gates below.
        const int fbConfirm,                       // 1 = capture for the refit verdict; 0 = drop here
        const int fbSameCharge,                    // 1 = require same charge sign (state()[2]) to contest
        const float fbAbsFloorDPhi,                // abs-floor box: max |dphi| (wrapped); large => off
        const float fbAbsFloorDQoP,                // abs-floor box: max |d q/pT|;         large => off
        const float fbAbsFloorDCotTheta,           // abs-floor box: max |d cotTheta|;     large => off
        uint32_t *__restrict__ contestedPairs,     // capped list, 2 uint32/pair {i, j}; null => off
        uint32_t *__restrict__ contestedCount,     // atomic append cursor (1 uint32); null => off
        const uint32_t contestedCap,               // pair-list capacity in pairs (slots)
        uint32_t *__restrict__ contestedOverflow,  // atomic overflow counter (nullable)
        // ranking / drop-authority variants. With finder 0, rankClusters 0, rankNHits 0 and guardCrossArm 0
        // the two ranking lambdas use the (nLayers, nHits) length key. hh is dereferenced only when
        // rankClusters or guardCrossArm is on; the launcher passes an empty view otherwise.
        const ::reco::TrackingRecHitConstView hh,  // for ::reco::isStub -> cluster count / pixel-core arm proxy
        const int fbFinderOnly,                    // finder mode: scan+diag but NEVER drop (own bucket [16])
        const int rankClusters,                    // length key = weighted CLUSTER count (stub=2) not nL/nH
        const int rankNHits,                       // length key = nHits ONLY (skip the nLayers primary)
        const int guardCrossArm,                   // cross-arm keep-longest corner guard (needs qual/chi2)
        const float guardVertPosMin,               // guard: min |dxy| proxy (|state[1]|, cm) to engage
        const float guardChi2Margin) const {       // guard: chi2/ndof margin the longer track must ALSO win by
      // nSigma2 is the cov-gate width of the shared-hit path; fbEtaReach/fbPhiReach the bin reach of the
      // fallback neighbourhood scan. The fb* parameters act only on the 0-shared fallback branch:
      //   fbNSigma2       cov-gate width of the fallback's compatible() call
      //   fbDropAbsEtaMax the fallback's drop-authority |eta| bound (image of kDedupFbDropAbsEtaMax);
      //                   out-of-bound candidates are counted in diag, never dropped
      //   fbEnable        master fallback-drop switch; 0 never drops but still scans and fills the diagnostics
      //   fbFinderOnly    the fallback scans and diag-counts but sets drop[] for no candidate (bucket [16])
      //   rankClusters    length key = weighted cluster count (raw-OT extra=1, core stub=2, pixel=1) instead
      //                   of (nLayers, nHits), in both beats() and the loser test jBeatsI()
      //   guardCrossArm   corner guard on keep-longest: for cross-arm pairs (pixel-core vs OT-only-core) at
      //                   |dxy| proxy > guardVertPosMin a longer track wins only if it also wins the
      //                   quality/chi2 tiebreak by guardChi2Margin
      const int32_t nT = tracks_view.nTracks();
      const int cParam[3] = {0, 2, 3};  // phi, signed q/pT, cotTheta
      const int cCov[3] = {0, 9, 12};   // their diagonal covariance offsets (iParam2iCov)
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nT)) {
        drop[i] = 0;
        const float etaI = tracks_view[i].eta();
        const int nlI = tracks_view[i].nLayers();
        const int nhI = ::reco::nHits(tracks_view, i);
        const auto qI = tracks_view[i].quality();
        const float c2I = tracks_view[i].chi2();
        const uint32_t iBeg = (i == 0) ? 0u : tracks_view[i - 1].hitOffsets();
        const uint32_t iEnd = tracks_view[i].hitOffsets();

        // length / arm helpers, called only when rankClusters or guardCrossArm is on (the launcher gates
        // that on a valid hit view hh). Weighted cluster count: raw-OT extra=1, core stub=2 (two rechits),
        // pixel=1, the same weight the extension caps on.
        auto lengthClusters = [&](int32_t t) -> int {
          const uint32_t b = (t == 0) ? 0u : tracks_view[t - 1].hitOffsets();
          const uint32_t e = tracks_view[t].hitOffsets();
          int c = 0;
          for (uint32_t a = b; a < e; ++a) {
            const uint32_t id = trackHit_view[a].id();
            c += caExtension::isOTId(id) ? 1 : (isStub(hh, int32_t(id)) ? 2 : 1);
          }
          return c;
        };
        // Arm proxy: a track has a pixel core iff it carries at least one plain pixel hit (not a bit30 OT
        // tag and not a stub). pixel-core vs OT-only-core is the cross-arm split.
        auto hasPixelCore = [&](int32_t t) -> bool {
          const uint32_t b = (t == 0) ? 0u : tracks_view[t - 1].hitOffsets();
          const uint32_t e = tracks_view[t].hitOffsets();
          for (uint32_t a = b; a < e; ++a) {
            const uint32_t id = trackHit_view[a].id();
            if (!caExtension::isOTId(id) && !isStub(hh, int32_t(id)))
              return true;
          }
          return false;
        };
        auto crossArm = [&](int32_t x, int32_t y) -> bool { return hasPixelCore(x) != hasPixelCore(y); };
        auto vertPosMax = [&](int32_t x, int32_t y) -> float {
          const float ax = std::abs(tracks_view[x].state()[1]);  // state[1] = tip (|dxyBS| proxy)
          const float ay = std::abs(tracks_view[y].state()[1]);
          return ax > ay ? ax : ay;
        };

        // Strict total order: is x a better member than y? (length -> quality -> chi2 -> index). Being
        // total, it names a unique loser. The length key is (nLayers, nHits), a single cluster count when
        // rankClusters is on, or nHits only when rankNHits is on, which takes precedence over rankClusters.
        auto beats = [&](int32_t x, int32_t y) -> bool {
          if (rankNHits != 0) {
            const int nhx = ::reco::nHits(tracks_view, x), nhy = ::reco::nHits(tracks_view, y);
            if (nhx != nhy)
              return nhx > nhy;
          } else if (rankClusters != 0) {
            const int cx = lengthClusters(x), cy = lengthClusters(y);
            if (cx != cy)
              return cx > cy;
          } else {
            const int nlx = tracks_view[x].nLayers(), nly = tracks_view[y].nLayers();
            if (nlx != nly)
              return nlx > nly;
            const int nhx = ::reco::nHits(tracks_view, x), nhy = ::reco::nHits(tracks_view, y);
            if (nhx != nhy)
              return nhx > nhy;
          }
          const auto qx = tracks_view[x].quality(), qy = tracks_view[y].quality();
          if (qx != qy)
            return qx > qy;
          const float cx = tracks_view[x].chi2(), cy = tracks_view[y].chi2();
          if (cx != cy)
            return cx < cy;
          return x < y;
        };
        // does j beat i? Same order as beats(j, i), with the cross-arm corner guard spliced into the
        // length-decided branch so a longer cross-arm track at large displacement cannot win on length alone.
        auto jBeatsI = [&](int32_t j) -> bool {
          bool lenDiff = false, jLonger = false;
          if (rankNHits != 0) {  // nHits-only length key; takes precedence over rankClusters
            const int nhj = ::reco::nHits(tracks_view, j);
            if (nhj != nhI) {
              lenDiff = true;
              jLonger = nhj > nhI;
            }
          } else if (rankClusters != 0) {
            const int ci = lengthClusters(i), cj = lengthClusters(j);
            if (ci != cj) {
              lenDiff = true;
              jLonger = cj > ci;
            }
          } else {
            const int nlj = tracks_view[j].nLayers();
            if (nlj != nlI) {
              lenDiff = true;
              jLonger = nlj > nlI;
            } else {
              const int nhj = ::reco::nHits(tracks_view, j);
              if (nhj != nhI) {
                lenDiff = true;
                jLonger = nhj > nhI;
              }
            }
          }
          if (lenDiff) {
            if (jLonger && guardCrossArm != 0 && crossArm(i, j) && vertPosMax(i, j) > guardVertPosMin) {
              const auto qj = tracks_view[j].quality();
              const float c2j = tracks_view[j].chi2();
              const bool qualWins = (qj > qI) || (c2j + guardChi2Margin <= c2I);
              if (!qualWins) {
                if (diag)
                  alpaka::atomicAdd(acc, &diag[17], 1u, alpaka::hierarchy::Blocks{});  // corner guard fired
                return false;  // keep i: length alone does not win in the at-risk corner
              }
            }
            return jLonger;
          }
          const auto qj = tracks_view[j].quality();
          if (qj != qI)
            return qj > qI;
          const float c2j = tracks_view[j].chi2();
          if (c2j != c2I)
            return c2j < c2I;
          return j < i;
        };
        // cov-scaled arm-invariant compatibility (all three parameters must pass). ns2 is the gate width:
        // the shared-hit path passes nSigma2, the fallback path fbNSigma2; the arithmetic is identical.
        auto compatible = [&](int32_t j, float ns2) -> bool {
          for (int t = 0; t < 3; ++t) {
            float dp = tracks_view[i].state()[cParam[t]] - tracks_view[j].state()[cParam[t]];
            if (t == 0) {
              while (dp > kTwinPi)
                dp -= kTwinTwoPi;
              while (dp < -kTwinPi)
                dp += kTwinTwoPi;
            }
            const float e2 = ns2 * (tracks_view[i].covariance()[cCov[t]] + tracks_view[j].covariance()[cCov[t]]);
            if (dp * dp > e2)
              return false;
          }
          return true;
        };
        // shared hit-id count between i and j (brute force; lists <= kTwinMaxMergedHits).
        auto sharedCount = [&](int32_t j) -> int {
          const uint32_t jBeg = (j == 0) ? 0u : tracks_view[j - 1].hitOffsets();
          const uint32_t jEnd = tracks_view[j].hitOffsets();
          int s = 0;
          for (uint32_t a = iBeg; a < iEnd; ++a) {
            const uint32_t ida = trackHit_view[a].id();
            for (uint32_t b = jBeg; b < jEnd; ++b)
              if (trackHit_view[b].id() == ida) {
                ++s;
                break;
              }
          }
          return s;
        };

        // shared-hit path: co-occurring candidates from the hit histogram
        bool loser = false;
        bool sawGateMiss = false;  // a better co-occurring partner exists but the cov gate rejected the pair
        int32_t bestPartner = -1;  // highest-ranked matched-and-better partner, for the diagnostics
        for (uint32_t a = iBeg; a < iEnd; ++a) {
          const uint32_t key = hitToTupleKey(trackHit_view[a].id(), nHits);
          if (key >= nKeys)
            continue;  // guarded (already counted in overflow during the build)
          for (auto p = hitAssoc->begin(key); p != hitAssoc->end(key); ++p) {
            const int32_t j = int32_t(*p);
            if (j == i)
              continue;
            if (!jBeatsI(j))
              continue;                     // only a better member can make i the loser
            if (!compatible(j, nSigma2)) {  // shared-hit path: the shared cov-gate width
              sawGateMiss = true;
              continue;
            }
            loser = true;
            if (bestPartner < 0 || beats(j, bestPartner))
              bestPartner = j;
          }
        }
        if (loser) {
          drop[i] = 1;
          if (diag) {
            const int shared = sharedCount(bestPartner);  // >=1 by construction (co-occurrence)
            const bool fwd =
                (std::abs(etaI) > kDedupFwdEta) || (std::abs(tracks_view[bestPartner].eta()) > kDedupFwdEta);
            const int region = fwd ? 1 : 0;
            const int bucket = (shared <= 0) ? 0 : (shared == 1 ? 1 : 2);
            alpaka::atomicAdd(acc, &diag[region * 3 + bucket], 1u, alpaka::hierarchy::Blocks{});
            alpaka::atomicAdd(acc, &diag[6 + region], 1u, alpaka::hierarchy::Blocks{});
          }
          continue;  // shared path owns this track; skip the fallback
        }
        // Diagnostic: this track survived the shared path only because the cov gate rejected every better
        // co-occurring partner. Counted per track, region by the track's own eta. Slots 8/9 = central/forward.
        if (diag && sawGateMiss) {
          const int missRegion = (std::abs(etaI) > kDedupFwdEta) ? 1 : 0;
          alpaka::atomicAdd(acc, &diag[8 + missRegion], 1u, alpaka::hierarchy::Blocks{});
        }

        // 0-shared forward fallback, active when the eta-phi binner is present
        if (etaPhiAssoc != nullptr) {
          const float phiI = ::reco::phi(tracks_view, i);
          const int pb = trackBinKey(0.f, phiI, kDedupFbPhiBins, 1, 0.f);
          const int eb =
              (trackBinKey(etaI, phiI, kDedupFbPhiBins, kDedupFbEtaSlabs, kDedupFbEtaMax) - pb) / kDedupFbPhiBins;
          bool fbLoser = false;
          int32_t fbPartner = -1;
          for (int de = -fbEtaReach; de <= fbEtaReach; ++de) {
            const int e = eb + de;
            if (e < 0 || e >= kDedupFbEtaSlabs)
              continue;
            for (int dp2 = -fbPhiReach; dp2 <= fbPhiReach; ++dp2) {
              const int b = (pb + dp2 + kDedupFbPhiBins) % kDedupFbPhiBins;
              const uint32_t bin = uint32_t(e * kDedupFbPhiBins + b);
              for (auto p = etaPhiAssoc->begin(bin); p != etaPhiAssoc->end(bin); ++p) {
                const int32_t j = int32_t(*p);
                if (j == i)
                  continue;
                if (!jBeatsI(j))
                  continue;
                if (!compatible(j, fbNSigma2))  // fallback path: fallback-only cov-gate width
                  continue;
                if (sharedCount(j) != 0)
                  continue;  // shared>=1 is the co-occurrence path's job; fallback is 0-shared only
                fbLoser = true;
                if (fbPartner < 0 || beats(j, fbPartner))
                  fbPartner = j;
              }
            }
          }
          if (fbLoser) {
            // Drop authority bounded to |eta| <= fbDropAbsEtaMax (the runtime image of
            // kDedupFbDropAbsEtaMax), which keeps the fallback out of the far-forward region where it costs
            // efficiency. fbEnable == 0 disables the drop authority; in both cases the candidate is still
            // counted in diag but never dropped.
            const bool inDropRegion = (etaI <= fbDropAbsEtaMax) && (etaI >= -fbDropAbsEtaMax);
            const bool legacyDrop = inDropRegion && (fbEnable != 0);
            // capture-for-confirm: with fbConfirm on, a droppable pair is deferred to the union-refit
            // verdict instead of dropped here, unless a pre-gate below rejects it outright (kept both,
            // diag-counted). With fbConfirm == 0 the whole block is skipped.
            bool captured = false;
            if (fbConfirm != 0 && contestedPairs != nullptr && legacyDrop) {
              bool preGateOk = true;
              // (a) same-charge requirement (charge = sign of signed q/pT = state()[2]).
              if (fbSameCharge != 0) {
                const float qi = tracks_view[i].state()[2];
                const float qj = tracks_view[fbPartner].state()[2];
                if ((qi >= 0.f) != (qj >= 0.f))
                  preGateOk = false;
              }
              // (b) absolute-floor box cuts on {|dphi| (wrapped), |d q/pT|, |d cotTheta|}; the sentinel
              // defaults never fire. They catch a large absolute gap passing the shared-cov Mahalanobis gate.
              if (preGateOk) {
                float dphi = tracks_view[i].state()[0] - tracks_view[fbPartner].state()[0];
                while (dphi > kTwinPi)
                  dphi -= kTwinTwoPi;
                while (dphi < -kTwinPi)
                  dphi += kTwinTwoPi;
                const float dqop = tracks_view[i].state()[2] - tracks_view[fbPartner].state()[2];
                const float dcot = tracks_view[i].state()[3] - tracks_view[fbPartner].state()[3];
                if (std::abs(dphi) > fbAbsFloorDPhi || std::abs(dqop) > fbAbsFloorDQoP ||
                    std::abs(dcot) > fbAbsFloorDCotTheta)
                  preGateOk = false;
              }
              if (preGateOk) {
                const uint32_t slot = alpaka::atomicAdd(acc, contestedCount, 1u, alpaka::hierarchy::Blocks{});
                if (slot < contestedCap) {
                  contestedPairs[2u * slot + 0u] = uint32_t(i);
                  contestedPairs[2u * slot + 1u] = uint32_t(fbPartner);
                  captured = true;
                  if (diag)
                    alpaka::atomicAdd(acc, &diag[10], 1u, alpaka::hierarchy::Blocks{});  // contested captured
                } else if (contestedOverflow != nullptr) {
                  // list full: pair is kept both (safe) and surfaced via LogWarning by the launcher.
                  alpaka::atomicAdd(acc, contestedOverflow, 1u, alpaka::hierarchy::Blocks{});
                }
              } else if (diag) {
                alpaka::atomicAdd(acc, &diag[11], 1u, alpaka::hierarchy::Blocks{});  // pre-gate rejected (kept both)
              }
            }
            // Immediate drop authority: with fbConfirm == 0 this is exactly legacyDrop; with fbConfirm != 0
            // no pair drops here, confirmed drops being applied later by the verdict kernel.
            (void)captured;
            // Finder mode (fbFinderOnly) scans and diag-counts but drops nothing, with its own bucket [16].
            const bool dropping = legacyDrop && (fbConfirm == 0) && (fbFinderOnly == 0);
            if (dropping)
              drop[i] = 1;  // out-of-bound / fallback-disabled / finder-mode candidates count but do not drop
            if (diag) {
              const bool fwd =
                  (std::abs(etaI) > kDedupFwdEta) || (std::abs(tracks_view[fbPartner].eta()) > kDedupFwdEta);
              const int region = fwd ? 1 : 0;
              alpaka::atomicAdd(acc, &diag[region * 3 + 0], 1u, alpaka::hierarchy::Blocks{});  // bucket 0 = 0-shared
              if (dropping)  // totals = ACTUAL drops only
                alpaka::atomicAdd(acc, &diag[6 + region], 1u, alpaka::hierarchy::Blocks{});
              if (fbFinderOnly != 0 && legacyDrop)  // finder mode suppressed a drop this pair would take
                alpaka::atomicAdd(acc, &diag[16], 1u, alpaka::hierarchy::Blocks{});
            }
          }
        }
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_countHitInTracks {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  HitToTuple *hitToTuple,
                                  uint32_t nHits) const {  // OT extras bin at nHits + otIdx
      const auto nKeys = hitToTuple->nOnes();
      for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(idx) == 0)
          break;  // guard
        for (auto h = foundNtuplets->begin(idx); h != foundNtuplets->end(idx); ++h) {
          auto const key = hitToTupleKey(*h, nHits);
          // Key-range guard: a hitContainer content overflow leaves unwritten hit ids in the CSR, so the
          // key can land outside [0, nOnes). Drop instead of writing outside off[]; the fill pass below
          // skips exactly the same keys and counts the drop.
          if (key < nKeys)
            hitToTuple->count(acc, key);
        }
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_fillHitInTracks {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  HitToTuple *hitToTuple,
                                  uint32_t nHits,
                                  Counters *counters) const {  // OT extras bin at nHits + otIdx
      const auto nKeys = hitToTuple->nOnes();
      for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(idx) == 0)
          break;  // guard
        for (auto h = foundNtuplets->begin(idx); h != foundNtuplets->end(idx); ++h) {
          auto const key = hitToTupleKey(*h, nHits);
          // Key-range guard mirroring the count pass; the drop is counted here, once per lost association.
          if (key < nKeys)
            hitToTuple->fill(acc, key, idx);
          else
            alpaka::atomicAdd(acc, &counters->nHitToTupleOverflow, 1ull, alpaka::hierarchy::Blocks{});
        }
      }
    }
  };

  // Content-buffer overflow repair, paired with the truncating bulkFill in OneToManyAssoc.h. When a tuple's
  // hit block does not fit the hit container, bulkFill plugs the tuple's offset but writes no content, so
  // after bulkFinalize the CSR describes blocks that were never written. Run right after bulkFinalize and
  // before anything walks tuple content, these two kernels clamp every offset to the start of the first
  // overflowed tuple: that tuple and all later ones become empty, which every walker already treats as no
  // more tuples. No-op when nothing overflowed (both words stay at their 0xFFFFFFFF memset value).
  // Exactly one thread finds the boundary tuple k (off[k] <= capacity < off[k+1]). clampInfo[0] = off[k],
  // clampInfo[1] = k, the first dropped tuple, which Kernel_fillHitDetIndices uses to cut nTracks so that no
  // empty slot is ever published (the SoA->legacy converter asserts nHits >= 3 on every published slot).
  class Kernel_findTupleContentOverflow {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  uint32_t *__restrict__ clampInfo) const {
      const uint32_t cap = uint32_t(foundNtuplets->capacity());
      const uint32_t nOff = uint32_t(foundNtuplets->totOnes());  // off[] has nOnes()+1 entries
      for (auto k : cms::alpakatools::uniform_elements(acc, nOff - 1)) {
        if (foundNtuplets->off[k] <= cap && foundNtuplets->off[k + 1] > cap) {
          clampInfo[0] = foundNtuplets->off[k];
          clampInfo[1] = uint32_t(k);
        }
      }
    }
  };

  class Kernel_clampTupleContentOverflow {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitContainer *__restrict__ foundNtuplets,
                                  uint32_t const *__restrict__ clampInfo) const {
      const uint32_t bound = clampInfo[0];
      for (auto j : cms::alpakatools::uniform_elements(acc, uint32_t(foundNtuplets->totOnes()))) {
        if (foundNtuplets->off[j] > bound)
          foundNtuplets->off[j] = bound;
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_fillHitDetIndices {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  TkHitSoAView track_hits_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  HitsConstView hh,
                                  cms::alpakatools::AtomicPairCounter *apc,
                                  uint32_t const *__restrict__ tupleClampInfo) const {
      // clamp the number of tracks to the capacity of the SoA
      auto ntracks = std::min<int>(apc->get().first, tracks_view.metadata().size() - 1);
      // ... and to the first tuple dropped by a content-buffer overflow: the slots from that tuple on are
      // empty after the repair and must not be published (0xFFFFFFFF = nothing overflowed, no cut).
      if (tupleClampInfo[1] < uint32_t(ntracks))
        ntracks = int(tupleClampInfo[1]);
      if (cms::alpakatools::once_per_grid(acc))
        tracks_view.nTracks() = ntracks;

      // copy offsets, clamped to the hit SoA capacity: on a content-buffer overflow the raw offset can
      // exceed what the copy loop below writes, and a CSR end past the copied region would make downstream
      // hit walks read unwritten rows (the offset for track 0 is always 0).
      const uint32_t hitRowCap = uint32_t(track_hits_view.metadata().size());
      for (auto idx : cms::alpakatools::uniform_elements(acc, ntracks)) {
        tracks_view[idx].hitOffsets() = std::min(foundNtuplets->off[idx + 1], hitRowCap);
        tracks_view[idx].ndof() = 0;  // stamped by the fit for fitted tuples
      }
      // Tail: the slots past the last track carry its CSR end offset, so nHits() reads zero there and a
      // reader that walks the SoA up to the first empty slot stops at the right place.
      const uint32_t hitEndTail = std::min(foundNtuplets->off[ntracks], hitRowCap);
      for (auto idx : cms::alpakatools::uniform_elements(acc, uint32_t(tracks_view.metadata().size())))
        if (int(idx) >= ntracks)
          tracks_view[idx].hitOffsets() = hitEndTail;
      // fill hit indices, clamped to the hit SoA capacity: foundNtuplets->size() is the AtomicPairCounter's
      // hits-in-tracks total, which on a tuple overflow exceeds what was actually written, so an unclamped
      // loop would read the container beyond its filled region.
      const uint32_t nHitsInTracks = std::min<uint32_t>(foundNtuplets->size(), track_hits_view.metadata().size());
      for (auto idx : cms::alpakatools::uniform_elements(acc, nHitsInTracks)) {
        // On content-buffer overflow the content is unwritten, so the hit index can be out of range; the
        // hit was already dropped and writing a garbage detId would corrupt.
        if (foundNtuplets->content[idx] >= (uint32_t)hh.metadata().size())
          continue;
        track_hits_view[idx].id() = foundNtuplets->content[idx];
        track_hits_view[idx].detId() = hh[foundNtuplets->content[idx]].detectorIndex();
        track_hits_view[idx].attached() = 0;  // CA-found; the extension stage flags its own additions
#ifdef CA_DEBUG
        printf("Kernel_fillHitDetIndices %d %d %d \n",
               idx,
               foundNtuplets->content[idx],
               track_hits_view.metadata().size());
#endif
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_doStatsForHitInTracks {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitToTuple const *__restrict__ hitToTuple,
                                  Counters *counters) const {
      auto &c = *counters;
      for (auto idx : cms::alpakatools::uniform_elements(acc, hitToTuple->nOnes())) {
        if (hitToTuple->size(idx) == 0)
          continue;  // SHALL NOT BE break
        alpaka::atomicAdd(acc, &c.nUsedHits, 1ull, alpaka::hierarchy::Blocks{});
        if (hitToTuple->size(idx) > 1)
          alpaka::atomicAdd(acc, &c.nDupHits, 1ull, alpaka::hierarchy::Blocks{});
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_countSharedHit {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  int *__restrict__ nshared,
                                  HitContainer const *__restrict__ ptuples,
                                  Quality const *__restrict__ quality,
                                  HitToTuple const *__restrict__ phitToTuple) const {
      constexpr auto loose = Quality::loose;

      auto &hitToTuple = *phitToTuple;
      auto const &foundNtuplets = *ptuples;
      for (auto idx : cms::alpakatools::uniform_elements(acc, hitToTuple.nOnes())) {
        if (hitToTuple.size(idx) < 2)
          continue;

        int nt = 0;

        // count "good" tracks
        for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
          if (quality[*it] < loose)
            continue;
          ++nt;
        }

        if (nt < 2)
          continue;

        // now mark  each track triplet as sharing a hit
        for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
          if (foundNtuplets.size(*it) > 3)
            continue;
          alpaka::atomicAdd(acc, &nshared[*it], 1, alpaka::hierarchy::Blocks{});
        }

      }  //  hit loop
    }
  };

  template <typename TrackerTraits>
  class Kernel_markSharedHit {
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  int const *__restrict__ nshared,
                                  HitContainer const *__restrict__ tuples,
                                  Quality *__restrict__ quality,
                                  bool dupPassThrough) const {
      // constexpr auto bad = Quality::bad;
      constexpr auto dup = Quality::dup;
      constexpr auto loose = Quality::loose;
      // constexpr auto strict = Quality::strict;

      // quality to mark rejected
      auto const reject = dupPassThrough ? loose : dup;
      for (auto idx : cms::alpakatools::uniform_elements(acc, tuples->nOnes())) {
        if (tuples->size(idx) == 0)
          break;  //guard
        if (quality[idx] <= reject)
          continue;
        if (nshared[idx] > 2)
          quality[idx] = reject;
      }
    }
  };

  // Track-parallel single-writer shared-hit removers: each thread owns one track, inspects the hit
  // buckets it belongs to (hitToTuple), reads every quality from the frozen scratch snapshot, and writes
  // only its own track's quality()
  template <typename TrackerTraits>
  class Kernel_rejectDuplicate {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  int32_t const *__restrict__ qualityScratch,
                                  HitToTuple const *__restrict__ phitToTuple,
                                  float fastDupNSigma2) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;

      auto &hitToTuple = *phitToTuple;
      auto qual = [&](uint32_t t) { return static_cast<Quality>(qualityScratch[t]); };
      auto score = [&](uint32_t it) { return tracks_view[it].chi2(); };

      // A track is rejected iff some compatible track sharing one of its hits is strictly better by
      // the total order (more layers, then higher quality, then lower chi2, then lower track index)
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(it) == 0)
          break;  // guard
        auto const qi = qual(it);
        if (qi <= reject)
          continue;
        auto const nli = tracks_view[it].nLayers();
        // Phase2OTStubs only: the duplicate winner ordering inserts the total hit count as a tie-break
        // between nLayers and quality (max nLayers -> max total hits -> max quality -> min chi2 -> min
        // index). reco::nHits() is the track's full CSR hit extent, so it separates duplicates that tie on
        // nLayers, where otherwise a pixel-rich prompt track could beat its OT-rich displaced twin on the
        // chi2 tie-break and lose its TID hits. Off on every other topology.
        constexpr bool kUseHitCountTieBreak = std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>;
        const uint32_t nhi = kUseHitCountTieBreak ? ::reco::nHits(tracks_view, it) : 0u;

        // get track parameters and covariances
        float iParams[nTrackParameters];
        float iCovs[nTrackParameters];
        for (int p{0}; p < nTrackParameters; ++p) {
          iParams[p] = tracks_view[it].state()(p);
          iCovs[p] = tracks_view[it].covariance()(iParam2iCov[p]);
        }
        auto incompatibleTrackParams = [&](uint32_t jt) -> bool {
          for (int p{0}; p < nTrackParameters; ++p) {
            const auto dpij = iParams[p] - tracks_view[jt].state()(p);
            const auto e2dpij = fastDupNSigma2 * (iCovs[p] + tracks_view[jt].covariance()(iParam2iCov[p]));
            if (dpij * dpij > e2dpij)
              return true;
          }
          return false;
        };

        bool dominated = false;
        for (auto hp = foundNtuplets->begin(it); hp != foundNtuplets->end(it) && !dominated; ++hp) {
          auto const h = *hp;
          if (h >= hitToTuple.nOnes())
            continue;  // key-range guard (hitContainer content overflow)
          for (auto jp = hitToTuple.begin(h); jp != hitToTuple.end(h); ++jp) {
            auto const jt = *jp;
            if (jt == it)
              continue;
            auto const qj = qual(jt);
            if (qj <= reject)
              continue;
            if (incompatibleTrackParams(jt))
              continue;
            auto const nlj = tracks_view[jt].nLayers();
            // jt dominates it by the total order (nLayers, [total hits], quality, score, then track index).
            // The score test stays a strict order even for a NaN score, so exactly one of a pair is demoted
            bool jBetterTail =
                (qj > qi || (qj == qi && (score(jt) < score(it) || (!(score(it) < score(jt)) && jt < it))));
            bool jBetter;
            if constexpr (kUseHitCountTieBreak) {
              const uint32_t nhj = ::reco::nHits(tracks_view, jt);
              jBetter = (nlj > nli) || (nlj == nli && (nhj > nhi || (nhj == nhi && jBetterTail)));
            } else {
              jBetter = (nlj > nli) || (nlj == nli && jBetterTail);
            }
            if (jBetter) {
              dominated = true;
              break;
            }
          }
        }
        if (dominated)
          tracks_view[it].quality() = reject;
      }
    }
  };

  // Phase-1 specialization (very forward triplets)
  template <>
  class Kernel_rejectDuplicate<pixelTopology::Phase1> {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  int32_t const *__restrict__ qualityScratch,
                                  HitToTuple const *__restrict__ phitToTuple,
                                  float fastDupNSigma2) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;

      auto &hitToTuple = *phitToTuple;
      auto qual = [&](uint32_t t) { return static_cast<Quality>(qualityScratch[t]); };
      auto score = [&](uint32_t it) { return std::abs(reco::tip(tracks_view, it)); };

      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(it) == 0)
          break;  // guard
        auto const qi = qual(it);
        if (qi <= reject)
          continue;
        auto const opi = tracks_view[it].state()(2);
        auto const e2opi = tracks_view[it].covariance()(9);
        auto const cti = tracks_view[it].state()(3);
        auto const e2cti = tracks_view[it].covariance()(12);
        auto const nli = tracks_view[it].nLayers();

        bool dominated = false;
        for (auto hp = foundNtuplets->begin(it); hp != foundNtuplets->end(it) && !dominated; ++hp) {
          auto const h = *hp;
          if (h >= hitToTuple.nOnes())
            continue;  // key-range guard (hitContainer content overflow)
          for (auto jp = hitToTuple.begin(h); jp != hitToTuple.end(h); ++jp) {
            auto const jt = *jp;
            if (jt == it)
              continue;
            auto const qj = qual(jt);
            if (qj <= reject)
              continue;
            auto const opj = tracks_view[jt].state()(2);
            auto const ctj = tracks_view[jt].state()(3);
            auto const dct = nSigma2Phase1 * (tracks_view[jt].covariance()(12) + e2cti);
            if ((cti - ctj) * (cti - ctj) > dct)
              continue;
            auto const dop = nSigma2Phase1 * (tracks_view[jt].covariance()(9) + e2opi);
            if ((opi - opj) * (opi - opj) > dop)
              continue;
            auto const nlj = tracks_view[jt].nLayers();
            // jt dominates it by the total order (nLayers, quality, score, then track index). The score
            // test stays a strict order even for a non-finite score (NaN), so exactly one of a duplicate
            // pair is always demoted
            bool jBetter =
                (nlj > nli) ||
                (nlj == nli &&
                 (qj > qi || (qj == qi && (score(jt) < score(it) || (!(score(it) < score(jt)) && jt < it)))));
            if (jBetter) {
              dominated = true;
              break;
            }
          }
        }
        if (dominated)
          tracks_view[it].quality() = reject;
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_sharedHitCleaner {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsConstView hh,
                                  uint32_t const *__restrict__ layerStarts,
                                  TkSoAView tracks_view,
                                  int nmin,
                                  bool dupPassThrough,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  int32_t const *__restrict__ qualityScratch,
                                  HitToTuple const *__restrict__ phitToTuple) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;
      // quality of longest track
      auto const longTqual = Quality::highPurity;

      auto &hitToTuple = *phitToTuple;
      auto qual = [&](uint32_t t) { return static_cast<Quality>(qualityScratch[t]); };
      uint32_t l1end = layerStarts[1];

      // Short track `it` (nLayers <= nmin) is killed if it shares a non-bpix1 hit with a longer track
      // (nLayers == maxNl >= 4 among the highPurity tracks of that hit). maxNl is a reduction over the
      // frozen snapshot, so this is order-independent
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(it) == 0)
          break;  // guard
        if (qual(it) <= reject)
          continue;
        auto const nlit = tracks_view[it].nLayers();
        if (nlit > nmin)
          continue;  // only short tracks are cleaned here

        bool kill = false;
        for (auto hp = foundNtuplets->begin(it); hp != foundNtuplets->end(it) && !kill; ++hp) {
          auto const h = *hp;
          if (h < l1end)
            continue;  // shared hit on bpix1
          if (h >= hitToTuple.nOnes())
            continue;  // key-range guard (hitContainer content overflow)
          int8_t maxNl = 0;
          if (hitToTuple.size(h) >= 2) {
            for (auto jp = hitToTuple.begin(h); jp != hitToTuple.end(h); ++jp) {
              if (qual(*jp) < longTqual)
                continue;
              maxNl = std::max(tracks_view[*jp].nLayers(), maxNl);
            }
          }

          // For Phase2OTStubs: several stubs sharing a lowerHitIdx have different hit indices but stand for
          // the same physical measurement, so for cleaning purposes they count as the same shared hit.
          if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
            if (h < static_cast<uint32_t>(hh.metadata().size()) && isStub(hh, h)) {
              auto const lowerHitIdx = hh[h].lowerHitIdx();
              if (lowerHitIdx != std::numeric_limits<uint32_t>::max()) {
                auto const offsetStubs = hh.offsetStubs();
                auto const nHits = static_cast<uint32_t>(hh.metadata().size());
                for (uint32_t otherIdx = offsetStubs; otherIdx < nHits; ++otherIdx) {
                  if (otherIdx == h)
                    continue;
                  if (otherIdx >= hitToTuple.nOnes())
                    continue;
                  if (!isStub(hh, otherIdx))
                    continue;
                  if (hh[otherIdx].lowerHitIdx() != lowerHitIdx)
                    continue;
                  for (auto jp = hitToTuple.begin(otherIdx); jp != hitToTuple.end(otherIdx); ++jp) {
                    if (qual(*jp) < longTqual)
                      continue;
                    maxNl = std::max(tracks_view[*jp].nLayers(), maxNl);
                  }
                }
              }
            }
          }

          if (maxNl >= 4 && nlit < maxNl)
            kill = true;
        }
        if (kill)
          tracks_view[it].quality() = reject;
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_tripletCleaner {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  int32_t const *__restrict__ qualityScratch,
                                  HitToTuple const *__restrict__ phitToTuple) const {
      // quality to mark rejected
      auto const reject = Quality::loose;
      /// min quality of good
      auto const good = Quality::strict;

      auto &hitToTuple = *phitToTuple;
      auto qual = [&](uint32_t t) { return static_cast<Quality>(qualityScratch[t]); };

      // Track `it` is rejected if, on one of its shared hits whose good-quality tracks are all
      // triplets, it is not the best-tip survivor (lower track index breaks ties)
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(it) == 0)
          break;  // guard
        if (qual(it) <= reject)
          continue;

        bool kill = false;
        for (auto hp = foundNtuplets->begin(it); hp != foundNtuplets->end(it) && !kill; ++hp) {
          auto const h = *hp;
          if (h >= hitToTuple.nOnes())
            continue;  // key-range guard (hitContainer content overflow)
          if (hitToTuple.size(h) < 2)
            continue;
          bool onlyTriplets = true;
          for (auto jp = hitToTuple.begin(h); jp != hitToTuple.end(h); ++jp) {
            if (qual(*jp) <= good)
              continue;
            onlyTriplets &= reco::isTriplet(tracks_view, *jp);
            if (!onlyTriplets)
              break;
          }
          if (!onlyTriplets)
            continue;
          float mc = maxScore;
          uint32_t im = tkNotFound;
          for (auto jp = hitToTuple.begin(h); jp != hitToTuple.end(h); ++jp) {
            auto const jt = *jp;
            if (qual(jt) >= good) {
              auto const t = std::abs(reco::tip(tracks_view, jt));
              if (t < mc || (t == mc && jt < im)) {
                mc = t;
                im = jt;
              }
            }
          }
          if (im != tkNotFound && it != im)
            kill = true;
        }
        if (kill)
          tracks_view[it].quality() = reject;
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_simpleTripletCleaner {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  int32_t const *__restrict__ qualityScratch,
                                  HitToTuple const *__restrict__ phitToTuple) const {
      // quality to mark rejected
      auto const reject = Quality::loose;
      /// min quality of good
      auto const good = Quality::loose;

      auto &hitToTuple = *phitToTuple;
      auto qual = [&](uint32_t t) { return static_cast<Quality>(qualityScratch[t]); };

      // Triplet `it` is rejected if, on one of its shared hits, it is not the best-tip survivor
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(it) == 0)
          break;  // guard
        if (qual(it) <= reject || !reco::isTriplet(tracks_view, it))
          continue;

        bool kill = false;
        for (auto hp = foundNtuplets->begin(it); hp != foundNtuplets->end(it) && !kill; ++hp) {
          auto const h = *hp;
          if (h >= hitToTuple.nOnes())
            continue;  // key-range guard (hitContainer content overflow)
          if (hitToTuple.size(h) < 2)
            continue;
          float mc = maxScore;
          uint32_t im = tkNotFound;
          for (auto jp = hitToTuple.begin(h); jp != hitToTuple.end(h); ++jp) {
            auto const jt = *jp;
            if (qual(jt) >= good) {
              auto const t = std::abs(reco::tip(tracks_view, jt));
              if (t < mc || (t == mc && jt < im)) {
                mc = t;
                im = jt;
              }
            }
          }
          if (im != tkNotFound && it != im)
            kill = true;
        }
        if (kill)
          tracks_view[it].quality() = reject;
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_print_found_ntuplets {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  HitsConstView hh,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  HitToTuple const *__restrict__ phitToTuple,
                                  uint32_t firstPrint,
                                  uint32_t lastPrint,
                                  int iev) const {
      constexpr auto loose = Quality::loose;

      for (auto i : cms::alpakatools::uniform_elements(acc, firstPrint, std::min(lastPrint, foundNtuplets->nOnes()))) {
        auto nh = foundNtuplets->size(i);
        if (nh < 3)
          continue;
        if (tracks_view[i].quality() < loose)
          continue;
        printf("TK: %d %d %d %d %f %f %f %f %f %f %f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
               10000 * iev + i,
               int(tracks_view[i].quality()),
               nh,
               tracks_view[i].nLayers(),
               reco::charge(tracks_view, i),
               tracks_view[i].pt(),
               tracks_view[i].eta(),
               reco::phi(tracks_view, i),
               reco::tip(tracks_view, i),
               reco::zip(tracks_view, i),
               tracks_view[i].chi2(),
               hh[*foundNtuplets->begin(i)].zGlobal(),
               hh[*(foundNtuplets->begin(i) + 1)].zGlobal(),
               hh[*(foundNtuplets->begin(i) + 2)].zGlobal(),
               nh > 3 ? hh[int(*(foundNtuplets->begin(i) + 3))].zGlobal() : 0,
               nh > 4 ? hh[int(*(foundNtuplets->begin(i) + 4))].zGlobal() : 0,
               nh > 5 ? hh[int(*(foundNtuplets->begin(i) + 5))].zGlobal() : 0,
               nh > 6 ? hh[int(*(foundNtuplets->begin(i) + nh - 1))].zGlobal() : 0);
      }
    }
  };

  class Kernel_printCounters {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc, Counters const *counters) const {
      auto const &c = *counters;
      printf(
          "||Counters | nEvents | nHits | nCells | nTuples | nFitTacks  |  nLooseTracks  |  nGoodTracks | nUsedHits | "
          "nDupHits | nFishCells | nKilledCells | nUsedCells | nZeroTrackCells ||\n");
      printf("Counters Raw %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld\n",
             c.nEvents,
             c.nHits,
             c.nCells,
             c.nTuples,
             c.nFitTracks,
             c.nLooseTracks,
             c.nGoodTracks,
             c.nUsedHits,
             c.nDupHits,
             c.nFishCells,
             c.nKilledCells,
             c.nEmptyCells,
             c.nZeroTrackCells);
      printf(
          "Counters Norm %lld ||  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.3f|  %.3f|  %.3f|  "
          "%.3f||\n",
          c.nEvents,
          c.nHits / double(c.nEvents),
          c.nCells / double(c.nEvents),
          c.nTuples / double(c.nEvents),
          c.nFitTracks / double(c.nEvents),
          c.nLooseTracks / double(c.nEvents),
          c.nGoodTracks / double(c.nEvents),
          c.nUsedHits / double(c.nEvents),
          c.nDupHits / double(c.nEvents),
          c.nFishCells / double(c.nCells),
          c.nKilledCells / double(c.nCells),
          c.nEmptyCells / double(c.nCells),
          c.nZeroTrackCells / double(c.nCells));
      printf(
          "Counters Overflow %lld ||  tupleOvf=%lld  cellOvf=%lld  tripletOvf=%lld  "
          "cellTrackOvf=%lld  hitToTupleOvf=%lld  hitToCellOvf=%lld ||\n",
          c.nEvents,
          c.nTupleOverflow,
          c.nCellOverflow,
          c.nTripletOverflow,
          c.nCellTrackOverflow,
          c.nHitToTupleOverflow,
          c.nHitToCellOverflow);
    }
  };
  // Merger gather/compact kernel. Single device-side gather: reads each input's actual nTracks() and last
  // hitOffsets() on device (no host readback) and compacts all track columns and trackHits columns (id,
  // detId, attached) of every input into a dense merged layout, shifting each input's hitOffsets by the
  // cumulative hit count of the inputs before it. Every thread recomputes the per-input cumulative track and
  // hit offsets independently (Phase 1). The per-track arm labels are passed as two scalars (arm0, arm1), so
  // no device copy of the host arm vector is needed.
  //
  // The output SoA is capacity-sized, so the tail [mergedNTracks, capacity) holds whatever the allocator
  // handed out. The kernel stamps Quality::bad over quality() there, which makes the downstream filterTracks
  // kernel (it iterates metadata().size(), not nTracks()) skip every tail slot; only the quality column
  // needs it, since filterTracks touches no other column when the gate fails.
  class Kernel_mergeGather {
  public:
    // Tie the eigen column element counts used by the copy below (5 for state, 15 for covariance) to the
    // layout, so resizing either column breaks the build here instead of mis-striding the copy.
    static_assert(::reco::Vector5f::RowsAtCompileTime == 5 && ::reco::Vector15f::RowsAtCompileTime == 15,
                  "the eigen columns of reco::TrackLayout changed size: the element count this copy "
                  "starts from and the step it adds after each eigen column must be updated together");
    // Tie the hardcoded column set below to the SoA layouts: adding or removing a column fails the build
    // here instead of leaving the new column silently uncopied.
    // TrackLayout (Phase 2 copy): quality, chi2, nLayers, eta, pt, state[5], covariance[15], hitOffsets,
    //   iteration, ndof = 10 columns, plus the nTracks scalar written in Phase 1.
    // TrackHitsLayout (Phase 3 copy): id, detId, attached = 3 columns.
    static_assert(::reco::TrackSoA::Descriptor::num_cols == 11,
                  "reco::TrackLayout column count changed: update the gather kernel's track column "
                  "enumeration in Kernel_mergeGather (quality, chi2, nLayers, eta, pt, state[5], "
                  "covariance[15], nTracks, hitOffsets, iteration, ndof)");
    static_assert(::reco::TrackHitSoA::Descriptor::num_cols == 3,
                  "reco::TrackHitsLayout column count changed: update the gather kernel's trackHits "
                  "column enumeration in Kernel_mergeGather (id, detId, attached)");

    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  ::reco::TrackSoAView outTrack_view,
                                  ::reco::TrackHitSoAView outHit_view,
                                  const ::reco::TrackSoAConstView inp0Track_view,
                                  const ::reco::TrackHitSoAConstView inp0Hit_view,
                                  const ::reco::TrackSoAConstView inp1Track_view,
                                  const ::reco::TrackHitSoAConstView inp1Hit_view,
                                  const int nInputs,
                                  int32_t *armBuf,
                                  const int32_t arm0,
                                  const int32_t arm1) const {
      // Phase 1: all threads read device-side nTracks and hitOffsets and compute the cumulative offsets
      // independently; grid thread 0 also writes the merged nTracks scalar.
      uint32_t nTks[2] = {0, 0};
      uint32_t cumulTks[3] = {0, 0, 0};
      uint32_t cumulHits[3] = {0, 0, 0};

      // The two inputs are passed as separate view arguments and the arrays above are sized for exactly two;
      // the host refuses more (PixelTracksSoAMerger throws) and the clamp keeps the loops inside the arrays.
      const int nInp = (nInputs < 2) ? nInputs : 2;

      for (int s = 0; s < nInp; ++s) {
        const uint32_t ntk = (s == 0) ? uint32_t(inp0Track_view.nTracks()) : uint32_t(inp1Track_view.nTracks());
        nTks[s] = ntk;
        cumulTks[s + 1] = cumulTks[s] + ntk;
        // Total hits for this input = last filled hitOffsets (CSR cumulative hit-end); 0 when ntk == 0.
        uint32_t totHitsS = 0;
        if (ntk > 0) {
          totHitsS = (s == 0) ? uint32_t(inp0Track_view[ntk - 1].hitOffsets())
                              : uint32_t(inp1Track_view[ntk - 1].hitOffsets());
        }
        cumulHits[s + 1] = cumulHits[s] + totHitsS;
      }

      const uint32_t outCap = uint32_t(outTrack_view.metadata().size());
      const uint32_t outHitCap = uint32_t(outHit_view.metadata().size());
      // Clamp the merged count to the output track capacity, the track-side twin of the hit-side clamp in
      // Phase 3. The output covers the sum of the inputs, so this cannot bind; it exists so that a capacity
      // mismatch degrades by dropping the excess instead of writing past the allocation.
      const uint32_t mergedNTracks = (cumulTks[nInp] < outCap) ? cumulTks[nInp] : outCap;

      if (alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0] == 0) {
        outTrack_view.nTracks() = mergedNTracks;
      }

      // Phase 2: all threads cooperate on a grid-stride copy of every track column.
      // The per-input copy range is [cumulTks[s], cumulTks[s+1]) in the output, [0, nTks[s]) in the input.
      for (int s = 0; s < nInp; ++s) {
        auto inpTrack_view = (s == 0) ? inp0Track_view : inp1Track_view;
        const uint32_t ntk = nTks[s];
        const uint32_t outBase = cumulTks[s];
        const uint32_t hitShift = cumulHits[s];
        const int32_t armLabel = (s == 0) ? arm0 : arm1;
        // Same clamp as above, applied to this input's slice of the output.
        const uint32_t ntkCopy = (outBase >= outCap) ? 0u : ((ntk < outCap - outBase) ? ntk : (outCap - outBase));

        for (uint32_t i : cms::alpakatools::uniform_elements(acc, ntkCopy)) {
          const uint32_t outIdx = outBase + i;
          // Copy every track column enumerated in the static_asserts above.
          outTrack_view[outIdx].quality() = inpTrack_view[i].quality();
          outTrack_view[outIdx].chi2() = inpTrack_view[i].chi2();
          outTrack_view[outIdx].ndof() = inpTrack_view[i].ndof();
          outTrack_view[outIdx].nLayers() = inpTrack_view[i].nLayers();
          outTrack_view[outIdx].eta() = inpTrack_view[i].eta();
          outTrack_view[outIdx].pt() = inpTrack_view[i].pt();
          for (uint32_t k = 0; k < 5; ++k)
            outTrack_view[outIdx].state()[k] = inpTrack_view[i].state()[k];
          for (uint32_t k = 0; k < 15; ++k)
            outTrack_view[outIdx].covariance()[k] = inpTrack_view[i].covariance()[k];
          outTrack_view[outIdx].iteration() = inpTrack_view[i].iteration();
          // Shift hitOffsets by the cumulative hit count of the previous inputs, clamped to the output hit
          // capacity so the CSR end offsets stay inside the block the hit copy below is clamped to.
          const uint32_t shifted = uint32_t(inpTrack_view[i].hitOffsets()) + hitShift;
          outTrack_view[outIdx].hitOffsets() = (shifted < outHitCap) ? shifted : outHitCap;
          // arm-label buffer, in the dense merged-SoA ordering of the track copy
          if (armBuf)
            armBuf[outIdx] = armLabel;
        }
      }

      // Phase 3: copy trackHits columns (id, detId, attached) for each input.
      for (int s = 0; s < nInp; ++s) {
        auto inpHit_view = (s == 0) ? inp0Hit_view : inp1Hit_view;
        auto inpTrack_view = (s == 0) ? inp0Track_view : inp1Track_view;
        const uint32_t ntk = nTks[s];
        const uint32_t outHitBase = cumulHits[s];
        // Total hits for this input (recomputed from the CSR end, same as Phase 1).
        uint32_t totHitsS = 0;
        if (ntk > 0)
          totHitsS = uint32_t(inpTrack_view[ntk - 1].hitOffsets());
        // Truncate the copy at the output hit capacity. The output is sized to the sum of the inputs'
        // hit-block capacities, so this binds only if an input's own hit total ran past its block.
        if (outHitBase >= outHitCap) {
          if (cms::alpakatools::once_per_grid(acc) && totHitsS > 0)
            printf("Warning!!!! mergeGather: input %d hit copy dropped entirely (base %u >= capacity %u)!\n",
                   s,
                   outHitBase,
                   outHitCap);
          totHitsS = 0;
        } else if (outHitBase + totHitsS > outHitCap) {
          if (cms::alpakatools::once_per_grid(acc))
            printf("Warning!!!! mergeGather: input %d hit copy truncated (%u of %u hits kept)!\n",
                   s,
                   outHitCap - outHitBase,
                   totHitsS);
          totHitsS = outHitCap - outHitBase;
        }

        for (uint32_t h : cms::alpakatools::uniform_elements(acc, totHitsS)) {
          outHit_view[outHitBase + h].id() = inpHit_view[h].id();
          outHit_view[outHitBase + h].detId() = inpHit_view[h].detId();
          outHit_view[outHitBase + h].attached() = inpHit_view[h].attached();
        }
      }

      // Phase 4: tail [mergedNTracks, outCap). quality() is stamped bad so the downstream filterTracks
      // kernel skips every tail slot; the arm label is stamped -1 and the iteration label notIteration so a
      // consumer that reads either column before testing quality() sees a defined value.
      const uint32_t hitEndTail = (cumulHits[nInp] < outHitCap) ? cumulHits[nInp] : outHitCap;
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, outCap)) {
        if (i >= mergedNTracks) {
          outTrack_view[i].quality() = pixelTrack::Quality::bad;
          outTrack_view[i].iteration() = pixelTrack::Iteration::notIteration;
          outTrack_view[i].hitOffsets() = hitEndTail;  // nHits() reads zero past the last track
          if (armBuf)
            armBuf[i] = -1;
        }
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caHitNtupletGeneratorKernels

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h
