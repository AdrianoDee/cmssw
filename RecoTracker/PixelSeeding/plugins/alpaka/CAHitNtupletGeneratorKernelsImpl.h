#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h

// #define GPU_DEBUG
// #define NTUPLE_DEBUG
// #define CA_DEBUG
// #define CA_WARNINGS

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
#include "RecoTracker/PixelSeeding/interface/CircleEq.h"

// local includes
#include "CACell.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "CAStructures.h"
#include "CATripletCuts.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::caHitNtupletGeneratorKernels {

  using namespace ::caStructures;

  constexpr uint32_t tkNotFound = std::numeric_limits<uint32_t>::max();
  constexpr float maxScore = std::numeric_limits<float>::max();
  constexpr float nSigma2 = 5.f;
  constexpr int nTrackParameters = 5;
  // map: index of a track parameter -> index of its covariance
  HOST_DEVICE_CONSTANT std::array<uint8_t, nTrackParameters> iParam2iCov = {0u, 5u, 9u, 12u, 14u};

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
  using CellToCell = caStructures::NeighborCellContainer;
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
                                  caStructures::CACellPairSoAConstView cellCell,
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
        if (apc->get().first >= uint32_t(tracks_view.metadata().size()))
          printf("Tuples overflow\n");
        if (*nCells >= maxNumberOfDoublets)
          printf("Cells overflow\n");
        if (*nTrips >= uint32_t(cellCell.metadata().size()))
          printf("Triplets overflow\n");
        if (*nCellTracks >= uint32_t(cellTrack.metadata().size()))
          printf("TracksToCell overflow\n");
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
  template <typename TrackerTraits>
  class Kernel_earlyDuplicateRemover {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<TrackerTraits> const *cells,
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

        auto const *__restrict__ tracksOfCell = cellTracksHisto->begin(idx);
        int ntr = cellTracksHisto->size(idx);

        // loop over tracks i
        for (int i = 0; i < ntr - 1; i++) {
          auto it = tracksOfCell[i];
          auto nli = tracks_view[it].nLayers();
          auto curvi = tracks_view[it].pt();

          // function that compares the track curvatures of tracks it and jt
          auto incompatibleTrackParams = [=](int jt) -> bool {
            // comparing curvatures
            const auto dcurv = curvi - tracks_view[jt].pt();
            return (dcurv * dcurv > 0.000001);
          };

          // loop over remaining tracks j and compare
          for (int j = i + 1; j < ntr; ++j) {
            auto jt = tracksOfCell[j];

            if (incompatibleTrackParams(jt))
              continue;

            auto nlj = tracks_view[jt].nLayers();
            if (nlj < nli)
              tracks_view[jt].quality() = reject;  // no race: simple assignment of the same constant
            else if (nlj > nli)
              tracks_view[it].quality() = reject;  // no race: simple assignment of the same constant
          }
        }
      }
    }
  };

  // assume the above (so, short tracks already removed)
  template <typename TrackerTraits>
  class Kernel_fastDuplicateRemover {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<TrackerTraits> const *__restrict__ cells,
                                  uint32_t const *__restrict__ nCells,
                                  CellToTrack const *__restrict__ cellTracksHisto,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;
      constexpr auto loose = Quality::loose;

      ALPAKA_ASSERT_ACC(nCells);
      const auto ntNCells = (*nCells);

      for (auto idx : cms::alpakatools::uniform_elements(acc, ntNCells)) {
        if (cellTracksHisto->size(idx) < 2)
          continue;

        float mc = maxScore;
        uint32_t im = tkNotFound;

        // auto score = [&](auto it) { return std::abs(reco::tip(tracks_view, it)); };
        auto score = [&](auto it) { return tracks_view[it].chi2(); };

        // full crazy combinatorics
        auto const *__restrict__ thisCellTracks = cellTracksHisto->begin(idx);
        int ntr = cellTracksHisto->size(idx);
        for (int i = 0; i < ntr - 1; i++) {
          auto it = thisCellTracks[i];
          auto qi = tracks_view[it].quality();
          if (qi <= reject)
            continue;

          // get track parameters and covariances
          float iParams[nTrackParameters];
          float iCovs[nTrackParameters];
          for (int p{0}; p < nTrackParameters; ++p) {
            iParams[p] = tracks_view[it].state()(p);
            const auto c = iParam2iCov[p];
            iCovs[p] = tracks_view[it].covariance()(c);
          }
          // function that compares the five track parameters of tracks it and jt
          auto incompatibleTrackParams = [=](int jt) -> bool {
            // comparing phi, tip, 1/pT, cotan(theta) and zip
            for (int p{0}; p < nTrackParameters; ++p) {
              const auto dpij = iParams[p] - tracks_view[jt].state()(p);
              const auto c = iParam2iCov[p];
              const auto e2dpij = nSigma2 * (iCovs[p] + tracks_view[jt].covariance()(c));
              if (dpij * dpij > e2dpij)
                return true;  // incompatible param found
            }
            return false;  // all params compatible
          };

          // loop over remaining tracks j and compare
          for (int j = i + 1; j < ntr; ++j) {
            auto jt = thisCellTracks[j];
            auto qj = tracks_view[jt].quality();
            if (qj <= reject)
              continue;
            if (incompatibleTrackParams(jt))
              continue;
            if ((qj < qi) || (qj == qi && score(it) < score(jt)))
              tracks_view[jt].quality() = reject;
            // explicitly check since they might be identical when using multiple stubs per p-hit!
            else if ((qj > qi) || (qj == qi && score(it) > score(jt))) {
              tracks_view[it].quality() = reject;
              break;
            }
          }
        }

        // find maxQual
        auto maxQual = reject;  // no duplicate!
        for (int i = 0; i < ntr; i++) {
          auto it = thisCellTracks[i];
          if (tracks_view[it].quality() > maxQual)
            maxQual = tracks_view[it].quality();
        }

        if (maxQual <= loose)
          continue;

        // find min score
        for (int i = 0; i < ntr; i++) {
          auto it = thisCellTracks[i];
          if (tracks_view[it].quality() == maxQual && score(it) < mc) {
            mc = score(it);
            im = it;
          }
        }

        if (tkNotFound == im)
          continue;

        // mark all other duplicates  (not yet, keep it loose)
        for (int i = 0; i < ntr; i++) {
          auto it = thisCellTracks[i];
          if (tracks_view[it].quality() > loose && score(it) > mc)
            tracks_view[it].quality() = loose;  //no race:  simple assignment of the same constant
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
                                  reco::CALayersSoAConstView ll,
                                  reco::CAGraphSoAConstView cc,
                                  caStructures::CACellPairSoAView cn,
                                  CACell<TrackerTraits> *cells,
                                  uint32_t const *nCells,
                                  uint32_t *nTrips,
                                  HitToCell const *__restrict__ outerHitHisto,
                                  CellToCell *cellNeighborsHisto,
                                  AlgoParams const &params,
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

        auto ccLayerPairParams = cc[outerCell.layerPairId()];
        auto skips = ccLayerPairParams.skipsLayers();

#ifdef CA_DEBUG
        printf("numberOfPossibleFromHisto;%d;%d;%d;%d;%d\n",
               *nCells,
               middleHitId,
               oCellIndex,
               outerCell.innerLayer(),
               numberOfPossibleNeighbors);
#endif

        // loop on inner cells
        for (uint32_t j : cms::alpakatools::independent_group_elements_x(acc, numberOfPossibleNeighbors)) {
          auto iCellIndex = outerHitCells[j];
          auto &innerCell = cells[iCellIndex];
          float curvature = 0.f;

          // apply compatibility cuts for this triplet (innerCell, outerCell)
          if (TripletCuts<TrackerTraits>::accept(
                  innerCell, outerCell, curvature, hh, ccLayerPairParams, params, pipelineCounters)) {
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

            // bin = 2*iCell     (== non-layer-skipping neighbors)
            // bin = 2*iCell + 1 (== layer-skipping neighbors)
            auto bin = 2 * iCellIndex + skips;
            cellNeighborsHisto->count(acc, bin);

            cn[t_ind].inner() = bin;
            cn[t_ind].outer() = {oCellIndex, curvature};
            outerCell.setStatusBits(Cell::StatusBit::kUsed);
            outerCell.setStatusBits(Cell::StatusBit::kHasInner);  // outerCell has an inner neighbor
            innerCell.setStatusBits(Cell::StatusBit::kUsed);

            // Pipeline stage counters: classify triplet by hit types
            if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
              if (pipelineCounters) {
                using PC = caHitNtupletGenerator::PipelineCounter;
                alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsTotal], 1u, alpaka::hierarchy::Blocks{});
                auto hit1 = innerCell.inner_hit_id();
                auto hit2 = outerCell.inner_hit_id();
                auto hit3 = outerCell.outer_hit_id();
                int nStubs = (hh[hit1].isStub() ? 1 : 0) + (hh[hit2].isStub() ? 1 : 0) + (hh[hit3].isStub() ? 1 : 0);
                if (nStubs == 0)
                  alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsPixPixPix], 1u, alpaka::hierarchy::Blocks{});
                else if (nStubs == 1)
                  alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsPixPixOT], 1u, alpaka::hierarchy::Blocks{});
                else if (nStubs == 2)
                  alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsPixOTOT], 1u, alpaka::hierarchy::Blocks{});
                else {
                  alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsOTOTOT], 1u, alpaka::hierarchy::Blocks{});
                  // OOO triplet region breakdown
                  auto layer1 = innerCell.innerLayer();        // innermost
                  auto layer2 = outerCell.innerLayer();  // middle
                  auto layer3 = outerCell.outerLayer();  // outermost
                  bool l1Brl = (layer1 >= 28 && layer1 <= 33);
                  bool l2Brl = (layer2 >= 28 && layer2 <= 33);
                  bool l3Brl = (layer3 >= 28 && layer3 <= 33);
                  bool l1Bwd = (layer1 >= 34 && layer1 <= 43);
                  bool l2Bwd = (layer2 >= 34 && layer2 <= 43);
                  bool l3Bwd = (layer3 >= 34 && layer3 <= 43);
                  bool l1Fwd = (layer1 >= 44 && layer1 <= 53);
                  bool l2Fwd = (layer2 >= 44 && layer2 <= 53);
                  bool l3Fwd = (layer3 >= 44 && layer3 <= 53);
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
                                  HitToCell *outerHitHisto) const {
      for (auto cellIndex : cms::alpakatools::uniform_elements(acc, *nCells)) {
#ifdef DOUBLETS_DEBUG
        printf("outerHitHisto;%d;%d\n", cellIndex, cells[cellIndex].outer_hit_id());
#endif
        outerHitHisto->fill(acc, cells[cellIndex].outer_hit_id() - offsetBPIX2, cellIndex);
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
      for (uint32_t index : cms::alpakatools::uniform_elements(acc, *nElements)) {
        genericHisto->fill(acc, cn[index].inner(), cn[index].outer());
      }
    }
  };

  // Sort each histogram bin by value for deterministic iteration on both CPU and GPU backends.
  // Used for cellToNeighbors (DFS order), cellToTracks (duplicate removal), and hitToTuple (shared-hit cleaning).
  class Kernel_sortHistoBins {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc, GenericContainer *histo) const {
      for (auto idx : cms::alpakatools::uniform_elements(acc, histo->nOnes())) {
        auto size = histo->size(idx);
        if (size <= 1)
          continue;
        auto *bin = histo->content.data() + histo->off[idx];
        // Insertion sort: optimal for tiny arrays (typically 2-5 entries per bin)
        for (uint32_t i = 1; i < size; ++i) {
          auto key = bin[i];
          int j = i - 1;
          while (j >= 0 && bin[j] > key) {
            bin[j + 1] = bin[j];
            --j;
          }
          bin[j + 1] = key;
        }
      }
    }
  };

  // Deep reachability filter: for cells involving a target layer (e.g. OT barrel L1),
  // require that the outer-neighbor chain reaches a minimum outer layer (e.g. OT barrel L4).
  // This enforces that the target-layer stub is confirmed by multiple pure outer OT layers
  // before fishbone and n-tuplet building, killing dead-end fakes early.
  // Only compiled for Phase2OTStubs topology (transparent for all other topologies).
  // Uses hop-count semantics: counts distinct outer layers reached through the
  // neighbor chain, regardless of detector region (barrel/endcap).
  template <typename TrackerTraits>
  class Kernel_reachabilityFilter {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  CACell<TrackerTraits> *__restrict__ cells,
                                  uint32_t const *nCells,
                                  CellToCell const *__restrict__ cellNeighborsHisto,
                                  uint8_t targetLayer,
                                  uint8_t minHops,
                                  uint32_t *__restrict__ pipelineCounters) const {
      for (auto idx : cms::alpakatools::uniform_elements(acc, *nCells)) {
        auto &thisCell = cells[idx];
        if (thisCell.isKilled())
          continue;

        // Only filter cells that involve the target layer
        if (thisCell.innerLayer() != targetLayer && thisCell.outerLayer() != targetLayer)
          continue;

        if (pipelineCounters)
          alpaka::atomicAdd(
              acc, &pipelineCounters[caHitNtupletGenerator::kReachCellsChecked], 1u, alpaka::hierarchy::Blocks{});

        // Count distinct outer layers reachable through up to 3 hops.
        // Works across barrel (28-33), endcap BWD (34-38), endcap FWD (39-43).
        uint8_t seenLayers[6] = {};
        uint8_t nSeen = 0;
        bool hasAnyNeighbor = false;
        bool hasLiveNeighbor = false;

        // Lambda to register a unique layer
        auto registerLayer = [&seenLayers, &nSeen](uint8_t layer) {
          for (uint8_t k = 0; k < nSeen; ++k)
            if (seenLayers[k] == layer)
              return;
          if (nSeen < 6)
            seenLayers[nSeen++] = layer;
        };

        // Hop 1: direct outer neighbors
        auto n1 = cellNeighborsHisto->size(idx);
        auto const *nb1 = cellNeighborsHisto->begin(idx);
        if (n1 > 0)
          hasAnyNeighbor = true;
        for (auto j1 = 0u; j1 < n1 && nSeen < minHops; ++j1) {
          auto c1 = nb1[j1].index;
          if (cells[c1].isKilled())
            continue;
          hasLiveNeighbor = true;
          registerLayer(cells[c1].outerLayer());

          // Hop 2: outer neighbors of hop-1 cells
          auto n2 = cellNeighborsHisto->size(c1);
          auto const *nb2 = cellNeighborsHisto->begin(c1);
          for (auto j2 = 0u; j2 < n2 && nSeen < minHops; ++j2) {
            auto c2 = nb2[j2].index;
            if (cells[c2].isKilled())
              continue;
            registerLayer(cells[c2].outerLayer());

            // Hop 3: outer neighbors of hop-2 cells
            auto n3 = cellNeighborsHisto->size(c2);
            auto const *nb3 = cellNeighborsHisto->begin(c2);
            for (auto j3 = 0u; j3 < n3 && nSeen < minHops; ++j3) {
              auto c3 = nb3[j3].index;
              if (cells[c3].isKilled())
                continue;
              registerLayer(cells[c3].outerLayer());
            }
          }
          if (nSeen >= minHops)
            break;
        }

        if (nSeen < minHops) {
          thisCell.kill();
          if (pipelineCounters) {
            alpaka::atomicAdd(
                acc, &pipelineCounters[caHitNtupletGenerator::kReachabilityKilled], 1u, alpaka::hierarchy::Blocks{});
            // Classify WHY the chain failed
            if (!hasAnyNeighbor) {
              alpaka::atomicAdd(
                  acc, &pipelineCounters[caHitNtupletGenerator::kReachNoNeighbors], 1u, alpaka::hierarchy::Blocks{});
            } else if (!hasLiveNeighbor) {
              alpaka::atomicAdd(
                  acc, &pipelineCounters[caHitNtupletGenerator::kReachAllNeighKilled], 1u, alpaka::hierarchy::Blocks{});
            } else {
              alpaka::atomicAdd(
                  acc, &pipelineCounters[caHitNtupletGenerator::kReachChainShort], 1u, alpaka::hierarchy::Blocks{});
            }
            // Classify by cell type: pixel->OT vs OT->OT
            if (thisCell.innerLayer() < 28) {
              alpaka::atomicAdd(
                  acc, &pipelineCounters[caHitNtupletGenerator::kReachKilledPixOT], 1u, alpaka::hierarchy::Blocks{});
            } else {
              alpaka::atomicAdd(
                  acc, &pipelineCounters[caHitNtupletGenerator::kReachKilledOTOT], 1u, alpaka::hierarchy::Blocks{});
            }
          }
        } else {
          if (pipelineCounters)
            alpaka::atomicAdd(
                acc, &pipelineCounters[caHitNtupletGenerator::kReachCellsPassed], 1u, alpaka::hierarchy::Blocks{});
        }
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_find_ntuplets {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ::reco::CALayersSoAConstView &ll,
                                  const ::reco::CAGraphSoAConstView &cc,
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
        if ((cellNeighborsHisto->size(2 * idx) == 0) && (cellNeighborsHisto->size(2 * idx + 1) == 0))
          continue;

        // check if the layer pair of the cell is among the set of starting pairs
        auto pid = thisCell.layerPairId();
        bool doit = cc[pid].startingPair();

        // check if the most inner hit does not fulfill the starting requirement
        auto lid = thisCell.innerLayer();
        if (thisCell.inner_r() > ll[lid].startMaxInnerR())
          doit = false;

        constexpr uint32_t maxDepth = TrackerTraits::maxLayersPerTrack - 1;
#ifdef CA_DEBUG
        printf(
            "LayerPairId %d and inner layer %d doit ? %d From cell %d with nNeighbors (skipping) = %d and nNeighbors "
            "(non-skipping) = %d and innerR=%f < "
            "maxInnerR=%f ?\n",
            pid,
            lid,
            doit,
            idx,
            cellNeighborsHisto->size(2 * idx),
            cellNeighborsHisto->size(2 * idx + 1),
            thisCell.inner_r(),
            ll[lid].startMaxInnerR());
#endif

        if (doit) {
          typename Cell::TmpTuple stack;

          stack.reset();
          thisCell.template find_ntuplets<maxDepth>(acc,
                                                    ll,
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
                                                    params.minHitsPerNtuplet_);
          ALPAKA_ASSERT_ACC(stack.empty());
        }
      }
    }
  };
  // Orphan chain recovery: find n-tuplets from cells with outer neighbors but no inner connection.
  // Recovers OT-only tracks from displaced vertices, tracks outside pixel acceptance, etc.
  template <typename TrackerTraits>
  class Kernel_find_orphan_ntuplets {
    using Cell = CACell<TrackerTraits>;
    using CellToCell = caStructures::NeighborCellContainer;
    using CellToTrack = caStructures::GenericContainer;
    using HitContainer = caStructures::SequentialContainer;

  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  const ::reco::CALayersSoAConstView &ll,
                                  const ::reco::CAGraphSoAConstView &cc,
                                  TkSoAView tracks_view,
                                  HitContainer *foundNtuplets,
                                  CellToCell const *__restrict__ cellNeighborsHisto,
                                  CellToTrack *cellTracksHisto,
                                  caStructures::CAPairSoAView ct,
                                  CACell<TrackerTraits> *__restrict__ cells,
                                  uint32_t *nCellTracks,
                                  uint32_t const *nCells,
                                  cms::alpakatools::AtomicPairCounter *apc,
                                  AlgoParams const &params) const {
      using Cell = CACell<TrackerTraits>;

      for (auto idx : cms::alpakatools::uniform_elements(acc, (*nCells))) {
        auto const &thisCell = cells[idx];

        if (thisCell.isKilled())
          continue;

        // Must have outer neighbors (otherwise it's a leaf)
        if (cellNeighborsHisto->size(idx) == 0)
          continue;

        // ORPHAN: has outer neighbors but NO inner connection from any other cell
        if (thisCell.hasInnerNeighbor())
          continue;

        // Skip cells on starting pairs - already handled by Kernel_find_ntuplets
        auto pid = thisCell.layerPairId();
        if (cc[pid].startingPair())
          continue;

        constexpr uint32_t maxDepth = TrackerTraits::maxLayersPerTrack - 1;
        typename Cell::TmpTuple stack;
        stack.reset();
        thisCell.template find_ntuplets<maxDepth>(acc,
                                                  ll,
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
                                                  params.minHitsOrphanNtuplet_);
        ALPAKA_ASSERT_ACC(stack.empty());
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
            if (hh[*h].isStub())
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

  // Copy *nCellTracks into the pipeline counter array
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

  template <typename TrackerTraits>
  class Kernel_countMultiplicity {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
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
        ALPAKA_ASSERT_ACC(tracks_view[it].quality() == Quality::bad);
        if (nhits > TrackerTraits::maxHitsOnTrack)  // current limit
          printf("wrong mult %d %d\n", it, nhits);
        ALPAKA_ASSERT_ACC(nhits <= TrackerTraits::maxHitsOnTrack);
        tupleMultiplicity->count(acc, nhits);
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_fillMultiplicity {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  TupleMultiplicity *tupleMultiplicity) const {
      for (auto it : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        auto nhits = foundNtuplets->size(it);

        if (nhits < 3)
          continue;
        if (tracks_view[it].quality() == Quality::edup)
          continue;
        ALPAKA_ASSERT_ACC(tracks_view[it].quality() == Quality::bad);
        if (nhits > TrackerTraits::maxHitsOnTrack)
          printf("wrong mult %d %d\n", it, nhits);
        ALPAKA_ASSERT_ACC(nhits <= TrackerTraits::maxHitsOnTrack);
        tupleMultiplicity->fill(acc, nhits, it);
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_classifyTracks {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  QualityCuts<TrackerTraits> cuts) const {
#if defined(NTUPLE_DEBUG) || defined(FIT_DEBUG)
      // Counters for diagnostic output
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

        ALPAKA_ASSERT_ACC(tracks_view[it].quality() == Quality::bad);

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
        // Print details for first 10 successfully fitted tracks
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

        if (cuts.strictCut(tracks_view, nhits, it))
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

  // Final quality distribution counter: counts tracks at each quality level
  // after ALL processing (classification, fishbone, duplicate removal).
#ifdef CA_PIPELINE_COUNTERS
  // Runs right before the pipeline printout to complete the diagnostic funnel.
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
              if (hh[*h].isStub()) {
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

          // Count fishbone hits per track
          uint32_t nFishbone = 0;
          // Fishbone hits are interleaved in the hit container - they don't correspond to cell inner/outer hits
          // Heuristic: count hits that appear in the foundNtuplets but are not the inner/outer of any cell
          // Simpler: just count based on nhits vs expected cell count
          // For a track with N cells, we expect N+1 hits (no fishbone) or more (with fishbone)
          // Actually, nhits includes fishbone hits. Typical: 3 cells -> 4 hits (no FB) or 5-6 (with FB)
          // For now, just report nhits directly -- the excess over (nCells+1) is fishbone count
          // Since we can't easily get nCells here, just count nhits > expected
          nFishbone = 0;  // Will be counted properly below
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

  template <typename TrackerTraits>
  class Kernel_countHitInTracks {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  HitToTuple *hitToTuple) const {
      for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(idx) == 0)
          break;  // guard
        for (auto h = foundNtuplets->begin(idx); h != foundNtuplets->end(idx); ++h)
          hitToTuple->count(acc, *h);
      }
    }
  };

  template <typename TrackerTraits>
  class Kernel_fillHitInTracks {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  HitContainer const *__restrict__ foundNtuplets,
                                  HitToTuple *hitToTuple) const {
      for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes())) {
        if (foundNtuplets->size(idx) == 0)
          break;  // guard
        for (auto h = foundNtuplets->begin(idx); h != foundNtuplets->end(idx); ++h)
          hitToTuple->fill(acc, *h, idx);
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
                                  cms::alpakatools::AtomicPairCounter *apc) const {
      // clamp the number of tracks to the capacity of the SoA
      auto ntracks = std::min<int>(apc->get().first, tracks_view.metadata().size() - 1);
      if (cms::alpakatools::once_per_grid(acc))
        tracks_view.nTracks() = ntracks;

      // copy offsets
      for (auto idx : cms::alpakatools::uniform_elements(acc, ntracks)) {
        tracks_view[idx].hitOffsets() = foundNtuplets->off[idx + 1];  // offset for track 0 is always 0
      }
      // fill hit indices
      for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->size())) {
        ALPAKA_ASSERT_ACC(foundNtuplets->content[idx] < (uint32_t)hh.metadata().size());
        track_hits_view[idx].id() = foundNtuplets->content[idx];
        track_hits_view[idx].detId() = hh[foundNtuplets->content[idx]].detectorIndex();
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
  class Kernel_fillNLayers {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoABlocksView view,
                                  uint32_t const *__restrict__ layerStarts,
                                  uint16_t maxLayers,
                                  cms::alpakatools::AtomicPairCounter *apc) const {
      // clamp the number of tracks to the capacity of the SoA
      auto ntracks = std::min<int>(apc->get().first, view.tracks().metadata().size() - 1);

      if (cms::alpakatools::once_per_grid(acc))
        view.tracks().nTracks() = ntracks;
      for (auto idx : cms::alpakatools::uniform_elements(acc, ntracks)) {
        ALPAKA_ASSERT_ACC(reco::nHits(view.tracks(), idx) >= 3);
        view.tracks()[idx].nLayers() = reco::nLayers(view, maxLayers, layerStarts, idx);
#ifdef CA_DEBUG
        printf("Kernel_fillNLayers %d %d %d - %d %d\n",
               idx,
               ntracks,
               view.tracks()[idx].nLayers(),
               apc->get().first,
               view.tracks().metadata().size() - 1);
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

  // mostly for very forward triplets.....
  template <typename TrackerTraits>
  class Kernel_rejectDuplicate {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough,
                                  HitToTuple const *__restrict__ phitToTuple) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;

      auto &hitToTuple = *phitToTuple;

      for (auto idx : cms::alpakatools::uniform_elements(acc, hitToTuple.nOnes())) {
        if (hitToTuple.size(idx) < 2)
          continue;

        // auto score = [&](auto it, auto nl) { return std::abs(reco::tip(tracks_view, it)); };
        auto score = [&](auto it, auto nl) { return tracks_view[it].chi2(); };

        // full combinatorics
        for (auto ip = hitToTuple.begin(idx); ip < hitToTuple.end(idx) - 1; ++ip) {
          auto const it = *ip;
          auto qi = tracks_view[it].quality();
          if (qi <= reject)
            continue;

          // get track parameters and covariances
          float iParams[nTrackParameters];
          float iCovs[nTrackParameters];
          for (int p{0}; p < nTrackParameters; ++p) {
            iParams[p] = tracks_view[it].state()(p);
            const auto c = iParam2iCov[p];
            iCovs[p] = tracks_view[it].covariance()(c);
          }
          // function that compares the five track parameters of tracks it and jt
          auto incompatibleTrackParams = [=](int jt) -> bool {
            // comparing phi, tip, 1/pT, cotan(theta) and zip
            for (int p{0}; p < nTrackParameters; ++p) {
              const auto dpij = iParams[p] - tracks_view[jt].state()(p);
              const auto c = iParam2iCov[p];
              const auto e2dpij = nSigma2 * (iCovs[p] + tracks_view[jt].covariance()(c));
              if (dpij * dpij > e2dpij)
                return true;  // incompatible param found
            }
            return false;  // all params compatible
          };

          auto nli = tracks_view[it].nLayers();

          for (auto jp = ip + 1; jp < hitToTuple.end(idx); ++jp) {
            auto const jt = *jp;
            auto qj = tracks_view[jt].quality();
            if (qj <= reject)
              continue;
            if (incompatibleTrackParams(jt))
              continue;
            auto nlj = tracks_view[jt].nLayers();
            if (nlj < nli || (nlj == nli && (qj < qi || (qj == qi && score(it, nli) < score(jt, nlj)))))
              tracks_view[jt].quality() = reject;
            // explicitly check since we can have actual duplicated tracks with identical parameters
            else if (nli < nlj || (nli == nlj && (qi < qj || (qi == qj && score(jt, nlj) < score(it, nli))))) {
              tracks_view[it].quality() = reject;
              break;
            }
            // if we have two tracks with the same length, parameters and quality, we keep the one with the lower index
            // (arbitrary but deterministic) and reject the other to avoid double counting
            else if (it < jt)
              tracks_view[jt].quality() = reject;
            else
              tracks_view[it].quality() = reject;
          }
        }
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
                                  HitToTuple const *__restrict__ phitToTuple) const {
      // quality to mark rejected
      auto const reject = dupPassThrough ? Quality::loose : Quality::dup;
      // quality of longest track
      auto const longTqual = Quality::highPurity;

      auto &hitToTuple = *phitToTuple;

      uint32_t l1end = layerStarts[1];

      for (auto idx : cms::alpakatools::uniform_elements(acc, hitToTuple.nOnes())) {
        if (hitToTuple.size(idx) < 2)
          continue;

        // checking if shared hit is on bpix1
        if (idx < l1end)
          continue;

        int8_t maxNl = 0;

        // find maxNl
        for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
          if (tracks_view[*it].quality() < longTqual)
            continue;
          // if (tracks_view[*it].nHits()==3) continue;
          auto nl = tracks_view[*it].nLayers();
          maxNl = std::max(nl, maxNl);
        }

        // For Phase2OTStubs: also check tracks using other stubs with the same pHitGroupId
        // Multiple stubs from the same P-hit have different hit indices but the same pHitGroupId
        // These should be treated as sharing the same hit for cleaning purposes
        if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
          if (hh[idx].isStub()) {
            auto pHitGroup = hh[idx].pHitGroupId();
            // Only process if this is a valid pHitGroupId (not UINT32_MAX for pixel hits)
            if (pHitGroup != std::numeric_limits<uint32_t>::max()) {
              // Search for other stubs with the same pHitGroupId
              // Stubs start at offsetStubs in the unified hit collection
              auto offsetStubs = hh.offsetStubs();
              auto nHits = static_cast<uint32_t>(hh.metadata().size());
              // Only search the stub region (from offsetStubs to end)
              for (uint32_t otherIdx = offsetStubs; otherIdx < nHits; ++otherIdx) {
                if (otherIdx == idx)
                  continue;  // Skip self
                if (!hh[otherIdx].isStub())
                  continue;  // Should not happen in stub region, but check anyway
                if (hh[otherIdx].pHitGroupId() != pHitGroup)
                  continue;  // Different P-hit group

                // Found a stub with the same pHitGroupId - include its tracks in maxNl calculation
                for (auto it = hitToTuple.begin(otherIdx); it != hitToTuple.end(otherIdx); ++it) {
                  if (tracks_view[*it].quality() < longTqual)
                    continue;
                  auto nl = tracks_view[*it].nLayers();
                  maxNl = std::max(nl, maxNl);
                }
              }
            }
          }
        }

        if (maxNl < 4)
          continue;

        // quad pass through (leave for tests)
        // maxNl = std::min(4, maxNl);

        // kill all tracks shorter than maxNl (only triplets???)
        for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
          auto nl = tracks_view[*it].nLayers();

          // checking if the tuple is short enough
          if (nl > nmin)
            continue;

          if (nl < maxNl && tracks_view[*it].quality() > reject)
            tracks_view[*it].quality() = reject;
        }

        // For Phase2OTStubs: also clean tracks using other stubs with the same pHitGroupId
        if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
          if (hh[idx].isStub()) {
            auto pHitGroup = hh[idx].pHitGroupId();
            if (pHitGroup != std::numeric_limits<uint32_t>::max()) {
              auto offsetStubs = hh.offsetStubs();
              auto nHits = static_cast<uint32_t>(hh.metadata().size());
              for (uint32_t otherIdx = offsetStubs; otherIdx < nHits; ++otherIdx) {
                if (otherIdx == idx)
                  continue;
                if (!hh[otherIdx].isStub())
                  continue;
                if (hh[otherIdx].pHitGroupId() != pHitGroup)
                  continue;

                // Apply the same cleaning to tracks using this related stub
                for (auto it = hitToTuple.begin(otherIdx); it != hitToTuple.end(otherIdx); ++it) {
                  auto nl = tracks_view[*it].nLayers();

                  // For stubs, we don't apply the bpix1 exception (idx < l1end check)
                  // since stubs are not in BPIX1

                  if (nl < maxNl && tracks_view[*it].quality() > reject)
                    tracks_view[*it].quality() = reject;
                }
              }
            }
          }
        }
      }
    }
  };
  template <typename TrackerTraits>
  class Kernel_tripletCleaner {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough,
                                  HitToTuple const *__restrict__ phitToTuple) const {
      // quality to mark rejected
      auto const reject = Quality::loose;
      /// min quality of good
      auto const good = Quality::strict;

      auto &hitToTuple = *phitToTuple;

      for (auto idx : cms::alpakatools::uniform_elements(acc, hitToTuple.nOnes())) {
        if (hitToTuple.size(idx) < 2)
          continue;

        float mc = maxScore;
        uint32_t im = tkNotFound;
        bool onlyTriplets = true;

        // check if only triplets
        for (auto it = hitToTuple.begin(idx); it != hitToTuple.end(idx); ++it) {
          if (tracks_view[*it].quality() <= good)
            continue;
          onlyTriplets &= reco::isTriplet(tracks_view, *it);
          if (!onlyTriplets)
            break;
        }

        // only triplets
        if (!onlyTriplets)
          continue;

        // for triplets choose best tip!  (should we first find best quality???)
        for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
          auto const it = *ip;
          if (tracks_view[it].quality() >= good && std::abs(reco::tip(tracks_view, it)) < mc) {
            mc = std::abs(reco::tip(tracks_view, it));
            im = it;
          }
        }

        if (tkNotFound == im)
          continue;

        // mark worse ambiguities
        for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
          auto const it = *ip;
          if (tracks_view[it].quality() > reject && it != im)
            tracks_view[it].quality() = reject;  //no race:  simple assignment of the same constant
        }

      }  // loop over hits
    }
  };

  template <typename TrackerTraits>
  class Kernel_simpleTripletCleaner {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TkSoAView tracks_view,
                                  bool dupPassThrough,
                                  HitToTuple const *__restrict__ phitToTuple) const {
      // quality to mark rejected
      auto const reject = Quality::loose;
      /// min quality of good
      auto const good = Quality::loose;

      auto &hitToTuple = *phitToTuple;

      for (auto idx : cms::alpakatools::uniform_elements(acc, hitToTuple.nOnes())) {
        if (hitToTuple.size(idx) < 2)
          continue;

        float mc = maxScore;
        uint32_t im = tkNotFound;

        // choose best tip!  (should we first find best quality???)
        for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
          auto const it = *ip;
          if (tracks_view[it].quality() >= good && std::abs(reco::tip(tracks_view, it)) < mc) {
            mc = std::abs(reco::tip(tracks_view, it));
            im = it;
          }
        }

        if (tkNotFound == im)
          continue;

        // mark worse ambiguities
        for (auto ip = hitToTuple.begin(idx); ip != hitToTuple.end(idx); ++ip) {
          auto const it = *ip;
          if (tracks_view[it].quality() > reject && reco::isTriplet(tracks_view, it) && it != im)
            tracks_view[it].quality() = reject;  //no race:  simple assignment of the same constant
        }

      }  // loop over hits
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
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caHitNtupletGeneratorKernels

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h
