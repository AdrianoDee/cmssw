#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGeneratorKernelsImpl_h

// #define GPU_DEBUG
// #define NTUPLE_DEBUG
// #define CA_DEBUG
#define CA_WARNINGS

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

namespace ALPAKA_ACCELERATOR_NAMESPACE::caHitNtupletGeneratorKernels {

  using namespace ::caStructures;

  constexpr uint32_t tkNotFound = std::numeric_limits<uint32_t>::max();
  constexpr float maxScore = std::numeric_limits<float>::max();
  constexpr float nSigma2 = 25.f;

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

        auto score = [&](auto it) { return std::abs(reco::tip(tracks_view, it)); };

        // full crazy combinatorics
        auto const *__restrict__ thisCellTracks = cellTracksHisto->begin(idx);
        int ntr = cellTracksHisto->size(idx);
        for (int i = 0; i < ntr - 1; i++) {
          auto it = thisCellTracks[i];
          auto qi = tracks_view[it].quality();
          if (qi <= reject)
            continue;
          auto opi = tracks_view[it].state()(2);
          auto e2opi = tracks_view[it].covariance()(9);
          auto cti = tracks_view[it].state()(3);
          auto e2cti = tracks_view[it].covariance()(12);
          for (int j = i + 1; j < ntr; ++j) {
            auto jt = thisCellTracks[j];
            auto qj = tracks_view[jt].quality();
            if (qj <= reject)
              continue;
            auto opj = tracks_view[jt].state()(2);
            auto ctj = tracks_view[jt].state()(3);
            auto dct = nSigma2 * (tracks_view[jt].covariance()(12) + e2cti);
            if ((cti - ctj) * (cti - ctj) > dct)
              continue;
            auto dop = nSigma2 * (tracks_view[jt].covariance()(9) + e2opi);
            if ((opi - opj) * (opi - opj) > dop)
              continue;
            if ((qj < qi) || (qj == qi && score(it) < score(jt)))
              tracks_view[jt].quality() = reject;
            else {
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
          if (tracks_view[it].quality() > loose && it != im)
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
      for (uint32_t cellIndex : cms::alpakatools::uniform_elements_y(acc, *nCells)) {
        auto &thisCell = cells[cellIndex];
        auto innerHitId = thisCell.inner_hit_id() - hh.offsetBPIX2();

        if (int(innerHitId) < 0)
          continue;

        auto const *__restrict__ outerHitCells = outerHitHisto->begin(innerHitId);
        auto const numberOfPossibleNeighbors = outerHitHisto->size(innerHitId);

#ifdef CA_DEBUG
        printf("numberOfPossibleFromHisto;%d;%d;%d;%d;%d\n",
               *nCells,
               innerHitId,
               cellIndex,
               thisCell.innerLayer(),
               numberOfPossibleNeighbors);
#endif
        auto ri = thisCell.inner_r(hh);
        auto zi = thisCell.inner_z(hh);
        auto ro = thisCell.outer_r(hh);
        auto zo = thisCell.outer_z(hh);
        auto thetaCut = ll[thisCell.innerLayer()].caThetaCut();

        // loop on inner cells
        for (uint32_t j : cms::alpakatools::independent_group_elements_x(acc, numberOfPossibleNeighbors)) {
          auto otherCell = outerHitCells[j];
          auto &oc = cells[otherCell];
          auto r1 = oc.inner_r(hh);
          auto z1 = oc.inner_z(hh);
          auto dcaCut = ll[oc.innerLayer()].caDCACut();
          auto dcaFloor = ll[oc.innerLayer()].caDCAFloor();

          // Check for SS stubs and count real stubs early -- needed for both
          // the relaxed theta cut and the later kappa/DCA logic.
          bool hasSSStub = false;
          int nStubs = 0;
          int nSS = 0;
          bool s1 = false, s2 = false, s3 = false;
          uint32_t hit1 = 0, hit2 = 0, hit3 = 0;
          if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
            hit1 = oc.inner_hit_id();
            hit2 = thisCell.inner_hit_id();
            hit3 = thisCell.outer_hit_id();

            auto isSSStub = [&](uint32_t hitId) {
              return hh[hitId].isStub() && hh[hitId].stubType() == ::reco::StubType::SS;
            };
            auto isRealStub = [&](uint32_t hitId) {
              return hh[hitId].isStub() && hh[hitId].stubType() != ::reco::StubType::PHitOnly;
            };

            bool ss1 = isSSStub(hit1);
            bool ss2 = isSSStub(hit2);
            bool ss3 = isSSStub(hit3);
            hasSSStub = ss1 || ss2 || ss3;
            nSS = int(ss1) + int(ss2) + int(ss3);
            s1 = isRealStub(hit1);
            s2 = isRealStub(hit2);
            s3 = isRealStub(hit3);
            nStubs = int(s1) + int(s2) + int(s3);
          }

          // SS stubs have ~1-2 cm z uncertainty. Select the per-layer theta cut
          // based on the number of SS stubs. A negative SoA value means "fall back
          // to the old hardcoded multiplier" for backward compatibility.
          float effectiveThetaCut = thetaCut;
          if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
            if (nSS >= 2) {
              auto cut2 = ll[thisCell.innerLayer()].caThetaCut2SS();
              effectiveThetaCut = (cut2 >= 0.f) ? cut2 : thetaCut * 3.0f;
            } else if (nSS >= 1) {
              auto cut1 = ll[thisCell.innerLayer()].caThetaCut1SS();
              effectiveThetaCut = (cut1 >= 0.f) ? cut1 : thetaCut * 2.0f;
            }
          }

          bool aligned = Cell::areAlignedRZ(r1, z1, ri, zi, ro, zo, params.ptmin_, effectiveThetaCut);
          bool dcaPassed;
          float curvature = 0.f;
          float tripletPhiResid = 0.f;  // geometric phi residual at middle hit [rad]
          if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
            if (hasSSStub) {
              // SS stubs have cm-scale (x,y) errors from coarse strip-length
              // measurement -- the 3-point circle fit is unreliable.
              // Instead, compare geometric kappa (from hit positions) vs stub-measured kappa.
              auto geomSigCut = ll[thisCell.innerLayer()].geomKappaSigmaCut();

              if (nStubs >= 1 && geomSigCut > 0.f) {
                // Load global coordinates once (shared between pre-filter and geomKappa)
                float x1g = hh[hit1].xGlobal();
                float y1g = hh[hit1].yGlobal();
                float x2g = hh[hit2].xGlobal();
                float y2g = hh[hit2].yGlobal();
                float x3g = hh[hit3].xGlobal();
                float y3g = hh[hit3].yGlobal();

                // Fast pre-reject: cross-product sign consistency.
                // For a genuine track curving smoothly, cross(1->2) and cross(2->3) have the same sign.
                // For random hit combinations, ~50% have opposite signs -> immediate rejection.
                float cross12 = x1g * y2g - y1g * x2g;
                float cross23 = x2g * y3g - y2g * x3g;
                dcaPassed = (cross12 * cross23 >= 0.f);

                if (dcaPassed) {
                  // Compute kappa + error for each stub hit
                  auto computeKappa = [&](uint32_t hitId, float r) {
                    float d = hh[hitId].dPhiDr();
                    float s = hh[hitId].dPhiDrError();
                    float den = 1.f + r * r * d * d;
                    float sqrt_den = std::sqrt(den);
                    return std::make_pair(d / sqrt_den, s / (den * sqrt_den));
                  };

                  // Weighted average of stub kappas
                  float w_sum = 0.f, wk_sum = 0.f;
                  if (s1) { auto [k, sk] = computeKappa(hit1, r1); float w = 1.f / (sk * sk); w_sum += w; wk_sum += w * k; }
                  if (s2) { auto [k, sk] = computeKappa(hit2, ri); float w = 1.f / (sk * sk); w_sum += w; wk_sum += w * k; }
                  if (s3) { auto [k, sk] = computeKappa(hit3, ro); float w = 1.f / (sk * sk); w_sum += w; wk_sum += w * k; }

                  float kappa_stub_avg = wk_sum / w_sum;
                  float sigma_stub_avg2 = 1.f / w_sum;

                  // Geometric kappa from inner-outer phi difference (1 atan2 instead of 2)
                  float cross13 = x1g * y3g - y1g * x3g;
                  float dot13 = x1g * x3g + y1g * y3g;
                  float dphi_13 = std::atan2(cross13, dot13);
                  float dr_13 = ro - r1;
                  float dphidr_geom = dphi_13 / dr_13;
                  float r_mid = 0.5f * (r1 + ro);
                  float den_g = 1.f + r_mid * r_mid * dphidr_geom * dphidr_geom;
                  float sqrt_den_g = std::sqrt(den_g);
                  float kappa_geom = dphidr_geom / sqrt_den_g;

                  // Geometric kappa error (~500 murad phi resolution)
                  constexpr float sigma_phi = 5e-4f;
                  float sk_geom = sigma_phi / (std::abs(dr_13) * den_g * sqrt_den_g);

                  // Significance test (squared form, no sqrt needed)
                  float combined_err2 = sk_geom * sk_geom + sigma_stub_avg2;
                  float dk = kappa_geom - kappa_stub_avg;
                  dcaPassed = (dk * dk < geomSigCut * geomSigCut * combined_err2);

                  // Phi residual at middle hit: check that the actual phi of the middle hit
                  // matches the phi predicted from inner hit + weighted-average stub kappa.
                  // Provides orthogonal fake rejection, especially for endcap disk-to-disk triplets.
                  if (dcaPassed) {
                    auto phiMiddleCut = ll[thisCell.innerLayer()].caPhiMiddleCut();
                    if (phiMiddleCut >= 0.f && nStubs >= 2) {
                      // Reuse cross12 from pre-filter; compute dot12 for dphi_21
                      float dot12 = x1g * x2g + y1g * y2g;
                      float dphi_21 = std::atan2(cross12, dot12);
                      float dphi_mid = dphi_21 - kappa_stub_avg * (ri - r1);
                      if (dphi_mid > float(M_PI))
                        dphi_mid -= 2.f * float(M_PI);
                      if (dphi_mid < -float(M_PI))
                        dphi_mid += 2.f * float(M_PI);
                      dcaPassed = (std::abs(dphi_mid) < phiMiddleCut);
                      if (!dcaPassed && pipelineCounters) {
                        using PC = caHitNtupletGenerator::PipelineCounter;
                        alpaka::atomicAdd(
                            acc, &pipelineCounters[PC::kTripletPhiMiddleRej], 1u, alpaka::hierarchy::Blocks{});
                      }
                    }
                  }

                  // Compute geometric phi residual for chain consistency
                  if (dcaPassed) {
                    float dot12 = x1g * x2g + y1g * y2g;
                    float dphi_12 = std::atan2(cross12, dot12);
                    float dr_12 = ri - r1;   // r_middle - r_inner
                    tripletPhiResid = dphi_12 - dphi_13 * (dr_12 / dr_13);
                  }
                }
              } else {
                dcaPassed = true;  // no stubs or cut disabled
              }
            } else {
              [dcaPassed, curvature] = thisCell.dcaCut(hh, oc, dcaCut, params.hardCurvCut_, dcaFloor);
              if (dcaPassed) {
                // Compute phi residual from hit global positions
                float x1 = oc.inner_x(hh), y1 = oc.inner_y(hh);
                float x2 = thisCell.inner_x(hh), y2 = thisCell.inner_y(hh);
                float x3 = thisCell.outer_x(hh), y3 = thisCell.outer_y(hh);
                float cross12 = x1 * y2 - y1 * x2;
                float dot12 = x1 * x2 + y1 * y2;
                float dphi_12 = std::atan2(cross12, dot12);
                float cross13 = x1 * y3 - y1 * x3;
                float dot13 = x1 * x3 + y1 * y3;
                float dphi_13 = std::atan2(cross13, dot13);
                float dr_12 = thisCell.inner_r(hh) - oc.inner_r(hh);
                float dr_13 = thisCell.outer_r(hh) - oc.inner_r(hh);
                tripletPhiResid = dphi_12 - dphi_13 * (dr_12 / dr_13);
              }
            }
          } else {
            [dcaPassed, curvature] = thisCell.dcaCut(hh, oc, dcaCut, params.hardCurvCut_, dcaFloor);
            if (dcaPassed) {
              // Compute phi residual from hit global positions
              float x1 = oc.inner_x(hh), y1 = oc.inner_y(hh);
              float x2 = thisCell.inner_x(hh), y2 = thisCell.inner_y(hh);
              float x3 = thisCell.outer_x(hh), y3 = thisCell.outer_y(hh);
              float cross12 = x1 * y2 - y1 * x2;
              float dot12 = x1 * x2 + y1 * y2;
              float dphi_12 = std::atan2(cross12, dot12);
              float cross13 = x1 * y3 - y1 * x3;
              float dot13 = x1 * x3 + y1 * y3;
              float dphi_13 = std::atan2(cross13, dot13);
              float dr_12 = thisCell.inner_r(hh) - oc.inner_r(hh);
              float dr_13 = thisCell.outer_r(hh) - oc.inner_r(hh);
              tripletPhiResid = dphi_12 - dphi_13 * (dr_12 / dr_13);
            }
          }

#ifdef CA_DEBUG
          // Compute theta alignment value for debug output
          float radius_diff = std::abs(r1 - ro);
          float distance_13_squared = radius_diff * radius_diff + (z1 - zo) * (z1 - zo);
          float pMin = params.ptmin_ * std::sqrt(distance_13_squared);
          float tan_val = std::abs(z1 * (ri - ro) + zi * (ro - r1) + zo * (r1 - ri));
          float thetaAlignVal = tan_val * pMin;
          float thetaThreshold = effectiveThetaCut * distance_13_squared * radius_diff;

          // Compute DCA value for debug output
          auto x1d = oc.inner_x(hh);
          auto y1d = oc.inner_y(hh);
          auto x2d = thisCell.inner_x(hh);
          auto y2d = thisCell.inner_y(hh);
          auto x3d = thisCell.outer_x(hh);
          auto y3d = thisCell.outer_y(hh);
          CircleEq<float> eq(x1d, y1d, x2d, y2d, x3d, y3d);
          float curvature = std::abs(eq.curvature());
          float dcaVal = std::abs(eq.dca0());
          float dcaThreshold = dcaCut * curvature;

          // Determine stub types for all three hits
          const char *stubType1 = "pixel";
          const char *stubType2 = "pixel";
          const char *stubType3 = "pixel";
          if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
            auto hit1 = oc.inner_hit_id();
            auto hit2 = thisCell.inner_hit_id();
            auto hit3 = thisCell.outer_hit_id();
            if (hh[hit1].isStub()) {
              stubType1 = (hh[hit1].stubType() == ::reco::StubType::SS) ? "SS" : "PS";
            }
            if (hh[hit2].isStub()) {
              stubType2 = (hh[hit2].stubType() == ::reco::StubType::SS) ? "SS" : "PS";
            }
            if (hh[hit3].isStub()) {
              stubType3 = (hh[hit3].stubType() == ::reco::StubType::SS) ? "SS" : "PS";
            }
          }

          printf("TripletCheck;%d;%d;%d;%d;%d;%d;%d;%d;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%d;%d;%s;%s;%s;%d\n",
                 cellIndex,                      // outer cell index
                 otherCell,                      // inner cell index
                 thisCell.layerPairId(),         // outer layer pair
                 oc.layerPairId(),               // inner layer pair
                 oc.innerLayer(),                // innermost layer
                 thisCell.innerLayer(),          // middle layer
                 thisCell.outerLayer(),          // outermost layer
                 hasSSStub ? 1 : 0,              // has SS stub (theta skipped)
                 thetaAlignVal,                  // theta alignment value
                 thetaThreshold,                 // theta threshold
                 thetaCut,                       // raw thetaCut parameter
                 curvature,                      // curvature
                 dcaVal,                         // DCA value
                 dcaThreshold,                   // DCA threshold
                 dcaCut,                         // raw dcaCut parameter
                 params.hardCurvCut_,            // hard curvature cut
                 aligned ? 1 : 0,                // theta passed
                 dcaPassed ? 1 : 0,              // DCA passed
                 stubType1,                      // hit1 stub type
                 stubType2,                      // hit2 stub type
                 stubType3,                      // hit3 stub type
                 (aligned && dcaPassed) ? 1 : 0  // overall passed
          );
#endif

          if (aligned && dcaPassed) {
            auto t_ind = alpaka::atomicAdd(acc, nTrips, 1u, alpaka::hierarchy::Blocks{});
#ifdef CA_DEBUG
              printf("Triplet no. %d %.5f %.5f (%d %d) - %d %d -> (%d, %d, %d, %d) \n",
                     t_ind,
                     thetaCut,
                     dcaCut,
                     thisCell.layerPairId(),
                     oc.layerPairId(),
                     otherCell,
                     cellIndex,
                     thisCell.inner_hit_id(),
                     thisCell.outer_hit_id(),
                     oc.inner_hit_id(),
                     oc.outer_hit_id());
#endif

#ifdef CA_DEBUG
              printf("filling cell no. %d %d: %d -> %d\n", t_ind, cellNeighborsHisto->size(), otherCell, cellIndex);
#endif

              if (t_ind >= maxTriplets) {
#ifdef CA_WARNINGS
                printf("Warning!!!! Too many cell->cell (triplets) associations (limit = %d)!\n", cn.metadata().size());
#endif
                alpaka::atomicSub(acc, nTrips, 1u, alpaka::hierarchy::Blocks{});
                break;
              }

              cellNeighborsHisto->count(acc, otherCell);

              cn[t_ind].inner() = otherCell;
              cn[t_ind].outer() = {cellIndex, curvature};
              cn[t_ind].phiResid() = caStructures::quantizePhiResid(tripletPhiResid);
            thisCell.setStatusBits(Cell::StatusBit::kUsed);
              thisCell.setStatusBits(Cell::StatusBit::kHasInner);  // thisCell (outer) has an inner neighbor
            oc.setStatusBits(Cell::StatusBit::kUsed);
  
            // Pipeline stage counters: classify triplet by hit types
            if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
              if (pipelineCounters) {
                using PC = caHitNtupletGenerator::PipelineCounter;
                alpaka::atomicAdd(acc, &pipelineCounters[PC::kTripletsTotal], 1u, alpaka::hierarchy::Blocks{});
                auto hit1 = oc.inner_hit_id();
                auto hit2 = thisCell.inner_hit_id();
                auto hit3 = thisCell.outer_hit_id();
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
                  auto layer1 = oc.innerLayer();        // innermost
                  auto layer2 = thisCell.innerLayer();  // middle
                  auto layer3 = thisCell.outerLayer();  // outermost
                  bool l1Brl = (layer1 >= 28 && layer1 <= 33);
                  bool l2Brl = (layer2 >= 28 && layer2 <= 33);
                  bool l3Brl = (layer3 >= 28 && layer3 <= 33);
                  bool l1Bwd = (layer1 >= 34 && layer1 <= 38);
                  bool l2Bwd = (layer2 >= 34 && layer2 <= 38);
                  bool l3Bwd = (layer3 >= 34 && layer3 <= 38);
                  bool l1Fwd = (layer1 >= 39 && layer1 <= 43);
                  bool l2Fwd = (layer2 >= 39 && layer2 <= 43);
                  bool l3Fwd = (layer3 >= 39 && layer3 <= 43);
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

  // Like Kernel_fillGenericPair but also writes the phiResid from the CAPairSoA into a parallel array.
  // Used for the cellToNeighbors histogram so phiResid travels alongside neighbor cell IDs.
  class Kernel_fillGenericPairWithPhiResid {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  caStructures::CACellPairSoAConstView cn,
                                  uint32_t const *nElements,
                                  GenericContainer *genericHisto,
                                  int16_t *__restrict__ phiResidStorage) const {
      for (uint32_t index : cms::alpakatools::uniform_elements(acc, *nElements)) {
        auto b = cn[index].inner();
        ALPAKA_ASSERT_ACC(b < genericHisto->nOnes());
        auto w = GenericContainer::atomicDecrement(acc, genericHisto->off[b]);
        ALPAKA_ASSERT_ACC(w > 0);
        genericHisto->content[w - 1] = cn[index].outer();
        phiResidStorage[w - 1] = cn[index].phiResid();
      }
    }
  };

  // Sort each histogram bin by content value, co-sorting the parallel phiResid array.
  class Kernel_sortHistoBinsWithPhiResid {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  GenericContainer *histo,
                                  int16_t *__restrict__ phiResidStorage) const {
      for (auto idx : cms::alpakatools::uniform_elements(acc, histo->nOnes())) {
        auto size = histo->size(idx);
        if (size <= 1)
          continue;
        auto offset = histo->off[idx];
        auto *bin = histo->content.data() + offset;
        auto *kbin = phiResidStorage + offset;
        // Insertion sort: co-sort phiResid alongside content
        for (uint32_t i = 1; i < size; ++i) {
          auto key = bin[i];
          auto kkey = kbin[i];
          int j = i - 1;
          while (j >= 0 && bin[j] > key) {
            bin[j + 1] = bin[j];
            kbin[j + 1] = kbin[j];
            --j;
          }
          bin[j + 1] = key;
          kbin[j + 1] = kkey;
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
          auto c1 = nb1[j1];
          if (cells[c1].isKilled())
            continue;
          hasLiveNeighbor = true;
          registerLayer(cells[c1].outerLayer());

          // Hop 2: outer neighbors of hop-1 cells
          auto n2 = cellNeighborsHisto->size(c1);
          auto const *nb2 = cellNeighborsHisto->begin(c1);
          for (auto j2 = 0u; j2 < n2 && nSeen < minHops; ++j2) {
            auto c2 = nb2[j2];
            if (cells[c2].isKilled())
              continue;
            registerLayer(cells[c2].outerLayer());

            // Hop 3: outer neighbors of hop-2 cells
            auto n3 = cellNeighborsHisto->size(c2);
            auto const *nb3 = cellNeighborsHisto->begin(c2);
            for (auto j3 = 0u; j3 < n3 && nSeen < minHops; ++j3) {
              auto c3 = nb3[j3];
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
                                  AlgoParams const &params,
                                  int16_t const *__restrict__ connectionPhiResid) const {
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
        auto lid = thisCell.innerLayer();
        if (thisCell.inner_r() > ll[lid].startMaxInnerR())
          doit = false;

        constexpr uint32_t maxDepth = TrackerTraits::maxDepth;
#ifdef CA_DEBUG
        printf(
            "LayerPairId %d and inner layer %d doit ? %d From cell %d with nNeighbors = %d and innerR=%f < "
            "maxInnerR=%f ?\n",
            pid,
            lid,
            doit,
            idx,
            cellNeighborsHisto->size(idx),
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
                                                    stack,
                                                    params.minHitsPerNtuplet_,
                                                    connectionPhiResid,
                                                    params.chainPhiResidCut_);
          ALPAKA_ASSERT_ACC(stack.empty());
        }
      }
    }
  };

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
                                  HitsConstView hh) const {
      // copy offsets
      for (auto idx : cms::alpakatools::uniform_elements(acc, foundNtuplets->nOnes() - 1)) {
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

        auto score = [&](auto it, auto nl) { return std::abs(reco::tip(tracks_view, it)); };

        // full combinatorics
        for (auto ip = hitToTuple.begin(idx); ip < hitToTuple.end(idx) - 1; ++ip) {
          auto const it = *ip;
          auto qi = tracks_view[it].quality();
          if (qi <= reject)
            continue;
          auto opi = tracks_view[it].state()(2);
          auto e2opi = tracks_view[it].covariance()(9);
          auto cti = tracks_view[it].state()(3);
          auto e2cti = tracks_view[it].covariance()(12);
          auto nli = tracks_view[it].nLayers();
          for (auto jp = ip + 1; jp < hitToTuple.end(idx); ++jp) {
            auto const jt = *jp;
            auto qj = tracks_view[jt].quality();
            if (qj <= reject)
              continue;
            auto opj = tracks_view[jt].state()(2);
            auto ctj = tracks_view[jt].state()(3);
            auto dct = nSigma2 * (tracks_view[jt].covariance()(12) + e2cti);
            if ((cti - ctj) * (cti - ctj) > dct)
              continue;
            auto dop = nSigma2 * (tracks_view[jt].covariance()(9) + e2opi);
            if ((opi - opj) * (opi - opj) > dop)
              continue;
            auto nlj = tracks_view[jt].nLayers();
            if (nlj < nli || (nlj == nli && (qj < qi || (qj == qi && score(it, nli) < score(jt, nlj)))))
              tracks_view[jt].quality() = reject;
            else {
              tracks_view[it].quality() = reject;
              break;
            }
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
