#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAPixelDoubletsAlgos_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAPixelDoubletsAlgos_h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Math/interface/approx_atan2.h"
#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"

#include "CACell.h"
#include "CAPipelineCounters.h"
#include "CAStructures.h"

// #define GPU_DEBUG
// #define DOUBLETS_DEBUG  // Very verbose - enable only for detailed debugging
// #define CA_WARNINGS

namespace ALPAKA_ACCELERATOR_NAMESPACE::caPixelDoublets {
  using namespace cms::alpakatools;
  using namespace ::caStructures;
  using namespace ::reco;

  using HitToCell = GenericContainer;

  template <typename TrackerTraits>
  using PhiBinner = PhiBinnerT<TrackerTraits>;
  //Move this ^ definition in CAStructures maybe

  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool moduleIsOuterLadderPhase1(int const moduleId) {
    return (0 == (moduleId / 8) % 2);
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool moduleIsOuterLadderPhase2(int const moduleId) {
    return (0 != (moduleId / 18) % 2);
  }

  template <typename TrackerTraits>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool moduleIsOuterLadder(const int moduleId);

  template <>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool moduleIsOuterLadder<pixelTopology::Phase1>(int const moduleId) {
    return moduleIsOuterLadderPhase1(moduleId);
  }

  template <>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool moduleIsOuterLadder<pixelTopology::HIonPhase1>(int const moduleId) {
    return moduleIsOuterLadderPhase1(moduleId);
  }

  template <>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool moduleIsOuterLadder<pixelTopology::Phase2>(int const moduleId) {
    return moduleIsOuterLadderPhase2(moduleId);
  }

  template <>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool moduleIsOuterLadder<pixelTopology::Phase2OT>(int const moduleId) {
    return moduleIsOuterLadderPhase2(moduleId);
  }

  template <>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool moduleIsOuterLadder<pixelTopology::Phase2OTStubs>(int const moduleId) {
    return moduleIsOuterLadderPhase2(moduleId);
  }

  template <typename TrackerTraits, alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool zSizeCut(
      const TAcc& acc, HitsConstView hh, ::reco::CALayersSoAConstView ll, AlgoParams const& params, int i, int o) {
    const uint32_t mi = hh[i].detectorIndex();
    const auto first_forward = ll.layerStarts()[4];
    const auto first_bpix2 = ll.layerStarts()[1];
    bool innerB1 = mi < first_bpix2;
    bool isOuterLadder = moduleIsOuterLadder<TrackerTraits>(mi);
    auto mes = (!innerB1) || isOuterLadder ? hh[i].clusterSizeY() : -1;
#ifdef DOUBLETS_DEBUG
    printf("i = %d o = %d mi = %d innerB1 = %d isOuterLadder = %d first_forward = %d first_bpix2 = %d\n",
           i,
           o,
           mi,
           innerB1,
           isOuterLadder,
           first_forward,
           first_bpix2);
#endif
    if (mes < 0)
      return false;

    const uint32_t mo = hh[o].detectorIndex();
    auto so = hh[o].clusterSizeY();

    auto dz = hh[i].zGlobal() - hh[o].zGlobal();
    auto dr = hh[i].rGlobal() - hh[o].rGlobal();

    auto innerBarrel = mi < first_forward;
    auto onlyBarrel = mo < first_forward;
#ifdef DOUBLETS_DEBUG
    printf("i = %d o = %d mo = %d innerB1 = %d isOuterLadder = %d \n", i, o, mo, innerBarrel, onlyBarrel);
#endif
    if (not innerBarrel and not onlyBarrel)
      return false;
    auto dy = innerB1 ? params.maxDYsize12_ : params.maxDYsize_;
#ifdef DOUBLETS_DEBUG
    printf("i = %d o = %d dy = %d maxDYsize12_ = %d maxDYsize_ = %d dzdrFact_ = %.2f maxDYPred_ = %d \n",
           i,
           o,
           dy,
           params.maxDYsize12_,
           params.maxDYsize_,
           params.dzdrFact_,
           params.maxDYPred_);
#endif
    return onlyBarrel
               ? so > 0 && std::abs(so - mes) > dy
               : innerBarrel && std::abs(mes - int(std::abs(dz / dr) * params.dzdrFact_ + 0.5f)) > params.maxDYPred_;
  }

  template <typename TrackerTraits, alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool clusterCut(
      const TAcc& acc, HitsConstView hh, ::reco::CALayersSoAConstView ll, AlgoParams const& params, uint32_t i) {
    const uint32_t mi = hh[i].detectorIndex();
    const auto first_bpix2 = ll.layerStarts()[1];
    const auto first_bpix3 = ll.layerStarts()[2];
    bool innerB1orB2 = mi < ll.layerStarts()[2];
#ifdef DOUBLETS_DEBUG
    printf(
        "i = %d mi = %d innerB1orB2 = %d innerB1 = %d innerB2 = %d minYsizeB1 = %d minYsizeB2 = %d isOuterLadder = %d "
        "mes = %d \n",
        i,
        mi,
        innerB1orB2,
        mi < first_bpix2,
        (mi >= first_bpix2) && (mi < first_bpix3),
        params.minYsizeB1_,
        params.minYsizeB2_,
        (0 == (mi / 8) % 2),
        (!(mi < first_bpix2)) || (0 == (mi / 8) % 2) ? hh[i].clusterSizeY() : -1);
#endif
    if (!innerB1orB2)
      return false;

    bool innerB1 = mi < first_bpix2;

    bool isOuterLadder = moduleIsOuterLadder<TrackerTraits>(mi);
    auto mes = (!innerB1) || isOuterLadder ? hh[i].clusterSizeY() : -1;

    if (innerB1)  // B1
      if (mes > 0 && mes < params.minYsizeB1_)
        return true;  // only long cluster  (5*8)
    bool innerB2 = (mi >= first_bpix2) && (mi < first_bpix3);
    if (innerB2)  // B2 and F1
      if (mes > 0 && mes < params.minYsizeB2_)
        return true;

    return false;
  }

  template <typename TrackerTraits, alpaka::concepts::Acc TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void doubletsFromHisto(const TAcc& acc,
                                                        uint32_t maxNumOfDoublets,
                                                        CACell<TrackerTraits>* cells,
                                                        uint32_t* nCells,
                                                        HitsConstView hh,
                                                        ::reco::CAGraphSoAConstView cc,
                                                        ::reco::CALayersSoAConstView ll,
                                                        uint32_t const* __restrict__ offsets,
                                                        PhiBinner<TrackerTraits> const* phiBinner,
                                                        HitToCell* outerHitHisto,
                                                        AlgoParams const& params,
                                                        uint32_t* __restrict__ pipelineCounters) {
    const bool doClusterCut = params.minYsizeB1_ > 0 or params.minYsizeB2_ > 0;
    const bool doZSizeCut = params.maxDYsize12_ > 0 or params.maxDYsize_ > 0 or params.maxDYPred_ > 0;

    const uint32_t nPairs = cc.metadata().size();
    using PhiHisto = PhiBinner<TrackerTraits>;
    ALPAKA_ASSERT_ACC(offsets);

    auto layerSize = [=](uint8_t li) { return offsets[li + 1] - offsets[li]; };

    // nPairs for the OT-extended CA is 98.
    // If it should become much bigger than 64, consider using a block-wide parallel prefix scan,
    // e.g. see  https://nvlabs.github.io/cub/classcub_1_1_warp_scan.html
    auto& innerLayerCumulativeSize = alpaka::declareSharedVar<uint32_t[TrackerTraits::nPairs], __COUNTER__>(acc);
    auto& ntot = alpaka::declareSharedVar<uint32_t, __COUNTER__>(acc);

#ifdef DOUBLETS_DEBUG
    if (cms::alpakatools::once_per_grid(acc))
      printf(
          "maxNumDoublets = %d  cc.metadata().size() = %d ll.metadata().size() = %d ptmin_ = "
          "%.2f doClusterCut = %d  doPtCut = %d doZSizeCut = %d\n",
          maxNumOfDoublets,
          cc.metadata().size(),
          ll.metadata().size(),
          params.ptmin_,
          doClusterCut,
          params.ptmin_ > 0,
          doZSizeCut);
#endif

    if (cms::alpakatools::once_per_block(acc)) {
      innerLayerCumulativeSize[0] = layerSize(cc.graph()[0][0]);
      for (uint32_t i = 1; i < nPairs; ++i) {
        innerLayerCumulativeSize[i] = innerLayerCumulativeSize[i - 1] + layerSize(cc.graph()[i][0]);
      }
      ntot = innerLayerCumulativeSize[nPairs - 1];
    }
    alpaka::syncBlockThreads(acc);

    // declared outside the loop, as it cannot go backward
    uint32_t pairLayerId = 0;

    // outermost parallel loop, using all grid elements along the slower dimension (Y or 0 in a 2D grid)
    for (uint32_t j : cms::alpakatools::uniform_elements_y(acc, ntot)) {
      // move to lower_bound ?
      while (j >= innerLayerCumulativeSize[pairLayerId++])
        ;
      --pairLayerId;

      ALPAKA_ASSERT_ACC(pairLayerId < nPairs);
      ALPAKA_ASSERT_ACC(j < innerLayerCumulativeSize[pairLayerId]);
      ALPAKA_ASSERT_ACC(0 == pairLayerId || j >= innerLayerCumulativeSize[pairLayerId - 1]);

      uint8_t inner = cc.graph()[pairLayerId][0];
      uint8_t outer = cc.graph()[pairLayerId][1];
      ALPAKA_ASSERT_ACC(outer > inner);

#ifdef CA_PIPELINE_COUNTERS
      // Helper to count per-cut doublet rejections for 3 groups:
      // Total (all pairs), OTEarly (inner L28-29), OTLate (inner L30-32)
      // perPair=true: called inside the X-loop (each X-thread has a unique pair → count from all)
      // perPair=false: called before the X-loop (per inner hit → count only from first X-thread
      //   to avoid stride-x overcounting)
      auto countRej = [&](int cut, bool perPair = true) {
        if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
          if (pipelineCounters) {
            // For per-inner-hit rejections (before X-loop), only count from first X-thread
            if (!perPair && alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[1] != 0)
              return;
            using namespace caHitNtupletGenerator;
            alpaka::atomicAdd(
                acc, &pipelineCounters[kDblRejBase + kGroupTotal * kNCuts + cut], 1u, alpaka::hierarchy::Blocks{});
            if (inner == 28 || inner == 29)
              alpaka::atomicAdd(
                  acc, &pipelineCounters[kDblRejBase + kGroupOTEarly * kNCuts + cut], 1u, alpaka::hierarchy::Blocks{});
            else if (inner >= 30 && inner <= 32)
              alpaka::atomicAdd(
                  acc, &pipelineCounters[kDblRejBase + kGroupOTLate * kNCuts + cut], 1u, alpaka::hierarchy::Blocks{});
          }
        }
      };
#endif

      auto hoff = PhiHisto::histOff(outer);
      auto i = (0 == pairLayerId) ? j : j - innerLayerCumulativeSize[pairLayerId - 1];
      i += offsets[inner];

      ALPAKA_ASSERT_ACC(i >= offsets[inner]);
      ALPAKA_ASSERT_ACC(i < offsets[inner + 1]);
#ifdef DOUBLETS_DEBUG
      printf("pairLayerId = %d i = %d inner = %d outer = %d offsets[inner] = %d offsets[inner + 1] = %d\n",
             pairLayerId,
             i,
             inner,
             outer,
             offsets[inner],
             offsets[inner + 1]);
#endif
      // found hit corresponding to our worker thread, now do the job
      if (hh[i].detectorIndex() > ll.layerStarts()[ll.metadata().size() - 1]) {  //TODO use cc
#ifdef DOUBLETS_DEBUG
        printf("Killed here 1\n");
#endif
#ifdef CA_PIPELINE_COUNTERS
        countRej(caHitNtupletGenerator::kCutInvalidHit, false);  // per-inner-hit, not per-pair
#endif
        continue;  // invalid
      }

      /* maybe clever, not effective when zoCut is on
      auto bpos = (mi%8)/4;  // if barrel is 1 for z>0
      auto fpos = (outer>3) & (outer<7);
      if ( ((inner<3) & (outer>3)) && bpos!=fpos) continue;
      */

      auto zi = hh[i].zGlobal();
      auto ri = hh[i].rGlobal();

      // cut on inner coordinate (z or r depending on layer)
      auto valInner = ll.isBarrel()[inner] ? zi : ri;
      if (valInner < cc.minInner()[pairLayerId] || valInner > cc.maxInner()[pairLayerId]) {
#ifdef DOUBLETS_DEBUG
        printf("Killed here 2 --> valInner: %f [index: %d], minInner: %f, maxInner: %f\n",
               valInner,
               hh[i].detectorIndex(),
               cc.minInner()[pairLayerId],
               cc.maxInner()[pairLayerId]);
#endif
#ifdef CA_PIPELINE_COUNTERS
        countRej(caHitNtupletGenerator::kCutInnerCoord, false);  // per-inner-hit, not per-pair
#endif
        continue;
      }

#ifdef DOUBLETS_DEBUG
      if (doClusterCut && outer > pixelTopology::last_barrel_layer)
        printf("clustCut: %d %d \n", i, clusterCut<TrackerTraits, TAcc>(acc, hh, ll, params, i));
#endif

      if (doClusterCut && outer > pixelTopology::last_barrel_layer &&
          clusterCut<TrackerTraits, TAcc>(acc, hh, ll, params, i)) {
#ifdef DOUBLETS_DEBUG
        printf("Killed here 4\n");
#endif
#ifdef CA_PIPELINE_COUNTERS
        countRej(caHitNtupletGenerator::kCutClusterCut, false);  // per-inner-hit, not per-pair
#endif
        continue;
      }

      auto mep = hh[i].iphi();

#ifdef DOUBLETS_DEBUG
      if (inner >= 28) {  // Only print for OT layers
        printf("Inner hit: idx=%d layer=%d iphi=%d detIndex=%d zi=%.2f ri=%.2f\n",
               i,
               inner,
               mep,
               hh[i].detectorIndex(),
               zi,
               ri);
      }
#endif

      // all cuts: true if fails
      auto ptcut = [&](int j, int16_t idphi) {
        // ptCut already converted to minRadius2T4 in CAHitNtupletGenerator.cc
        auto ro = hh[j].rGlobal();
        auto dphi = short2phi(idphi);
        return dphi * dphi * (cc.ptCuts()[pairLayerId] - ri * ro) > (ro - ri) * (ro - ri);
      };
      auto z0cutoff = [&](int j) {
        auto zo = hh[j].zGlobal();
        auto ro = hh[j].rGlobal();
        auto dr = ro - ri;
#ifdef DOUBLETS_DEBUG
        printf("dr: %4.3f, %4.3f, %4.3f --> %d\n", ri, ro, dr, (dr > cc.maxDR()[pairLayerId]));
        printf("zi: %4.3f, zo: %4.3f, std::abs((zi * ro - ri * zo)): %4.3f --> %d\n",
               zi,
               zo,
               std::abs((zi * ro - ri * zo)),
               (std::abs((zi * ro - ri * zo)) > cc.z0Cuts()[pairLayerId] * dr));
#endif
        return dr > cc.maxDR()[pairLayerId] || dr < 0 || std::abs((zi * ro - ri * zo)) > cc.z0Cuts()[pairLayerId] * dr;
      };

      auto iphicut = cc.phiCuts()[pairLayerId];

      auto kl = PhiHisto::bin(int16_t(mep - iphicut));
      auto kh = PhiHisto::bin(int16_t(mep + iphicut));
      auto incr = [](auto& k) { return k = (k + 1) % PhiHisto::nbins(); };

#ifdef GPU_DEBUG
      // Only print for first few pairs to avoid flooding
      if (pairLayerId < 5 && i == 0) {
        auto innerLayer = cc.graph()[pairLayerId][0];
        auto outerLayer = cc.graph()[pairLayerId][1];
        printf(
            "[CAPixelDoublets] Pair %d: layers (%d->%d) | phiCut=%d | minIn=%.1f maxIn=%.1f | minOut=%.1f maxOut=%.1f "
            "| maxDR=%.1f | minDZ=%.1f maxDZ=%.1f\n",
            pairLayerId,
            innerLayer,
            outerLayer,
            cc.phiCuts()[pairLayerId],
            cc.minInner()[pairLayerId],
            cc.maxInner()[pairLayerId],
            cc.minOuter()[pairLayerId],
            cc.maxOuter()[pairLayerId],
            cc.maxDR()[pairLayerId],
            cc.minDZ()[pairLayerId],
            cc.maxDZ()[pairLayerId]);
      }
#endif

      auto khh = kh;
      incr(khh);
      for (auto kk = kl; kk != khh; incr(kk)) {
        //#ifdef GPU_DEBUG
        //        if (kk != kl && kk != kh)
        //          nmin += phiBinner->size(kk + hoff);
        //#endif

        auto const* __restrict__ p = phiBinner->begin(kk + hoff);
        auto const* __restrict__ e = phiBinner->end(kk + hoff);
        auto const maxpIndex = e - p;

#ifdef DOUBLETS_DEBUG
        if (outer >= 28) {  // Only print for OT layers
          printf("PhiBinner search: inner=%d outer=%d kk=%d hoff=%d phiBin=%d maxpIndex=%d\n",
                 inner,
                 outer,
                 kk,
                 hoff,
                 kk + hoff,
                 int(maxpIndex));
        }
#endif

        // innermost parallel loop, using the block elements along the faster dimension (X or 1 in a 2D grid)
        for (uint32_t pIndex : cms::alpakatools::independent_group_elements_x(acc, maxpIndex)) {
          // FIXME implement alpaka::ldg and use it here? or is it const* __restrict__ enough?
          auto oi = p[pIndex];
          ALPAKA_ASSERT_ACC(oi >= offsets[outer]);
          ALPAKA_ASSERT_ACC(oi < offsets[outer + 1]);
#ifdef DOUBLETS_DEBUG
          printf("Exploring couple i: %d o: %d\n", i, oi);
#endif
          auto mo = hh[oi].detectorIndex();

          // invalid - use TrackerTraits::numberOfModules for correct limit per tracker configuration
          if (mo >= TrackerTraits::numberOfModules) {
#ifdef DOUBLETS_DEBUG
            printf("Killed here 4 --> mo: %d >= numberOfModules: %d\n", mo, TrackerTraits::numberOfModules);
#endif
#ifdef CA_PIPELINE_COUNTERS
            countRej(caHitNtupletGenerator::kCutInvalidModule);
#endif
            continue;
          }

          auto zo = hh[oi].zGlobal();
          auto ro = hh[oi].rGlobal();

          // cut on outer coordinate (z or r depending on layer)
          auto valOuter = ll.isBarrel()[outer] ? zo : ro;
          if (valOuter < cc.minOuter()[pairLayerId] || valOuter > cc.maxOuter()[pairLayerId]) {
#ifdef DOUBLETS_DEBUG
            printf("Killed here 5 --> valOuter: %f [index: %d], minOuter: %f, maxOuter: %f\n",
                   valOuter,
                   mo,
                   cc.minOuter()[pairLayerId],
                   cc.maxOuter()[pairLayerId]);
#endif
#ifdef CA_PIPELINE_COUNTERS
            countRej(caHitNtupletGenerator::kCutOuterCoord);
#endif
            continue;
          }

          auto dz = zo - zi;

          // cut on signed dz
          if (dz < cc.minDZ()[pairLayerId] || dz > cc.maxDZ()[pairLayerId]) {
#ifdef DOUBLETS_DEBUG
            printf("Killed here 5 --> dz: %f [index: %d], minDZ: %f, maxDZ: %f\n",
                   dz,
                   mo,
                   cc.minDZ()[pairLayerId],
                   cc.maxDZ()[pairLayerId]);
#endif
#ifdef CA_PIPELINE_COUNTERS
            countRej(caHitNtupletGenerator::kCutDzRange);
#endif
            continue;
          }

          if (z0cutoff(oi)) {
#ifdef DOUBLETS_DEBUG
            printf("Killed here 5\n");
#endif
#ifdef CA_PIPELINE_COUNTERS
            countRej(caHitNtupletGenerator::kCutZ0);
#endif
            continue;
          }

          auto mop = hh[oi].iphi();
          uint16_t idphi = std::min(std::abs(int16_t(mop - mep)), std::abs(int16_t(mep - mop)));

          if (idphi > iphicut) {
#ifdef DOUBLETS_DEBUG
            printf("Killed here 6 --> idphi: %d, iphicut: %d\n", idphi, iphicut);
#endif
#ifdef CA_PIPELINE_COUNTERS
            countRej(caHitNtupletGenerator::kCutPhi);
#endif
            continue;
          }
#ifdef DOUBLETS_DEBUG
          printf("zSizeCut: %d %d %d \n", i, oi, zSizeCut<TrackerTraits, TAcc>(acc, hh, ll, params, i, oi));
#endif
          if (doZSizeCut && zSizeCut<TrackerTraits, TAcc>(acc, hh, ll, params, i, oi)) {
#ifdef DOUBLETS_DEBUG
            printf("Killed here 7\n");
#endif
#ifdef CA_PIPELINE_COUNTERS
            countRej(caHitNtupletGenerator::kCutZSize);
#endif
            continue;
          }

          if (cc.ptCuts()[pairLayerId] > 0. && ptcut(oi, idphi)) {
#ifdef DOUBLETS_DEBUG
            printf("Killed here 8\n");
#endif
#ifdef CA_PIPELINE_COUNTERS
            countRej(caHitNtupletGenerator::kCutPt);
#endif
            continue;
          }

          // Stub-stub pairwise compatibility cut using unified kappa (half-curvature) comparison.
          //
          // The stub formation kernel computes dPhiDr = dphi / dr_effective for all module types,
          // where dr_effective = separation / (cosTilt + sinTilt * z/r) projects the sensor gap
          // onto the radial direction. This makes dPhiDr a curvature proxy for all module types:
          //   - Flat barrel: dPhiDr = dphi/dr ~= kappa
          //   - Tilted barrel: parallax correction + dr_effective account for tilt -> dPhiDr ~= kappa
          //   - Endcap: dr_effective = separation * r/z cancels the dip angle -> dPhiDr ~= kappa
          //
          // The kappa transformation kappa = dPhiDr / sqrt(1 + r^2 * dPhiDr^2) extracts the
          // radius-independent half-curvature. For barrel stubs this removes the r-dependence;
          // for endcap stubs it is a near-identity transformation (since r^2 * dPhiDr^2 << 1).
          //
          // All stub-stub transitions are handled uniformly: flat-flat, flat-tilted, tilted-tilted,
          // disk-disk, flat-disk, tilted-disk.
          //
          // Pairs involving pixel hits or PHitOnly stubs are skipped.
          // Controlled by per-pair stubSigmaCut (negative = disabled).
          if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
            auto stubSigmaCut = cc.stubSigmaCut()[pairLayerId];
            if (stubSigmaCut > 0.f && hh[i].isStub() && hh[oi].isStub() &&
                hh[i].stubType() != ::reco::StubType::PHitOnly && hh[oi].stubType() != ::reco::StubType::PHitOnly) {
              // Unified kappa-corrected significance for all stub-stub pairs
              float d_i = hh[i].dPhiDr(), s_i = hh[i].dPhiDrError();
              float den_i = 1.f + ri * ri * d_i * d_i;
              float sqrt_den_i = std::sqrt(den_i);
              float k_i = d_i / sqrt_den_i;
              float sk_i = s_i / (den_i * sqrt_den_i);

              float d_o = hh[oi].dPhiDr(), s_o = hh[oi].dPhiDrError();
              float den_o = 1.f + ro * ro * d_o * d_o;
              float sqrt_den_o = std::sqrt(den_o);
              float k_o = d_o / sqrt_den_o;
              float sk_o = s_o / (den_o * sqrt_den_o);

              float combined_err2 = sk_i * sk_i + sk_o * sk_o;
              float significance = std::abs(k_i - k_o) / std::sqrt(combined_err2);

              if (significance > stubSigmaCut) {
#ifdef DOUBLETS_DEBUG
                auto flags_i = hh[i].stubFlags();
                auto flags_o = hh[oi].stubFlags();
                printf("Killed here 10: stub sigma cut (sig=%.2f > cut=%.2f, barrel_i=%d barrel_o=%d)\n",
                       significance,
                       stubSigmaCut,
                       (int)::reco::StubFlags::isBarrel(flags_i),
                       (int)::reco::StubFlags::isBarrel(flags_o));
#endif
#ifdef CA_PIPELINE_COUNTERS
                countRej(caHitNtupletGenerator::kCutStubSigma);
#endif
                continue;
              }
            }
          }

          // Pixel-to-stub direction consistency check using kappa comparison.
          // When only the outer hit is a stub, compute the doublet's kappa from the
          // pixel-stub geometry and compare with the stub's own kappa measurement.
          // This reuses stubSigmaCut as the significance threshold.
          if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
            auto stubSigmaCut = cc.stubSigmaCut()[pairLayerId];
            if (stubSigmaCut > 0.f && !hh[i].isStub() && hh[oi].isStub() &&
                hh[oi].stubType() != ::reco::StubType::PHitOnly) {
              auto signed_dphi = short2phi(int16_t(mop - mep));
              auto dr = ro - ri;
              if (dr > 0.f) {
                float dphidr_doublet = signed_dphi / dr;
                float den_d = 1.f + ro * ro * dphidr_doublet * dphidr_doublet;
                float sqrt_den_d = std::sqrt(den_d);
                float k_doublet = dphidr_doublet / sqrt_den_d;

                float d_o = hh[oi].dPhiDr(), s_o = hh[oi].dPhiDrError();
                float den_o = 1.f + ro * ro * d_o * d_o;
                float sqrt_den_o = std::sqrt(den_o);
                float k_stub = d_o / sqrt_den_o;
                float sk_stub = s_o / (den_o * sqrt_den_o);

                if (sk_stub > 0.f) {
                  float significance = std::abs(k_doublet - k_stub) / sk_stub;
                  if (significance > stubSigmaCut) {
#ifdef DOUBLETS_DEBUG
                    printf("Killed here 11: pixel-stub kappa cut (sig=%.2f > cut=%.2f)\n",
                           significance,
                           stubSigmaCut);
#endif
#ifdef CA_PIPELINE_COUNTERS
                    countRej(caHitNtupletGenerator::kCutPixStub);
#endif
                    continue;
                  }
                }
              }
            }
          }

          auto ind = alpaka::atomicAdd(acc, nCells, 1u, alpaka::hierarchy::Blocks{});
          if (ind >= maxNumOfDoublets or ind >= uint32_t(outerHitHisto->capacity())) {
#ifdef CA_WARNINGS
            printf("Warning!!!! Too many cells (maxNumOfDoublets = %d - nHitsToCell = %d)!\n",
                   maxNumOfDoublets,
                   outerHitHisto->capacity());
#endif
            alpaka::atomicSub(acc, nCells, 1u, alpaka::hierarchy::Blocks{});
            break;
          }

          outerHitHisto->count(acc, oi - hh.offsetBPIX2());
          cells[ind].init(hh, pairLayerId, inner, outer, i, oi);
#ifdef DOUBLETS_DEBUG
          printf("doublet: %d layerPair: %d inner: %d outer: %d i: %d oi: %d\n", ind, pairLayerId, inner, outer, i, oi);
#endif
          // Pipeline stage counters: classify doublet by hit type and layer pair
          if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
            if (pipelineCounters) {
              using PC = caHitNtupletGenerator::PipelineCounter;
              alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsTotal], 1u, alpaka::hierarchy::Blocks{});
              bool innerIsStub = hh[i].isStub();
              bool outerIsStub = hh[oi].isStub();
              if (!innerIsStub && !outerIsStub)
                alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsPixPix], 1u, alpaka::hierarchy::Blocks{});
              else if (!innerIsStub && outerIsStub)
                alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsPixOT], 1u, alpaka::hierarchy::Blocks{});
              else {
                alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsOTOT], 1u, alpaka::hierarchy::Blocks{});
                // Per-layer-pair breakdown for OT-OT doublets
                int key = int(inner) * 100 + int(outer);
                // Flat/tilted classification helper for barrel-barrel pairs
                auto classifyFlatTilted = [&](PC pairCounter, PC ffCounter, PC ftCounter, PC ttCounter) {
                  alpaka::atomicAdd(acc, &pipelineCounters[pairCounter], 1u, alpaka::hierarchy::Blocks{});
                  bool iFlat = (hh[i].stubFlags() & 0x02) != 0;
                  bool oFlat = (hh[oi].stubFlags() & 0x02) != 0;
                  if (iFlat && oFlat)
                    alpaka::atomicAdd(acc, &pipelineCounters[ffCounter], 1u, alpaka::hierarchy::Blocks{});
                  else if (!iFlat && !oFlat)
                    alpaka::atomicAdd(acc, &pipelineCounters[ttCounter], 1u, alpaka::hierarchy::Blocks{});
                  else
                    alpaka::atomicAdd(acc, &pipelineCounters[ftCounter], 1u, alpaka::hierarchy::Blocks{});
                };
                switch (key) {
                  // OT barrel consecutive (5 pairs) with flat/tilted breakdown
                  case 2829: classifyFlatTilted(PC::kDoubletsL28L29, PC::kDoubletsL28L29_FF, PC::kDoubletsL28L29_FT, PC::kDoubletsL28L29_TT); break;
                  case 2930: classifyFlatTilted(PC::kDoubletsL29L30, PC::kDoubletsL29L30_FF, PC::kDoubletsL29L30_FT, PC::kDoubletsL29L30_TT); break;
                  case 3031: classifyFlatTilted(PC::kDoubletsL30L31, PC::kDoubletsL30L31_FF, PC::kDoubletsL30L31_FT, PC::kDoubletsL30L31_TT); break;
                  case 3132: classifyFlatTilted(PC::kDoubletsL31L32, PC::kDoubletsL31L32_FF, PC::kDoubletsL31L32_FT, PC::kDoubletsL31L32_TT); break;
                  case 3233: classifyFlatTilted(PC::kDoubletsL32L33, PC::kDoubletsL32L33_FF, PC::kDoubletsL32L33_FT, PC::kDoubletsL32L33_TT); break;
                  // OT barrel to backward disk 1 (6 pairs)
                  case 2834: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsL28D1B], 1u, alpaka::hierarchy::Blocks{}); break;
                  case 2934: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsL29D1B], 1u, alpaka::hierarchy::Blocks{}); break;
                  case 3034: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsL30D1B], 1u, alpaka::hierarchy::Blocks{}); break;
                  case 3134: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsL31D1B], 1u, alpaka::hierarchy::Blocks{}); break;
                  case 3234: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsL32D1B], 1u, alpaka::hierarchy::Blocks{}); break;
                  case 3334: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsL33D1B], 1u, alpaka::hierarchy::Blocks{}); break;
                  case 3335: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsL33D2B], 1u, alpaka::hierarchy::Blocks{}); break;
                  // OT barrel to forward disk 1 (6 pairs)
                  case 2839: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsL28D1F], 1u, alpaka::hierarchy::Blocks{}); break;
                  case 2939: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsL29D1F], 1u, alpaka::hierarchy::Blocks{}); break;
                  case 3039: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsL30D1F], 1u, alpaka::hierarchy::Blocks{}); break;
                  case 3139: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsL31D1F], 1u, alpaka::hierarchy::Blocks{}); break;
                  case 3239: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsL32D1F], 1u, alpaka::hierarchy::Blocks{}); break;
                  case 3339: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsL33D1F], 1u, alpaka::hierarchy::Blocks{}); break;
                  case 3340: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsL33D2F], 1u, alpaka::hierarchy::Blocks{}); break;
                  // Backward disk chain (4 pairs)
                  case 3435: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsD1BD2B], 1u, alpaka::hierarchy::Blocks{}); break;
                  case 3536: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsD2BD3B], 1u, alpaka::hierarchy::Blocks{}); break;
                  case 3637: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsD3BD4B], 1u, alpaka::hierarchy::Blocks{}); break;
                  case 3738: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsD4BD5B], 1u, alpaka::hierarchy::Blocks{}); break;
                  // Forward disk chain (4 pairs)
                  case 3940: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsD1FD2F], 1u, alpaka::hierarchy::Blocks{}); break;
                  case 4041: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsD2FD3F], 1u, alpaka::hierarchy::Blocks{}); break;
                  case 4142: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsD3FD4F], 1u, alpaka::hierarchy::Blocks{}); break;
                  case 4243: alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsD4FD5F], 1u, alpaka::hierarchy::Blocks{}); break;
                  default:   alpaka::atomicAdd(acc, &pipelineCounters[PC::kDoubletsOTOther], 1u, alpaka::hierarchy::Blocks{}); break;
                }
              }
            }
          }
        }
      }
    }  // loop in block...
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caPixelDoublets

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAPixelDoubletsAlgos_h
