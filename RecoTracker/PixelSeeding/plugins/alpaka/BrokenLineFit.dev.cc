// #define BROKENLINE_DEBUG
// #define BL_DUMP_HITS
// #define GPU_DEBUG
// #define FIT_DEBUG
// #define FIT_ERROR_DEBUG  // Detailed error transformation debug for OT hits
#include <cstdint>

#if defined(FIT_DEBUG) || defined(GPU_DEBUG) || defined(FIT_ERROR_DEBUG)
#include <iostream>
#endif

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"

#include "HelixFit.h"

using OutputSoAView = reco::TrackSoAView;
using TupleMultiplicity = caStructures::GenericContainer;
using Tuples = caStructures::SequentialContainer;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <int N>
  class Kernel_BLFastFit {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  Tuples const* __restrict__ foundNtuplets,
                                  TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                  ::reco::TrackingRecHitConstView hh,
                                  ::reco::CAModulesConstView cm,
                                  typename caStructures::tindex_type* __restrict__ ptkids,
                                  double* __restrict__ phits,
                                  float* __restrict__ phits_ge,
                                  double* __restrict__ pfast_fit,
                                  uint32_t nHitsL,
                                  uint32_t nHitsH,
                                  int32_t offset,
                                  ::reco::OTRecHitsConstView otRecHits = ::reco::OTRecHitsConstView(),
                                  ::reco::StubsConstView stubs = ::reco::StubsConstView(),
                                  int32_t offsetStubs = -1) const {
      constexpr uint32_t hitsInFit = N;
      constexpr auto invalidTkId = std::numeric_limits<typename caStructures::tindex_type>::max();

      ALPAKA_ASSERT_ACC(hitsInFit <= nHitsL);
      ALPAKA_ASSERT_ACC(nHitsL <= nHitsH);
      ALPAKA_ASSERT_ACC(phits);
      ALPAKA_ASSERT_ACC(pfast_fit);
      ALPAKA_ASSERT_ACC(foundNtuplets);
      ALPAKA_ASSERT_ACC(tupleMultiplicity);

      // look in bin for this hit multiplicity
      int totTK = tupleMultiplicity->end(nHitsH) - tupleMultiplicity->begin(nHitsL);
      ALPAKA_ASSERT_ACC(totTK <= int(tupleMultiplicity->size()));
      ALPAKA_ASSERT_ACC(totTK >= 0);

#ifdef BROKENLINE_DEBUG
      if (cms::alpakatools::once_per_grid(acc)) {
        printf("%d total Ntuple\n", tupleMultiplicity->size());
        printf("%d Ntuple of size %d/%d for %d hits to fit\n", totTK, nHitsL, nHitsH, hitsInFit);
      }
#endif
      const auto nt = riemannFit::maxNumberOfConcurrentFits;
      for (auto local_idx : cms::alpakatools::uniform_elements(acc, nt)) {
        auto tuple_idx = local_idx + offset;
        if ((int)tuple_idx >= totTK) {
          ptkids[local_idx] = invalidTkId;
          break;
        }
        // get it from the ntuple container (one to one to helix)
        auto tkid = *(tupleMultiplicity->begin(nHitsL) + tuple_idx);
        ALPAKA_ASSERT_ACC(tkid < foundNtuplets->nOnes());

        ptkids[local_idx] = tkid;

        auto nHits = foundNtuplets->size(tkid);

        ALPAKA_ASSERT_ACC(nHits >= nHitsL);
        ALPAKA_ASSERT_ACC(nHits <= nHitsH);

        riemannFit::Map3xNd<N> hits(phits + local_idx);
        riemannFit::Map4d fast_fit(pfast_fit + local_idx);
        riemannFit::Map6xNf<N> hits_ge(phits_ge + local_idx);

        // Prepare data structure
        auto const* hitId = foundNtuplets->begin(tkid);

        // #define YERR_FROM_DC
#ifdef YERR_FROM_DC
        // try to compute more precise error in y
        auto dx = hh[hitId[hitsInFit - 1]].xGlobal() - hh[hitId[0]].xGlobal();
        auto dy = hh[hitId[hitsInFit - 1]].yGlobal() - hh[hitId[0]].yGlobal();
        auto dz = hh[hitId[hitsInFit - 1]].zGlobal() - hh[hitId[0]].zGlobal();
        float ux, uy, uz;
#endif

        // Uniform sampling: select hitsInFit hits uniformly from the full track
        uint32_t selectedHits[N];
        uint32_t nSelected = 0;

        {
          float incr = std::max(1.f, float(nHits) / float(hitsInFit));
          float fn = 0;
          for (uint32_t i = 0; i < hitsInFit; ++i) {
            int j = int(fn + 0.5f);  // round
            if (hitsInFit - 1 == i)
              j = nHits - 1;  // force last hit to ensure max lever arm
            ALPAKA_ASSERT_ACC(j < int(nHits));
            selectedHits[nSelected++] = j;
            fn += incr;
          }
        }

        ALPAKA_ASSERT_ACC(nSelected == hitsInFit);

#ifdef BL_DUMP_HITS
        // Pre-scan: detect whether track has stubs and classify hit types for diagnostics
        bool trackHasStubs = false;
        {
          // Count original (pre-sampling) hit composition over ALL hits on the track
          uint32_t oPix = 0, oPS = 0, oSS = 0, oPHit = 0;
          for (uint32_t oi = 0; oi < nHits; ++oi) {
            auto origHit = hitId[oi];
            bool origIsStub = (offsetStubs >= 0 && hh[origHit].isStub());
            if (!origIsStub) {
              oPix++;
            } else {
              auto st = hh[origHit].stubType();
              if (st == ::reco::StubType::PS)
                oPS++;
              else if (st == ::reco::StubType::SS)
                oSS++;
              else
                oPHit++;
            }
          }

          // Count selected (sampled for fit) hit composition
          uint32_t nPix = 0, nPS = 0, nSS = 0, nPHit = 0;
          uint8_t hitTypes[N];
          for (uint32_t si = 0; si < hitsInFit; ++si) {
            auto selHit = hitId[selectedHits[si]];
            bool selIsStub = (offsetStubs >= 0 && hh[selHit].isStub());
            if (!selIsStub) {
              hitTypes[si] = 0;
              nPix++;
            } else {
              auto st = hh[selHit].stubType();
              if (st == ::reco::StubType::PS) {
                hitTypes[si] = 1;
                nPS++;
              } else if (st == ::reco::StubType::SS) {
                hitTypes[si] = 2;
                nSS++;
              } else {
                hitTypes[si] = 3;
                nPHit++;
              }
            }
          }
          // Print if at least one stub hit is on the track (original composition)
          trackHasStubs = (oPS + oSS + oPHit > 0);
          if (trackHasStubs) {
            printf("BLFit: tkid=%d nHits=%d fitN=%d offsetStubs=%d orig=[oPix=%d oPS=%d oSS=%d oPHit=%d] sel=[",
                   tkid, nHits, hitsInFit, offsetStubs, oPix, oPS, oSS, oPHit);
            for (uint32_t si = 0; si < hitsInFit; ++si) {
              const char* label = (hitTypes[si] == 0)   ? "pix"
                                  : (hitTypes[si] == 1) ? "PS"
                                  : (hitTypes[si] == 2) ? "SS"
                                                        : "PHit";
              if (si > 0)
                printf(" ");
              printf("%s(%d:%d)", label, selectedHits[si], hitId[selectedHits[si]]);
            }
            printf("] nPix=%d nPS=%d nSS=%d nPHit=%d\n", nPix, nPS, nSS, nPHit);
          }
        }
#endif

        for (uint32_t i = 0; i < hitsInFit; ++i) {
          int j = selectedHits[i];
          auto hit = hitId[j];
          float ge[6];

          // Check if this hit is a stub and needs special handling
          // Views are valid if offsetStubs >= 0 (they have data pointers set)
          bool isStub = (offsetStubs >= 0 && hh[hit].isStub());
          bool isSSStub = false;
          uint32_t sensorHitIdx = hit;  // Default to using merged hit index

          if (isStub) {
            auto stubType = hh[hit].stubType();
            if (stubType == ::reco::StubType::SS) {
              // SS stub: fetch inner sensor hit from OTRecHitsSoA
              isSSStub = true;
              uint32_t stubIdx = hit - offsetStubs;
              sensorHitIdx = stubs[stubIdx].innerHitIdx();
            }
            // PS stubs use merged data directly (already has inner/pixel sensor)
          }

#ifdef YERR_FROM_DC
          auto const& dp = cm->detParams(hh.detectorIndex(hit));
          auto status = hh[hit].chargeAndStatus().status;
          int qbin = CPEFastParametrisation::kGenErrorQBins - 1 - status.qBin;
          ALPAKA_ASSERT_ACC(qbin >= 0 && qbin < 5);
          bool nok = (status.isBigY | status.isOneY);
          // compute cotanbeta and use it to recompute error
          dp.frame.rotation().multiply(dx, dy, dz, ux, uy, uz);
          auto cb = std::abs(uy / uz);
          int bin =
              int(cb * (float(phase1PixelTopology::pixelThickess) / float(phase1PixelTopology::pixelPitchY)) * 8.f) - 4;
          int low_value = 0;
          int high_value = CPEFastParametrisation::kNumErrorBins - 1;
          // return estimated bin value truncated to [0, 15]
          bin = std::clamp(bin, low_value, high_value);
          float yerr = dp.sigmay[bin] * 1.e-4f;  // toCM
          yerr *= dp.yfact[qbin];                // inflate
          yerr *= yerr;
          yerr += dp.apeYY;
          yerr = nok ? hh[hit].yerrLocal() : yerr;
          dp.frame.toGlobal(hh[hit].xerrLocal(), 0, yerr, ge);
#else
          if (isSSStub) {
            // Use pre-computed global errors from OT sensor hit
            ge[0] = otRecHits[sensorHitIdx].xGlobalErr();
            ge[1] = otRecHits[sensorHitIdx].xyGlobalErr();
            ge[2] = otRecHits[sensorHitIdx].yGlobalErr();
            ge[3] = otRecHits[sensorHitIdx].xzGlobalErr();
            ge[4] = otRecHits[sensorHitIdx].yzGlobalErr();
            ge[5] = otRecHits[sensorHitIdx].zGlobalErr();
          } else {
            // Use standard frame transformation for pixel hits and PS stubs
            auto const& frame = cm.detFrame(hh.detectorIndex(hit));
            frame.toGlobal(hh[hit].xerrLocal(), 0, hh[hit].yerrLocal(), ge);
          }
#endif

#ifdef FIT_ERROR_DEBUG
          // Debug output for OT hits only (detectorIndex >= pixel layers)
          // This helps compare error transformation between CAExtension (Phase2OT) and CA+Stubs (Phase2OTStubs)
          {
            auto detIdx = hh.detectorIndex(hit);
            // OT modules typically have detectorIndex > ~1500 (after all pixel modules)
            // For Phase2, pixel ends around ~1500, OT starts after
            // We check isStub flag to distinguish Phase2OTStubs path
            bool isOTHit = (detIdx > 1500);  // Rough threshold to skip pixel hits
            if (isOTHit) {
              auto const& frame = cm.detFrame(detIdx);
              auto const& rot = frame.rotation();

              // Compute what frame.toGlobal would give (for comparison with pre-computed SS errors)
              float ge_frame[6];
              frame.toGlobal(hh[hit].xerrLocal(), 0, hh[hit].yerrLocal(), ge_frame);

              printf("=== FIT_ERROR_DEBUG tkid=%d hit_idx=%d fit_slot=%d ===\n", tkid, hit, i);
              printf("  detectorIndex: %u\n", detIdx);
              printf("  isStub: %d  stubType: %u (0=PS, 1=SS, 2=PHitOnly)\n",
                     isStub ? 1 : 0, isStub ? hh[hit].stubType() : 255);
              printf("  offsetStubs: %d  sensorHitIdx: %u\n", offsetStubs, sensorHitIdx);
              printf("  Local errors: xerrLocal=%.6e  yerrLocal=%.6e\n",
                     hh[hit].xerrLocal(), hh[hit].yerrLocal());
              printf("  Global position (merged): x=%.4f  y=%.4f  z=%.4f  r=%.4f\n",
                     hh[hit].xGlobal(), hh[hit].yGlobal(), hh[hit].zGlobal(),
                     sqrt(hh[hit].xGlobal() * hh[hit].xGlobal() + hh[hit].yGlobal() * hh[hit].yGlobal()));
              if (isSSStub) {
                printf("  Global position (SS sensor): x=%.4f  y=%.4f  z=%.4f\n",
                       otRecHits[sensorHitIdx].xGlobal(),
                       otRecHits[sensorHitIdx].yGlobal(),
                       otRecHits[sensorHitIdx].zGlobal());
              }
              printf("  Frame position: x=%.4f  y=%.4f  z=%.4f\n",
                     frame.x(), frame.y(), frame.z());
              printf("  Frame rotation (row1): xx=%.6f  xy=%.6f  xz=%.6f\n",
                     rot.xx(), rot.xy(), rot.xz());
              printf("  Frame rotation (row2): yx=%.6f  yy=%.6f  yz=%.6f\n",
                     rot.yx(), rot.yy(), rot.yz());
              printf("  Frame rotation (row3): zx=%.6f  zy=%.6f  zz=%.6f\n",
                     rot.zx(), rot.zy(), rot.zz());
              printf("  Global errors (used): ge[0]=%.6e  ge[2]=%.6e  ge[5]=%.6e\n",
                     ge[0], ge[2], ge[5]);
              printf("  Global errors (full): ge[0-5]=%.6e %.6e %.6e %.6e %.6e %.6e\n",
                     ge[0], ge[1], ge[2], ge[3], ge[4], ge[5]);
              if (isSSStub) {
                printf("  Global errors (frame.toGlobal): ge[0]=%.6e  ge[2]=%.6e  ge[5]=%.6e\n",
                       ge_frame[0], ge_frame[2], ge_frame[5]);
                printf("  Global errors (frame, full): ge[0-5]=%.6e %.6e %.6e %.6e %.6e %.6e\n",
                       ge_frame[0], ge_frame[1], ge_frame[2], ge_frame[3], ge_frame[4], ge_frame[5]);
                printf("  ERROR DIFF (pre-computed - frame): dge[0]=%.6e  dge[2]=%.6e  dge[5]=%.6e\n",
                       ge[0] - ge_frame[0], ge[2] - ge_frame[2], ge[5] - ge_frame[5]);
              }
              printf("  Errors [um]: sqrt_ge0=%.2f  sqrt_ge2=%.2f  sqrt_ge5=%.2f\n",
                     1.e4 * sqrt(ge[0]), 1.e4 * sqrt(ge[2]), 1.e4 * sqrt(ge[5]));
              printf("=== END FIT_ERROR_DEBUG ===\n");
            }
          }
#endif

#ifdef BL_DUMP_HITS
          // Per-hit error diagnostic: print ge[] for every hit on stub-containing tracks
          // Limited to first 100 track IDs to avoid overwhelming output
          if (trackHasStubs && tkid < 100) {
            const char* hitType = "pix";
            if (isStub) {
              if (isSSStub)
                hitType = "SS";
              else if (hh[hit].stubType() == ::reco::StubType::PS)
                hitType = "PS";
              else
                hitType = "PHit";
            }
            float localXerr = hh[hit].xerrLocal();
            float localYerr = hh[hit].yerrLocal();
            printf("  BLErr tkid=%d hit %d/%d: type=%-4s hitIdx=%d localErr=[%.2e, %.2e] "
                   "ge=[%.2e %.2e %.2e %.2e %.2e %.2e] sqrt(ge)=[%.1f %.1f %.1f]um\n",
                   tkid, i, hitsInFit, hitType, hit,
                   localXerr, localYerr,
                   ge[0], ge[1], ge[2], ge[3], ge[4], ge[5],
                   1.e4f * sqrtf(std::abs(ge[0])),
                   1.e4f * sqrtf(std::abs(ge[2])),
                   1.e4f * sqrtf(std::abs(ge[5])));
          }
#endif

          // Fill position - use sensor hit for SS stubs, merged hit otherwise
          if (isSSStub) {
            hits.col(i) << otRecHits[sensorHitIdx].xGlobal(), otRecHits[sensorHitIdx].yGlobal(),
                otRecHits[sensorHitIdx].zGlobal();
          } else {
            hits.col(i) << hh[hit].xGlobal(), hh[hit].yGlobal(), hh[hit].zGlobal();
          }
          hits_ge.col(i) << ge[0], ge[1], ge[2], ge[3], ge[4], ge[5];
        }
        brokenline::fastFit(acc, hits, fast_fit);
#if 0
      printf("Fast Fit: %f, %f, %f, %f\n", fast_fit(0), fast_fit(1), fast_fit(2), fast_fit(3));
#endif

#ifdef BROKENLINE_DEBUG
        // any NaN value should cause the track to be rejected at a later stage
        ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(0)));
        ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(1)));
        ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(2)));
        ALPAKA_ASSERT_ACC(not alpaka::math::isnan(acc, fast_fit(3)));
#endif
      }
    }
  };

  template <int N, typename TrackerTraits>
  struct Kernel_BLFit {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  TupleMultiplicity const* __restrict__ tupleMultiplicity,
                                  double bField,
                                  OutputSoAView results_view,
                                  typename caStructures::tindex_type const* __restrict__ ptkids,
                                  double* __restrict__ phits,
                                  float* __restrict__ phits_ge,
                                  double* __restrict__ pfast_fit) const {
      ALPAKA_ASSERT_ACC(results_view.pt().data());
      ALPAKA_ASSERT_ACC(results_view.eta().data());
      ALPAKA_ASSERT_ACC(results_view.chi2().data());
      ALPAKA_ASSERT_ACC(pfast_fit);

      constexpr auto invalidTkId = std::numeric_limits<typename caStructures::tindex_type>::max();

      // same as above...
      // look in bin for this hit multiplicity
      const auto nt = riemannFit::maxNumberOfConcurrentFits;
      for (auto local_idx : cms::alpakatools::uniform_elements(acc, nt)) {
        if (invalidTkId == ptkids[local_idx])
          break;
        auto tkid = ptkids[local_idx];

        ALPAKA_ASSERT_ACC(tkid < tupleMultiplicity->capacity());

        riemannFit::Map3xNd<N> hits(phits + local_idx);
        riemannFit::Map4d fast_fit(pfast_fit + local_idx);
        riemannFit::Map6xNf<N> hits_ge(phits_ge + local_idx);

        brokenline::PreparedBrokenLineData<N> data;

        brokenline::karimaki_circle_fit circle;
        riemannFit::LineFit line;

        brokenline::prepareBrokenLineData(acc, hits, fast_fit, bField, data);
        brokenline::lineFit(acc, hits_ge, fast_fit, bField, data, line);
        brokenline::circleFit(acc, hits, hits_ge, fast_fit, bField, data, circle);

        reco::copyFromCircle(results_view, circle.par, circle.cov, line.par, line.cov, 1.f / float(bField), tkid);
        results_view[tkid].pt() = float(bField) / float(std::abs(circle.par(2)));
        results_view[tkid].eta() = alpaka::math::asinh(acc, line.par(0));
        results_view[tkid].chi2() = (circle.chi2 + line.chi2) / (2 * N - 5);

#ifdef BROKENLINE_DEBUG
        if (!(circle.chi2 >= 0) || !(line.chi2 >= 0))
          printf("kernelBLFit failed! %f/%f\n", circle.chi2, line.chi2);
        printf("kernelBLFit size %d for %d hits of tkid %d circle.par(0,1,2): %f,%f,%f\n",
               N,
               N,
               tkid,
               circle.par(0),
               circle.par(1),
               circle.par(2));
        printf("kernelBLHits line.par(0,1): %d %f,%f\n", tkid, line.par(0), line.par(1));
        printf("kernelBLHits chi2_circle: %f chi2_line: %f, cov(0-3)_circle: %e, %e, %e cov(1-2)_line %e,%e\n",
               circle.chi2,
               line.chi2,
               circle.cov(0, 0),
               circle.cov(1, 1),
               circle.cov(2, 2),
               line.cov(0, 0),
               line.cov(1, 1));
#endif
      }
    }
  };

  template <typename TrackerTraits>
  void HelixFit<TrackerTraits>::launchBrokenLineKernels(const ::reco::TrackingRecHitConstView& hv,
                                                        const ::reco::CAModulesConstView& cm,
                                                        uint32_t hitsInFit,
                                                        uint32_t maxNumberOfTuples,
                                                        Queue& queue) {
    ALPAKA_ASSERT_ACC(tuples_);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting HelixFit<TrackerTraits>::launchBrokenLineKernels" << std::endl;
#endif

    uint32_t blockSize = 64;
    uint32_t numberOfBlocks = cms::alpakatools::divide_up_by(maxNumberOfConcurrentFits_, blockSize);
    const WorkDiv1D workDivTriplets = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    const WorkDiv1D workDivQuadsPenta = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks / 4, blockSize);

    //  Fit internals
    auto tkidDevice =
        cms::alpakatools::make_device_buffer<typename caStructures::tindex_type[]>(queue, maxNumberOfConcurrentFits_);
    constexpr auto maxN = TrackerTraits::maxHitsOnTrackForFullFit;
    auto hitsDevice = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix3xNd<maxN>) / sizeof(double));
    auto hits_geDevice = cms::alpakatools::make_device_buffer<float[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix6xNf<maxN>) / sizeof(float));
    auto fast_fit_resultsDevice = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Vector4d) / sizeof(double));

    for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
      // fit triplets

      alpaka::exec<Acc1D>(queue,
                          workDivTriplets,
                          Kernel_BLFastFit<3>{},
                          tuples_,
                          tupleMultiplicity_,
                          hv,
                          cm,
                          tkidDevice.data(),
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          3,
                          3,
                          offset);
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_BLFastFit(3) -> done! " << std::endl;
#endif
      alpaka::exec<Acc1D>(queue,
                          workDivTriplets,
                          Kernel_BLFit<3, TrackerTraits>{},
                          tupleMultiplicity_,
                          bField_,
                          outputSoa_,
                          tkidDevice.data(),
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data());
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_BLFit(3) -> done! " << std::endl;
#endif

      if (fitNas4_) {
        // fit all as 4
        riemannFit::rolling_fits<4, TrackerTraits::maxHitsOnTrack, 1>([this,
                                                                       &hv,
                                                                       &cm,
                                                                       &tkidDevice,
                                                                       &hitsDevice,
                                                                       &hits_geDevice,
                                                                       &fast_fit_resultsDevice,
                                                                       &offset,
                                                                       &queue,
                                                                       &workDivQuadsPenta](auto i) {
          alpaka::exec<Acc1D>(queue,
                              workDivQuadsPenta,
                              Kernel_BLFastFit<4>{},
                              tuples_,
                              tupleMultiplicity_,
                              hv,
                              cm,
                              tkidDevice.data(),
                              hitsDevice.data(),
                              hits_geDevice.data(),
                              fast_fit_resultsDevice.data(),
                              4,
                              4,
                              offset);

          alpaka::exec<Acc1D>(queue,
                              workDivQuadsPenta,
                              Kernel_BLFit<4, TrackerTraits>{},
                              tupleMultiplicity_,
                              bField_,
                              outputSoa_,
                              tkidDevice.data(),
                              hitsDevice.data(),
                              hits_geDevice.data(),
                              fast_fit_resultsDevice.data());
        });
      } else {
        riemannFit::rolling_fits<4, TrackerTraits::maxHitsOnTrackForFullFit, 1>([this,
                                                                                 &hv,
                                                                                 &cm,
                                                                                 &tkidDevice,
                                                                                 &hitsDevice,
                                                                                 &hits_geDevice,
                                                                                 &fast_fit_resultsDevice,
                                                                                 &offset,
                                                                                 &queue,
                                                                                 &workDivQuadsPenta](auto i) {
          alpaka::exec<Acc1D>(queue,
                              workDivQuadsPenta,
                              Kernel_BLFastFit<i>{},
                              tuples_,
                              tupleMultiplicity_,
                              hv,
                              cm,
                              tkidDevice.data(),
                              hitsDevice.data(),
                              hits_geDevice.data(),
                              fast_fit_resultsDevice.data(),
                              i,
                              i,
                              offset);

          alpaka::exec<Acc1D>(queue,
                              workDivQuadsPenta,
                              Kernel_BLFit<i, TrackerTraits>{},
                              tupleMultiplicity_,
                              bField_,
                              outputSoa_,
                              tkidDevice.data(),
                              hitsDevice.data(),
                              hits_geDevice.data(),
                              fast_fit_resultsDevice.data());
#ifdef GPU_DEBUG
          alpaka::wait(queue);
          std::cout << "Kernel_BLFastFit(" << i << ") and Kernel_BLFit(" << i << ") -> done! " << std::endl;
#endif
        });

        static_assert(TrackerTraits::maxHitsOnTrackForFullFit < TrackerTraits::maxHitsOnTrack);

        //Fit all the rest using the maximum from previous call
        alpaka::exec<Acc1D>(queue,
                            workDivQuadsPenta,
                            Kernel_BLFastFit<TrackerTraits::maxHitsOnTrackForFullFit>{},
                            tuples_,
                            tupleMultiplicity_,
                            hv,
                            cm,
                            tkidDevice.data(),
                            hitsDevice.data(),
                            hits_geDevice.data(),
                            fast_fit_resultsDevice.data(),
                            TrackerTraits::maxHitsOnTrackForFullFit,
                            TrackerTraits::maxHitsOnTrack - 1,
                            offset);

        alpaka::exec<Acc1D>(queue,
                            workDivQuadsPenta,
                            Kernel_BLFit<TrackerTraits::maxHitsOnTrackForFullFit, TrackerTraits>{},
                            tupleMultiplicity_,
                            bField_,
                            outputSoa_,
                            tkidDevice.data(),
                            hitsDevice.data(),
                            hits_geDevice.data(),
                            fast_fit_resultsDevice.data());
      }

    }  // loop on concurrent fits
  }

  // Overloaded version for stub-based tracking
  // offsetStubs is passed from the device container (has cached host-side value)
  // Do NOT call hv.offsetStubs() from host - that would dereference device memory and segfault!
  template <typename TrackerTraits>
  void HelixFit<TrackerTraits>::launchBrokenLineKernels(const ::reco::TrackingRecHitConstView& hv,
                                                        const ::reco::CAModulesConstView& cm,
                                                        uint32_t hitsInFit,
                                                        uint32_t maxNumberOfTuples,
                                                        Queue& queue,
                                                        const ::reco::OTRecHitsConstView& otRecHits,
                                                        const ::reco::StubsConstView& stubs,
                                                        int32_t offsetStubs) {
    ALPAKA_ASSERT_ACC(tuples_);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting HelixFit<TrackerTraits>::launchBrokenLineKernels with stubs" << std::endl;
#endif

#if defined(FIT_DEBUG) || defined(GPU_DEBUG)
    alpaka::wait(queue);
    std::cout << "FIT_DEBUG: launchBrokenLineKernels (stubs) - maxNumberOfTuples=" << maxNumberOfTuples
              << " maxNumberOfConcurrentFits=" << maxNumberOfConcurrentFits_ << " fitNas4=" << fitNas4_
              << " bField=" << bField_ << " offsetStubs=" << offsetStubs << std::endl;
    std::cout << "FIT_DEBUG: TrackerTraits::maxHitsOnTrack=" << TrackerTraits::maxHitsOnTrack
              << " maxHitsOnTrackForFullFit=" << TrackerTraits::maxHitsOnTrackForFullFit << std::endl;
#endif

    // offsetStubs is now passed as a parameter (from device container's cached host-side value)

    uint32_t blockSize = 64;
    uint32_t numberOfBlocks = cms::alpakatools::divide_up_by(maxNumberOfConcurrentFits_, blockSize);
    const WorkDiv1D workDivTriplets = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    const WorkDiv1D workDivQuadsPenta = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks / 4, blockSize);

    //  Fit internals
    auto tkidDevice =
        cms::alpakatools::make_device_buffer<typename caStructures::tindex_type[]>(queue, maxNumberOfConcurrentFits_);
    constexpr auto maxN = TrackerTraits::maxHitsOnTrackForFullFit;
    auto hitsDevice = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix3xNd<maxN>) / sizeof(double));
    auto hits_geDevice = cms::alpakatools::make_device_buffer<float[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix6xNf<maxN>) / sizeof(float));
    auto fast_fit_resultsDevice = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Vector4d) / sizeof(double));

    for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
      // fit triplets
      alpaka::exec<Acc1D>(queue,
                          workDivTriplets,
                          Kernel_BLFastFit<3>{},
                          tuples_,
                          tupleMultiplicity_,
                          hv,
                          cm,
                          tkidDevice.data(),
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          3,
                          3,
                          offset,
                          otRecHits,
                          stubs,
                          offsetStubs);

      alpaka::exec<Acc1D>(queue,
                          workDivTriplets,
                          Kernel_BLFit<3, TrackerTraits>{},
                          tupleMultiplicity_,
                          bField_,
                          outputSoa_,
                          tkidDevice.data(),
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data());

      if (fitNas4_) {
        // fit all as 4
        riemannFit::rolling_fits<4, TrackerTraits::maxHitsOnTrack, 1>([this,
                                                                       &hv,
                                                                       &cm,
                                                                       &tkidDevice,
                                                                       &hitsDevice,
                                                                       &hits_geDevice,
                                                                       &fast_fit_resultsDevice,
                                                                       &offset,
                                                                       &queue,
                                                                       &workDivQuadsPenta,
                                                                       otRecHits,
                                                                       stubs,
                                                                       offsetStubs](auto i) {
          alpaka::exec<Acc1D>(queue,
                              workDivQuadsPenta,
                              Kernel_BLFastFit<4>{},
                              tuples_,
                              tupleMultiplicity_,
                              hv,
                              cm,
                              tkidDevice.data(),
                              hitsDevice.data(),
                              hits_geDevice.data(),
                              fast_fit_resultsDevice.data(),
                              4,
                              4,
                              offset,
                              otRecHits,
                              stubs,
                              offsetStubs);

          alpaka::exec<Acc1D>(queue,
                              workDivQuadsPenta,
                              Kernel_BLFit<4, TrackerTraits>{},
                              tupleMultiplicity_,
                              bField_,
                              outputSoa_,
                              tkidDevice.data(),
                              hitsDevice.data(),
                              hits_geDevice.data(),
                              fast_fit_resultsDevice.data());
        });

        static_assert(TrackerTraits::maxHitsOnTrackForFullFit < TrackerTraits::maxHitsOnTrack);

        //Fit all the rest using the maximum from previous call
        alpaka::exec<Acc1D>(queue,
                            workDivQuadsPenta,
                            Kernel_BLFastFit<TrackerTraits::maxHitsOnTrackForFullFit>{},
                            tuples_,
                            tupleMultiplicity_,
                            hv,
                            cm,
                            tkidDevice.data(),
                            hitsDevice.data(),
                            hits_geDevice.data(),
                            fast_fit_resultsDevice.data(),
                            TrackerTraits::maxHitsOnTrackForFullFit,
                            TrackerTraits::maxHitsOnTrack - 1,
                            offset,
                            otRecHits,
                            stubs,
                            offsetStubs);

        alpaka::exec<Acc1D>(queue,
                            workDivQuadsPenta,
                            Kernel_BLFit<TrackerTraits::maxHitsOnTrackForFullFit, TrackerTraits>{},
                            tupleMultiplicity_,
                            bField_,
                            outputSoa_,
                            tkidDevice.data(),
                            hitsDevice.data(),
                            hits_geDevice.data(),
                            fast_fit_resultsDevice.data());
      } else {
        // Rolling fits for multiplicities 4 to maxHitsOnTrackForFullFit
        riemannFit::rolling_fits<4, TrackerTraits::maxHitsOnTrackForFullFit, 1>([this,
                                                                                 &hv,
                                                                                 &cm,
                                                                                 &tkidDevice,
                                                                                 &hitsDevice,
                                                                                 &hits_geDevice,
                                                                                 &fast_fit_resultsDevice,
                                                                                 &offset,
                                                                                 &queue,
                                                                                 &workDivQuadsPenta,
                                                                                 otRecHits,
                                                                                 stubs,
                                                                                 offsetStubs](auto i) {
          alpaka::exec<Acc1D>(queue,
                              workDivQuadsPenta,
                              Kernel_BLFastFit<i>{},
                              tuples_,
                              tupleMultiplicity_,
                              hv,
                              cm,
                              tkidDevice.data(),
                              hitsDevice.data(),
                              hits_geDevice.data(),
                              fast_fit_resultsDevice.data(),
                              i,
                              i,
                              offset,
                              otRecHits,
                              stubs,
                              offsetStubs);

          alpaka::exec<Acc1D>(queue,
                              workDivQuadsPenta,
                              Kernel_BLFit<i, TrackerTraits>{},
                              tupleMultiplicity_,
                              bField_,
                              outputSoa_,
                              tkidDevice.data(),
                              hitsDevice.data(),
                              hits_geDevice.data(),
                              fast_fit_resultsDevice.data());
#ifdef GPU_DEBUG
          alpaka::wait(queue);
          std::cout << "Stub Kernel_BLFastFit(" << i << ") and Kernel_BLFit(" << i << ") -> done! " << std::endl;
#endif
        });

        static_assert(TrackerTraits::maxHitsOnTrackForFullFit < TrackerTraits::maxHitsOnTrack);

        // Fit all the rest using maxHitsOnTrackForFullFit hits
        alpaka::exec<Acc1D>(queue,
                            workDivQuadsPenta,
                            Kernel_BLFastFit<TrackerTraits::maxHitsOnTrackForFullFit>{},
                            tuples_,
                            tupleMultiplicity_,
                            hv,
                            cm,
                            tkidDevice.data(),
                            hitsDevice.data(),
                            hits_geDevice.data(),
                            fast_fit_resultsDevice.data(),
                            TrackerTraits::maxHitsOnTrackForFullFit,
                            TrackerTraits::maxHitsOnTrack - 1,
                            offset,
                            otRecHits,
                            stubs,
                            offsetStubs);

        alpaka::exec<Acc1D>(queue,
                            workDivQuadsPenta,
                            Kernel_BLFit<TrackerTraits::maxHitsOnTrackForFullFit, TrackerTraits>{},
                            tupleMultiplicity_,
                            bField_,
                            outputSoa_,
                            tkidDevice.data(),
                            hitsDevice.data(),
                            hits_geDevice.data(),
                            fast_fit_resultsDevice.data());
      }
    }  // loop on concurrent fits
  }

  template class HelixFit<pixelTopology::Phase1>;
  template class HelixFit<pixelTopology::Phase2>;
  template class HelixFit<pixelTopology::Phase2OT>;
  template class HelixFit<pixelTopology::Phase2OTStubs>;
  template class HelixFit<pixelTopology::HIonPhase1>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
