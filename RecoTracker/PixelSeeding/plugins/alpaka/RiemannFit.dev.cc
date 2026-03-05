// #define FIT_DEBUG
#include <cstdint>

#if defined(FIT_DEBUG) || defined(GPU_DEBUG)
#include <iostream>
#endif

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/RiemannFit.h"

#include "HelixFit.h"
#include "CAStructures.h"

using OutputSoAView = reco::TrackSoAView;
using TupleMultiplicity = caStructures::GenericContainer;
using Tuples = caStructures::SequentialContainer;

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace alpaka;
  using namespace cms::alpakatools;

  template <int N, typename TrackerTraits>
  class Kernel_FastFit {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  Tuples const *__restrict__ foundNtuplets,
                                  TupleMultiplicity const *__restrict__ tupleMultiplicity,
                                  uint32_t nHits,
                                  ::reco::TrackingRecHitConstView hh,
                                  ::reco::CAModulesConstView cm,
                                  double *__restrict__ phits,
                                  float *__restrict__ phits_ge,
                                  double *__restrict__ pfast_fit,
                                  uint32_t offset,
                                  ::reco::OTRecHitsConstView otRecHits = ::reco::OTRecHitsConstView(),
                                  ::reco::StubsConstView stubs = ::reco::StubsConstView(),
                                  int32_t offsetStubs = -1) const {
      constexpr uint32_t hitsInFit = N;

      ALPAKA_ASSERT_ACC(hitsInFit <= nHits);

      ALPAKA_ASSERT_ACC(pfast_fit);
      ALPAKA_ASSERT_ACC(foundNtuplets);
      ALPAKA_ASSERT_ACC(tupleMultiplicity);

      // look in bin for this hit multiplicity

#ifdef RIEMANN_DEBUG
      const uint32_t threadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
      if (cms::alpakatools::once_per_grid(acc))
        printf("%d Ntuple of size %d for %d hits to fit\n", tupleMultiplicity->size(nHits), nHits, hitsInFit);
#endif

      const auto nt = riemannFit::maxNumberOfConcurrentFits;
      for (auto local_idx : cms::alpakatools::uniform_elements(acc, nt)) {
        auto tuple_idx = local_idx + offset;
        if (tuple_idx >= tupleMultiplicity->size(nHits))
          break;

        // get it from the ntuple container (one to one to helix)
        auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_idx);
        ALPAKA_ASSERT_ACC(tkid < foundNtuplets->nOnes());

        ALPAKA_ASSERT_ACC(foundNtuplets->size(tkid) == nHits);

        riemannFit::Map3xNd<N> hits(phits + local_idx);
        riemannFit::Map4d fast_fit(pfast_fit + local_idx);
        riemannFit::Map6xNf<N> hits_ge(phits_ge + local_idx);

        // Prepare data structure
        auto const *hitId = foundNtuplets->begin(tkid);
        for (unsigned int i = 0; i < hitsInFit; ++i) {
          auto hit = hitId[i];
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

          // Fill errors - use pre-computed global errors for SS stubs
          if (isSSStub) {
            ge[0] = otRecHits[sensorHitIdx].xGlobalErr();
            ge[1] = otRecHits[sensorHitIdx].xyGlobalErr();
            ge[2] = otRecHits[sensorHitIdx].yGlobalErr();
            ge[3] = otRecHits[sensorHitIdx].xzGlobalErr();
            ge[4] = otRecHits[sensorHitIdx].yzGlobalErr();
            ge[5] = otRecHits[sensorHitIdx].zGlobalErr();
          } else {
            cm.detFrame(hh.detectorIndex(hit)).toGlobal(hh[hit].xerrLocal(), 0, hh[hit].yerrLocal(), ge);
          }

          // Fill position - use sensor hit for SS stubs, merged hit otherwise
          if (isSSStub) {
            hits.col(i) << otRecHits[sensorHitIdx].xGlobal(),
                           otRecHits[sensorHitIdx].yGlobal(),
                           otRecHits[sensorHitIdx].zGlobal();
          } else {
            hits.col(i) << hh[hit].xGlobal(), hh[hit].yGlobal(), hh[hit].zGlobal();
          }
          hits_ge.col(i) << ge[0], ge[1], ge[2], ge[3], ge[4], ge[5];
        }
        riemannFit::fastFit(acc, hits, fast_fit);

#ifdef RIEMANN_DEBUG
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
  class Kernel_CircleFit {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TupleMultiplicity const *__restrict__ tupleMultiplicity,
                                  uint32_t nHits,
                                  double bField,
                                  double *__restrict__ phits,
                                  float *__restrict__ phits_ge,
                                  double *__restrict__ pfast_fit_input,
                                  riemannFit::CircleFit *circle_fit,
                                  uint32_t offset) const {
      ALPAKA_ASSERT_ACC(circle_fit);
      ALPAKA_ASSERT_ACC(N <= nHits);

      // same as above...

      // look in bin for this hit multiplicity
      const auto nt = riemannFit::maxNumberOfConcurrentFits;
      for (auto local_idx : cms::alpakatools::uniform_elements(acc, nt)) {
        auto tuple_idx = local_idx + offset;
        if (tuple_idx >= tupleMultiplicity->size(nHits))
          break;

        riemannFit::Map3xNd<N> hits(phits + local_idx);
        riemannFit::Map4d fast_fit(pfast_fit_input + local_idx);
        riemannFit::Map6xNf<N> hits_ge(phits_ge + local_idx);

        riemannFit::VectorNd<N> rad = (hits.block(0, 0, 2, N).colwise().norm());

        riemannFit::Matrix2Nd<N> hits_cov = riemannFit::Matrix2Nd<N>::Zero();
        riemannFit::loadCovariance2D(acc, hits_ge, hits_cov);

        circle_fit[local_idx] =
            riemannFit::circleFit(acc, hits.block(0, 0, 2, N), hits_cov, fast_fit, rad, bField, true);

#ifdef RIEMANN_DEBUG
//    auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_idx);
//  printf("kernelCircleFit circle.par(0,1,2): %d %f,%f,%f\n", tkid,
//         circle_fit[local_idx].par(0), circle_fit[local_idx].par(1), circle_fit[local_idx].par(2));
#endif
      }
    }
  };

  template <int N, typename TrackerTraits>
  class Kernel_LineFit {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const &acc,
                                  TupleMultiplicity const *__restrict__ tupleMultiplicity,
                                  uint32_t nHits,
                                  double bField,
                                  OutputSoAView results_view,
                                  double *__restrict__ phits,
                                  float *__restrict__ phits_ge,
                                  double *__restrict__ pfast_fit_input,
                                  riemannFit::CircleFit *__restrict__ circle_fit,
                                  uint32_t offset) const {
      ALPAKA_ASSERT_ACC(circle_fit);
      ALPAKA_ASSERT_ACC(N <= nHits);

      // same as above...

      // look in bin for this hit multiplicity
      const auto nt = riemannFit::maxNumberOfConcurrentFits;
      for (auto local_idx : cms::alpakatools::uniform_elements(acc, nt)) {
        auto tuple_idx = local_idx + offset;
        if (tuple_idx >= tupleMultiplicity->size(nHits))
          break;

        // get it for the ntuple container (one to one to helix)
        int32_t tkid = *(tupleMultiplicity->begin(nHits) + tuple_idx);

        riemannFit::Map3xNd<N> hits(phits + local_idx);
        riemannFit::Map4d fast_fit(pfast_fit_input + local_idx);
        riemannFit::Map6xNf<N> hits_ge(phits_ge + local_idx);

        auto const &line_fit = riemannFit::lineFit(acc, hits, hits_ge, circle_fit[local_idx], fast_fit, bField, true);

        riemannFit::fromCircleToPerigee(acc, circle_fit[local_idx]);

        reco::copyFromCircle(results_view,
                             circle_fit[local_idx].par,
                             circle_fit[local_idx].cov,
                             line_fit.par,
                             line_fit.cov,
                             1.f / float(bField),
                             tkid);
        results_view[tkid].pt() = bField / std::abs(circle_fit[local_idx].par(2));
        results_view[tkid].eta() = asinhf(line_fit.par(0));
        results_view[tkid].chi2() = (circle_fit[local_idx].chi2 + line_fit.chi2) / (2 * N - 5);

#ifdef RIEMANN_DEBUG
        printf("kernelLineFit size %d for %d hits circle.par(0,1,2): %d %f,%f,%f\n",
               N,
               nHits,
               tkid,
               circle_fit[local_idx].par(0),
               circle_fit[local_idx].par(1),
               circle_fit[local_idx].par(2));
        printf("kernelLineFit line.par(0,1): %d %f,%f\n", tkid, line_fit.par(0), line_fit.par(1));
        printf("kernelLineFit chi2 cov %f/%f %e,%e,%e,%e,%e\n",
               circle_fit[local_idx].chi2,
               line_fit.chi2,
               circle_fit[local_idx].cov(0, 0),
               circle_fit[local_idx].cov(1, 1),
               circle_fit[local_idx].cov(2, 2),
               line_fit.cov(0, 0),
               line_fit.cov(1, 1));
#endif
      }
    }
  };

  template <typename TrackerTraits>
  void HelixFit<TrackerTraits>::launchRiemannKernels(const ::reco::TrackingRecHitConstView &hv,
                                                     const ::reco::CAModulesConstView &cm,
                                                     uint32_t nhits,
                                                     uint32_t maxNumberOfTuples,
                                                     Queue &queue) {
    assert(tuples_);

    auto blockSize = 64;
    auto numberOfBlocks = (maxNumberOfConcurrentFits_ + blockSize - 1) / blockSize;
    const auto workDivTriplets = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    const auto workDivQuadsPenta = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks / 4, blockSize);

    //  Fit internals
    constexpr auto maxN = TrackerTraits::maxHitsOnTrackForFullFit;
    auto hitsDevice = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix3xNd<maxN>) / sizeof(double));
    auto hits_geDevice = cms::alpakatools::make_device_buffer<float[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix6xNf<maxN>) / sizeof(float));
    auto fast_fit_resultsDevice = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Vector4d) / sizeof(double));
    auto circle_fit_resultsDevice_holder =
        cms::alpakatools::make_device_buffer<char[]>(queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::CircleFit));
    riemannFit::CircleFit *circle_fit_resultsDevice_ =
        (riemannFit::CircleFit *)(circle_fit_resultsDevice_holder.data());

    for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
      // triplets
      alpaka::exec<Acc1D>(queue,
                          workDivTriplets,
                          Kernel_FastFit<3, TrackerTraits>{},
                          tuples_,
                          tupleMultiplicity_,
                          3,
                          hv,
                          cm,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          offset);

      alpaka::exec<Acc1D>(queue,
                          workDivTriplets,
                          Kernel_CircleFit<3, TrackerTraits>{},
                          tupleMultiplicity_,
                          3,
                          bField_,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          circle_fit_resultsDevice_,
                          offset);

      alpaka::exec<Acc1D>(queue,
                          workDivTriplets,
                          Kernel_LineFit<3, TrackerTraits>{},
                          tupleMultiplicity_,
                          3,
                          bField_,
                          outputSoa_,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          circle_fit_resultsDevice_,
                          offset);

      // quads
      alpaka::exec<Acc1D>(queue,
                          workDivQuadsPenta,
                          Kernel_FastFit<4, TrackerTraits>{},
                          tuples_,
                          tupleMultiplicity_,
                          4,
                          hv,
                          cm,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          offset);

      alpaka::exec<Acc1D>(queue,
                          workDivQuadsPenta,
                          Kernel_CircleFit<4, TrackerTraits>{},
                          tupleMultiplicity_,
                          4,
                          bField_,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          circle_fit_resultsDevice_,
                          offset);

      alpaka::exec<Acc1D>(queue,
                          workDivQuadsPenta,
                          Kernel_LineFit<4, TrackerTraits>{},
                          tupleMultiplicity_,
                          4,
                          bField_,
                          outputSoa_,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          circle_fit_resultsDevice_,
                          offset);

      if (fitNas4_) {
        // penta
        alpaka::exec<Acc1D>(queue,
                            workDivQuadsPenta,
                            Kernel_FastFit<4, TrackerTraits>{},
                            tuples_,
                            tupleMultiplicity_,
                            5,
                            hv,
                            cm,
                            hitsDevice.data(),
                            hits_geDevice.data(),
                            fast_fit_resultsDevice.data(),
                            offset);

        alpaka::exec<Acc1D>(queue,
                            workDivQuadsPenta,
                            Kernel_CircleFit<4, TrackerTraits>{},
                            tupleMultiplicity_,
                            5,
                            bField_,
                            hitsDevice.data(),
                            hits_geDevice.data(),
                            fast_fit_resultsDevice.data(),
                            circle_fit_resultsDevice_,
                            offset);

        alpaka::exec<Acc1D>(queue,
                            workDivQuadsPenta,
                            Kernel_LineFit<4, TrackerTraits>{},
                            tupleMultiplicity_,
                            5,
                            bField_,
                            outputSoa_,
                            hitsDevice.data(),
                            hits_geDevice.data(),
                            fast_fit_resultsDevice.data(),
                            circle_fit_resultsDevice_,
                            offset);
      } else {
        // penta all 5
        alpaka::exec<Acc1D>(queue,
                            workDivQuadsPenta,
                            Kernel_FastFit<5, TrackerTraits>{},
                            tuples_,
                            tupleMultiplicity_,
                            5,
                            hv,
                            cm,
                            hitsDevice.data(),
                            hits_geDevice.data(),
                            fast_fit_resultsDevice.data(),
                            offset);

        alpaka::exec<Acc1D>(queue,
                            workDivQuadsPenta,
                            Kernel_CircleFit<5, TrackerTraits>{},
                            tupleMultiplicity_,
                            5,
                            bField_,
                            hitsDevice.data(),
                            hits_geDevice.data(),
                            fast_fit_resultsDevice.data(),
                            circle_fit_resultsDevice_,
                            offset);

        alpaka::exec<Acc1D>(queue,
                            workDivQuadsPenta,
                            Kernel_LineFit<5, TrackerTraits>{},
                            tupleMultiplicity_,
                            5,
                            bField_,
                            outputSoa_,
                            hitsDevice.data(),
                            hits_geDevice.data(),
                            fast_fit_resultsDevice.data(),
                            circle_fit_resultsDevice_,
                            offset);
      }
    }
  }

  // Overloaded version for stub-based tracking
  // offsetStubs is passed from the device container (has cached host-side value)
  // Do NOT call hv.offsetStubs() from host - that would dereference device memory and segfault!
  template <typename TrackerTraits>
  void HelixFit<TrackerTraits>::launchRiemannKernels(const ::reco::TrackingRecHitConstView &hv,
                                                     const ::reco::CAModulesConstView &cm,
                                                     uint32_t nhits,
                                                     uint32_t maxNumberOfTuples,
                                                     Queue &queue,
                                                     const ::reco::OTRecHitsConstView &otRecHits,
                                                     const ::reco::StubsConstView &stubs,
                                                     int32_t offsetStubs) {
    assert(tuples_);

#if defined(FIT_DEBUG) || defined(GPU_DEBUG)
    alpaka::wait(queue);
    std::cout << "FIT_DEBUG: launchRiemannKernels (stubs) - maxNumberOfTuples=" << maxNumberOfTuples
              << " maxNumberOfConcurrentFits=" << maxNumberOfConcurrentFits_ << " bField=" << bField_
              << " offsetStubs=" << offsetStubs << std::endl;
    std::cout << "FIT_DEBUG: TrackerTraits::maxHitsOnTrack=" << TrackerTraits::maxHitsOnTrack << std::endl;
#endif

    // offsetStubs is now passed as a parameter (from device container's cached host-side value)

    auto blockSize = 64;
    auto numberOfBlocks = (maxNumberOfConcurrentFits_ + blockSize - 1) / blockSize;
    const auto workDivTriplets = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    const auto workDivQuadsPenta = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks / 4, blockSize);

    //  Fit internals
    constexpr auto maxN = TrackerTraits::maxHitsOnTrackForFullFit;
    auto hitsDevice = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix3xNd<maxN>) / sizeof(double));
    auto hits_geDevice = cms::alpakatools::make_device_buffer<float[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix6xNf<maxN>) / sizeof(float));
    auto fast_fit_resultsDevice = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::Vector4d) / sizeof(double));
    auto circle_fit_resultsDevice_holder =
        cms::alpakatools::make_device_buffer<char[]>(queue, maxNumberOfConcurrentFits_ * sizeof(riemannFit::CircleFit));
    riemannFit::CircleFit *circle_fit_resultsDevice_local =
        (riemannFit::CircleFit *)(circle_fit_resultsDevice_holder.data());

    for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
      // triplets
      alpaka::exec<Acc1D>(queue,
                          workDivTriplets,
                          Kernel_FastFit<3, TrackerTraits>{},
                          tuples_,
                          tupleMultiplicity_,
                          3,
                          hv,
                          cm,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          offset,
                          otRecHits,
                          stubs,
                          offsetStubs);

      alpaka::exec<Acc1D>(queue,
                          workDivTriplets,
                          Kernel_CircleFit<3, TrackerTraits>{},
                          tupleMultiplicity_,
                          3,
                          bField_,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          circle_fit_resultsDevice_local,
                          offset);

      alpaka::exec<Acc1D>(queue,
                          workDivTriplets,
                          Kernel_LineFit<3, TrackerTraits>{},
                          tupleMultiplicity_,
                          3,
                          bField_,
                          outputSoa_,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          circle_fit_resultsDevice_local,
                          offset);

      // quadruplets
      alpaka::exec<Acc1D>(queue,
                          workDivQuadsPenta,
                          Kernel_FastFit<4, TrackerTraits>{},
                          tuples_,
                          tupleMultiplicity_,
                          4,
                          hv,
                          cm,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          offset,
                          otRecHits,
                          stubs,
                          offsetStubs);

      alpaka::exec<Acc1D>(queue,
                          workDivQuadsPenta,
                          Kernel_CircleFit<4, TrackerTraits>{},
                          tupleMultiplicity_,
                          4,
                          bField_,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          circle_fit_resultsDevice_local,
                          offset);

      alpaka::exec<Acc1D>(queue,
                          workDivQuadsPenta,
                          Kernel_LineFit<4, TrackerTraits>{},
                          tupleMultiplicity_,
                          4,
                          bField_,
                          outputSoa_,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          circle_fit_resultsDevice_local,
                          offset);

      // quintuplets
      alpaka::exec<Acc1D>(queue,
                          workDivQuadsPenta,
                          Kernel_FastFit<5, TrackerTraits>{},
                          tuples_,
                          tupleMultiplicity_,
                          5,
                          hv,
                          cm,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          offset,
                          otRecHits,
                          stubs,
                          offsetStubs);

      alpaka::exec<Acc1D>(queue,
                          workDivQuadsPenta,
                          Kernel_CircleFit<5, TrackerTraits>{},
                          tupleMultiplicity_,
                          5,
                          bField_,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          circle_fit_resultsDevice_local,
                          offset);

      alpaka::exec<Acc1D>(queue,
                          workDivQuadsPenta,
                          Kernel_LineFit<5, TrackerTraits>{},
                          tupleMultiplicity_,
                          5,
                          bField_,
                          outputSoa_,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          circle_fit_resultsDevice_local,
                          offset);

      // sextuplets (6 hits) - fit all 6 hits
      alpaka::exec<Acc1D>(queue,
                          workDivQuadsPenta,
                          Kernel_FastFit<6, TrackerTraits>{},
                          tuples_,
                          tupleMultiplicity_,
                          6,
                          hv,
                          cm,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          offset,
                          otRecHits,
                          stubs,
                          offsetStubs);

      alpaka::exec<Acc1D>(queue,
                          workDivQuadsPenta,
                          Kernel_CircleFit<6, TrackerTraits>{},
                          tupleMultiplicity_,
                          6,
                          bField_,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          circle_fit_resultsDevice_local,
                          offset);

      alpaka::exec<Acc1D>(queue,
                          workDivQuadsPenta,
                          Kernel_LineFit<6, TrackerTraits>{},
                          tupleMultiplicity_,
                          6,
                          bField_,
                          outputSoa_,
                          hitsDevice.data(),
                          hits_geDevice.data(),
                          fast_fit_resultsDevice.data(),
                          circle_fit_resultsDevice_local,
                          offset);

      // 7+ hits - fit using first 6 hits (simplified fit for longer tracks)
      // Loop over multiplicities 7 to maxHitsOnTrack-1, fitting first 6 hits of each
      for (uint32_t nHits = 7; nHits < TrackerTraits::maxHitsOnTrack; ++nHits) {
        alpaka::exec<Acc1D>(queue,
                            workDivQuadsPenta,
                            Kernel_FastFit<6, TrackerTraits>{},
                            tuples_,
                            tupleMultiplicity_,
                            nHits,
                            hv,
                            cm,
                            hitsDevice.data(),
                            hits_geDevice.data(),
                            fast_fit_resultsDevice.data(),
                            offset,
                            otRecHits,
                            stubs,
                            offsetStubs);

        alpaka::exec<Acc1D>(queue,
                            workDivQuadsPenta,
                            Kernel_CircleFit<6, TrackerTraits>{},
                            tupleMultiplicity_,
                            nHits,
                            bField_,
                            hitsDevice.data(),
                            hits_geDevice.data(),
                            fast_fit_resultsDevice.data(),
                            circle_fit_resultsDevice_local,
                            offset);

        alpaka::exec<Acc1D>(queue,
                            workDivQuadsPenta,
                            Kernel_LineFit<6, TrackerTraits>{},
                            tupleMultiplicity_,
                            nHits,
                            bField_,
                            outputSoa_,
                            hitsDevice.data(),
                            hits_geDevice.data(),
                            fast_fit_resultsDevice.data(),
                            circle_fit_resultsDevice_local,
                            offset);
      }
    }
  }

  template class HelixFit<pixelTopology::Phase1>;
  template class HelixFit<pixelTopology::Phase2>;
  template class HelixFit<pixelTopology::Phase2OT>;
  template class HelixFit<pixelTopology::Phase2OTStubs>;
  template class HelixFit<pixelTopology::HIonPhase1>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
