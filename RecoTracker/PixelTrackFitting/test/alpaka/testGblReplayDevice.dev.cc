// Checks that the host material march of test/gblTestMaterial.h reproduces the device one: for each
// fixture the device prep (brokenline::prepareGblFitData and the beamline->hit0 segmentXX0GapSplit) is
// run and gblTestMaterial::fillMatData must match every number on the host: matXX0 per gap, the per-gap
// two-thin split (gapD1, gapW1), innerXX0 and its own split.
//
// Tolerance: the two marches are the same arithmetic in the same order, differing only in
// alpaka::math::sqrt vs std::sqrt and in contraction (~1e-14 accumulated). One benign failure mode: a
// last-bit difference in the path length L across an integer boundary changes the sample count
// nseg = int(2*L) by one and moves a single gap by a per-cent amount.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BrokenLine.h"         // upstream host fastFit
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"  // the device march under test
#include "RecoTracker/PixelTrackFitting/test/gblTestFixtures.h"
#include "RecoTracker/PixelTrackFitting/test/gblTestMaterial.h"  // the host copy gblReplay uses

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
namespace blh = ::brokenline;
namespace bld = ALPAKA_ACCELERATOR_NAMESPACE::brokenline;

namespace {

  //!< same arithmetic in the same order, ~1e-14 accumulated.
  constexpr double kTolMarch = 1e-9;

  template <int N>
  struct MatKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  double const* hitsIn,
                                  double const* ffIn,
                                  float const* rho,
                                  double bField,
                                  double* out) const {
      for ([[maybe_unused]] auto lane : cms::alpakatools::uniform_elements(acc, 1)) {
        Eigen::Matrix<double, 3, N> hits;
        for (int c = 0; c < N; ++c)
          for (int r = 0; r < 3; ++r)
            hits(r, c) = hitsIn[3 * c + r];
        const Eigen::Vector4d ff(ffIn[0], ffIn[1], ffIn[2], ffIn[3]);

        bld::PreparedGblData<N> data;
        double gapD1[N], gapW1[N];
        bld::prepareGblFitData(acc, hits, ff, bField, rho, data, /*matCached=*/nullptr, gapD1, gapW1);
        const double rHit0 = alpaka::math::sqrt(acc, hits(0, 0) * hits(0, 0) + hits(1, 0) * hits(1, 0));
        double innerD1 = 0., innerW1 = 0.;
        if (data.innerXX0 > 0.)
          bld::segmentXX0GapSplit(acc, rho, 0., 0., rHit0, hits(2, 0), innerD1, innerW1);

        for (int i = 0; i < N; ++i) {
          out[i] = data.matXX0(i);
          out[N + i] = gapD1[i];
          out[2 * N + i] = gapW1[i];
        }
        out[3 * N] = data.innerXX0;
        out[3 * N + 1] = innerD1;
        out[3 * N + 2] = innerW1;
      }
    }
  };

  struct MaxRel {
    double v = 0.;
    bool any = false;
    void add(double got, double ref) {
      if (std::abs(ref) > 1e-30) {
        any = true;
        v = std::max(v, std::abs(got - ref) / std::abs(ref));
      }
    }
  };

  template <int N>
  void runFixture(Queue& queue, const char* devName, const char* label, const double D[N][9], const float* rhoDev) {
    constexpr int kOut = 3 * N + 3;
    Eigen::Matrix<double, 3, N> hits;
    for (int k = 0; k < N; ++k) {
      hits(0, k) = D[k][0];
      hits(1, k) = D[k][1];
      hits(2, k) = D[k][2];
    }
    Eigen::Vector4d ff;
    blh::fastFit(hits, ff);

    auto hits_h = cms::alpakatools::make_host_buffer<double[], Platform>(3 * N);
    auto ff_h = cms::alpakatools::make_host_buffer<double[], Platform>(4);
    for (int c = 0; c < N; ++c)
      for (int r = 0; r < 3; ++r)
        hits_h[3 * c + r] = hits(r, c);
    for (int j = 0; j < 4; ++j)
      ff_h[j] = ff(j);

    auto hits_d = cms::alpakatools::make_device_buffer<double[]>(queue, 3 * N);
    auto ff_d = cms::alpakatools::make_device_buffer<double[]>(queue, 4);
    auto out_d = cms::alpakatools::make_device_buffer<double[]>(queue, kOut);
    auto out_h = cms::alpakatools::make_host_buffer<double[], Platform>(kOut);
    alpaka::memcpy(queue, hits_d, hits_h);
    alpaka::memcpy(queue, ff_d, ff_h);
    alpaka::memset(queue, out_d, 0);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(1, 1),
                        MatKernel<N>{},
                        hits_d.data(),
                        ff_d.data(),
                        rhoDev,
                        gblTestFixtures::kB,
                        out_d.data());
    alpaka::memcpy(queue, out_h, out_d);
    alpaka::wait(queue);

    gblTestMaterial::MatData<N> md;
    gblTestMaterial::fillMatData<N>(hits, md);

    MaxRel mat, gd1, gw1, inner;
    for (int i = 0; i < N; ++i) {
      mat.add(md.matXX0[i], out_h[i]);
      gd1.add(md.gapD1[i], out_h[N + i]);
      gw1.add(md.gapW1[i], out_h[2 * N + i]);
    }
    inner.add(md.innerXX0, out_h[3 * N]);
    inner.add(md.innerD1, out_h[3 * N + 1]);
    inner.add(md.innerW1, out_h[3 * N + 2]);

    std::printf("  %-28s N=%2d | matXX0=%.2e gapD1=%.2e gapW1=%.2e inner=%.2e   (host innerXX0=%.6g)  [%s]\n",
                label,
                N,
                mat.v,
                gd1.v,
                gw1.v,
                inner.v,
                md.innerXX0,
                devName);

    // positive controls: an all-zero march would agree with anything.
    REQUIRE(mat.any);
    REQUIRE(gd1.any);
    REQUIRE(gw1.any);
    REQUIRE(inner.any);
    REQUIRE(md.innerXX0 > 0.);

    REQUIRE(mat.v < kTolMarch);
    REQUIRE(gd1.v < kTolMarch);
    REQUIRE(gw1.v < kTolMarch);
    REQUIRE(inner.v < kTolMarch);
  }

}  // namespace

TEST_CASE("gblReplay host material march vs the device for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty())
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, test skipped.");

  auto rho_h = cms::alpakatools::make_host_buffer<float[], Platform>(blMaterialMap::kSize);
  std::copy_n(blMaterialMap::blMaterialMapData(), blMaterialMap::kSize, rho_h.data());

  using namespace gblTestFixtures;
  for (auto const& device : devices) {
    auto queue = Queue(device);
    auto rho_d = cms::alpakatools::make_device_buffer<float[]>(queue, blMaterialMap::kSize);
    alpaka::memcpy(queue, rho_d, rho_h);
    alpaka::wait(queue);
    const std::string dn = alpaka::getName(device);
    std::printf("\n=== %s : gblTestMaterial (host) vs brokenline::prepareGblFitData (device) ===\n", dn.c_str());
    runFixture<4>(queue, dn.c_str(), "barrel eta~0 (IT only)", BARREL0, rho_d.data());
    runFixture<10>(queue, dn.c_str(), "central eta~0.36 IT+OT", CENTRAL, rho_d.data());
    runFixture<10>(queue, dn.c_str(), "forward eta~3.2 IT only", FWD, rho_d.data());
    runFixture<10>(queue, dn.c_str(), "tilted eta~1.0 OT barrel", TILT, rho_d.data());
    runFixture<10>(queue, dn.c_str(), "REAL barrel eta~0.27", BARRELR027, rho_d.data());
    runFixture<4>(queue, dn.c_str(), "displaced OT only", DISPLACED1, rho_d.data());
  }
}
