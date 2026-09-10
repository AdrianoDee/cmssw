// testGblTiltedDevice.dev.cc
//
// The tilted-OT-barrel measurement model, on the device. One real track (SingleMu Eta1p0 Pt100,
// eta = 1.087, N = 10) whose OT barrel modules at hits 4, 5 and 6 are tilted about 43 deg out of z. The
// production node chain is built on the device, copied back, and seven variants are solved with the
// device gblFitPca and the host twin:
//
//   (a) as built        measPrec = inv2(Ruv * ge3 * Ruv^T), the shipping measurement model
//   (b) obliquity-free  measPrec = A^T diag(1/xel, 1/yel) A with A from the true module axes, residual
//                       A^-1 A^-T r; not the KF measurement, which needs H = A^-T
//   (c) isotropic scat  scatPrec(1,1) := scatPrec(0,0), dropping the cos^2(lambda) anisotropy
//   (d) stiff kinks     scatPrec *= 1e6 (scattering frozen)
//   (e) MS x0.1         scatPrec *= 10
//   (f) MS x10          scatPrec *= 0.1
//   (g) all-precise     measPrec := diag(1e8, 1e8) (sigma = 1 um, isotropic)
//
// Asserted: device equals host twin on every variant and quantity; sigma(1/R) is non-decreasing as the
// kinks loosen, (d) <= (e) <= (a) <= (f), since process noise cannot shrink the posterior covariance;
// (c) <= (a), because cos^2(lambda) <= 1; and sigma(1/R)_(a)/sigma(1/R)_(b) stays inside
// [kRatioMin, kRatioMax], which a curvature blow-up on the tilted modules would trip.
// Not asserted: (g) against (a), diag(1e8,1e8) not dominating the as-built precision on a tilted module,
// and absolute sigmas, which move with every material or scattering change.

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
#include "RecoTracker/PixelTrackFitting/interface/BrokenLine.h"                // upstream host fastFit
#include "RecoTracker/PixelTrackFitting/interface/GeneralBrokenLine.h"         // host twin
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"         // device prep under test
#include "RecoTracker/PixelTrackFitting/interface/alpaka/GeneralBrokenLine.h"  // device twin under test
#include "RecoTracker/PixelTrackFitting/test/gblTestFixtures.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
namespace gblh = ::generalBrokenLine;
namespace blh = ::brokenline;
namespace gbld = ALPAKA_ACCELERATOR_NAMESPACE::generalBrokenLine;
namespace bld = ALPAKA_ACCELERATOR_NAMESPACE::brokenline;

namespace {

  constexpr int kN = 10;              // hits in the TILT fixture
  constexpr int kNodes = 2 * kN + 1;  // the 2N+1 two-thin-scatterer layout
  constexpr int kM = 2 * kN;          // gblFitPca<M> spans M+1 nodes
  constexpr int kVariants = 7;

  // Device vs host twin. This fixture is ill-conditioned by construction (a near-singular projected
  // covariance at three of ten hits), so the FMA difference between a GPU and the host is amplified past
  // the 1e-9 a well-conditioned fixture reaches. Variant (d) is excluded, see kTolTwinStiff.
  constexpr double kTolTwin = 1e-7;
  // Variant (d) only. Multiplying every kink precision by 1e6 multiplies the condition number of the band
  // system by the same factor (kappa_chi2 ~ 5e13 against ~1e7 elsewhere); at that conditioning a double
  // carries about two correct digits of chi2, so (d)'s chi2 differs at the per-cent level between host and
  // GPU with neither more right. sigma(1/R), the quantity the monotonicity check uses, is unaffected.
  constexpr double kTolTwinStiff = 1e-2;      // (d) cov / corrections
  constexpr double kTolTwinStiffChi2 = 1e-1;  // (d) chi2
  constexpr int kStiffVariant = 3;            //!< index of (d) in the variant list
  // Bracket on sigma(1/R)_(a) / sigma(1/R)_(b), which sits close to 1: wide enough that a material or
  // scattering retune does not trip it, narrow enough that a blow-up on the tilted modules would.
  constexpr double kRatioMin = 0.5;
  constexpr double kRatioMax = 2.0;
  //!< slack on the monotonicity theorems, to absorb round-off in two solves of nearly the same system.
  constexpr double kMonoSlack = 1e-9;

  // Model knobs: the field map and the three trajectory-model corrections are off (plain constant-field model).
  constexpr double kELossPerX0 = 0.0;
  constexpr double kBFieldOrigin = 0.0;
  constexpr bool kTrajectoryCorrections = false;
  constexpr bool kScatteringLogAtTotal = false;
  constexpr bool kElossCumulative = false;

  //!< scalars the build kernel reports alongside the node chain.
  enum BuildOut { kOutUsedSplit = 0, kOutQCharge, kOutSTransverse0, kOutInnerXX0, kOutMatSum, kBuildOutSize };

  // Kernel 1: the production node builder.
  struct BuildKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  double const* hitsIn,
                                  float const* geIn,
                                  double const* ffIn,
                                  float const* rho,
                                  double bField,
                                  gbld::GblNodeData* nodes,
                                  double* out) const {
      for ([[maybe_unused]] auto lane : cms::alpakatools::uniform_elements(acc, 1)) {
        Eigen::Matrix<double, 3, kN> hits;
        Eigen::Matrix<float, 6, kN> hits_ge;
        for (int c = 0; c < kN; ++c) {
          for (int r = 0; r < 3; ++r)
            hits(r, c) = hitsIn[3 * c + r];
          for (int r = 0; r < 6; ++r)
            hits_ge(r, c) = geIn[6 * c + r];
        }
        const Eigen::Vector4d ff(ffIn[0], ffIn[1], ffIn[2], ffIn[3]);

        bld::PreparedGblData<kN> data;
        double gapD1[kN], gapW1[kN];
        bld::prepareGblFitData(acc, hits, ff, bField, rho, data, /*matCached=*/nullptr, gapD1, gapW1);
        const double rHit0 = alpaka::math::sqrt(acc, hits(0, 0) * hits(0, 0) + hits(1, 0) * hits(1, 0));
        double innerD1 = 0., innerW1 = 0.;
        if (data.innerXX0 > 0.)
          bld::segmentXX0GapSplit(acc, rho, 0., 0., rHit0, hits(2, 0), innerD1, innerW1);

        const bool usedSplit = gbld::prepareGblDataSplit<Acc1D, kN>(acc,
                                                                    hits,
                                                                    hits_ge,
                                                                    ff,
                                                                    bField,
                                                                    data.qCharge,
                                                                    data.sTransverse,
                                                                    data.sTotal,
                                                                    data.matXX0,
                                                                    gapD1,
                                                                    gapW1,
                                                                    data.innerXX0,
                                                                    innerD1,
                                                                    innerW1,
                                                                    nodes,
                                                                    /*msScale=*/1.0,
                                                                    kELossPerX0,
                                                                    /*bMap=*/nullptr,
                                                                    kBFieldOrigin,
                                                                    kTrajectoryCorrections,
                                                                    kScatteringLogAtTotal,
                                                                    kElossCumulative);
        double matSum = 0.;
        for (int i = 0; i < kN; ++i)
          matSum += data.matXX0(i);
        out[kOutUsedSplit] = usedSplit ? 1. : 0.;
        out[kOutQCharge] = double(data.qCharge);
        out[kOutSTransverse0] = double(data.sTransverse(0));
        out[kOutInnerXX0] = data.innerXX0;
        out[kOutMatSum] = matSum;
      }
    }
  };

  // Kernel 2: solve V independent chains, one per lane.
  // Layout of `nodesAll`: V chains of kNodes nodes, contiguous. `scratch`: V blocks of
  // kGblScratchDoubles<kM>. `out`: V blocks of kFitOutSize.
  enum FitOut { kFitCov = 0, kFitCorr = 25, kFitChi2 = 30, kFitOutSize = 31 };

  struct SolveKernel {
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc, gbld::GblNodeData const* nodesAll, double* scratch, double* out, int nVariants) const {
      for (auto v : cms::alpakatools::uniform_elements(acc, nVariants)) {
        gbld::Vector5d corr = gbld::Vector5d::Zero();
        double chi2 = 0.;
        const auto cov = gbld::gblFitPca<Acc1D, kM>(acc,
                                                    nodesAll + int(v) * kNodes,
                                                    scratch + int(v) * gbld::kGblScratchDoubles<kM>,
                                                    &corr,
                                                    /*fullDelta=*/nullptr,
                                                    &chi2);
        double* o = out + int(v) * kFitOutSize;
        for (int a = 0; a < 5; ++a) {
          o[kFitCorr + a] = corr(a);
          for (int b = 0; b < 5; ++b)
            o[kFitCov + a * 5 + b] = cov(a, b);
        }
        o[kFitChi2] = chi2;
      }
    }
  };

  //!< max relative difference with a positive control (a 0-vs-0 comparison always passes).
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

}  // namespace

TEST_CASE("GBL tilted-module measurement model for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty())
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, test skipped.");

  using namespace gblTestFixtures;
  constexpr double kB = gblTestFixtures::kB;

  Eigen::Matrix<double, 3, kN> hits;
  Eigen::Matrix<float, 6, kN> hits_ge = Eigen::Matrix<float, 6, kN>::Zero();
  for (int k = 0; k < kN; ++k) {
    hits(0, k) = TILT[k][0];
    hits(1, k) = TILT[k][1];
    hits(2, k) = TILT[k][2];
    hits_ge.col(k) << TILT[k][3], TILT[k][4], TILT[k][5], TILT[k][6], TILT[k][7], TILT[k][8];
  }
  Eigen::Vector4d ff;
  blh::fastFit(hits, ff);

  auto rho_h = cms::alpakatools::make_host_buffer<float[], Platform>(blMaterialMap::kSize);
  std::copy_n(blMaterialMap::blMaterialMapData(), blMaterialMap::kSize, rho_h.data());

  for (auto const& device : devices) {
    auto queue = Queue(device);
    const std::string devName = alpaka::getName(device);

    auto rho_d = cms::alpakatools::make_device_buffer<float[]>(queue, blMaterialMap::kSize);
    alpaka::memcpy(queue, rho_d, rho_h);

    auto hits_h = cms::alpakatools::make_host_buffer<double[], Platform>(3 * kN);
    auto ge_h = cms::alpakatools::make_host_buffer<float[], Platform>(6 * kN);
    auto ff_h = cms::alpakatools::make_host_buffer<double[], Platform>(4);
    for (int c = 0; c < kN; ++c) {
      for (int r = 0; r < 3; ++r)
        hits_h[3 * c + r] = hits(r, c);
      for (int r = 0; r < 6; ++r)
        ge_h[6 * c + r] = hits_ge(r, c);
    }
    for (int j = 0; j < 4; ++j)
      ff_h[j] = ff(j);

    auto hits_d = cms::alpakatools::make_device_buffer<double[]>(queue, 3 * kN);
    auto ge_d = cms::alpakatools::make_device_buffer<float[]>(queue, 6 * kN);
    auto ff_d = cms::alpakatools::make_device_buffer<double[]>(queue, 4);
    auto nodes_d = cms::alpakatools::make_device_buffer<gbld::GblNodeData[]>(queue, kNodes);
    auto bout_d = cms::alpakatools::make_device_buffer<double[]>(queue, kBuildOutSize);
    auto nodes_h = cms::alpakatools::make_host_buffer<gbld::GblNodeData[], Platform>(kNodes);
    auto bout_h = cms::alpakatools::make_host_buffer<double[], Platform>(kBuildOutSize);
    alpaka::memcpy(queue, hits_d, hits_h);
    alpaka::memcpy(queue, ge_d, ge_h);
    alpaka::memcpy(queue, ff_d, ff_h);
    alpaka::memset(queue, bout_d, 0);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(1, 1),
                        BuildKernel{},
                        hits_d.data(),
                        ge_d.data(),
                        ff_d.data(),
                        rho_d.data(),
                        kB,
                        nodes_d.data(),
                        bout_d.data());
    alpaka::memcpy(queue, nodes_h, nodes_d);
    alpaka::memcpy(queue, bout_h, bout_d);
    alpaka::wait(queue);

    // TILT is a prompt ten-hit track: production always takes the 2N+1 two-thin-scatterer layout for it.
    if (!(bout_h[kOutUsedSplit] > 0.5))
      FAIL("prepareGblDataSplit refused the TILT fixture: the 2N+1 layout is expected for a prompt N=10 track.");
    REQUIRE(bout_h[kOutInnerXX0] > 0.);
    REQUIRE(bout_h[kOutMatSum] > 0.);
    const int qCharge = int(bout_h[kOutQCharge]);

    // hit i sits at the (i+1)-th measurement node of the chain.
    std::vector<int> hitNode;
    hitNode.reserve(kN);
    for (int k = 0; k < kNodes; ++k)
      if (nodes_h[k].hasMeas)
        hitNode.push_back(k);
    REQUIRE(int(hitNode.size()) == kN);

    // Build the seven variants host-side, with test-local algebra.
    std::vector<std::vector<gblh::GblNodeData>> hv(kVariants, std::vector<gblh::GblNodeData>(kNodes));
    auto dv_h = cms::alpakatools::make_host_buffer<gbld::GblNodeData[], Platform>(kVariants * kNodes);
    for (int k = 0; k < kNodes; ++k) {
      const gbld::GblNodeData& d = nodes_h[k];
      for (int v = 0; v < kVariants; ++v) {
        hv[v][k].jacToPrev = d.jacToPrev;
        hv[v][k].measPrec = d.measPrec;
        hv[v][k].scatPrec = d.scatPrec;
        hv[v][k].measResidual = d.measResidual;
        hv[v][k].hasMeas = d.hasMeas;
        hv[v][k].hasScat = d.hasScat;
      }
    }
    // (b) module frame: the curvilinear (U,V) frame at each hit, exactly as the device builder forms it
    // (alpaka/GeneralBrokenLine.h emitHit), then proL2m maps a (U,V) offset onto the true module axes.
    {
      const double cx = ff(0), cy = ff(1), R = ff(2);
      const double slope = -double(qCharge) / ff(3);
      const double invn = 1. / std::sqrt(1. + slope * slope);
      for (int i = 0; i < kN; ++i) {
        const double rx = (hits(0, i) - cx) / R, ry = (hits(1, i) - cy) / R;
        const Eigen::Vector3d T(double(qCharge) * ry * invn, -double(qCharge) * rx * invn, slope * invn);
        const double un = 1. / std::sqrt(T.x() * T.x() + T.y() * T.y());
        const Eigen::Vector3d U(-T.y() * un, T.x() * un, 0.);
        const Eigen::Vector3d V = T.cross(U);
        const Eigen::Vector3d lx(TILT_LX[i][0], TILT_LX[i][1], TILT_LX[i][2]);
        const Eigen::Vector3d ly(TILT_LY[i][0], TILT_LY[i][1], TILT_LY[i][2]);
        Eigen::Matrix2d proL2m;
        proL2m << lx.dot(U), lx.dot(V), ly.dot(U), ly.dot(V);
        Eigen::Matrix2d locPrec = Eigen::Matrix2d::Zero();
        locPrec(0, 0) = 1. / TILT_XEL[i];
        locPrec(1, 1) = 1. / TILT_YEL[i];
        const Eigen::Matrix2d aInv = proL2m.inverse();
        hv[1][hitNode[i]].measPrec = proL2m.transpose() * locPrec * proL2m;
        // Keep the residual consistent with that precision, so this variant's chi2 stays interpretable:
        // the device's residual is A^T r_local (see testGblMeasFrameDevice), so the local residual is
        // A^-T r_dev and the obliquity-free model's own residual is A^-1 r_local = A^-1 A^-T r_dev.
        hv[1][hitNode[i]].measResidual = aInv * (aInv.transpose() * nodes_h[hitNode[i]].measResidual);
      }
    }
    for (int k = 0; k < kNodes; ++k) {
      if (hv[0][k].hasScat) {
        hv[2][k].scatPrec(1, 1) = hv[0][k].scatPrec(0, 0);  // (c) isotropic kink
        hv[3][k].scatPrec *= 1.e6;                          // (d) stiff kinks: MS frozen
        hv[4][k].scatPrec *= 10.;                           // (e) MS x0.1
        hv[5][k].scatPrec *= 0.1;                           // (f) MS x10
      }
      if (hv[0][k].hasMeas) {  // (g) all measurements precise and isotropic, sigma = 1 um
        hv[6][k].measPrec = Eigen::Matrix2d::Zero();
        hv[6][k].measPrec(0, 0) = hv[6][k].measPrec(1, 1) = 1.e8;
      }
    }
    for (int v = 0; v < kVariants; ++v)
      for (int k = 0; k < kNodes; ++k) {
        gbld::GblNodeData d;
        d.jacToPrev = hv[v][k].jacToPrev;
        d.measPrec = hv[v][k].measPrec;
        d.scatPrec = hv[v][k].scatPrec;
        d.measResidual = hv[v][k].measResidual;
        d.hasMeas = hv[v][k].hasMeas;
        d.hasScat = hv[v][k].hasScat;
        dv_h[v * kNodes + k] = d;
      }

    // Solve all seven on the device, and the same seven on the host twin.
    auto dv_d = cms::alpakatools::make_device_buffer<gbld::GblNodeData[]>(queue, kVariants * kNodes);
    auto scr_d = cms::alpakatools::make_device_buffer<double[]>(queue, kVariants * gbld::kGblScratchDoubles<kM>);
    auto fout_d = cms::alpakatools::make_device_buffer<double[]>(queue, kVariants * kFitOutSize);
    auto fout_h = cms::alpakatools::make_host_buffer<double[], Platform>(kVariants * kFitOutSize);
    alpaka::memcpy(queue, dv_d, dv_h);
    alpaka::memset(queue, fout_d, 0);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(1, kVariants),
                        SolveKernel{},
                        dv_d.data(),
                        scr_d.data(),
                        fout_d.data(),
                        kVariants);
    alpaka::memcpy(queue, fout_h, fout_d);
    alpaka::wait(queue);

    const double slope = -double(qCharge) / ff(3);
    const double sec2 = 1. + slope * slope;
    const char* vname[kVariants] = {"(a) as built",
                                    "(b) module frame",
                                    "(c) isotropic scat",
                                    "(d) stiff kinks",
                                    "(e) MS x0.1",
                                    "(f) MS x10",
                                    "(g) all precise"};
    double sig1R[kVariants] = {};
    std::printf("\n=== %s : TILT eta=1.087 Pt100, tilted OT barrel at hits 4,5,6 ===\n", devName.c_str());
    for (int v = 0; v < kVariants; ++v) {
      gblh::Vector5d hostCorr;
      double hostChi2 = 0.;
      const gblh::Matrix5d hostCov = gblh::gblFitPca(hv[v], &hostCorr, nullptr, &hostChi2);
      MaxRel cov, corr, chi2;
      const double* o = &fout_h[v * kFitOutSize];
      for (int a = 0; a < 5; ++a) {
        corr.add(o[kFitCorr + a], hostCorr(a));
        for (int b = 0; b < 5; ++b)
          cov.add(o[kFitCov + a * 5 + b], hostCov(a, b));
      }
      chi2.add(o[kFitChi2], hostChi2);
      sig1R[v] = kB * std::sqrt(std::abs(sec2 * o[kFitCov]));
      std::printf(
          "  %-19s sigma(1/R)=%.4e  sigma(d0)=%8.2f um  sigma(z0)=%8.2f um  chi2=%9.3f | twin cov=%.1e corr=%.1e "
          "chi2=%.1e\n",
          vname[v],
          sig1R[v],
          1.e4 * std::sqrt(std::abs(o[kFitCov + 3 * 5 + 3])),
          1.e4 * std::sqrt(std::abs(o[kFitCov + 4 * 5 + 4])),
          o[kFitChi2],
          cov.v,
          corr.v,
          chi2.v);
      REQUIRE(cov.any);
      REQUIRE(corr.any);
      REQUIRE(chi2.any);
      const double tolCov = (v == kStiffVariant) ? kTolTwinStiff : kTolTwin;
      const double tolChi2 = (v == kStiffVariant) ? kTolTwinStiffChi2 : kTolTwin;
      REQUIRE(cov.v < tolCov);
      REQUIRE(corr.v < tolCov);
      REQUIRE(chi2.v < tolChi2);
      REQUIRE(std::isfinite(sig1R[v]));
      REQUIRE(sig1R[v] > 0.);
    }

    std::printf(
        "  sigma(1/R) (a)/(b) = %.3f  [asserted inside the bracket, see the file head]   (g)/(a) = %.3f "
        " [(g) is NOT bounded]\n",
        sig1R[0] / sig1R[1],
        sig1R[6] / sig1R[0]);

    // theorems: loosening the kinks cannot tighten the curvature, and an isotropic kink is the tighter one.
    REQUIRE(sig1R[3] <= sig1R[4] * (1. + kMonoSlack));  // stiff  <= MS x0.1
    REQUIRE(sig1R[4] <= sig1R[0] * (1. + kMonoSlack));  // MS x0.1 <= MS x1
    REQUIRE(sig1R[0] <= sig1R[5] * (1. + kMonoSlack));  // MS x1   <= MS x10
    REQUIRE(sig1R[2] <= sig1R[0] * (1. + kMonoSlack));  // isotropic kink is more precise than cos^2(lambda)

    // the bracket on the ratio (see the file head).
    REQUIRE(sig1R[0] >= kRatioMin * sig1R[1]);
    REQUIRE(sig1R[0] <= kRatioMax * sig1R[1]);
  }
}
