// Device-twin check of the alpaka GeneralBrokenLine functions (curvilinearJacobian, gblFitPca,
// gblHelixAtPca): on every available backend they must reproduce the CPU twin
// (interface/GeneralBrokenLine.h) to 1e-9 on a small synthetic node chain. Every measurement node carries
// a non-zero residual and the host references are asserted non-zero, so no check can pass vacuously.

#include <cmath>
#include <cstdio>
#include <vector>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/PixelTrackFitting/interface/GeneralBrokenLine.h"         // CPU twin (host reference)
#include "RecoTracker/PixelTrackFitting/interface/alpaka/GeneralBrokenLine.h"  // device twin under test

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
// The using-directive above pulls the device twin's generalBrokenLine into scope, so the host twin needs
// the global-scope qualification.
namespace gblh = ::generalBrokenLine;                              // host (CPU twin)
namespace gbld = ALPAKA_ACCELERATOR_NAMESPACE::generalBrokenLine;  // device twin

namespace {
  constexpr int kN = 4;  // hits -> nNodes = kN + 1
}

// writes gblFitPca cov to out[0..24] and curvilinearJacobian to out[25..49].
struct GblKernel {
  ALPAKA_FN_ACC void operator()(Acc1D const& acc, gbld::GblNodeData const* nodes, double const* seg, double* out) const {
    for ([[maybe_unused]] auto i : cms::alpakatools::uniform_elements(acc, 1)) {
      gbld::Vector5d gcorr;
      double fd[2 * kN + 3];
      double chi2d = 0.;
      // gblFitPca takes its bordered-band solver scratch explicitly; here a local block of stride 1.
      double gblScratch[gbld::kGblScratchDoubles<kN>];
      const auto cov = gbld::gblFitPca<Acc1D, kN>(acc, nodes, gblScratch, &gcorr, fd, &chi2d);
      out[86 + 2 * kN + 3] = chi2d;  // native GBL chi2 (after the fullDelta block out[86..86+2kN+2])
      for (int a = 0; a < 5; ++a)
        for (int b = 0; b < 5; ++b)
          out[a * 5 + b] = cov(a, b);
      const gbld::Vector3d p1(seg[0], seg[1], seg[2]), p2(seg[3], seg[4], seg[5]);
      const gbld::Vector3d dx(seg[6], seg[7], seg[8]), h(seg[9], seg[10], seg[11]);
      const auto jac = gbld::curvilinearJacobian(acc, p1, p2, dx, seg[12], h, seg[13]);
      for (int a = 0; a < 5; ++a)
        for (int b = 0; b < 5; ++b)
          out[25 + a * 5 + b] = jac(a, b);
      // corrections out[50..54], gblHelixAtPca helixPar out[55..59] and helixCov out[60..84].
      for (int a = 0; a < 5; ++a)
        out[50 + a] = gcorr(a);
      Eigen::Vector4d ff(seg[14], seg[15], seg[16], seg[17]);
      gbld::Vector5d hp;
      gbld::Matrix5d hc;
      gbld::gblHelixAtPca(acc, ff, int(seg[18]), seg[19], seg[20], seg[21], gcorr, cov, hp, hc);
      for (int a = 0; a < 5; ++a)
        out[55 + a] = hp(a);
      for (int a = 0; a < 5; ++a)
        for (int b = 0; b < 5; ++b)
          out[60 + a * 5 + b] = hc(a, b);
      // the full solution vector fullDelta, out[86..86+2kN+2].
      for (int a = 0; a < 2 * kN + 3; ++a)
        out[86 + a] = fd[a];
    }
  }
};

TEST_CASE("GeneralBrokenLine device twin vs CPU twin for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty())
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, test skipped.");

  // one representative segment Jacobian, artificial but valid: invertible blocks, SPD system
  Eigen::Vector3d p1(0.90, 0.10, 0.42);
  p1.normalize();
  Eigen::Vector3d p2(0.88, 0.14, 0.42);
  p2.normalize();
  const Eigen::Vector3d dxv(-3.0, -0.3, -1.4);
  const Eigen::Vector3d hv(0., 0., 0.0113921);
  const double qbp = -1. / 50., sstep = 4.0;
  const auto jac = gblh::curvilinearJacobian(p1, p2, dxv, qbp, hv, sstep);

  Eigen::Matrix2d measP;
  measP << 1.0e6, 0., 0., 1.0e4;
  Eigen::Matrix2d scatP;
  scatP << 1.0e7, 0., 0., 1.2e7;

  // build identical node arrays for host (vector) and device (host buffer of the device struct).
  std::vector<gblh::GblNodeData> hostNodes(kN + 1);
  auto devNodes_h = cms::alpakatools::make_host_buffer<gbld::GblNodeData[], Platform>(kN + 1);
  for (int k = 0; k <= kN; ++k) {
    gbld::GblNodeData d;  // node 0 (PCA): default, no measurement/scatterer
    if (k >= 1) {
      // deterministic, node-dependent and non-zero on every measurement node
      const Eigen::Vector2d resid(1.5e-3 * double(k), -0.8e-3 * double(k + 1));
      hostNodes[k].jacToPrev = jac;
      hostNodes[k].measPrec = measP;
      hostNodes[k].measResidual = resid;
      hostNodes[k].hasMeas = true;
      d.jacToPrev = jac;
      d.measPrec = measP;
      d.measResidual = resid;
      d.hasMeas = true;
    }
    if (k >= 1 && k <= kN - 1) {
      hostNodes[k].scatPrec = scatP;
      hostNodes[k].hasScat = true;
      d.scatPrec = scatP;
      d.hasScat = true;
    }
    devNodes_h[k] = d;
  }
  gblh::Vector5d hostCorr;
  Eigen::VectorXd hostFd;
  double hostChi2 = 0.;
  const auto hostCov = gblh::gblFitPca(hostNodes, &hostCorr, &hostFd, &hostChi2);
  // host gblHelixAtPca on a made-up but valid fast-fit and reference
  const Eigen::Vector4d ff(5.0, 3.0, 100.0, 2.4);
  const int qCh = 1;
  const double bF = 0.0113921, sT0 = 2.0, hZ0 = 1.0;
  gblh::Vector5d hostHp;
  gblh::Matrix5d hostHc;
  gblh::gblHelixAtPca(ff, qCh, bF, sT0, hZ0, hostCorr, hostCov, hostHp, hostHc);
  // the relative comparisons below skip any reference that is numerically zero, so require first that the
  // fit actually moved.
  REQUIRE(hostCorr.cwiseAbs().maxCoeff() > 1e-30);
  REQUIRE(hostFd.cwiseAbs().maxCoeff() > 1e-30);
  REQUIRE(hostHp.cwiseAbs().maxCoeff() > 1e-30);
  REQUIRE(std::abs(hostChi2) > 1e-30);

  // seg inputs for the device curvilinearJacobian and gblHelixAtPca checks
  constexpr int kSeg = 22;
  auto seg_h = cms::alpakatools::make_host_buffer<double[], Platform>(kSeg);
  for (int j = 0; j < 3; ++j) {
    seg_h[j] = p1[j];
    seg_h[3 + j] = p2[j];
    seg_h[6 + j] = dxv[j];
    seg_h[9 + j] = hv[j];
  }
  seg_h[12] = qbp;
  seg_h[13] = sstep;
  for (int j = 0; j < 4; ++j)
    seg_h[14 + j] = ff[j];
  seg_h[18] = qCh;
  seg_h[19] = bF;
  seg_h[20] = sT0;
  seg_h[21] = hZ0;

  for (auto const& device : cms::alpakatools::devices<Platform>()) {
    auto queue = Queue(device);
    constexpr int kOut = 86 + 2 * kN + 3 + 1;  // 86 prior + fullDelta(2kN+3) + chi2(1)
    auto nodes_d = cms::alpakatools::make_device_buffer<gbld::GblNodeData[]>(queue, kN + 1);
    auto seg_d = cms::alpakatools::make_device_buffer<double[]>(queue, kSeg);
    auto out_d = cms::alpakatools::make_device_buffer<double[]>(queue, kOut);
    auto out_h = cms::alpakatools::make_host_buffer<double[], Platform>(kOut);
    alpaka::memcpy(queue, nodes_d, devNodes_h);
    alpaka::memcpy(queue, seg_d, seg_h);
    auto div = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
    alpaka::exec<Acc1D>(queue, div, GblKernel{}, nodes_d.data(), seg_d.data(), out_d.data());
    alpaka::memcpy(queue, out_h, out_d);
    alpaka::wait(queue);

    double covMaxRel = 0., jacMaxRel = 0., corrMaxRel = 0., hpMaxRel = 0., hcMaxRel = 0.;
    for (int a = 0; a < 5; ++a) {
      if (std::abs(hostCorr(a)) > 1e-30)
        corrMaxRel = std::max(corrMaxRel, std::abs(out_h[50 + a] - hostCorr(a)) / std::abs(hostCorr(a)));
      if (std::abs(hostHp(a)) > 1e-30)
        hpMaxRel = std::max(hpMaxRel, std::abs(out_h[55 + a] - hostHp(a)) / std::abs(hostHp(a)));
      for (int b = 0; b < 5; ++b) {
        const double rc = hostCov(a, b), rj = jac(a, b), rh = hostHc(a, b);
        if (std::abs(rc) > 1e-30)
          covMaxRel = std::max(covMaxRel, std::abs(out_h[a * 5 + b] - rc) / std::abs(rc));
        if (std::abs(rj) > 1e-30)
          jacMaxRel = std::max(jacMaxRel, std::abs(out_h[25 + a * 5 + b] - rj) / std::abs(rj));
        if (std::abs(rh) > 1e-30)
          hcMaxRel = std::max(hcMaxRel, std::abs(out_h[60 + a * 5 + b] - rh) / std::abs(rh));
      }
    }
    double fdMaxRel = 0.;
    for (int a = 0; a < 2 * kN + 3; ++a)
      if (std::abs(hostFd(a)) > 1e-30)
        fdMaxRel = std::max(fdMaxRel, std::abs(out_h[86 + a] - hostFd(a)) / std::abs(hostFd(a)));
    const double chi2Rel =
        (std::abs(hostChi2) > 1e-30) ? std::abs(out_h[86 + 2 * kN + 3] - hostChi2) / std::abs(hostChi2) : 0.;
    std::printf(
        "  %s: gblFitPca cov=%.2e corr=%.2e | curvJac=%.2e | gblHelixAtPca par=%.2e cov=%.2e | fullDelta=%.2e "
        "chi2=%.2e\n",
        alpaka::getName(device).c_str(),
        covMaxRel,
        corrMaxRel,
        jacMaxRel,
        hpMaxRel,
        hcMaxRel,
        fdMaxRel,
        chi2Rel);
    REQUIRE(covMaxRel < 1e-9);
    REQUIRE(jacMaxRel < 1e-9);
    REQUIRE(corrMaxRel < 1e-9);
    REQUIRE(hpMaxRel < 1e-9);
    REQUIRE(hcMaxRel < 1e-9);
    REQUIRE(fdMaxRel < 1e-9);
    // chi2 = chi2ref - delta.b cancels two large nearly equal numbers (~2e6 here), so its best attainable
    // relative precision is eps * 2e6 ~ 4e-10; 1e-7 leaves headroom for FMA contraction on CUDA.
    REQUIRE(chi2Rel < 1e-7);
  }
}
