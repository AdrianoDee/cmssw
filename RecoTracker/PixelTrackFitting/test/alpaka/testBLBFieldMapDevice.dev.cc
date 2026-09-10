// The normalized (Bz, Br) solenoid map (interface/BLBFieldMap.h) on the device, with no EventSetup and
// no MagneticField product: a synthetic analytic lattice is uploaded, the device reads it through the
// same blBFieldMap::bBendAt / bBendAndBrAt the fit kernels call, and an analytic oracle checks it.
//
//     Bz(r,z)/Bz(0,0) = 1 - 7.5e-2 * (z/kZMax)^2      quadratic in z  -> bilinear leaves a known residual
//     Br(r,z)/Bz(0,0) = 1.8e-2 * (r/kRMax) * (z/kZMax) bilinear       -> reproduced exactly
// The magnitudes are the CMS ones: Bz falls ~7.5 % at |z| = 2.7 m and |Br/Bz| reaches ~1.8 % in the
// outer-tracker endcap.
//
// Asserted:
//   1. at lattice nodes the device returns exactly bz - br * tanLambdaCosAlpha from the two stored
//      floats, which tests the r-major ir*kNZ + iz indexing, the Bz-then-Br block order and the sign;
//   2. device == host for the same constexpr function on the same values, to 1e-14 relative;
//   3. at cell centres the interpolated value tracks the continuous profile to 5e-5: linear
//      interpolation of a*z^2 over a step h overshoots the midpoint by a*h^2/4, here 2.4e-5;
//   4. queries outside the sampled box are clamped to the boundary, not extrapolated;
//   5. bBendAndBrAt plus brNorm * tanLambdaCosAlpha reproduces the pure-Bz bBendAt value (the two
//      functions duplicate their index block, bBendAt being on the hot path);
//   6. the node builder run with the map and with bMap == nullptr gives different corrections, so the
//      map is not silently ignored.

#include <algorithm>
#include <array>
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

#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BrokenLine.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/GeneralBrokenLine.h"
#include "RecoTracker/PixelTrackFitting/test/gblTestFixtures.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
namespace blh = ::brokenline;
namespace gbld = ALPAKA_ACCELERATOR_NAMESPACE::generalBrokenLine;
namespace bld = ALPAKA_ACCELERATOR_NAMESPACE::brokenline;

namespace {

  constexpr double kBzCurv = 7.5e-2;   // Bz/Bz00 = 1 - kBzCurv * (z/kZMax)^2
  constexpr double kBrScale = 1.8e-2;  // Br/Bz00 = kBrScale * (r/kRMax) * (z/kZMax)
  constexpr double kTlca = 0.7;        // a representative tanLambda * cos(alpha)

  //!< device vs host for the same constexpr function on the same buffer.
  constexpr double kTolExact = 1e-14;
  //!< bilinear-vs-continuous bound at cell centres, a*h^2/4 = 2.4e-5.
  constexpr double kTolBilinear = 5e-5;

  double bzAnalytic(double r, double z) {
    (void)r;
    const double t = z / double(blBFieldMap::kZMax);
    return 1. - kBzCurv * t * t;
  }
  double brAnalytic(double r, double z) {
    return kBrScale * (r / double(blBFieldMap::kRMax)) * (z / double(blBFieldMap::kZMax));
  }

  struct FieldKernel {
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc, float const* map, double const* pts, double* out, int nPts, double tlca) const {
      for (auto i : cms::alpakatools::uniform_elements(acc, nPts)) {
        const double r = pts[2 * int(i)], z = pts[2 * int(i) + 1];
        double br = 0.;
        out[3 * int(i) + 0] = blBFieldMap::bBendAt(map, r, z, tlca);
        out[3 * int(i) + 1] = blBFieldMap::bBendAndBrAt(map, r, z, tlca, br);
        out[3 * int(i) + 2] = br;
      }
    }
  };

  //!< the FWD fixture through the production node builder, with and without the map.
  constexpr int kNf = 10;
  constexpr int kNodesF = 2 * kNf + 1;

  struct WiringKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  double const* hitsIn,
                                  float const* geIn,
                                  double const* ffIn,
                                  float const* rho,
                                  float const* bMap,
                                  double bField,
                                  gbld::GblNodeData* nodes,
                                  double* scratch,
                                  double* out) const {
      for (auto v : cms::alpakatools::uniform_elements(acc, 2)) {
        Eigen::Matrix<double, 3, kNf> hits;
        Eigen::Matrix<float, 6, kNf> hits_ge;
        for (int c = 0; c < kNf; ++c) {
          for (int r = 0; r < 3; ++r)
            hits(r, c) = hitsIn[3 * c + r];
          for (int r = 0; r < 6; ++r)
            hits_ge(r, c) = geIn[6 * c + r];
        }
        const Eigen::Vector4d ff(ffIn[0], ffIn[1], ffIn[2], ffIn[3]);
        bld::PreparedGblData<kNf> data;
        double gapD1[kNf], gapW1[kNf];
        bld::prepareGblFitData(acc, hits, ff, bField, rho, data, /*matCached=*/nullptr, gapD1, gapW1);
        const double rHit0 = alpaka::math::sqrt(acc, hits(0, 0) * hits(0, 0) + hits(1, 0) * hits(1, 0));
        double innerD1 = 0., innerW1 = 0.;
        if (data.innerXX0 > 0.)
          bld::segmentXX0GapSplit(acc, rho, 0., 0., rHit0, hits(2, 0), innerD1, innerW1);

        gbld::GblNodeData* myNodes = nodes + int(v) * kNodesF;
        const bool ok = gbld::prepareGblDataSplit<Acc1D, kNf>(acc,
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
                                                              myNodes,
                                                              /*msScale=*/1.0,
                                                              /*eLossPerX0=*/0.0,
                                                              /*bMap=*/(int(v) == 1) ? bMap : nullptr,
                                                              /*bFieldOrigin=*/bField,
                                                              /*trajectoryCorrections=*/false,
                                                              /*scatteringLogAtTotal=*/false,
                                                              /*elossCumulative=*/false);
        gbld::Vector5d corr = gbld::Vector5d::Zero();
        double chi2 = 0.;
        const auto cov = gbld::gblFitPca<Acc1D, 2 * kNf>(
            acc, myNodes, scratch + int(v) * gbld::kGblScratchDoubles<2 * kNf>, &corr, nullptr, &chi2);
        double* o = out + int(v) * 32;
        o[0] = ok ? 1. : 0.;
        for (int a = 0; a < 5; ++a) {
          o[1 + a] = corr(a);
          o[6 + a] = cov(a, a);
        }
        o[11] = chi2;
      }
    }
  };

}  // namespace

TEST_CASE("blBFieldMap on the device for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty())
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, test skipped.");

  using blBFieldMap::kNNodes;
  using blBFieldMap::kNR;
  using blBFieldMap::kNValues;
  using blBFieldMap::kNZ;
  const double dr = double(blBFieldMap::kRMax) / double(kNR - 1);
  const double dz = 2. * double(blBFieldMap::kZMax) / double(kNZ - 1);

  // the synthetic lattice
  std::vector<float> mapHost(kNValues);
  for (int ir = 0; ir < kNR; ++ir)
    for (int iz = 0; iz < kNZ; ++iz) {
      const double r = ir * dr, z = -double(blBFieldMap::kZMax) + iz * dz;
      mapHost[ir * kNZ + iz] = float(bzAnalytic(r, z));
      mapHost[kNNodes + ir * kNZ + iz] = float(brAnalytic(r, z));
    }

  // query points: every lattice node, every cell centre, and four out-of-box probes
  std::vector<double> pts;
  std::vector<int> nodeIr, nodeIz;  // lattice-node bookkeeping for assertion 1
  for (int ir = 0; ir < kNR; ++ir)
    for (int iz = 0; iz < kNZ; ++iz) {
      pts.push_back(ir * dr);
      pts.push_back(-double(blBFieldMap::kZMax) + iz * dz);
      nodeIr.push_back(ir);
      nodeIz.push_back(iz);
    }
  const int nNodesPts = int(nodeIr.size());
  for (int ir = 0; ir + 1 < kNR; ++ir)
    for (int iz = 0; iz + 1 < kNZ; ++iz) {
      pts.push_back((ir + 0.5) * dr);
      pts.push_back(-double(blBFieldMap::kZMax) + (iz + 0.5) * dz);
    }
  const int nCentres = (kNR - 1) * (kNZ - 1);
  const double rMax = double(blBFieldMap::kRMax), zMax = double(blBFieldMap::kZMax);
  const double clampProbes[4][2] = {{2. * rMax, 0.}, {-5., 0.}, {10., 3. * zMax}, {10., -3. * zMax}};
  const double clampRefs[4][2] = {{rMax, 0.}, {0., 0.}, {10., zMax}, {10., -zMax}};
  for (int i = 0; i < 4; ++i) {
    pts.push_back(clampProbes[i][0]);
    pts.push_back(clampProbes[i][1]);
  }
  for (int i = 0; i < 4; ++i) {
    pts.push_back(clampRefs[i][0]);
    pts.push_back(clampRefs[i][1]);
  }
  const int nPts = int(pts.size()) / 2;
  const int iClamp = nNodesPts + nCentres;

  // the forward fixture, for the wiring check
  Eigen::Matrix<double, 3, kNf> fhits;
  Eigen::Matrix<float, 6, kNf> fge = Eigen::Matrix<float, 6, kNf>::Zero();
  for (int k = 0; k < kNf; ++k) {
    fhits(0, k) = gblTestFixtures::FWD[k][0];
    fhits(1, k) = gblTestFixtures::FWD[k][1];
    fhits(2, k) = gblTestFixtures::FWD[k][2];
    const double zz = (gblTestFixtures::FWD[k][8] > 0.) ? gblTestFixtures::FWD[k][8] : 1.0;
    fge.col(k) << gblTestFixtures::FWD[k][3], gblTestFixtures::FWD[k][4], gblTestFixtures::FWD[k][5],
        gblTestFixtures::FWD[k][6], gblTestFixtures::FWD[k][7], zz;
  }
  Eigen::Vector4d fff;
  blh::fastFit(fhits, fff);

  auto rho_h = cms::alpakatools::make_host_buffer<float[], Platform>(blMaterialMap::kSize);
  std::copy_n(blMaterialMap::blMaterialMapData(), blMaterialMap::kSize, rho_h.data());

  for (auto const& device : devices) {
    auto queue = Queue(device);
    const std::string dn = alpaka::getName(device);

    auto map_h = cms::alpakatools::make_host_buffer<float[], Platform>(kNValues);
    std::copy_n(mapHost.data(), kNValues, map_h.data());
    auto pts_h = cms::alpakatools::make_host_buffer<double[], Platform>(2 * nPts);
    std::copy_n(pts.data(), 2 * nPts, pts_h.data());
    auto map_d = cms::alpakatools::make_device_buffer<float[]>(queue, kNValues);
    auto pts_d = cms::alpakatools::make_device_buffer<double[]>(queue, 2 * nPts);
    auto fout_d = cms::alpakatools::make_device_buffer<double[]>(queue, 3 * nPts);
    auto fout_h = cms::alpakatools::make_host_buffer<double[], Platform>(3 * nPts);
    alpaka::memcpy(queue, map_d, map_h);
    alpaka::memcpy(queue, pts_d, pts_h);
    // nPts is a few thousand, more than one block can hold, so grid-stride it
    constexpr int kThreads = 256;
    const int kBlocks = (nPts + kThreads - 1) / kThreads;
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(kBlocks, kThreads),
                        FieldKernel{},
                        map_d.data(),
                        pts_d.data(),
                        fout_d.data(),
                        nPts,
                        kTlca);
    alpaka::memcpy(queue, fout_h, fout_d);
    alpaka::wait(queue);

    // 1. lattice nodes: exactly the two stored floats combined by the bending law.
    double worstNode = 0.;
    for (int i = 0; i < nNodesPts; ++i) {
      const double bz = double(mapHost[nodeIr[i] * kNZ + nodeIz[i]]);
      const double br = double(mapHost[kNNodes + nodeIr[i] * kNZ + nodeIz[i]]);
      const double want = bz - br * kTlca;
      worstNode = std::max(worstNode, std::abs(fout_h[3 * i] - want) / std::max(1e-30, std::abs(want)));
    }

    // 2. device == host for the same constexpr function, and 5. the two entry points stay in sync.
    double worstHost = 0., worstSync = 0.;
    for (int i = 0; i < nPts; ++i) {
      const double r = pts[2 * i], z = pts[2 * i + 1];
      const double hostVal = blBFieldMap::bBendAt(mapHost.data(), r, z, kTlca);
      worstHost = std::max(worstHost, std::abs(fout_h[3 * i] - hostVal) / std::max(1e-30, std::abs(hostVal)));
      const double pureBz = blBFieldMap::bBendAt(mapHost.data(), r, z, 0.);
      const double rebuilt = fout_h[3 * i + 1] + fout_h[3 * i + 2] * kTlca;
      worstSync = std::max(worstSync, std::abs(rebuilt - pureBz) / std::max(1e-30, std::abs(pureBz)));
    }

    // 3. cell centres vs the continuous profile.
    double worstBilinear = 0.;
    for (int c = 0; c < nCentres; ++c) {
      const int i = nNodesPts + c;
      const double r = pts[2 * i], z = pts[2 * i + 1];
      const double want = bzAnalytic(r, z) - brAnalytic(r, z) * kTlca;
      worstBilinear = std::max(worstBilinear, std::abs(fout_h[3 * i] - want));
    }

    // 4. clamping.
    double worstClamp = 0.;
    for (int i = 0; i < 4; ++i)
      worstClamp = std::max(worstClamp, std::abs(fout_h[3 * (iClamp + i)] - fout_h[3 * (iClamp + 4 + i)]));

    std::printf(
        "\n=== %s : blBFieldMap synthetic lattice (%d nodes, %d centres) ===\n", dn.c_str(), nNodesPts, nCentres);
    std::printf(
        "  node exactness=%.2e  device-vs-host=%.2e  bBendAndBrAt sync=%.2e  bilinear-vs-analytic=%.2e  "
        "clamp=%.2e\n",
        worstNode,
        worstHost,
        worstSync,
        worstBilinear,
        worstClamp);
    REQUIRE(worstNode < kTolExact);
    REQUIRE(worstHost < kTolExact);
    REQUIRE(worstSync < kTolExact);
    REQUIRE(worstBilinear < kTolBilinear);
    REQUIRE(worstClamp < 1e-15);

    // 6. wiring: the node builder must see the map
    auto rho_d = cms::alpakatools::make_device_buffer<float[]>(queue, blMaterialMap::kSize);
    auto fh_h = cms::alpakatools::make_host_buffer<double[], Platform>(3 * kNf);
    auto fg_h = cms::alpakatools::make_host_buffer<float[], Platform>(6 * kNf);
    auto fq_h = cms::alpakatools::make_host_buffer<double[], Platform>(4);
    for (int c = 0; c < kNf; ++c) {
      for (int r = 0; r < 3; ++r)
        fh_h[3 * c + r] = fhits(r, c);
      for (int r = 0; r < 6; ++r)
        fg_h[6 * c + r] = fge(r, c);
    }
    for (int j = 0; j < 4; ++j)
      fq_h[j] = fff(j);
    auto fh_d = cms::alpakatools::make_device_buffer<double[]>(queue, 3 * kNf);
    auto fg_d = cms::alpakatools::make_device_buffer<float[]>(queue, 6 * kNf);
    auto fq_d = cms::alpakatools::make_device_buffer<double[]>(queue, 4);
    auto wn_d = cms::alpakatools::make_device_buffer<gbld::GblNodeData[]>(queue, 2 * kNodesF);
    auto ws_d = cms::alpakatools::make_device_buffer<double[]>(queue, 2 * gbld::kGblScratchDoubles<2 * kNf>);
    auto wo_d = cms::alpakatools::make_device_buffer<double[]>(queue, 2 * 32);
    auto wo_h = cms::alpakatools::make_host_buffer<double[], Platform>(2 * 32);
    alpaka::memcpy(queue, rho_d, rho_h);
    alpaka::memcpy(queue, fh_d, fh_h);
    alpaka::memcpy(queue, fg_d, fg_h);
    alpaka::memcpy(queue, fq_d, fq_h);
    alpaka::memset(queue, wo_d, 0);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(1, 2),
                        WiringKernel{},
                        fh_d.data(),
                        fg_d.data(),
                        fq_d.data(),
                        rho_d.data(),
                        map_d.data(),
                        gblTestFixtures::kB,
                        wn_d.data(),
                        ws_d.data(),
                        wo_d.data());
    alpaka::memcpy(queue, wo_h, wo_d);
    alpaka::wait(queue);

    REQUIRE(wo_h[0] > 0.5);   // the split layout was built without the map
    REQUIRE(wo_h[32] > 0.5);  // and with it
    double maxCorrRel = 0.;
    for (int a = 0; a < 5; ++a) {
      const double ref = wo_h[1 + a];
      if (std::abs(ref) > 1e-30)
        maxCorrRel = std::max(maxCorrRel, std::abs(wo_h[32 + 1 + a] - ref) / std::abs(ref));
    }
    std::printf(
        "  wiring (FWD fixture): max relative change of the corrections with the map = %.3e  chi2 %.6g -> %.6g\n",
        maxCorrRel,
        wo_h[11],
        wo_h[32 + 11]);
    // only that the map reaches the node builder; the size of the shift is a physics number
    REQUIRE(maxCorrRel > 1e-9);
  }
}
