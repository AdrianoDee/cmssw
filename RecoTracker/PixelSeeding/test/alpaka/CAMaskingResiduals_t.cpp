// Check of the hit residual helpers used by CAMaskSoAProducer (RecoTracker/PixelSeeding/interface/CAMaskingUtilities.h)
// against the track parameters produced by the broken-line fit of the CA.
//
// Hits are generated on an exact helix (beam-spot frame), fitted with the same sequence of calls as Kernel_BLFit
// (fastFit, prepareBrokenLineData, lineFit, circleFit, reco::copyFromCircle with b = 1/B), including the subsampling of
// long tracks done in Kernel_BLFastFit; then the residuals of all hits (fitted or not) must be ~0, and a hit displaced
// by a known amount must give that residual.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include <alpaka/alpaka.hpp>

#include <Eigen/Core>

#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/PixelSeeding/interface/CAMaskingUtilities.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
namespace chc = caHitCompatibility;

namespace {

  constexpr double kBField = 1. / 87.7792;  // 1/inverseBzAtOriginInGeV for 3.8 T, as in CAHitNtuplet.cc
  constexpr int kMaxHits = 8;

  // One track per kernel call: same calls and same conventions as Kernel_BLFit (BrokenLineFit.h).
  template <int N>
  struct Kernel_testBLFit {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  double const* hitsIn,   // 3 x N, column major
                                  float const* hitsGeIn,  // 6 x N, column major
                                  double bField,
                                  reco::TrackSoAView tracks,
                                  int32_t tkid) const {
      if (not cms::alpakatools::once_per_grid(acc))
        return;
      ::riemannFit::Matrix3xNd<N> hits;
      Eigen::Matrix<float, 6, N> hits_ge;  // riemannFit::Matrix6xNf<N> in HelixFit.h
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 3; ++j)
          hits(j, i) = hitsIn[3 * i + j];
        for (int j = 0; j < 6; ++j)
          hits_ge(j, i) = hitsGeIn[6 * i + j];
      }
      ::riemannFit::Vector4d fast_fit;
      brokenline::fastFit(acc, hits, fast_fit);

      brokenline::PreparedBrokenLineData<N> data;
      brokenline::karimaki_circle_fit circle;
      ::riemannFit::LineFit line;

      brokenline::prepareBrokenLineData(acc, hits, fast_fit, bField, data);
      brokenline::lineFit(acc, hits_ge, fast_fit, bField, data, line);
      brokenline::circleFit(acc, hits, hits_ge, fast_fit, bField, data, circle);

      reco::copyFromCircle(tracks, circle.par, circle.cov, line.par, line.cov, 1.f / float(bField), tkid);
    }
  };

  // Independent analytic helix in the Karimaki convention:
  // PCA = d (sin phi, -cos phi), direction (cos phi, sin phi), centre C = (1/rho + d) (sin phi, -cos phi).
  struct TrueHelix {
    double phi, d, rho, cotTheta, z0;

    void at(double s, double& x, double& y, double& z) const {
      double const xc = (1. / rho + d) * std::sin(phi);
      double const yc = -(1. / rho + d) * std::cos(phi);
      double const xp = d * std::sin(phi), yp = -d * std::cos(phi);
      double const theta0 = std::atan2(yp - yc, xp - xc);
      double const theta = theta0 - rho * s;  // clockwise for rho > 0
      x = xc + std::cos(theta) / std::abs(rho);
      y = yc + std::sin(theta) / std::abs(rho);
      z = z0 + cotTheta * s;
    }
    void centre(double& xc, double& yc) const {
      xc = (1. / rho + d) * std::sin(phi);
      yc = -(1. / rho + d) * std::cos(phi);
    }
    // arc length at which the helix reaches the radius r (first crossing), < 0 if never
    double sAtRadius(double r) const {
      double const sMax = M_PI / std::abs(rho);
      double x, y, z;
      at(sMax, x, y, z);
      if (std::hypot(x, y) < r)
        return -1.;
      double lo = 0., hi = sMax;
      for (int i = 0; i < 200; ++i) {
        double const mid = 0.5 * (lo + hi);
        at(mid, x, y, z);
        (std::hypot(x, y) < r ? lo : hi) = mid;
      }
      return 0.5 * (lo + hi);
    }
  };

  struct Hit {
    float x, y, z;
    bool barrel;
  };

  struct Summary {
    int nTracks = 0, nHits = 0, nFailed = 0;
    double maxT = 0, maxL = 0;            // exact hits
    double maxDT = 0, maxDL = 0;          // displaced hits: |measured - expected|
    double maxDPar[5] = {0, 0, 0, 0, 0};  // |fitted - true| parameters (phi, tip, rho, cotTheta, zip)
  };

  // the hit subsampling of Kernel_BLFastFit
  template <int N>
  std::vector<int> fittedHits(int nHits) {
    std::vector<int> idx;
    float incr = std::max(1.f, float(nHits) / float(N));
    float n = 0;
    for (int i = 0; i < N; ++i) {
      int j = int(n + 0.5f);
      if (N - 1 == i)
        j = nHits - 1;
      n += incr;
      idx.push_back(j);
    }
    return idx;
  }

  template <int N>
  void testTrack(Queue& queue,
                 reco::TrackSoAView tracks,
                 TrueHelix const& th,
                 std::vector<Hit> const& hitsAll,
                 char const* label,
                 Summary& sum,
                 bool verbose) {
    int const nHits = hitsAll.size();
    if (nHits < N)
      return;
    auto const idx = fittedHits<N>(nHits);
    double hitsIn[3 * N];
    float hitsGe[6 * N];
    for (int i = 0; i < N; ++i) {
      auto const& h = hitsAll[idx[i]];
      hitsIn[3 * i + 0] = h.x;
      hitsIn[3 * i + 1] = h.y;
      hitsIn[3 * i + 2] = h.z;
      // global covariance (xx, xy, yy, xz, yz, zz): ~15 um in r-phi, ~30 um in z (r for endcap hits)
      float const sT = 15.e-4f, sL = 30.e-4f;
      float const phiHit = std::atan2(h.y, h.x);
      float const c = std::cos(phiHit), s = std::sin(phiHit);
      if (h.barrel) {
        hitsGe[6 * i + 0] = sT * sT * s * s;
        hitsGe[6 * i + 1] = -sT * sT * s * c;
        hitsGe[6 * i + 2] = sT * sT * c * c;
        hitsGe[6 * i + 3] = 0.f;
        hitsGe[6 * i + 4] = 0.f;
        hitsGe[6 * i + 5] = sL * sL;
      } else {
        hitsGe[6 * i + 0] = sT * sT * s * s + sL * sL * c * c;
        hitsGe[6 * i + 1] = (sL * sL - sT * sT) * s * c;
        hitsGe[6 * i + 2] = sT * sT * c * c + sL * sL * s * s;
        hitsGe[6 * i + 3] = 0.f;
        hitsGe[6 * i + 4] = 0.f;
        hitsGe[6 * i + 5] = 1.e-8f;
      }
    }
    int32_t const tkid = 0;
    auto const workDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
    alpaka::exec<Acc1D>(queue, workDiv, Kernel_testBLFit<N>{}, hitsIn, hitsGe, kBField, tracks, tkid);
    alpaka::wait(queue);

    auto const state = tracks[tkid].state();
    auto const helix = chc::makeTrackHelix(state(0), state(1), state(2), state(3), state(4), float(kBField));

    double dphi = std::remainder(double(state(0)) - th.phi, 2. * M_PI);
    double const dpar[5] = {std::abs(dphi),
                            std::abs(state(1) - th.d),
                            std::abs(helix.rho - th.rho) / std::abs(th.rho),
                            std::abs(state(3) - th.cotTheta),
                            std::abs(state(4) - th.z0)};
    for (int i = 0; i < 5; ++i)
      sum.maxDPar[i] = std::max(sum.maxDPar[i], dpar[i]);

    double xc, yc;
    th.centre(xc, yc);
    constexpr float delta = 0.0100f;  // 100 um

    bool failed = false;
    for (int k = 0; k < nHits; ++k) {
      auto const& h = hitsAll[k];
      auto const res = chc::computeResiduals(helix, h.x, h.y, h.z, h.barrel);
      sum.maxT = std::max(sum.maxT, double(std::abs(res.transverse)));
      sum.maxL = std::max(sum.maxL, double(std::abs(res.longitudinal)));
      ++sum.nHits;

      // displaced copies of the hit
      double dT = 0, dL = 0;
      {
        // radially w.r.t. the true centre (perpendicular to the circle): transverse residual must become +delta
        double const ux = h.x - xc, uy = h.y - yc, un = std::hypot(ux, uy);
        float const xd = h.x + delta * ux / un, yd = h.y + delta * uy / un;
        auto const r = chc::computeResiduals(helix, xd, yd, h.z, h.barrel);
        dT = std::abs((r.transverse - res.transverse) - delta);
        if (h.barrel)  // the closest point on the circle is unchanged, and so is the z residual
          dL = std::abs(r.longitudinal - res.longitudinal);
      }
      if (h.barrel) {
        // along z: longitudinal residual must become +delta, transverse unchanged
        auto const r = chc::computeResiduals(helix, h.x, h.y, h.z + delta, h.barrel);
        dL = std::max(dL, double(std::abs((r.longitudinal - res.longitudinal) - delta)));
        dT = std::max(dT, double(std::abs(r.transverse - res.transverse)));
      } else {
        // along r (radially w.r.t. the origin): longitudinal (r) residual must become +delta
        double const rr = std::hypot(h.x, h.y);
        float const xd = h.x + delta * h.x / rr, yd = h.y + delta * h.y / rr;
        auto const r = chc::computeResiduals(helix, xd, yd, h.z, h.barrel);
        dL = std::max(dL, double(std::abs((r.longitudinal - res.longitudinal) - delta)));
      }
      sum.maxDT = std::max(sum.maxDT, dT);
      sum.maxDL = std::max(sum.maxDL, dL);
      if (std::abs(res.transverse) > 1.e-4f or std::abs(res.longitudinal) > 1.e-4f or dT > 1.e-4 or dL > 1.e-4)
        failed = true;
      if (verbose)
        printf(
            "    hit %d %s %s r=%7.2f z=%8.2f  resT=%+8.3f um resL=%+8.3f um  |dispT-100um|=%.3f um |dispL-100um|=%.3f "
            "um\n",
            k,
            h.barrel ? "B" : "E",
            std::find(idx.begin(), idx.end(), k) != idx.end() ? "fit" : "---",
            std::hypot(h.x, h.y),
            h.z,
            1.e4 * res.transverse,
            1.e4 * res.longitudinal,
            1.e4 * dT,
            1.e4 * dL);
    }
    ++sum.nTracks;
    if (failed) {
      ++sum.nFailed;
      printf("FAILED: %s\n", label);
    }
    if (verbose or failed)
      printf(
          "  %s N=%d: fitted (phi, tip, rho, cotTheta, zip) = (%.6f, %.6f, %.6e, %.6f, %.6f), true = (%.6f, %.6f, "
          "%.6e, "
          "%.6f, %.6f)\n",
          label,
          N,
          state(0),
          state(1),
          helix.rho,
          state(3),
          state(4),
          th.phi,
          th.d,
          th.rho,
          th.cotTheta,
          th.z0);
  }

}  // namespace

int main() {
  auto const platform = Platform{};
  auto const device = alpaka::getDevByIdx(platform, 0);
  Queue queue(device);

  // host SoA for one track, filled by the fit kernel (serial backend: host memory)
  constexpr int nTracks = 1;
  auto const bufSize = reco::TrackSoA::computeDataSize(nTracks);
  std::unique_ptr<std::byte, decltype(std::free)*> buffer{
      reinterpret_cast<std::byte*>(aligned_alloc(reco::TrackSoA::alignment, bufSize)), std::free};
  reco::TrackSoA layout(buffer.get(), nTracks);
  reco::TrackSoAView tracks{layout};

  // radii of barrel-like layers (pixel barrel + first OT barrel layers) and z of endcap-like disks
  std::vector<double> const barrelR = {3.0, 6.1, 10.4, 14.6, 23.0, 35.7, 50.8};
  std::vector<double> const diskZ = {25.0, 32.0, 40.0, 50.0, 63.0, 80.0};

  struct Geo {
    char const* name;
    double cotTheta, tip, zip;
    bool endcap;
  };
  std::vector<Geo> const geos = {{"barrel_ct+0.4", 0.4, 0.02, -3.1, false},
                                 {"barrel_ct-1.1", -1.1, -0.05, 2.4, false},
                                 {"endcap_ct+3.5", 3.5, 0.01, 1.0, true},
                                 {"endcap_ct-4.2", -4.2, -0.03, -2.0, true}};
  std::vector<double> const pts = {0.3, 0.5, 1.0, 10.0, 100.0};

  Summary sum4, sum6;
  int itk = 0;
  for (auto const& g : geos) {
    for (double pt : pts) {
      for (int q : {+1, -1}) {
        TrueHelix th;
        th.phi = std::remainder(0.3 + 1.37 * itk, 2. * M_PI);
        th.d = g.tip;
        th.rho = q * kBField / pt;
        th.cotTheta = g.cotTheta;
        th.z0 = g.zip;
        ++itk;

        std::vector<Hit> hits;
        // two barrel hits first for the endcap tracks, then disks
        for (double r : barrelR) {
          if (g.endcap and hits.size() >= 2)
            break;
          double const s = th.sAtRadius(r);
          if (s < 0)
            break;
          double x, y, z;
          th.at(s, x, y, z);
          hits.push_back({float(x), float(y), float(z), true});
        }
        if (g.endcap) {
          for (double zd : diskZ) {
            double const s = (std::copysign(zd, g.cotTheta) - th.z0) / th.cotTheta;
            if (s > M_PI / std::abs(th.rho))
              break;
            double x, y, z;
            th.at(s, x, y, z);
            hits.push_back({float(x), float(y), float(z), false});
          }
        }
        if (int(hits.size()) > kMaxHits)
          hits.resize(kMaxHits);

        char label[128];
        snprintf(label, sizeof(label), "%s pt=%.1f q=%+d phi=%+.3f nHits=%zu", g.name, pt, q, th.phi, hits.size());
        bool const verbose = (pt == 1.0 and q == -1) or (pt == 0.3 and q == +1 and g.endcap);
        if (verbose)
          printf("%s\n", label);
        testTrack<4>(queue, tracks, th, hits, label, sum4, verbose);
        testTrack<6>(queue, tracks, th, hits, label, sum6, false);
      }
    }
  }

  for (auto const* s : {&sum4, &sum6}) {
    printf("\n%s: %d tracks, %d hits, %d failed\n",
           s == &sum4 ? "fit with 4 hits" : "fit with 6 hits",
           s->nTracks,
           s->nHits,
           s->nFailed);
    printf("  max |residual| exact hits:  transverse %.4f um, longitudinal %.4f um\n", 1.e4 * s->maxT, 1.e4 * s->maxL);
    printf("  max |residual - 100 um| displaced hits: transverse %.4f um, longitudinal %.4f um\n",
           1.e4 * s->maxDT,
           1.e4 * s->maxDL);
    printf("  max |fitted - true|: phi %.2e, tip %.2e cm, rho (relative) %.2e, cotTheta %.2e, zip %.2e cm\n",
           s->maxDPar[0],
           s->maxDPar[1],
           s->maxDPar[2],
           s->maxDPar[3],
           s->maxDPar[4]);
  }

  // straight-line limit and endcap guard of the helpers
  {
    auto const h = chc::makeTrackHelix(0.5f, 0.01f, 0.f, 0.f, 0.f, float(kBField));
    float const x = 10.f * std::cos(0.5f) + 0.01f * std::sin(0.5f), y = 10.f * std::sin(0.5f) - 0.01f * std::cos(0.5f);
    float const t = chc::transverseResidual(h, x, y);
    float const s = chc::transverseArcLength(h, x, y);
    float const l = chc::longitudinalResidualR(h, x, y, 30.f);
    printf("\nrho = 0: transverse residual of a point on the line %.4f um, arc length %.6f cm (expected 10)\n",
           1.e4 * t,
           s);
    printf("cotTheta = 0: endcap residual %g (expected %g)\n", l, chc::kInvalidResidual);
    if (std::abs(t) > 1.e-4f or std::abs(s - 10.f) > 1.e-4f or l != chc::kInvalidResidual)
      ++sum4.nFailed;
  }

  int const nFailed = sum4.nFailed + sum6.nFailed;
  printf("\n%s\n", nFailed == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
  return nFailed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
