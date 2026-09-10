#ifndef RecoTracker_PixelTrackFitting_test_gblTestMaterial_h
#define RecoTracker_PixelTrackFitting_test_gblTestMaterial_h

// Host copy of the Geant4-material-map march that the GBL fit runs on the device, for gblReplay: the host
// fit (interface/BrokenLine.h) carries no material march, and the device march is ALPAKA_FN_ACC code a
// plain host tool cannot call. The functions below transcribe the device ones with
// `alpaka::math::sqrt(acc, x)` replaced by `std::sqrt(x)`; same lattice, samples, weights and accumulation
// order, and testGblReplayDevice asserts the two give the same doubles on every fixture and backend.
// The table itself is not duplicated: blMaterialMapData() is the one compiled-in copy.

#include <cmath>

#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"

namespace gblTestMaterial {

  // X/X0 integrated along the straight segment (r0,z0)->(r1,z1) through the density map.
  // `trapezoid` picks the quadrature weights over the same samples: false gives every sample the full
  // weight dl, true gives the two endpoint samples half weight (the exact segment length integrated).
  // cf. ALPAKA_ACCELERATOR_NAMESPACE::brokenline::segmentXX0.
  inline double segmentXX0(double r0, double z0, double r1, double z1, bool trapezoid = false) {
    const float* rho = blMaterialMap::blMaterialMapData();
    const double L = std::sqrt((r1 - r0) * (r1 - r0) + (z1 - z0) * (z1 - z0));
    int nseg = int(2. * L);
    if (nseg < 2)
      nseg = 2;
    const double dl = L / (nseg - 1);
    double xx0 = 0.;
    for (int k = 0; k < nseg; ++k) {
      const double f = double(k) / (nseg - 1);
      const double w = (trapezoid && (k == 0 || k == nseg - 1)) ? 0.5 * dl : dl;
      xx0 += blMaterialMap::rhoAt(rho, float(r0 + f * (r1 - r0)), float(z0 + f * (z1 - z0))) * w;
    }
    return xx0;
  }

  // Two-equivalent-thin-scatterer split of the material along (r0,z0)->(r1,z1), rectangle weights.
  // Returns W = int q dl, d1 = S2/S1 (path distance of the interior scatterer upstream of the (r1,z1) end)
  // and w1 = S1^2/(S2 W) (its share of the scattering variance).
  // cf. ALPAKA_ACCELERATOR_NAMESPACE::brokenline::segmentXX0Moments.
  inline double segmentXX0Moments(double r0, double z0, double r1, double z1, double& d1, double& w1) {
    const float* rho = blMaterialMap::blMaterialMapData();
    const double L = std::sqrt((r1 - r0) * (r1 - r0) + (z1 - z0) * (z1 - z0));
    int nseg = int(2. * L);
    if (nseg < 2)
      nseg = 2;
    const double dl = L / (nseg - 1);
    double W = 0., S1 = 0., S2 = 0.;
    for (int k = 0; k < nseg; ++k) {
      const double f = double(k) / (nseg - 1);
      const double q = blMaterialMap::rhoAt(rho, float(r0 + f * (r1 - r0)), float(z0 + f * (z1 - z0))) * dl;
      const double d = (1. - f) * L;
      W += q;
      S1 += q * d;
      S2 += q * d * d;
    }
    d1 = 0.;
    w1 = 0.;
    if (W > 0. && S1 > 0. && S2 > 0.) {
      d1 = S2 / S1;
      w1 = S1 * S1 / (S2 * W);
    }
    return W;
  }

  // The same partition with trapezoid weights, the one the kernel calls for the gaps and for the
  // beamline->hit0 segment alike. Returns the same W as segmentXX0(..., trapezoid=true).
  // cf. ALPAKA_ACCELERATOR_NAMESPACE::brokenline::segmentXX0GapSplit.
  inline double segmentXX0GapSplit(double r0, double z0, double r1, double z1, double& d1, double& w1) {
    const float* rho = blMaterialMap::blMaterialMapData();
    const double L = std::sqrt((r1 - r0) * (r1 - r0) + (z1 - z0) * (z1 - z0));
    int nseg = int(2. * L);
    if (nseg < 2)
      nseg = 2;
    const double dl = L / (nseg - 1);
    double W = 0., S1overL = 0., S2overL2 = 0.;
    for (int k = 0; k < nseg; ++k) {
      const double f = double(k) / (nseg - 1);
      const double w = (k == 0 || k == nseg - 1) ? 0.5 * dl : dl;
      const double q = blMaterialMap::rhoAt(rho, float(r0 + f * (r1 - r0)), float(z0 + f * (z1 - z0))) * w;
      W += q;
      S1overL += q * (1. - f);
      S2overL2 += q * (1. - f) * (1. - f);
    }
    d1 = 0.;
    w1 = 0.;
    if (W > 0. && S1overL > 0. && S2overL2 > 0.) {
      d1 = L * S2overL2 / S1overL;
      w1 = (S1overL * S1overL) / (S2overL2 * W);
    }
    return W;
  }

  // The material rows of PreparedGblData<n>.
  template <int N>
  struct MatData {
    double matXX0[N] = {};  // slot g = the WHOLE of gap g->g+1 (slot N-1 is zero)
    double gapD1[N] = {};   // gap g's interior equivalent-scatterer path distance from its arrival hit [cm]
    double gapW1[N] = {};   // gap g's interior equivalent-scatterer share of the variance, in (0,1]
    double innerXX0 = 0.;   // beamline (IP) -> first hit, beam pipe + upstream material
    double innerD1 = 0.;    // the same two-thin split for the upstream segment
    double innerW1 = 0.;
  };

  // The material section of brokenline::prepareGblFitData (alpaka/BrokenLine.h), on the host.
  // hits is any 3xN Eigen-like object indexable as hits(row, col): rows 0,1,2 = x,y,z [cm].
  // The beamline->hit0 term is always integrated, as in the device march, so OT-only stub tracks carry
  // the upstream material too.
  template <int N, typename M3xN>
  inline void fillMatData(const M3xN& hits, MatData<N>& md) {
    auto rOf = [&](int j) {
      return std::sqrt(double(hits(0, j)) * double(hits(0, j)) + double(hits(1, j)) * double(hits(1, j)));
    };
    for (int i = 0; i < N; ++i) {
      md.matXX0[i] = 0.;
      md.gapD1[i] = 0.;
      md.gapW1[i] = 0.;
    }
    for (int g = 0; g + 1 < N; ++g)
      md.matXX0[g] = segmentXX0GapSplit(rOf(g), hits(2, g), rOf(g + 1), hits(2, g + 1), md.gapD1[g], md.gapW1[g]);
    md.innerXX0 = segmentXX0(0., 0., rOf(0), hits(2, 0), /*trapezoid=*/true);
    md.innerD1 = 0.;
    md.innerW1 = 0.;
    if (md.innerXX0 > 0.)
      segmentXX0GapSplit(0., 0., rOf(0), hits(2, 0), md.innerD1, md.innerW1);
  }

}  // namespace gblTestMaterial

#endif
