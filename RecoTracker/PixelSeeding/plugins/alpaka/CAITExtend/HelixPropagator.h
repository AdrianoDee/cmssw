#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAITExtend_HelixPropagator_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAITExtend_HelixPropagator_h

#include <cmath>

#include <alpaka/alpaka.hpp>

#include <Eigen/Core>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// Self-contained device-side helix propagation used by the IT inward-extension
// stage. Input is the standard 5-parameter helix at the beam spot:
//   state = (phi0, d0, kappa, cotTheta, z0)
// where kappa = state(2) carries the sign of the track charge and equals
// (1/pT) * (0.003 * B[T]).  See DataFormats/TrackSoA/interface/TracksSoA.h.
//
// Variance propagation is the linearised (leading-order) form: enough to size
// search windows but not a substitute for a refit. Multiple-scattering is added
// with a single uniform material density per radial gap, matching the convention
// used by the Broken-line fitter.

namespace caITExtend {

  struct HelixCrossing {
    float phi;        // predicted azimuth at the target layer
    float z;          // predicted z
    float r;          // predicted r (== target for barrel, derived for disk)
    float sigma2Phi;  // variance of phi prediction (including MS)
    float sigma2Z;    // variance of z prediction (including MS)
    bool valid;       // false if target unreachable (e.g. low-pT looper)
  };

  // Linearised multiple-scattering contribution per unit path length.
  // theta_MS^2 ~ (0.0136/(beta*p))^2 * (deltaR/X0).  With a single uniform
  // material density per cm (call it `density`, dimensionless x/X0 per cm),
  // we have theta_MS^2 = (0.0136*kappa/(0.003*B))^2 * deltaR * density.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float msAngle2(float kappa, float bField, float deltaR, float density) {
    const float oneOverP = std::abs(kappa) / (0.003f * bField);  // 1/pT, ignoring 1/sin(theta) factor here
    const float k = 0.0136f * oneOverP;
    return k * k * deltaR * density;
  }

  // Helix crossing of a cylinder at radius targetR.  Returns valid = false if
  // the helix turns around before reaching targetR (low-pT loopers, or |d0|+1/kappa < targetR).
  ALPAKA_FN_ACC ALPAKA_FN_INLINE HelixCrossing helixCrossingToR(
      const reco::Vector5f& state, const reco::Vector15f& cov, float targetR, float bField, float materialDensity) {
    HelixCrossing c{};
    c.valid = false;

    const float phi0 = state(0);
    const float d0 = state(1);
    const float kappa = state(2);
    const float cotTheta = state(3);
    const float z0 = state(4);

    // Special-case very high pT (negligible curvature): straight-line approx.
    // The 1e-7 cutoff corresponds to pT >> 1 TeV in any realistic field.
    if (std::abs(kappa) < 1.e-7f) {
      // Straight line of impact parameter d0 in direction phi0 crossing R requires |d0| < R.
      if (std::abs(d0) >= targetR)
        return c;
      const float ds = std::sqrt(targetR * targetR - d0 * d0);  // path length along xy
      // x(t) = -d0 sin phi0 + t cos phi0,  y(t) = d0 cos phi0 + t sin phi0
      // at t = ds, |xy| = R and the azimuth is phi0 + atan(d0/ds)
      c.phi = phi0 + std::atan2(d0, ds);
      c.r = targetR;
      c.z = z0 + cotTheta * ds;
      // Linearised variance:
      //   sigma2Phi ~ cov_phi_phi + (1/R)^2 cov_d0_d0  (leading term)
      //   sigma2Z   ~ cov_z0_z0 + ds^2 cov_cot_cot
      const float invR = 1.f / targetR;
      c.sigma2Phi = cov(0) + invR * invR * cov(5);
      c.sigma2Z = cov(14) + ds * ds * cov(12);
      // Multiple-scattering: deltaR taken as (targetR - |d0|) -- a positive radial gap.
      const float deltaR = targetR - std::abs(d0);
      const float msPhi = msAngle2(kappa, bField, deltaR, materialDensity);
      const float sinTheta2 = 1.f / (1.f + cotTheta * cotTheta);
      c.sigma2Phi += msPhi / sinTheta2;
      c.sigma2Z += msPhi * cotTheta * cotTheta / sinTheta2;
      c.valid = true;
      return c;
    }

    const float rho = 1.f / kappa;       // signed helix radius
    const float rc = d0 + rho;           // signed center-to-IP distance, sign of rho
    const float rcAbs = std::abs(rc);

    // Reachability: need targetR <= |rc| + |rho|  AND  targetR >= ||rc| - |rho|| (annulus).
    const float rhoAbs = std::abs(rho);
    if (targetR > rcAbs + rhoAbs)
      return c;
    if (targetR + rhoAbs < rcAbs)
      return c;

    // cos(alpha - phi0) = (rc^2 + rho^2 - R^2) / (2 rc rho)
    const float cosArg = (rc * rc + rho * rho - targetR * targetR) / (2.f * rc * rho);
    // Clamp against round-off
    const float cosArgClamped = std::fmin(1.f, std::fmax(-1.f, cosArg));
    const float dAlpha = std::acos(cosArgClamped);  // [0, pi]; choose the inward branch

    // The PCA is at phi0 (azimuth angle on the helix circle). The "inward" crossing
    // (smaller arc length) is the one with smaller |dAlpha|. acos already returns
    // the principal value in [0, pi], which corresponds to the first crossing of R
    // moving outward from the PCA -- this is what we want for IT extension (the
    // track goes outward from PCA to the IT layer at R).
    // Signed arc angle: for kappa > 0 the helix bends counter-clockwise (sign of
    // dPhi same as sign of rho).
    const float signRho = (rho >= 0.f) ? 1.f : -1.f;
    const float dPhi = signRho * dAlpha;

    c.phi = phi0 + dPhi;
    c.r = targetR;

    const float arcLength = rho * dPhi;  // signed arc length in xy
    c.z = z0 + cotTheta * arcLength;

    // ----- variance propagation (leading order, see header doc) ----------
    // partial(phi)/partial(phi0) = 1
    // partial(phi)/partial(d0)   ~ -sin(dAlpha) / (rho * sin(dAlpha)) -- regularise by ||
    //                              For small dAlpha: ~ 1/sqrt(2 R^2 - ...) approx 1/R
    // partial(phi)/partial(kappa) -- non-trivial; bounded by R/2 for small kappaR
    // Use the small-curvature leading-order forms for the variance only; the
    // floors clamp under-estimation:
    const float invR = 1.f / targetR;
    const float halfR = 0.5f * targetR;
    // dphi/dd0 ~ 1/sqrt(R^2 - d0^2 - ...) -> approximate by 1/R for IT radii
    // dphi/dkappa ~ R/2 for small kappa*R
    c.sigma2Phi = cov(0)                                   // sigma2(phi0)
                  + invR * invR * cov(5)                   // (dphi/dd0)^2 sigma2(d0)
                  + halfR * halfR * cov(9)                 // (dphi/dkappa)^2 sigma2(kappa)
                  + 2.f * invR * cov(1)                    // cross phi0,d0
                  + 2.f * halfR * cov(2)                   // cross phi0,kappa
                  + 2.f * invR * halfR * cov(6);           // cross d0,kappa

    // dz/dcotTheta ~ arcLength
    // dz/dphi ~ small (longitudinal coupling)
    // dz/dz0 = 1
    c.sigma2Z = cov(14) + arcLength * arcLength * cov(12) + 2.f * arcLength * cov(13);

    // Multiple-scattering term, single-density convention.
    const float deltaR = targetR - std::abs(d0);  // signed radial gap; clamp to >=0
    const float dR = (deltaR > 0.f) ? deltaR : 0.f;
    const float msPhi = msAngle2(kappa, bField, dR, materialDensity);
    const float sinTheta2 = 1.f / (1.f + cotTheta * cotTheta);
    c.sigma2Phi += msPhi / sinTheta2;
    c.sigma2Z += msPhi * cotTheta * cotTheta / sinTheta2;

    c.valid = true;
    return c;
  }

  // Helix crossing of a disk plane at z = targetZ.  Returns valid = false if
  // the helix's cotTheta has wrong sign for reaching targetZ from z0.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE HelixCrossing helixCrossingToZ(
      const reco::Vector5f& state, const reco::Vector15f& cov, float targetZ, float bField, float materialDensity) {
    HelixCrossing c{};
    c.valid = false;

    const float phi0 = state(0);
    const float d0 = state(1);
    const float kappa = state(2);
    const float cotTheta = state(3);
    const float z0 = state(4);

    // Sign check: (targetZ - z0) must have the same sign as cotTheta for
    // forward-only propagation. (We don't follow backward-going re-crossings.)
    const float dZ = targetZ - z0;
    if ((dZ * cotTheta) <= 0.f)
      return c;

    // Arc length along the helix to reach z = targetZ:
    //   arcLength = dZ / cotTheta
    const float arcLength = dZ / cotTheta;
    // Azimuth advance: dPhi = arcLength * kappa  (signed)
    const float dPhi = arcLength * kappa;

    // For straight-line / small-curvature: rExpected ~ sqrt(d0^2 + arcLength^2).
    // For curved: use the circle parameterisation.
    float rPred;
    if (std::abs(kappa) < 1.e-7f) {
      rPred = std::sqrt(d0 * d0 + arcLength * arcLength);
    } else {
      const float rho = 1.f / kappa;
      const float rc = d0 + rho;
      // Position on circle at angle dPhi from PCA:
      //   x = -rc sin(phi0+dPhi) + rho sin(phi0+dPhi) ... -- use the geometric identity
      // |xy|^2 = rc^2 + rho^2 - 2 rc rho cos(dPhi)
      const float r2 = rc * rc + rho * rho - 2.f * rc * rho * std::cos(dPhi);
      rPred = std::sqrt(std::max(0.f, r2));
    }

    c.phi = phi0 + dPhi;
    c.z = targetZ;
    c.r = rPred;

    // Variance propagation (leading order):
    // sigma2(phi at z=zT) = sigma2(phi0) + arcLength^2 * (sigma2(kappa) via dPhi/dkappa)
    //                      + (dArcLength/dCotTheta = -arcLength/cotTheta)^2 * sigma2(cot)
    const float dArcDCot = -arcLength / cotTheta;
    c.sigma2Phi = cov(0) + arcLength * arcLength * cov(9) + 2.f * arcLength * cov(2)
                  + kappa * dArcDCot * kappa * dArcDCot * cov(12);

    // sigma2(r at z=zT): dominated by dphi propagation through |xy| variation.
    // For window sizing on disks we use sigma2Z := this is r-window variance:
    // sigma2_r = (dRpred/dd0)^2 cov(d0) + (dRpred/dCotTheta)^2 cov(cot) + ...
    // Leading order: dRpred/dd0 ~ d0 / rPred (small), dRpred/dCot ~ arcLength * dArcDCot / rPred.
    const float rSafe = (rPred > 1.e-3f) ? rPred : 1.e-3f;
    const float dRdd0 = d0 / rSafe;
    const float dRdcot = (arcLength * dArcDCot) / rSafe;
    c.sigma2Z = dRdd0 * dRdd0 * cov(5) + dRdcot * dRdcot * cov(12) + 2.f * dRdd0 * dRdcot * cov(7);

    // Multiple-scattering: scale by approximate radial path d|xy|.
    const float dR = std::max(0.f, rPred - std::abs(d0));
    const float msPhi = msAngle2(kappa, bField, dR, materialDensity);
    const float sinTheta2 = 1.f / (1.f + cotTheta * cotTheta);
    c.sigma2Phi += msPhi / sinTheta2;
    c.sigma2Z += msPhi * cotTheta * cotTheta / sinTheta2;

    c.valid = true;
    return c;
  }

}  // namespace caITExtend

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAITExtend_HelixPropagator_h
