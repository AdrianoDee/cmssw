#ifndef RecoTracker_PixelSeeding_interface_CAMaskingUtilities_h
#define RecoTracker_PixelSeeding_interface_CAMaskingUtilities_h

// Hit-to-track compatibility used by CAMaskSoAProducer to decide which hits of an accepted track are masked
// for the next CA iteration (the analogue of the chi2 cut of the legacy cluster removal).
//
// Plain float math, usable both on host and in device code: the same functions can be called by a host-side
// analyzer to recompute the residuals, e.g. to tune the cuts on truth distributions.
//
// Conventions (taken from the CA fit, see RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h,
// RecoTracker/PixelTrackFitting/interface/alpaka/FitUtils.h and reco::copyFromCircle in
// DataFormats/TrackSoA/interface/TracksSoA.h):
//   - the track state is {phi, tip, state2, cotTheta, zip}, given at the point of closest approach (PCA) to the
//     origin of the frame of the hits (the beam spot for the Alpaka CA);
//   - (phi, tip, rho) are the Karimaki circle parameters: the PCA is tip * (sin(phi), -cos(phi)), the direction of
//     motion at the PCA is (cos(phi), sin(phi)) and the centre of the circle is (1/rho + tip) * (sin(phi), -cos(phi));
//     rho > 0 means clockwise rotation seen from +z (positive charge for Bz > 0);
//   - state2 = rho / B, with B = 1 / inverseBzAtOriginInGeV(), i.e. state2 = q/pT;
//   - z along the track is z(s) = zip + cotTheta * s, with s the transverse arc length from the PCA.

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

namespace caHitCompatibility {

  // Value returned for a residual that cannot be computed (e.g. an endcap hit for a track with cotTheta = 0):
  // larger than any sensible cut, so that the hit is considered not compatible.
  constexpr float kInvalidResidual = 1.e9f;

  // Hit categories with separate cuts.
  enum HitRegion : uint8_t { kPixelBarrel = 0, kPixelEndcap = 1, kOuterTracker = 2, kNumHitRegions = 3 };

  // Track helix at the PCA.
  struct TrackHelix {
    float sinPhi;  // (cos(phi), sin(phi)) is the direction of motion at the PCA
    float cosPhi;
    float tip;  // signed transverse impact parameter [cm]
    float rho;  // signed curvature [1/cm], > 0 for clockwise rotation seen from +z
    float cotTheta;
    float zip;  // longitudinal impact parameter [cm]
  };

  // Residuals of a hit with respect to the helix [cm].
  struct HitResiduals {
    float transverse;    // signed distance from the circle in the transverse plane, > 0 outside the circle
    float longitudinal;  // z_hit - z_track(s_hit) for barrel-like hits, r_hit - r_track(z_hit) for endcap-like hits
  };

  // Maximum absolute residuals [cm] for each HitRegion; a cut <= 0 is not applied.
  struct HitResidualCuts {
    float maxTransverse[kNumHitRegions];
    float maxLongitudinal[kNumHitRegions];
  };

  // Build the helix from the TracksSoA state {phi, tip, state2, cotTheta, zip} and the field B in the units of the
  // CA, i.e. B = 1 / MagneticField::inverseBzAtOriginInGeV().
  ALPAKA_FN_HOST_ACC inline TrackHelix makeTrackHelix(
      float phi, float tip, float state2, float cotTheta, float zip, float bField) {
    TrackHelix helix;
    helix.sinPhi = std::sin(phi);
    helix.cosPhi = std::cos(phi);
    helix.tip = tip;
    helix.rho = state2 * bField;
    helix.cotTheta = cotTheta;
    helix.zip = zip;
    return helix;
  }

  // Signed distance of the point (x, y) from the circle, > 0 outside the circle. Numerically stable for rho -> 0,
  // where it becomes the signed distance from the straight line (Karimaki, NIM A305 (1991) 187).
  ALPAKA_FN_HOST_ACC inline float transverseResidual(TrackHelix const& helix, float x, float y) {
    // coordinates of the point w.r.t. the PCA: eps is minus the component along the direction towards the centre
    // (for rho > 0), b is the component along the direction of motion
    float const eps = helix.tip - (x * helix.sinPhi - y * helix.cosPhi);
    float const b = x * helix.cosPhi + y * helix.sinPhi;
    float const a = 2.f * eps + helix.rho * (eps * eps + b * b);
    // 1 + rho * a = (rho * |P - C|)^2 >= 0, protect against rounding
    float const u = std::sqrt(std::max(0.f, 1.f + helix.rho * a));
    float const d = a / (1.f + u);
    return helix.rho < 0.f ? -d : d;
  }

  // Signed transverse arc length from the PCA to the point of the circle closest to (x, y); positive in the
  // direction of motion. Valid up to half a turn.
  ALPAKA_FN_HOST_ACC inline float transverseArcLength(TrackHelix const& helix, float x, float y) {
    float const eps = helix.tip - (x * helix.sinPhi - y * helix.cosPhi);
    float const b = x * helix.cosPhi + y * helix.sinPhi;
    float const absRho = std::abs(helix.rho);
    if (absRho < 1.e-9f)
      return b;
    // turning angle around the centre, in [0, pi]
    float const alpha = std::atan2(absRho * std::abs(b), 1.f + helix.rho * eps);
    return std::copysign(alpha / absRho, b);
  }

  // Position in the transverse plane after the arc length s from the PCA.
  ALPAKA_FN_HOST_ACC inline void transversePosition(TrackHelix const& helix, float s, float& x, float& y) {
    // along the direction of motion: sin(rho s) / rho; towards the centre: (1 - cos(rho s)) / rho
    float along = s;
    float across = 0.f;
    if (std::abs(helix.rho) >= 1.e-9f) {
      float const sh = std::sin(0.5f * helix.rho * s);
      along = std::sin(helix.rho * s) / helix.rho;
      across = 2.f * sh * sh / helix.rho;
    }
    x = (helix.tip + across) * helix.sinPhi + along * helix.cosPhi;
    y = -(helix.tip + across) * helix.cosPhi + along * helix.sinPhi;
  }

  // z_hit - z_track at the transverse arc length of the hit (barrel-like hits).
  ALPAKA_FN_HOST_ACC inline float longitudinalResidualZ(TrackHelix const& helix, float x, float y, float z) {
    float const s = transverseArcLength(helix, x, y);
    return z - (helix.zip + helix.cotTheta * s);
  }

  // r_hit - r_track at the z of the hit (endcap-like hits).
  ALPAKA_FN_HOST_ACC inline float longitudinalResidualR(TrackHelix const& helix, float x, float y, float z) {
    if (std::abs(helix.cotTheta) < 1.e-6f)
      return kInvalidResidual;
    float const s = (z - helix.zip) / helix.cotTheta;
    float xt, yt;
    transversePosition(helix, s, xt, yt);
    return std::sqrt(x * x + y * y) - std::sqrt(xt * xt + yt * yt);
  }

  ALPAKA_FN_HOST_ACC inline HitResiduals computeResiduals(
      TrackHelix const& helix, float x, float y, float z, bool barrelLike) {
    HitResiduals res;
    res.transverse = transverseResidual(helix, x, y);
    res.longitudinal = barrelLike ? longitudinalResidualZ(helix, x, y, z) : longitudinalResidualR(helix, x, y, z);
    return res;
  }

  // Region of a hit in the global hit numbering of the CA (pixel hits first, then outer-tracker hits).
  //   - globalIndex >= nPixelHits: outer tracker (the Phase-2 CA uses only OT barrel layers);
  //   - pixel hits: barrel if detectorIndex < nPixelBarrelModules (pixel modules are numbered in the order of
  //     TrackerGeometry::dets(), barrel first).
  ALPAKA_FN_HOST_ACC inline HitRegion hitRegion(uint32_t globalIndex,
                                                uint32_t nPixelHits,
                                                uint32_t detectorIndex,
                                                uint32_t nPixelBarrelModules) {
    if (globalIndex >= nPixelHits)
      return kOuterTracker;
    return detectorIndex < nPixelBarrelModules ? kPixelBarrel : kPixelEndcap;
  }

  // The longitudinal residual is computed in z for barrel-like regions, in r for the pixel endcap.
  ALPAKA_FN_HOST_ACC inline bool isBarrelLike(HitRegion region) { return region != kPixelEndcap; }

  ALPAKA_FN_HOST_ACC inline bool isCompatible(HitResidualCuts const& cuts, HitRegion region, HitResiduals const& res) {
    float const maxT = cuts.maxTransverse[region];
    float const maxL = cuts.maxLongitudinal[region];
    bool const okT = maxT <= 0.f or std::abs(res.transverse) < maxT;
    bool const okL = maxL <= 0.f or std::abs(res.longitudinal) < maxL;
    return okT and okL;
  }

}  // namespace caHitCompatibility

#endif  // RecoTracker_PixelSeeding_interface_CAMaskingUtilities_h
