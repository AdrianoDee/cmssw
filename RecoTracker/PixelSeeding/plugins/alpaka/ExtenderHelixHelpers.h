#ifndef RecoTracker_PixelSeeding_plugins_alpaka_ExtenderHelixHelpers_h
#define RecoTracker_PixelSeeding_plugins_alpaka_ExtenderHelixHelpers_h

// Helix-propagation helpers for the OT hit-attach walk (sole consumer: CAExtension.dev.cc).
// Conventions match BrokenLine.h so arc-length signs / PCA position / curvature direction agree.

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  struct HelixState {
    float phi0;
    float tip;
    float cotTheta;
    float zip;
    float rho;
    float xc, yc;
    float alphaOrigin;
  };

  // Build a HelixState from raw BL state parameters (phi0, tip, invPt, cotTheta, zip).
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE HelixState
  makeHelixStateFromParams(TAcc const& acc, float phi0, float tip, float invPt, float cotTheta, float zip, float bf) {
    HelixState h;
    h.phi0 = phi0;
    h.tip = tip;
    h.cotTheta = cotTheta;
    h.zip = zip;
    h.rho = (invPt != 0.f) ? 1.f / (invPt * bf) : 1e9f;
    const float sp = alpaka::math::sin(acc, h.phi0);
    const float cp = alpaka::math::cos(acc, h.phi0);
    const float x0 = tip * sp;
    const float y0 = -tip * cp;
    h.xc = x0 + h.rho * sp;
    h.yc = y0 - h.rho * cp;
    h.alphaOrigin = alpaka::math::atan2(acc, -h.yc, -h.xc);
    return h;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE HelixState
  makeHelixState(TAcc const& acc, const ::reco::TrackSoAConstView tracks, int i, float bf) {
    return makeHelixStateFromParams(acc,
                                    tracks[i].state()(0),
                                    tracks[i].state()(1),
                                    tracks[i].state()(2),
                                    tracks[i].state()(3),
                                    tracks[i].state()(4),
                                    bf);
  }

  struct Prediction {
    float phi;        // azimuth of the intersection (rad in [-pi, pi])
    float secondary;  // z for barrel prediction, r for endcap prediction
    float arcS;       // signed forward arc length from origin direction
    bool valid;
  };

  static constexpr float kExtenderPi = 3.14159265f;
  static constexpr float kExtenderMaxArcLengthCm = 250.f;

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float foldPi(float a) {
    if (a > kExtenderPi)
      a -= 2.f * kExtenderPi;
    if (a < -kExtenderPi)
      a += 2.f * kExtenderPi;
    return a;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE Prediction predictOnBarrel(TAcc const& acc, HelixState const& h, float R) {
    Prediction out{0.f, 0.f, 0.f, false};
    const float dc2 = h.xc * h.xc + h.yc * h.yc;
    const float absRho = alpaka::math::abs(acc, h.rho);
    const float dc = alpaka::math::sqrt(acc, dc2);
    if (R < dc - absRho - 1e-3f || R > dc + absRho + 1e-3f)
      return out;
    const float K = R * R + dc2 - h.rho * h.rho;
    float c = K / (2.f * dc * R);
    if (c > 1.f)
      c = 1.f;
    if (c < -1.f)
      c = -1.f;
    const float phi_c = alpaka::math::atan2(acc, h.yc, h.xc);
    const float dphi = alpaka::math::acos(acc, c);
    const float thetaA = phi_c + dphi;
    const float thetaB = phi_c - dphi;
    auto arcAt = [&](float theta) {
      const float x = R * alpaka::math::cos(acc, theta);
      const float y = R * alpaka::math::sin(acc, theta);
      const float alphaH = alpaka::math::atan2(acc, y - h.yc, x - h.xc);
      return h.rho * foldPi(h.alphaOrigin - alphaH);
    };
    const float sA = arcAt(thetaA);
    const float sB = arcAt(thetaB);
    float chosenS, chosenTheta;
    const bool aOk = sA > 0.f;
    const bool bOk = sB > 0.f;
    if (aOk && bOk) {
      if (sA <= sB) {
        chosenS = sA;
        chosenTheta = thetaA;
      } else {
        chosenS = sB;
        chosenTheta = thetaB;
      }
    } else if (aOk) {
      chosenS = sA;
      chosenTheta = thetaA;
    } else if (bOk) {
      chosenS = sB;
      chosenTheta = thetaB;
    } else {
      return out;
    }
    if (chosenS > kExtenderMaxArcLengthCm)
      return out;
    out.phi = foldPi(chosenTheta);
    out.secondary = h.zip + chosenS * h.cotTheta;
    out.arcS = chosenS;
    out.valid = true;
    return out;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE Prediction predictOnEndcap(TAcc const& acc, HelixState const& h, float zLayer) {
    Prediction out{0.f, 0.f, 0.f, false};
    if (alpaka::math::abs(acc, h.cotTheta) < 1e-4f)
      return out;
    const float arcS = (zLayer - h.zip) / h.cotTheta;
    if (arcS <= 0.f || arcS > kExtenderMaxArcLengthCm)
      return out;
    const float alphaH = h.alphaOrigin - arcS / h.rho;
    const float absRho = alpaka::math::abs(acc, h.rho);
    const float x = h.xc + absRho * alpaka::math::cos(acc, alphaH);
    const float y = h.yc + absRho * alpaka::math::sin(acc, alphaH);
    const float r = alpaka::math::sqrt(acc, x * x + y * y);
    out.phi = alpaka::math::atan2(acc, y, x);
    out.secondary = r;
    out.arcS = arcS;
    out.valid = true;
    return out;
  }

  // dphi/dr of the fitted helix at the layer surface (the track side of the stub-bend row), the
  // closed-form derivative of the crossing solve in predictOnBarrel/predictOnEndcap, kept separate to
  // preserve that path's floating-point contraction. Returns 0 when the crossing is degenerate.
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float extBendPredDPhiDr(TAcc const& acc,
                                                         HelixState const& h,
                                                         bool isBarrel,
                                                         float surf) {
    constexpr float kEps = 1.0e-6f;
    const Prediction p = isBarrel ? predictOnBarrel(acc, h, surf) : predictOnEndcap(acc, h, surf);
    if (!p.valid)
      return 0.f;
    const float absRho = alpaka::math::abs(acc, h.rho);
    if (isBarrel) {
      const float dc2 = h.xc * h.xc + h.yc * h.yc;
      const float dc = alpaka::math::sqrt(acc, dc2);
      if (!(dc > kEps) || !(surf > kEps))
        return 0.f;
      const float Aa = dc2 - h.rho * h.rho;
      float c0 = (surf * surf + Aa) / (2.f * dc * surf);
      c0 = alpaka::math::min(acc, 1.f, alpaka::math::max(acc, -1.f, c0));
      const float sq2 = 1.f - c0 * c0;
      if (!(sq2 > 1.0e-8f))
        return 0.f;
      const float sq = alpaka::math::sqrt(acc, sq2);
      const float cp1 = 1.f / (2.f * dc) - Aa / (2.f * dc * surf * surf);
      const float phic = alpaka::math::atan2(acc, h.yc, h.xc);
      const float sBranch = (foldPi(p.phi - phic) >= 0.f) ? 1.f : -1.f;
      return -sBranch * cp1 / sq;  // dphi/dR
    }
    const float cot = h.cotTheta;
    if (!(alpaka::math::abs(acc, cot) > 1.0e-4f) || !(absRho > kEps))
      return 0.f;
    const float a1 = -1.f / (h.rho * cot);  // dalphaH/dz, constant in z
    const float alphaH = h.alphaOrigin - p.arcS / h.rho;
    const float sA = alpaka::math::sin(acc, alphaH);
    const float cA = alpaka::math::cos(acc, alphaH);
    const float x = h.xc + absRho * cA;
    const float y = h.yc + absRho * sA;
    const float r2 = x * x + y * y;
    if (!(r2 > 1.0e-6f))
      return 0.f;
    const float rr = alpaka::math::sqrt(acc, r2);
    const float xp = -absRho * sA * a1;
    const float yp = absRho * cA * a1;
    const float dphidz = (x * yp - y * xp) / r2;
    const float drdz = (x * xp + y * yp) / rr;
    return (alpaka::math::abs(acc, drdz) > 1.0e-9f) ? dphidz / drdz : 0.f;
  }

  // Analytic state Jacobian of the bend prediction, H_b = d(dphi/dr)/d(d0, 1/pT, cot, z0). The phi0
  // partial is identically zero by rotation invariance and is not returned; `g` is the value
  // extBendPredDPhiDr returned, passed in to carry the branch sign. Returns false if degenerate.
  //
  // rho = 1/(invPt*bf), T = d0 + rho, A = |T| = |centre|, P = |rho|, rho_i = drho/d(invPt) =
  // -rho^2 bf, dA/dd0 = sgn(T), dA/d(invPt) = sgn(T) rho_i.
  // Barrel (radius R), with Aa = A^2 - rho^2, c = (R^2 + Aa)/(2 A R), s = sqrt(1 - c^2),
  // c' = 1/(2A) - Aa/(2 A R^2) and g = -sigma c'/s, differentiating ln|g| = ln|c'| - ln s:
  //     dg/du = g ( (dc'/du)/c' + c (dc/du)/s^2 ),                                    (1)
  //     dc/du = (dAa/du)/(2 A R) - c (dA/du)/A,   dc'/du = -(dA/du / A) c' - (dAa/du)/(2 A R^2),
  //     dAa = 2 A dA - 2 rho drho.  A, Aa and R do not depend on cot or z0, hence
  //     dH_b/dcot == dH_b/dz0 == 0 exactly.                                            (2)
  // Endcap (surface z), with arc = (z - z0)/cot, beta = arc/rho, N = P - A cos(beta),
  // D = -A sin(beta), r^2 = A^2 + P^2 - 2 A P cos(beta) and
  //     g = dphi/dr = N / (D r),                                                        (3)
  //     dN = dP - cos(beta) dA + A sin(beta) dbeta,   dD = -sin(beta) dA - A cos(beta) dbeta,
  //     dr = [A dA + P dP - cos(beta)(A dP + P dA) + A P sin(beta) dbeta] / r,
  //     dg/du = g ( dN/N - dD/D - dr/r ),                                              (4)
  //     dP/d(invPt) = sgn(rho) rho_i, dbeta/d(invPt) = arc bf, dbeta/dcot = -beta/cot,
  //     dbeta/dz0 = -1/(rho cot).
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool extBendPredDPhiDrGrad(TAcc const& acc,
                                                            HelixState const& h,
                                                            bool isBarrel,
                                                            float surf,
                                                            float bf,
                                                            float g,
                                                            float& hTip,
                                                            float& hInvPt,
                                                            float& hCot,
                                                            float& hZip) {
    constexpr float kEpsG = 1.0e-6f;
    hTip = 0.f;
    hInvPt = 0.f;
    hCot = 0.f;
    hZip = 0.f;
    if (g == 0.f || !alpaka::math::isfinite(acc, g))
      return false;
    const float rho = h.rho;
    const float T = h.tip + rho;
    const float sT = (T >= 0.f) ? 1.f : -1.f;
    const float A = alpaka::math::sqrt(acc, h.xc * h.xc + h.yc * h.yc);
    if (!(A > kEpsG))
      return false;
    const float rhoI = -rho * rho * bf;  // d rho / d(1/pT)
    const float dATip = sT;
    const float dAPt = sT * rhoI;
    if (isBarrel) {
      const float R = surf;
      if (!(R > kEpsG))
        return false;
      const float Aa = A * A - rho * rho;
      float c = (R * R + Aa) / (2.f * A * R);
      c = alpaka::math::min(acc, 1.f, alpaka::math::max(acc, -1.f, c));
      const float s2 = 1.f - c * c;
      if (!(s2 > 1.0e-8f))
        return false;
      const float cp = 1.f / (2.f * A) - Aa / (2.f * A * R * R);
      if (!(alpaka::math::abs(acc, cp) > 0.f))
        return false;
      // eq (1), once per non-trivial parameter. drho is zero for d0 and rhoI for 1/pT.
      const float dAaTip = 2.f * A * dATip;
      const float dcTip = dAaTip / (2.f * A * R) - c * dATip / A;
      const float dcpTip = -(dATip / A) * cp - dAaTip / (2.f * A * R * R);
      hTip = g * (dcpTip / cp + c * dcTip / s2);
      const float dAaPt = 2.f * A * dAPt - 2.f * rho * rhoI;
      const float dcPt = dAaPt / (2.f * A * R) - c * dAPt / A;
      const float dcpPt = -(dAPt / A) * cp - dAaPt / (2.f * A * R * R);
      hInvPt = g * (dcpPt / cp + c * dcPt / s2);
      return true;  // eq (2): the cot and z0 partials stay at the zero they were set to
    }
    const float cot = h.cotTheta;
    if (!(alpaka::math::abs(acc, cot) > 1.0e-4f))
      return false;
    const float P = alpaka::math::abs(acc, rho);
    if (!(P > kEpsG))
      return false;
    const float arc = (surf - h.zip) / cot;
    const float beta = arc / rho;
    const float cb = alpaka::math::cos(acc, beta);
    const float sb = alpaka::math::sin(acc, beta);
    const float Nn = P - A * cb;
    const float Dd = -A * sb;
    const float r2 = A * A + P * P - 2.f * A * P * cb;
    if (!(r2 > 1.0e-6f) || Nn == 0.f || Dd == 0.f)
      return false;
    const float r = alpaka::math::sqrt(acc, r2);
    const float sRho = (rho >= 0.f) ? 1.f : -1.f;
    // eq (4), evaluated four times with the (dA, dP, dbeta) triple of each parameter.
    const float invN = 1.f / Nn, invD = 1.f / Dd, invR = 1.f / r;
    auto row = [&](float dA, float dP, float dB) {
      const float dN = dP - cb * dA + A * sb * dB;
      const float dD = -sb * dA - A * cb * dB;
      const float dr = (A * dA + P * dP - cb * (A * dP + P * dA) + A * P * sb * dB) * invR;
      return g * (dN * invN - dD * invD - dr * invR);
    };
    hTip = row(dATip, 0.f, 0.f);
    hInvPt = row(dAPt, sRho * rhoI, arc * bf);
    hCot = row(0.f, 0.f, -beta / cot);
    hZip = row(0.f, 0.f, -1.f / (rho * cot));
    return true;
  }

  // VecDual: forward-mode AD carrying a value plus a 5-component gradient in one pass, so the endcap
  // r(params) evaluates each transcendental once; a finite-difference gradient would amplify
  // ULP-level roundoff of the crossing solve into visible chi2 shifts.

  struct VecDual {
    float v;     // value
    float d[5];  // partials w.r.t. {phi0, tip, invPt, cotTheta, zip}
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator+(VecDual a, VecDual b) {
    VecDual r;
    r.v = a.v + b.v;
    for (int k = 0; k < 5; ++k)
      r.d[k] = a.d[k] + b.d[k];
    return r;
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator-(VecDual a, VecDual b) {
    VecDual r;
    r.v = a.v - b.v;
    for (int k = 0; k < 5; ++k)
      r.d[k] = a.d[k] - b.d[k];
    return r;
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator-(VecDual a, float b) {
    VecDual r;
    r.v = a.v - b;
    for (int k = 0; k < 5; ++k)
      r.d[k] = a.d[k];
    return r;
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator-(float a, VecDual b) {
    VecDual r;
    r.v = a - b.v;
    for (int k = 0; k < 5; ++k)
      r.d[k] = -b.d[k];
    return r;
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator-(VecDual a) {  // unary minus
    VecDual r;
    r.v = -a.v;
    for (int k = 0; k < 5; ++k)
      r.d[k] = -a.d[k];
    return r;
  }
  // d(a*b) = a.d*b.v + a.v*b.d
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator*(VecDual a, VecDual b) {
    VecDual r;
    r.v = a.v * b.v;
    for (int k = 0; k < 5; ++k)
      r.d[k] = a.d[k] * b.v + a.v * b.d[k];
    return r;
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator*(VecDual a, float b) {
    VecDual r;
    r.v = a.v * b;
    for (int k = 0; k < 5; ++k)
      r.d[k] = a.d[k] * b;
    return r;
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator*(float a, VecDual b) {
    VecDual r;
    r.v = a * b.v;
    for (int k = 0; k < 5; ++k)
      r.d[k] = a * b.d[k];
    return r;
  }
  // d(a/b) = (a.d*b.v - a.v*b.d) / (b.v*b.v)
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator/(VecDual a, VecDual b) {
    VecDual r;
    r.v = a.v / b.v;
    for (int k = 0; k < 5; ++k)
      r.d[k] = (a.d[k] * b.v - a.v * b.d[k]) / (b.v * b.v);
    return r;
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator/(VecDual a, float b) {
    VecDual r;
    r.v = a.v / b;
    for (int k = 0; k < 5; ++k)
      r.d[k] = a.d[k] / b;
    return r;
  }
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual operator/(float a, VecDual b) {
    VecDual r;
    r.v = a / b.v;
    for (int k = 0; k < 5; ++k)
      r.d[k] = (-a * b.d[k]) / (b.v * b.v);
    return r;
  }

  // Value via alpaka::math::*, derivative by the chain rule; each transcendental is evaluated once.
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual dsin(TAcc const& acc, VecDual u) {
    // d(sin u) = cos(u.v) * u.d
    const float c = alpaka::math::cos(acc, u.v);
    VecDual r;
    r.v = alpaka::math::sin(acc, u.v);
    for (int k = 0; k < 5; ++k)
      r.d[k] = c * u.d[k];
    return r;
  }
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual dcos(TAcc const& acc, VecDual u) {
    // d(cos u) = -sin(u.v) * u.d
    const float s = alpaka::math::sin(acc, u.v);
    VecDual r;
    r.v = alpaka::math::cos(acc, u.v);
    for (int k = 0; k < 5; ++k)
      r.d[k] = -s * u.d[k];
    return r;
  }
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual dsqrt(TAcc const& acc, VecDual u) {
    // d(sqrt u) = u.d / (2*sqrt(u.v))
    const float s = alpaka::math::sqrt(acc, u.v);
    VecDual r;
    r.v = s;
    for (int k = 0; k < 5; ++k)
      r.d[k] = u.d[k] / (2.f * s);
    return r;
  }
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual dabs(TAcc const& acc, VecDual u) {
    // d(|u|) = sign(u.v) * u.d
    const float sgn = (u.v >= 0.f) ? 1.f : -1.f;
    VecDual r;
    r.v = alpaka::math::abs(acc, u.v);
    for (int k = 0; k < 5; ++k)
      r.d[k] = sgn * u.d[k];
    return r;
  }
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE VecDual datan2(TAcc const& acc, VecDual y, VecDual x) {
    // d(atan2(y,x)) = (x.v*y.d - y.v*x.d) / (x.v*x.v + y.v*y.v)
    const float den = x.v * x.v + y.v * y.v;
    VecDual r;
    r.v = alpaka::math::atan2(acc, y.v, x.v);
    for (int k = 0; k < 5; ++k)
      r.d[k] = (x.v * y.d[k] - y.v * x.d[k]) / den;
    return r;
  }

  // Exact dr/d{phi0,tip,invPt,cotTheta,zip} at fixed z=zh in one pass: the closed-form endcap
  // r(params) in VecDual arithmetic with identity-seeded inputs, writing the five partials to Jr[5].
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rWithGrad5(
      TAcc const& acc, float phi0, float tip, float invPt, float cotTheta, float zip, float zh, float bf, float Jr[5]) {
    const VecDual dPhi0{phi0, {1.f, 0.f, 0.f, 0.f, 0.f}};
    const VecDual dTip{tip, {0.f, 1.f, 0.f, 0.f, 0.f}};
    const VecDual dInvPt{invPt, {0.f, 0.f, 1.f, 0.f, 0.f}};
    const VecDual dCot{cotTheta, {0.f, 0.f, 0.f, 1.f, 0.f}};
    const VecDual dZip{zip, {0.f, 0.f, 0.f, 0.f, 1.f}};

    // makeHelixStateFromParams: rho = 1/(invPt*bf); xc=(tip+rho)*sin(phi0); yc=-(tip+rho)*cos(phi0).
    const VecDual rho = 1.f / (dInvPt * bf);
    const VecDual sp = dsin(acc, dPhi0);
    const VecDual cp = dcos(acc, dPhi0);
    const VecDual tr = dTip + rho;  // tip + rho
    const VecDual xc = tr * sp;
    const VecDual yc = -tr * cp;
    const VecDual alphaOrigin = datan2(acc, -yc, -xc);

    // predictOnEndcap: arcS=(zh-zip)/cot; alphaH=alphaOrigin-arcS/rho; x=xc+|rho|cos; y=yc+|rho|sin; r=sqrt.
    const VecDual arcS = (zh - dZip) / dCot;
    const VecDual alphaH = alphaOrigin - arcS / rho;
    const VecDual absRho = dabs(acc, rho);
    const VecDual x = xc + absRho * dcos(acc, alphaH);
    const VecDual y = yc + absRho * dsin(acc, alphaH);
    const VecDual r = dsqrt(acc, x * x + y * y);
    for (int k = 0; k < 5; ++k)
      Jr[k] = r.d[k];
  }

  // RunningHelix: helix state + covariance maintained per track during the extender's hit-attach
  // loop. The BL input cov is block-diagonal (circle 3x3 / line 2x2); barrel measurements update both
  // blocks independently, endcap ones only the line block. Jacobian signs:
  //   dphi/dphi0=+1  dphi/dtip=-1/r  dphi/d(1/pT)=-bf*s^2/(2r)  dz/dzip=+1  dz/dcot=+s
  struct RunningHelix {
    float phi0, tip, invPt, cotTheta, zip;
    // Circle block (3x3 symmetric, packed): V(phi), cov(phi,tip),
    // cov(phi,1/pT), V(tip), cov(tip,1/pT), V(1/pT).
    float vPhi, cPhiTip, cPhiPt, vTip, cTipPt, vPt;
    // Line block (2x2 symmetric, packed): V(cotTh), cov(cotTh,zip), V(zip).
    float vCot, cCotZip, vZip;
    // Derived helix geometry, refreshed by recomputeHelix().
    float rho;
    float xc, yc;
    float alphaOrigin;

    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void recomputeHelix(TAcc const& acc, float bf) {
      rho = (invPt != 0.f) ? 1.f / (invPt * bf) : 1e9f;
      const float sp = alpaka::math::sin(acc, phi0);
      const float cp = alpaka::math::cos(acc, phi0);
      const float x0 = tip * sp;
      const float y0 = -tip * cp;
      xc = x0 + rho * sp;
      yc = y0 - rho * cp;
      alphaOrigin = alpaka::math::atan2(acc, -yc, -xc);
    }

    // Read-only HelixState view, so predictOn{Barrel,Endcap} need not be templated on RunningHelix.
    ALPAKA_FN_ACC ALPAKA_FN_INLINE HelixState helix() const {
      HelixState h;
      h.phi0 = phi0;
      h.tip = tip;
      h.cotTheta = cotTheta;
      h.zip = zip;
      h.rho = rho;
      h.xc = xc;
      h.yc = yc;
      h.alphaOrigin = alphaOrigin;
      return h;
    }

    // Kalman update on the circle (3x3) block from a 1-D phi measurement; sigPhi2Hit is the hit+MS
    // innovation (the propagation contribution is already in P). Returns chi^2 against the pre-update
    // state. Only P is downdated; the per-gap process noise P += Q is injected by the caller.
    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float updateCircleFromPhi(
        TAcc const& acc, float dPhi, float sigPhi2Hit, float r, float bf, float s) {
      const float invR = 1.f / alpaka::math::max(acc, r, 1.f);
      const float J0 = 1.f;
      const float J1 = -invR;
      const float J2 = -bf * s * s * 0.5f * invR;
      // P H^T (3-vector): row i = sum_j H_j * P(i, j)
      const float PH0 = J0 * vPhi + J1 * cPhiTip + J2 * cPhiPt;
      const float PH1 = J0 * cPhiTip + J1 * vTip + J2 * cTipPt;
      const float PH2 = J0 * cPhiPt + J1 * cTipPt + J2 * vPt;
      // S = H P H^T + R (scalar)
      const float HPHT = J0 * PH0 + J1 * PH1 + J2 * PH2;
      const float S = HPHT + sigPhi2Hit;
      if (!(S > 0.f))
        return 0.f;
      const float invS = 1.f / S;
      const float chi2 = dPhi * dPhi * invS;
      // K = P H^T / S (3-vector). Call sites pass d = pred - meas; the textbook
      // Kalman mean update is state += K*(meas - pred) = state -= K*dPhi.
      const float K0 = PH0 * invS;
      const float K1 = PH1 * invS;
      const float K2 = PH2 * invS;
      phi0 -= K0 * dPhi;
      tip -= K1 * dPhi;
      invPt -= K2 * dPhi;
      // P_new = P - K (H P) = P - (P H^T)(P H^T)^T / S  (symmetric rank-1)
      vPhi -= PH0 * PH0 * invS;
      vTip -= PH1 * PH1 * invS;
      vPt -= PH2 * PH2 * invS;
      cPhiTip -= PH0 * PH1 * invS;
      cPhiPt -= PH0 * PH2 * invS;
      cTipPt -= PH1 * PH2 * invS;
      return chi2;
    }

    // Kalman update on the line (2x2) block from a 1-D sec measurement.
    // Barrel: sec=z (exact). Endcap: sec=r, approximate on cot/zip, to keep the blocks diagonal.
    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float updateLineFromSec(TAcc const& acc, float dSec, float sigSec2Hit, float s) {
      // sec = zip + s * cotTh  -> H = (J_cot, J_zip) = (s, 1)
      const float J0 = s;    // cotTh
      const float J1 = 1.f;  // zip
      const float PH0 = J0 * vCot + J1 * cCotZip;
      const float PH1 = J0 * cCotZip + J1 * vZip;
      const float HPHT = J0 * PH0 + J1 * PH1;
      const float S = HPHT + sigSec2Hit;
      if (!(S > 0.f))
        return 0.f;
      const float invS = 1.f / S;
      const float chi2 = dSec * dSec * invS;
      const float K0 = PH0 * invS;
      const float K1 = PH1 * invS;
      // Call sites pass d = pred - meas; textbook update is state -= K*dSec.
      cotTheta -= K0 * dSec;
      zip -= K1 * dSec;
      vCot -= PH0 * PH0 * invS;
      vZip -= PH1 * PH1 * invS;
      cCotZip -= PH0 * PH1 * invS;
      return chi2;
    }

    // Kalman update on the line (2x2) block with a general measurement row H = (Jcot, Jzip); for the
    // endcap, H = (Jr_cot, Jr_zip) from rWithGrad5, the r-at-fixed-z measurement being exactly
    // representable by the 2x2 line filter. dSec = pred - meas (state -= K*dSec); sigSec2Hit = noise.
    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float updateLineFromSecH(
        TAcc const& acc, float dSec, float sigSec2Hit, float Jcot, float Jzip) {
      const float J0 = Jcot;
      const float J1 = Jzip;
      const float PH0 = J0 * vCot + J1 * cCotZip;
      const float PH1 = J0 * cCotZip + J1 * vZip;
      const float HPHT = J0 * PH0 + J1 * PH1;
      const float S = HPHT + sigSec2Hit;
      if (!(S > 0.f))
        return 0.f;
      const float invS = 1.f / S;
      const float chi2 = dSec * dSec * invS;
      const float K0 = PH0 * invS;
      const float K1 = PH1 * invS;
      // Call sites pass d = pred - meas; textbook update is state -= K*dSec.
      cotTheta -= K0 * dSec;
      zip -= K1 * dSec;
      vCot -= PH0 * PH0 * invS;
      vZip -= PH1 * PH1 * invS;
      cCotZip -= PH0 * PH1 * invS;
      return chi2;
    }
  };

  // Initialize a RunningHelix from a track's BL state and cov; derived geometry is recomputed.
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE RunningHelix
  makeRunningHelix(TAcc const& acc, const ::reco::TrackSoAConstView tracks, int i, float bf) {
    RunningHelix h;
    h.phi0 = tracks[i].state()(0);
    h.tip = tracks[i].state()(1);
    h.invPt = tracks[i].state()(2);
    h.cotTheta = tracks[i].state()(3);
    h.zip = tracks[i].state()(4);
    // Circle block from cov entries (0, 1, 2, 5, 6, 9); line block from (12, 13, 14).
    h.vPhi = tracks[i].covariance()(0);
    h.cPhiTip = tracks[i].covariance()(1);
    h.cPhiPt = tracks[i].covariance()(2);
    h.vTip = tracks[i].covariance()(5);
    h.cTipPt = tracks[i].covariance()(6);
    h.vPt = tracks[i].covariance()(9);
    h.vCot = tracks[i].covariance()(12);
    h.cCotZip = tracks[i].covariance()(13);
    h.vZip = tracks[i].covariance()(14);
    h.recomputeHelix(acc, bf);
    return h;
  }

  // Diagnostic: exact full-5x5 covariance-only joint (phi,r) Kalman update, without the state update.
  // C[15] is packed symmetric, in the reco::TrackSoA cov layout, and takes the walk's measurement rows.
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void updateShadowCov(
      TAcc const& acc, float* C, const float Hp[5], const float Hs[5], float R00, float R11) {
    constexpr int off[5] = {0, 4, 7, 9, 10};
    auto cidx = [&](int a, int b) {
      const int lo = a < b ? a : b;
      const int hi = a < b ? b : a;
      return off[lo] + hi;
    };
    float Pp[5], Ps[5];
    for (int a = 0; a < 5; ++a) {
      float sp = 0.f, ss = 0.f;
      for (int b = 0; b < 5; ++b) {
        const float cab = C[cidx(a, b)];
        sp += cab * Hp[b];
        ss += cab * Hs[b];
      }
      Pp[a] = sp;
      Ps[a] = ss;
    }
    float S00 = R00, S11 = R11, S01 = 0.f;
    for (int a = 0; a < 5; ++a) {
      S00 += Hp[a] * Pp[a];
      S11 += Hs[a] * Ps[a];
      S01 += Hp[a] * Ps[a];
    }
    const float det = S00 * S11 - S01 * S01;
    if (!(det > 0.f) || !alpaka::math::isfinite(acc, det))
      return;
    const float i00 = S11 / det, i11 = S00 / det, i01 = -S01 / det;
    for (int a = 0; a < 5; ++a)
      for (int b = a; b < 5; ++b)
        C[cidx(a, b)] -= i00 * Pp[a] * Pp[b] + i01 * (Pp[a] * Ps[b] + Ps[a] * Pp[b]) + i11 * Ps[a] * Ps[b];
  }

  // Project a packed symmetric 5x5 (layout off[5]={0,4,7,9,10}) onto a measurement row H: H^T C H.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float projectCov(const float* C, const float H[5]) {
    constexpr int off[5] = {0, 4, 7, 9, 10};
    float acc2 = 0.f;
    for (int a = 0; a < 5; ++a) {
      acc2 += C[off[a] + a] * H[a] * H[a];
      for (int b = a + 1; b < 5; ++b)
        acc2 += 2.f * C[off[a] + b] * H[a] * H[b];
    }
    return acc2;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_ExtenderHelixHelpers_h
