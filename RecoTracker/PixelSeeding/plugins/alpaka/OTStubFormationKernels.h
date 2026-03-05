#ifndef RecoTracker_PixelSeeding_plugins_alpaka_OTStubFormationKernels_h
#define RecoTracker_PixelSeeding_plugins_alpaka_OTStubFormationKernels_h

// Uncomment to enable PS stub position debug output (very verbose, one printf per PS stub)
// #define PS_STUB_POSITION_DEBUG

// Uncomment to enable diagnostic counters for stub formation efficiency analysis
// #define STUB_DIAGNOSTIC_COUNTERS

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Math/interface/approx_atan2.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometrySoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace otStubFormation {

    // Work division for kernels
    using Idx = uint32_t;

    // Compute parallax correction for tilted modules
    // This is the geometric offset in local-x caused by the track coming from the IP
    // to a tilted module. Uses tiltAngle convention where angle is measured from radial.
    //
    // IMPORTANT: The tiltAngle is defined using the PHYSICAL inner->outer direction,
    // not the topological lower->upper direction. This ensures consistent behavior
    // regardless of whether the module is flipped. The input position (gx, gy, gz)
    // should be the PHYSICAL INNER hit position for consistency.
    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeParallaxCorrection(TAcc const& acc,
                                                                   float gx_inner,
                                                                   float gy_inner,
                                                                   float gz_inner,
                                                                   float tiltAngle,
                                                                   float separation) {
      // Compute radial distance in xy-plane
      float gR_inner = alpaka::math::sqrt(acc, gx_inner * gx_inner + gy_inner * gy_inner);
      if (gR_inner < 1e-6f)
        return 0.0f;  // At origin, undefined

      // Precompute trig functions for tilt angle
      float cosTheta = alpaka::math::cos(acc, tiltAngle);
      float sinTheta = alpaka::math::sin(acc, tiltAngle);

      // Transform position vector from origin to inner cluster into local frame
      // With our tilt angle convention (theta = atan2(dz_phys, dr_phys), physical inner->outer):
      //   local-z unit vector in (r,z) = (cos theta, sin theta)   [along inner->outer]
      //   local-x unit vector in (r,z) = (-sin theta, cos theta)  [perpendicular, 90 deg CCW from local-z]
      // So the transformation is:
      //   lV_x = r * (-sin(theta)) + z * cos(theta) = -r * sin(theta) + z * cos(theta)
      //   lV_z = r * cos(theta) + z * sin(theta)
      float lV_x = -gR_inner * sinTheta + gz_inner * cosTheta;
      float lV_z = gR_inner * cosTheta + gz_inner * sinTheta;

      // Check for valid local z-component
      if (alpaka::math::abs(acc, lV_z) < 1e-6f) {
        return 0.0f;
      }

      // Compute normalized slope in local x-z plane and parallax correction
      float lV_norm_x = lV_x / lV_z;
      float parallCorr = lV_norm_x * separation;

      if (!alpaka::math::isfinite(acc, parallCorr)) {
        return 0.0f;
      }

      return parallCorr;
    }

    // Counting kernel: Count stubs per module
    class CountStubsKernel {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    ::reco::OTRecHitsConstView hits,
                                    ::reco::OTHitModuleConstView moduleView,
                                    ::reco::StackedModuleGeometryConstView geometry,
                                    int32_t const* barrelFlatMaxCSDiff,
                                    int32_t const* barrelTiltedMaxCSDiff,
                                    int32_t const* endcapMaxCSDiff,
                                    int32_t const* barrelFlatMaxCS,
                                    int32_t const* barrelTiltedMaxCS,
                                    int32_t const* endcapMaxCS,
                                    int32_t const* barrelFlatMaxCSSum,
                                    int32_t const* barrelTiltedMaxCSSum,
                                    int32_t const* endcapMaxCSSum,
                                    uint32_t* stubCounts,
                                    uint32_t nModules) const {
        // One thread per module
        for (auto iModule : cms::alpakatools::uniform_elements(acc, nModules)) {
          // Debug disabled for production
          // if (iModule == 0) {
          //   printf("CountStubs kernel running: nModules=%u\n", nModules);
          // }

          uint32_t count = 0;

          // Get hit ranges for this module
          uint32_t hitStart = moduleView[iModule].moduleStart();
          uint32_t hitEnd = moduleView[iModule + 1].moduleStart();
          uint32_t upperStart = moduleView[iModule].upperSensorStart();

          uint32_t nLower = upperStart - hitStart;
          uint32_t nUpper = hitEnd - upperStart;

          // Debug disabled for production
          // if ((nLower > 0 || nUpper > 0) && iModule < 13000) {
          //   printf("CountStubs: Module %u: hitStart=%u, hitEnd=%u, upperStart=%u, nLower=%u, nUpper=%u\n",
          //          iModule,
          //          hitStart,
          //          hitEnd,
          //          upperStart,
          //          nLower,
          //          nUpper);
          // }

          // Handle modules with no upper hits (P-hit recovery for PS modules)
          if (nUpper == 0) {
            // For PS modules: recover P-hits that have no matching S-hits
            // These become PHitOnly stub entries (no bend measurement)
            uint8_t moduleType = geometry[iModule].moduleType();
            bool isPS = (moduleType == 0 || moduleType == 1);
            if (isPS && nLower > 0) {
              // Count each P-hit as a PHitOnly stub entry
              stubCounts[iModule + 1] = nLower;
#ifdef STUB_DIAGNOSTIC_COUNTERS
              printf("STUB_DIAG module=%u type=PS totalPHits=%u pHitOnlyEntries=%u "
                     "reason=noUpperHit\n",
                     iModule, nLower, nLower);
#endif
            } else {
              // SS modules or no lower hits: no stubs
              stubCounts[iModule + 1] = 0;
            }
            continue;
          }

          // Skip if no lower hits
          if (nLower == 0) {
            stubCounts[iModule + 1] = 0;
            continue;
          }

          // Get geometry parameters
          float maxBend = geometry[iModule].maxBend();
          bool isFlipped = geometry[iModule].isFlipped();
          bool isBarrel = geometry[iModule].isBarrel();
          bool isFlat = geometry[iModule].isFlat();
          uint8_t layer = geometry[iModule].layer();

          // Per-layer cluster size cuts
          int32_t maxCSDiff = isBarrel ? (isFlat ? barrelFlatMaxCSDiff[layer] : barrelTiltedMaxCSDiff[layer])
                                       : endcapMaxCSDiff[layer];
          int32_t maxCS = isBarrel ? (isFlat ? barrelFlatMaxCS[layer] : barrelTiltedMaxCS[layer])
                                   : endcapMaxCS[layer];
          int32_t maxCSSum = isBarrel ? (isFlat ? barrelFlatMaxCSSum[layer] : barrelTiltedMaxCSSum[layer])
                                      : endcapMaxCSSum[layer];

#ifdef STUB_DIAGNOSTIC_COUNTERS
          uint8_t moduleType = geometry[iModule].moduleType();
          bool isPS = (moduleType == 0 || moduleType == 1);
          // Track stubs per lower hit (P-hit for PS modules)
          uint32_t pHitsWithStub = 0;
          uint32_t pHitsLostBendCut = 0;
          uint32_t maxStubsPerPHit = 0;
#endif

          // Count valid stub combinations
          for (uint32_t iLower = hitStart; iLower < upperStart; ++iLower) {
            auto const& lowerHit = hits[iLower];
            float phi_lower = alpaka::math::atan2(acc, lowerHit.yGlobal(), lowerHit.xGlobal());

#ifdef STUB_DIAGNOSTIC_COUNTERS
            uint32_t stubsForThisLowerHit = 0;
#endif

            for (uint32_t iUpper = upperStart; iUpper < hitEnd; ++iUpper) {
              auto const& upperHit = hits[iUpper];

              // Same-sign local-y cut: reject if hits are on opposite strip sides
              if (lowerHit.yLocal() * upperHit.yLocal() < 0.f)
                continue;

              // Cluster size compatibility check (per-layer cuts)
              if (alpaka::math::abs(acc, int32_t(lowerHit.clusterSize()) - int32_t(upperHit.clusterSize())) > maxCSDiff)
                continue;
              if (int32_t(lowerHit.clusterSize()) > maxCS || int32_t(upperHit.clusterSize()) > maxCS)
                continue;
              if (int32_t(lowerHit.clusterSize()) + int32_t(upperHit.clusterSize()) > maxCSSum)
                continue;

              float phi_upper = alpaka::math::atan2(acc, upperHit.yGlobal(), upperHit.xGlobal());

              // Calculate bend = phi_outer - phi_inner
              // If flipped: lower is outer, upper is inner
              // If not flipped: upper is outer, lower is inner
              float bend = isFlipped ? (phi_lower - phi_upper) : (phi_upper - phi_lower);
              if (bend > M_PI)
                bend -= 2.0f * M_PI;
              if (bend < -M_PI)
                bend += 2.0f * M_PI;

              // Apply bend cut
              if (alpaka::math::abs(acc, bend) < maxBend) {
                count++;
#ifdef STUB_DIAGNOSTIC_COUNTERS
                stubsForThisLowerHit++;
#endif
              }
            }

#ifdef STUB_DIAGNOSTIC_COUNTERS
            if (stubsForThisLowerHit > 0) {
              pHitsWithStub++;
              if (stubsForThisLowerHit > maxStubsPerPHit) {
                maxStubsPerPHit = stubsForThisLowerHit;
              }
            } else {
              // This lower hit (P-hit for PS) formed no stubs - all pairs failed bend cut
              pHitsLostBendCut++;
            }
#endif
          }

          // Write to offset position for exclusive scan
          stubCounts[iModule + 1] = count;

#ifdef STUB_DIAGNOSTIC_COUNTERS
          // Print per-module diagnostic for modules with lower hits (P-hits for PS)
          if (nLower > 0) {
            const char* modType = isPS ? "PS" : "SS";
            printf("STUB_DIAG module=%u type=%s totalPHits=%u stubsFormed=%u pHitsWithStub=%u pHitsLost=%u "
                   "lostReason=bendCut maxStubsPerPHit=%u\n",
                   iModule, modType, nLower, count, pHitsWithStub, pHitsLostBendCut, maxStubsPerPHit);
          }
#endif

          // Debug disabled for production
          // if (count > 0 && iModule < 13000) {
          //   printf("CountStubs: Module %u counted %u stubs (maxBend=%.6f)\n", iModule, count, maxBend);
          // }
        }
      }
    };

    // Formation kernel: Create actual stubs
    // CA module offset parameters (computed in producer)
    struct CAModuleOffsets {
      static constexpr uint32_t nPixelModules = 4000;
      uint32_t barrelStartGeom;
      uint32_t backwardStartGeom;
      uint32_t forwardStartGeom;
      uint32_t nBarrelModules;
      uint32_t nBackwardModules;
      uint32_t nForwardModules;
    };

    class FormStubsKernel {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    ::reco::OTRecHitsConstView hits,
                                    ::reco::OTHitModuleConstView moduleView,
                                    ::reco::StackedModuleGeometryConstView geometry,
                                    int32_t const* barrelFlatMaxCSDiff,
                                    int32_t const* barrelTiltedMaxCSDiff,
                                    int32_t const* endcapMaxCSDiff,
                                    int32_t const* barrelFlatMaxCS,
                                    int32_t const* barrelTiltedMaxCS,
                                    int32_t const* endcapMaxCS,
                                    int32_t const* barrelFlatMaxCSSum,
                                    int32_t const* barrelTiltedMaxCSSum,
                                    int32_t const* endcapMaxCSSum,
                                    uint32_t const* stubOffsets,
                                    ::reco::StubsView stubs,
                                    uint32_t nModules,
                                    CAModuleOffsets caOffsets) const {
        // One thread per module
        for (auto iModule : cms::alpakatools::uniform_elements(acc, nModules)) {
          // Debug disabled for production
          // if (iModule == 0) {
          //   printf("FormStubs kernel running: nModules=%u\n", nModules);
          // }

          // Get hit ranges for this module
          uint32_t hitStart = moduleView[iModule].moduleStart();
          uint32_t hitEnd = moduleView[iModule + 1].moduleStart();
          uint32_t upperStart = moduleView[iModule].upperSensorStart();

          uint32_t nLower = upperStart - hitStart;
          uint32_t nUpper = hitEnd - upperStart;

          // Get geometry parameters
          float maxBend = geometry[iModule].maxBend();
          [[maybe_unused]] float separation = geometry[iModule].sensorSeparation();  // mm (kept for reference)
          uint8_t moduleType = geometry[iModule].moduleType();
          bool isPS = (moduleType == 0 || moduleType == 1);  // P or SinPS
          bool isBarrel = geometry[iModule].isBarrel();
          bool isFlat = geometry[iModule].isFlat();
          bool isFlipped = geometry[iModule].isFlipped();
          uint8_t layer = geometry[iModule].layer();

          // Per-layer cluster size cuts
          int32_t maxCSDiff = isBarrel ? (isFlat ? barrelFlatMaxCSDiff[layer] : barrelTiltedMaxCSDiff[layer])
                                       : endcapMaxCSDiff[layer];
          int32_t maxCS = isBarrel ? (isFlat ? barrelFlatMaxCS[layer] : barrelTiltedMaxCS[layer])
                                   : endcapMaxCS[layer];
          int32_t maxCSSum = isBarrel ? (isFlat ? barrelFlatMaxCSSum[layer] : barrelTiltedMaxCSSum[layer])
                                      : endcapMaxCSSum[layer];

          // Output offset for this module's stubs
          uint32_t stubIdx = stubOffsets[iModule];
          uint32_t maxStubIdx = stubOffsets[iModule + 1];  // End offset for this module

          // Handle PHitOnly recovery for PS modules with P-hits but no S-hits
          if (nUpper == 0) {
            if (isPS && nLower > 0) {
              // Create PHitOnly stub entries for each P-hit
              // These have position but no bend measurement
              uint32_t caModuleIndex = caOffsets.nPixelModules + iModule;

              for (uint32_t iLower = hitStart; iLower < upperStart; ++iLower) {
                if (stubIdx >= maxStubIdx)
                  break;

                auto const& pHit = hits[iLower];

                // Fill position from P-hit
                stubs[stubIdx].xGlobal() = pHit.xGlobal();
                stubs[stubIdx].yGlobal() = pHit.yGlobal();
                stubs[stubIdx].zGlobal() = pHit.zGlobal();
                float xg = pHit.xGlobal();
                float yg = pHit.yGlobal();
                stubs[stubIdx].rGlobal() = alpaka::math::sqrt(acc, xg * xg + yg * yg);
                stubs[stubIdx].xLocal() = pHit.xLocal();
                stubs[stubIdx].yLocal() = pHit.yLocal();
                stubs[stubIdx].xerrLocal() = pHit.xerrLocal();
                stubs[stubIdx].yerrLocal() = pHit.yerrLocal();
                stubs[stubIdx].iphi() = unsafe_atan2s<7>(yg, xg);

                // Set stub-specific fields to invalid values (no bend measurement)
                stubs[stubIdx].dPhiDr() = 0.0f;
                stubs[stubIdx].dPhiDrError() = 999.0f;  // Large error indicates invalid
                stubs[stubIdx].bend() = 0.0f;
                stubs[stubIdx].ptEst() = 0.0f;  // No pT estimate without bend

                // Detector index for CA
                stubs[stubIdx].detectorIndex() = static_cast<uint16_t>(caModuleIndex);

                // Hit indices: P-hit is inner, no outer hit
                stubs[stubIdx].innerHitIdx() = iLower;
                stubs[stubIdx].outerHitIdx() = UINT32_MAX;  // Invalid marker

                // Stub type: PHitOnly
                stubs[stubIdx].stubType() = ::reco::StubType::PHitOnly;

                // Flags: valid, with appropriate barrel/flat/layer info
                stubs[stubIdx].flags() = ::reco::StubFlags::makeFlags(isBarrel, isFlat, true, layer);

                // P-hit group ID: use unique stubIdx so each stub is treated independently
                // (set to iLower to re-enable grouping of stubs sharing the same P-hit)
                stubs[stubIdx].pHitGroupId() = stubIdx;

                stubIdx++;
              }
            }
            continue;
          }

          // Skip if no lower hits
          if (nLower == 0)
            continue;

          // Debug disabled for production
          // if (iModule < 13000 && (nLower > 0 || nUpper > 0)) {
          //   printf("FormStubs: Module %u: stubIdx=%u, maxStubIdx=%u, nLower=%u, nUpper=%u\n",
          //          iModule,
          //          stubIdx,
          //          maxStubIdx,
          //          nLower,
          //          nUpper);
          // }

          // Form stubs from all combinations
          for (uint32_t iLower = hitStart; iLower < upperStart; ++iLower) {
            auto const& lowerHit = hits[iLower];
            float phi_lower = alpaka::math::atan2(acc, lowerHit.yGlobal(), lowerHit.xGlobal());

            for (uint32_t iUpper = upperStart; iUpper < hitEnd; ++iUpper) {
              auto const& upperHit = hits[iUpper];

              // Same-sign local-y cut: reject if hits are on opposite strip sides
              if (lowerHit.yLocal() * upperHit.yLocal() < 0.f)
                continue;

              // Cluster size compatibility check (per-layer cuts)
              if (alpaka::math::abs(acc, int32_t(lowerHit.clusterSize()) - int32_t(upperHit.clusterSize())) > maxCSDiff)
                continue;
              if (int32_t(lowerHit.clusterSize()) > maxCS || int32_t(upperHit.clusterSize()) > maxCS)
                continue;
              if (int32_t(lowerHit.clusterSize()) + int32_t(upperHit.clusterSize()) > maxCSSum)
                continue;

              float phi_upper = alpaka::math::atan2(acc, upperHit.yGlobal(), upperHit.xGlobal());

              // Determine which hit is physically inner/outer based on isFlipped
              // Note: We avoid using ternary operator with references (auto const& = cond ? a : b)
              // as this pattern can cause issues in GPU device code with SoA proxy objects,
              // leading to NULL pointer dereference. Instead, we select indices and access
              // the hits directly when needed.
              uint32_t innerIdx = isFlipped ? iUpper : iLower;
              uint32_t outerIdx = isFlipped ? iLower : iUpper;

              // Calculate bend = phi_outer - phi_inner
              float bend = isFlipped ? (phi_lower - phi_upper) : (phi_upper - phi_lower);
              if (bend > M_PI)
                bend -= 2.0f * M_PI;
              if (bend < -M_PI)
                bend += 2.0f * M_PI;

              // Apply bend cut
              if (alpaka::math::abs(acc, bend) >= maxBend)
                continue;

              // Check bounds before writing
              if (stubIdx >= maxStubIdx)
                continue;

              // --- pT estimate from exact circle through (0,0), inner, outer ---
              // Access hits directly using selected indices to avoid issues with ternary-selected references
              float xi = hits[innerIdx].xGlobal();
              float yi = hits[innerIdx].yGlobal();
              float xo = hits[outerIdx].xGlobal();
              float yo = hits[outerIdx].yGlobal();

              // h1 = xi*yo - xo*yi
              float h1 = xi * yo - xo * yi;

              // h3 = (xi-xo)^2 + (yi-yo)^2
              float dx = xi - xo;
              float dy = yi - yo;
              float h3 = dx * dx + dy * dy;

              // r_i^2 and r_o^2
              float ri2 = xi * xi + yi * yi;
              float ro2 = xo * xo + yo * yo;

              // Protect against degeneracies (nearly straight line)
              constexpr float eps = 1e-9f;

              float rho = 0.0f;
              float ptEst = 0.0f;

              if (alpaka::math::abs(acc, h1) > eps && h3 > eps) {
                float num = alpaka::math::sqrt(acc, ri2 * ro2 * h3);
                rho = num / (2.0f * alpaka::math::abs(acc, h1));

                // Convert radius -> pT (rho must be in cm!)
                constexpr float B_TESLA = 3.8f;
                constexpr float kConv = 0.003f * B_TESLA;

                ptEst = kConv * rho;  // GeV
              } else {
                // Nearly straight track -> very high pT
                ptEst = 1e6f;
              }

              // Valid stub - fill properties
              // For PS modules, use the PIXEL hit position (better spatial precision)
              // In CMSSW convention, the pixel sensor is always the "lower" sensor in the
              // stack reference frame for PS modules, so lowerHit is the pixel hit.
              // For SS modules, use average of both sensors for global, inner for local
              //
              // IMPORTANT: Compute stub position in local variables first, then write to
              // the stubs SoA once. Do NOT read back from stubs SoA after writing, because
              // the SoA view uses __restrict__ qualified pointers and the compiler will
              // generate unnecessary global memory reads instead of reusing register values.
              float xg, yg, zg;
              if (isPS) {
                xg = lowerHit.xGlobal();
                yg = lowerHit.yGlobal();
                zg = lowerHit.zGlobal();
                stubs[stubIdx].xGlobal() = xg;
                stubs[stubIdx].yGlobal() = yg;
                stubs[stubIdx].zGlobal() = zg;
                stubs[stubIdx].xLocal() = lowerHit.xLocal();
                stubs[stubIdx].yLocal() = lowerHit.yLocal();
                stubs[stubIdx].xerrLocal() = lowerHit.xerrLocal();
                stubs[stubIdx].yerrLocal() = lowerHit.yerrLocal();
#ifdef PS_STUB_POSITION_DEBUG
                // Debug: print PS stub position from lowerHit (pixel sensor)
                // Compare with P-hit positions from Phase2OTRecHitsSoAConverter
                float stub_r = alpaka::math::sqrt(acc, xg * xg + yg * yg);
                printf("PS_STUB module=%u stubIdx=%u lowerHitIdx=%u: "
                       "x=%.6f y=%.6f z=%.6f r=%.4f "
                       "iphi_lower=%d xerrLocal=%.4e yerrLocal=%.4e sensorDetId=%u\n",
                       iModule, stubIdx, iLower,
                       xg, yg, zg, stub_r,
                       lowerHit.iphi(), lowerHit.xerrLocal(), lowerHit.yerrLocal(), lowerHit.sensorDetId());
#endif
              } else {
                xg = 0.5f * (lowerHit.xGlobal() + upperHit.xGlobal());
                yg = 0.5f * (lowerHit.yGlobal() + upperHit.yGlobal());
                zg = 0.5f * (lowerHit.zGlobal() + upperHit.zGlobal());
                stubs[stubIdx].xGlobal() = xg;
                stubs[stubIdx].yGlobal() = yg;
                stubs[stubIdx].zGlobal() = zg;
                // Access inner hit directly using selected index
                stubs[stubIdx].xLocal() = hits[innerIdx].xLocal();
                stubs[stubIdx].yLocal() = hits[innerIdx].yLocal();
                stubs[stubIdx].xerrLocal() = hits[innerIdx].xerrLocal();
                stubs[stubIdx].yerrLocal() = hits[innerIdx].yerrLocal();
              }
              float rg = alpaka::math::sqrt(acc, xg * xg + yg * yg);
              stubs[stubIdx].rGlobal() = rg;

              // iPhi - use same encoding as pixel hits (16-bit signed, full circle = [-32768, 32767])
              stubs[stubIdx].iphi() = unsafe_atan2s<7>(yg, xg);

              // Calculate dPhiDr from GLOBAL positions of inner and outer hits.
              // This ensures compatibility with the CA doublet direction check, which
              // computes dPhiDr as (phi_outer - phi_inner) / (r_outer - r_inner).
              //
              // For tilted modules, the raw phi difference includes a parallax contribution
              // that depends on where along the module (in local y) the hit occurs. This
              // causes a wider dPhiDr distribution for tilted modules even with fixed pT.
              // We correct for this by subtracting the parallax contribution from dphi.
              //
              // Parallax correction for dphi is applied ONLY for:
              // - Barrel tilted modules (isBarrel && !isFlat)
              // - Endcap modules (!isBarrel)
              // For barrel flat modules, no dphi parallax correction is needed because
              // the local-x direction (pitch direction) aligns with global z, not phi.
              //
              // Note: xi, yi, xo, yo, ri2, ro2 already computed above for pT estimate
              {
                float phi_inner = alpaka::math::atan2(acc, yi, xi);
                float phi_outer = alpaka::math::atan2(acc, yo, xo);
                float ri = alpaka::math::sqrt(acc, ri2);
                float ro = alpaka::math::sqrt(acc, ro2);

                float dphi_raw = phi_outer - phi_inner;
                // Handle phi wraparound
                if (dphi_raw > M_PI)
                  dphi_raw -= 2.0f * M_PI;
                if (dphi_raw < -M_PI)
                  dphi_raw += 2.0f * M_PI;

                // Get tilt parameters for parallax and effective dr calculation
                float tiltAngle = geometry[iModule].tiltAngle();
                float sinTilt = geometry[iModule].sinTilt();
                float cosTilt = geometry[iModule].cosTilt();
                float separation_cm = geometry[iModule].sensorSeparation() * 0.1f;  // mm to cm

                // Compute parallax correction for dphi
                // Only apply for tilted barrel and endcap modules
                // For flat barrel modules, local-x is aligned with z, not phi, so no dphi correction
                float dphi;
                bool applyDphiParallax = !isFlat || !isBarrel;  // tilted barrel or endcap
                if (applyDphiParallax) {
                  float pC = computeParallaxCorrection(acc, xi, yi, hits[innerIdx].zGlobal(), tiltAngle, separation_cm);
                  float dphi_parallax = (ri > 1e-6f) ? pC / ri : 0.0f;
                  dphi = dphi_raw - dphi_parallax;
                } else {
                  // Flat barrel: no dphi parallax correction
                  dphi = dphi_raw;
                }

                float dr_geometric = ro - ri;

                // Use stub position (already in local variables) for effective dr calculation
                float r_stub = rg;
                float z_stub = zg;

                // Compute effective dr: project sensor separation onto radial direction
                // dr_effective = separation / (cosTilt + sinTilt * z/r)
                // Note: tiltAngle = atan2(dz, dr) measured from radial axis, so sinTilt > 0 for +z tilt
                float denominator = cosTilt + sinTilt * z_stub / r_stub;

                float dr_effective;
                if (alpaka::math::abs(acc, denominator) > 1e-6f) {
                  dr_effective = separation_cm / denominator;
                } else {
                  // Fallback to geometric dr if denominator is too small (track parallel to module)
                  dr_effective = dr_geometric;
                }

                // dPhiDr: rate of change of phi with radius (rad/cm)
                // Use parallax-corrected dphi and effective dr which accounts for tilted module geometry
                stubs[stubIdx].dPhiDr() =
                    (alpaka::math::abs(acc, dr_effective) > 1e-6f) ? dphi / dr_effective : 0.0f;

                // Error on dPhiDr: propagate position errors through the calculation
                // Use global phi variance σ²(φ) = (sin²φ·Cxx - 2·sinφ·cosφ·Cxy + cos²φ·Cyy) / r²
                // where Cxx, Cxy, Cyy are the global position error covariance matrix elements.
                // This correctly accounts for the local-y error contribution which leaks into phi
                // for modules where the strip direction is not exactly radial (important for endcap).
                float sinphi_i = alpaka::math::sin(acc, phi_inner);
                float cosphi_i = alpaka::math::cos(acc, phi_inner);
                float sig2phi_i = (ri2 > 0.0f) ? (sinphi_i * sinphi_i * hits[innerIdx].xGlobalErr() -
                                                   2.0f * sinphi_i * cosphi_i * hits[innerIdx].xyGlobalErr() +
                                                   cosphi_i * cosphi_i * hits[innerIdx].yGlobalErr()) /
                                                      ri2
                                                : 0.0f;
                float sinphi_o = alpaka::math::sin(acc, phi_outer);
                float cosphi_o = alpaka::math::cos(acc, phi_outer);
                float sig2phi_o = (ro2 > 0.0f) ? (sinphi_o * sinphi_o * hits[outerIdx].xGlobalErr() -
                                                   2.0f * sinphi_o * cosphi_o * hits[outerIdx].xyGlobalErr() +
                                                   cosphi_o * cosphi_o * hits[outerIdx].yGlobalErr()) /
                                                      ro2
                                                : 0.0f;
                float dphi_err = alpaka::math::sqrt(acc, sig2phi_i + sig2phi_o);
                stubs[stubIdx].dPhiDrError() =
                    (alpaka::math::abs(acc, dr_effective) > 1e-6f) ? dphi_err / alpaka::math::abs(acc, dr_effective) : 0.0f;
              }

              // Bend (raw global phi difference, used for bend cut and pT estimate)
              stubs[stubIdx].bend() = bend;

              // Pt estimate
              stubs[stubIdx].ptEst() = ptEst;

              // Detector index - map to CA module index
              // Since StackedModuleGeometry is sorted in CA order (barrel by layer, backward, forward),
              // the CA module index is simply nPixelModules + geometry index
              uint32_t caModuleIndex = caOffsets.nPixelModules + iModule;
              stubs[stubIdx].detectorIndex() = static_cast<uint16_t>(caModuleIndex);

              // Hit indices (physical inner/outer)
              stubs[stubIdx].innerHitIdx() = innerIdx;
              stubs[stubIdx].outerHitIdx() = outerIdx;

              // Stub type
              stubs[stubIdx].stubType() = isPS ? ::reco::StubType::PS : ::reco::StubType::SS;

              // Flags
              stubs[stubIdx].flags() = ::reco::StubFlags::makeFlags(isBarrel, isFlat, true, layer);

              // P-hit group ID: use unique stubIdx so each stub is treated independently
              // (set to iLower to re-enable grouping of stubs sharing the same P-hit)
              stubs[stubIdx].pHitGroupId() = stubIdx;

              // Debug disabled for production
              // if (iModule < 13000) {
              //   printf("FormStubs: Module %u formed stub at index %u: pos=(%.3f,%.3f,%.3f) bend=%.6f\n",
              //          iModule,
              //          stubIdx,
              //          stubs[stubIdx].xGlobal(),
              //          stubs[stubIdx].yGlobal(),
              //          stubs[stubIdx].zGlobal(),
              //          bend);
              // }

              stubIdx++;
            }
          }
        }
      }
    };

    // Kernel to fill moduleStart array from stubOffsets
    class FillModuleStartKernel {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    uint32_t const* stubOffsets,
                                    ::reco::StubModuleView stubModuleView,
                                    uint32_t nModules) const {
        // One thread per module (+ 1 for the final total)
        for (auto i : cms::alpakatools::uniform_elements(acc, nModules + 1)) {
          stubModuleView[i].moduleStart() = stubOffsets[i];
        }
      }
    };

#ifdef STUB_DIAGNOSTIC_COUNTERS
    // Diagnostic kernel: Compute event-level summary statistics for stub formation
    // This kernel runs after stub formation to aggregate statistics across all modules
    class DiagnosticSummaryKernel {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    ::reco::OTRecHitsConstView hits,
                                    ::reco::OTHitModuleConstView moduleView,
                                    ::reco::StackedModuleGeometryConstView geometry,
                                    uint32_t const* stubOffsets,
                                    uint32_t nModules) const {
        // Single thread computes event-level summary
        // This is inefficient but acceptable for diagnostic purposes
        for (auto idx : cms::alpakatools::uniform_elements(acc, 1u)) {
          (void)idx;  // Unused, we just need one thread

          uint32_t totalPHits_PS = 0;
          uint32_t totalPHits_SS = 0;
          uint32_t pHitsWithStub_PS = 0;
          uint32_t pHitsWithStub_SS = 0;
          uint32_t pHitsLostNoUpper_PS = 0;
          uint32_t pHitsLostNoUpper_SS = 0;
          uint32_t pHitsLostBendCut_PS = 0;
          uint32_t pHitsLostBendCut_SS = 0;
          uint32_t totalStubs = 0;
          uint32_t maxStubsPerPHit_PS = 0;
          uint32_t maxStubsPerPHit_SS = 0;
          uint32_t pHitsMultiStub_PS = 0;  // P-hits that formed >1 stub
          uint32_t pHitsMultiStub_SS = 0;

          for (uint32_t iModule = 0; iModule < nModules; ++iModule) {
            uint32_t hitStart = moduleView[iModule].moduleStart();
            uint32_t hitEnd = moduleView[iModule + 1].moduleStart();
            uint32_t upperStart = moduleView[iModule].upperSensorStart();

            uint32_t nLower = upperStart - hitStart;
            uint32_t nUpper = hitEnd - upperStart;

            uint8_t moduleType = geometry[iModule].moduleType();
            bool isPS = (moduleType == 0 || moduleType == 1);
            bool isFlipped = geometry[iModule].isFlipped();
            float maxBend = geometry[iModule].maxBend();

            if (isPS) {
              totalPHits_PS += nLower;
            } else {
              totalPHits_SS += nLower;
            }

            if (nLower == 0 || nUpper == 0) {
              // Lower hits lost due to no upper hits
              if (nLower > 0 && nUpper == 0) {
                if (isPS) {
                  pHitsLostNoUpper_PS += nLower;
                } else {
                  pHitsLostNoUpper_SS += nLower;
                }
              }
              continue;
            }

            // Count stubs and track per-lower-hit statistics
            for (uint32_t iLower = hitStart; iLower < upperStart; ++iLower) {
              auto const& lowerHit = hits[iLower];
              float phi_lower = alpaka::math::atan2(acc, lowerHit.yGlobal(), lowerHit.xGlobal());

              uint32_t stubsForThisLowerHit = 0;

              for (uint32_t iUpper = upperStart; iUpper < hitEnd; ++iUpper) {
                auto const& upperHit = hits[iUpper];

                // Same-sign local-y cut: reject if hits are on opposite strip sides
                if (lowerHit.yLocal() * upperHit.yLocal() < 0.f)
                  continue;

                float phi_upper = alpaka::math::atan2(acc, upperHit.yGlobal(), upperHit.xGlobal());

                float bend = isFlipped ? (phi_lower - phi_upper) : (phi_upper - phi_lower);
                if (bend > M_PI)
                  bend -= 2.0f * M_PI;
                if (bend < -M_PI)
                  bend += 2.0f * M_PI;

                if (alpaka::math::abs(acc, bend) < maxBend) {
                  stubsForThisLowerHit++;
                  totalStubs++;
                }
              }

              if (stubsForThisLowerHit > 0) {
                if (isPS) {
                  pHitsWithStub_PS++;
                  if (stubsForThisLowerHit > maxStubsPerPHit_PS) {
                    maxStubsPerPHit_PS = stubsForThisLowerHit;
                  }
                  if (stubsForThisLowerHit > 1) {
                    pHitsMultiStub_PS++;
                  }
                } else {
                  pHitsWithStub_SS++;
                  if (stubsForThisLowerHit > maxStubsPerPHit_SS) {
                    maxStubsPerPHit_SS = stubsForThisLowerHit;
                  }
                  if (stubsForThisLowerHit > 1) {
                    pHitsMultiStub_SS++;
                  }
                }
              } else {
                // Lost to bend cut
                if (isPS) {
                  pHitsLostBendCut_PS++;
                } else {
                  pHitsLostBendCut_SS++;
                }
              }
            }
          }

          // Print event-level summary
          printf("STUB_DIAG_EVENT type=PS totalPHits=%u pHitsWithStub=%u pHitsLost=%u "
                 "lostNoUpper=%u lostBendCut=%u totalStubs=%u maxStubsPerPHit=%u pHitsMultiStub=%u\n",
                 totalPHits_PS, pHitsWithStub_PS, pHitsLostNoUpper_PS + pHitsLostBendCut_PS,
                 pHitsLostNoUpper_PS, pHitsLostBendCut_PS, totalStubs, maxStubsPerPHit_PS, pHitsMultiStub_PS);
          printf("STUB_DIAG_EVENT type=SS totalPHits=%u pHitsWithStub=%u pHitsLost=%u "
                 "lostNoUpper=%u lostBendCut=%u totalStubs=%u maxStubsPerPHit=%u pHitsMultiStub=%u\n",
                 totalPHits_SS, pHitsWithStub_SS, pHitsLostNoUpper_SS + pHitsLostBendCut_SS,
                 pHitsLostNoUpper_SS, pHitsLostBendCut_SS, totalStubs, maxStubsPerPHit_SS, pHitsMultiStub_SS);
          printf("STUB_DIAG_EVENT type=ALL totalPHits=%u pHitsWithStub=%u pHitsLost=%u totalStubs=%u\n",
                 totalPHits_PS + totalPHits_SS,
                 pHitsWithStub_PS + pHitsWithStub_SS,
                 pHitsLostNoUpper_PS + pHitsLostBendCut_PS + pHitsLostNoUpper_SS + pHitsLostBendCut_SS,
                 totalStubs);
        }
      }
    };
#endif  // STUB_DIAGNOSTIC_COUNTERS

  }  // namespace otStubFormation

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_OTStubFormationKernels_h
