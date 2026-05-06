#ifndef RecoTracker_PixelSeeding_plugins_alpaka_OTStubFormationVectorHitStyleKernels_h
#define RecoTracker_PixelSeeding_plugins_alpaka_OTStubFormationVectorHitStyleKernels_h

// Note: STUB_DIAGNOSTIC_COUNTERS is defined in OTStubFormationKernels.h which is included below

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometrySoA.h"
#include "RecoTracker/PixelSeeding/plugins/alpaka/OTStubFormationKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace otStubFormationVectorHitStyle {
#if 0
    // Device function to compute parallax correction using local coordinate transformation
    //
    // This replicates the VectorHit parallax correction logic using tilt angle information
    // to approximate the full surface transformation from global to local coordinates.
    //
    // Physics Background:
    // -------------------
    // A charged particle from the primary vertex (origin) traverses two stacked sensors.
    // Due to the finite distance between sensors (separation), the hit positions appear
    // shifted relative to each other (parallax effect). The magnitude depends on:
    // 1. The particle trajectory angle relative to the sensor normal
    // 2. The sensor separation distance
    // 3. The sensor orientation (tilt angle)
    //
    // Original VectorHit Approach:
    // ---------------------------
    // GlobalVector gV = gPosLower - origin;                     // Vector from origin to lower cluster
    // LocalVector lV = geomDetUnit_low->surface().toLocal(gV);  // Full surface transformation
    // LocalVector lV_norm = lV / lV.z();                        // Normalize: z-component = 1
    // parallCorr = lV_norm.x() * separation;                    // x-offset due to parallax
    //
    // Our GPU Approximation:
    // ----------------------
    // We approximate the surface transformation using:
    // 1. Cylindrical coordinates (r, phi, z) at the sensor position
    // 2. Tilt angle theta (angle of sensor normal from radial direction)
    // 3. Rotation matrices to transform from global to local frame
    //
    // Coordinate Frame Definitions:
    // -----------------------------
    // Global frame: (x, y, z) - Standard CMS global coordinates
    //
    // Cylindrical frame at sensor position (r, phi, z):
    //   r = (cos(phi), sin(phi), 0) = (gx/r, gy/r, 0)       [radial, outward]
    //   phi = (-sin(phi), cos(phi), 0) = (-gy/r, gx/r, 0)   [tangent, counter-clockwise]
    //   z = (0, 0, 1)                                       [beam direction]
    //
    // Local sensor frame (depends on tilt angle theta):
    //   x_local: Precise measurement direction (pitch direction, perpendicular to strips)
    //   y_local: Strip direction (imprecise, along strips) = phi
    //   z_local: Normal to sensor surface (pointing from inner to outer sensor)
    //
    // Tilt Angle Convention (simplified pure geometry):
    //   theta = atan2(upper.z - lower.z, upper.r - lower.r)
    //   This is the angle from +r axis to the lower->upper vector, measured CCW in the (r,z) plane.
    //   - Flat barrel: dr > 0, dz ~= 0 -> theta ~= 0
    //   - Tilted barrel at +z: positive angle (vector tilted toward +z)
    //   - Tilted barrel at -z: negative angle (vector tilted toward -z)
    //   - Forward endcap (+z): dr ~= 0, dz > 0 -> theta ~= +90 deg
    //   - Backward endcap (-z): dr ~= 0, dz < 0 -> theta ~= -90 deg
    //
    // Local coordinate system in the (r, z) plane:
    //   z_local = (cos(theta), sin(theta)) in (r, z) basis  [along lower->upper direction]
    //   x_local = (-sin(theta), cos(theta)) in (r, z) basis [90 deg CCW from z_local]
    //
    // For flat barrel sensors (theta = 0):
    //   z_local = (1, 0) = radial direction
    //   x_local = (0, 1) = z direction (along beam)
    //
    // For forward endcap (theta = +90 deg):
    //   z_local = (0, 1) = +z direction
    //   x_local = (-1, 0) = -r direction (inward)
    //
    // Transformation Derivation:
    // --------------------------
    // Given global position vector gV = (gx, gy, gz) from origin to lower cluster:
    //
    // Step 1: Convert to cylindrical coordinates
    //   r = sqrt(gx^2 + gy^2)
    //   For position vector from origin: V_r = r, V_phi = 0, V_z = gz
    //
    // Step 2: Transform to local frame using dot products with local unit vectors
    //   lV_x = (r, gz) . x_local = (r, gz) . (-sin(theta), cos(theta))
    //        = -r*sin(theta) + gz*cos(theta)
    //
    //   lV_y = (r, gz) . y_local = 0  [position vector from origin has no tangent component]
    //
    //   lV_z = (r, gz) . z_local = (r, gz) . (cos(theta), sin(theta))
    //        = r*cos(theta) + gz*sin(theta)
    //
    // Step 3: Compute normalized direction (slope in local x-z plane)
    //   lV_norm_x = lV_x / lV_z
    //
    // Step 4: Compute parallax correction
    //   parallCorr = lV_norm_x * separation
    //
    // Special Cases:
    // --------------
    // Flat barrel (theta = 0), point at (r, z):
    //   lV_x = -r*0 + z*1 = z
    //   lV_z = r*1 + z*0 = r
    //   lV_norm_x = z/r = tan(lambda)  [lambda = angle from xy-plane]
    //   parallCorr = (z/r) * separation
    //
    // Forward endcap (theta = +90 deg), point at (r, z) with z > 0:
    //   lV_x = -r*1 + z*0 = -r
    //   lV_z = r*0 + z*1 = z
    //   lV_norm_x = -r/z
    //   parallCorr = (-r/z) * separation
    //
#endif
    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeParallaxCorrection(
        TAcc const& acc,
        float gx_lower,
        float gy_lower,
        float gz_lower,
        // NEW: unit vector (global) from center(lower) -> center(upper)
        float globalLowUpNormX,
        float globalLowUpNormY,
        float globalLowUpNormZ,
        // NEW: direction of LOWER local-x axis expressed in global coords (unit)
        float localXInGlobalX,
        float localXInGlobalY,
        float localXInGlobalZ,
        // still needed: magnitude of separation between sensors (in cm/mm as consistent with coords)
        float separation,
        bool debug = false,
        uint32_t moduleId = 0) {
      constexpr float eps = 1e-6f;

      // gV: vector from origin to LOWER hit/cluster (global)
      // (If you later want to use a non-zero beamspot, subtract that here instead of assuming origin=(0,0,0).)
      float gVx = gx_lower;
      float gVy = gy_lower;
      float gVz = gz_lower;

      // zLoc direction: along the stack from LOWER -> UPPER (global, assumed normalized already)
      float zLocX = globalLowUpNormX;
      float zLocY = globalLowUpNormY;
      float zLocZ = globalLowUpNormZ;

      // Safety: ensure zLoc is valid (normalized-ish and non-zero)
      float zLocN2 = zLocX * zLocX + zLocY * zLocY + zLocZ * zLocZ;
      if (zLocN2 < eps)
        return 0.0f;

      // If not perfectly normalized, normalize it
      if (alpaka::math::abs(acc, zLocN2 - 1.0f) > 1e-3f) {
        float invN = 1.0f / alpaka::math::sqrt(acc, zLocN2);
        zLocX *= invN;
        zLocY *= invN;
        zLocZ *= invN;
      }

      // xAxis (global) is the direction of LOWER local-x, but it may not be exactly orthogonal to zLoc.
      // Project it into the plane orthogonal to zLoc to get a proper in-plane x direction.
      float x0X = localXInGlobalX;
      float x0Y = localXInGlobalY;
      float x0Z = localXInGlobalZ;

      float x0N2 = x0X * x0X + x0Y * x0Y + x0Z * x0Z;
      if (x0N2 < eps)
        return 0.0f;

      if (alpaka::math::abs(acc, x0N2 - 1.0f) > 1e-3f) {
        float invN = 1.0f / alpaka::math::sqrt(acc, x0N2);
        x0X *= invN;
        x0Y *= invN;
        x0Z *= invN;
      }

      // xPerp = x0 - (x0*zLoc) zLoc
      float x0DotZ = x0X * zLocX + x0Y * zLocY + x0Z * zLocZ;
      float xPerpX = x0X - x0DotZ * zLocX;
      float xPerpY = x0Y - x0DotZ * zLocY;
      float xPerpZ = x0Z - x0DotZ * zLocZ;

      float xPerpN2 = xPerpX * xPerpX + xPerpY * xPerpY + xPerpZ * xPerpZ;
      if (xPerpN2 < eps) {
        // Degenerate: provided localX is (nearly) parallel to stack direction
        if (debug) {
          printf("    [ParallaxDebug Module %u] WARNING: xPerp too small (localX || stack), returning 0\n", moduleId);
        }
        return 0.0f;
      }

      float invXPerpN = 1.0f / alpaka::math::sqrt(acc, xPerpN2);
      float xLocX = xPerpX * invXPerpN;
      float xLocY = xPerpY * invXPerpN;
      float xLocZ = xPerpZ * invXPerpN;

      // Local components needed for VectorHit-style correction:
      //   lVx = gV * xLoc
      //   lVz = gV * zLoc
      float lVx = gVx * xLocX + gVy * xLocY + gVz * xLocZ;
      float lVz = gVx * zLocX + gVy * zLocY + gVz * zLocZ;

      if (debug) {
        printf("    [ParallaxDebug Module %u] Inputs: gPos=(%.4f, %.4f, %.4f) sep=%.4f\n",
               moduleId,
               gx_lower,
               gy_lower,
               gz_lower,
               separation);
        printf("    [ParallaxDebug Module %u] zLoc(L->U)=(%.6f, %.6f, %.6f)\n", moduleId, zLocX, zLocY, zLocZ);
        printf(
            "    [ParallaxDebug Module %u] x0(localX in global)=(%.6f, %.6f, %.6f)  xLoc(projected)=(%.6f, %.6f, "
            "%.6f)\n",
            moduleId,
            localXInGlobalX,
            localXInGlobalY,
            localXInGlobalZ,
            xLocX,
            xLocY,
            xLocZ);
        printf("    [ParallaxDebug Module %u] Local comps: lVx=%.6f lVz=%.6f\n", moduleId, lVx, lVz);
      }

      // Protect against near-parallel-to-plane direction (would blow up slope)
      if (alpaka::math::abs(acc, lVz) < eps) {
        if (debug) {
          printf("    [ParallaxDebug Module %u] WARNING: |lVz| too small, returning 0\n", moduleId);
        }
        return 0.0f;
      }

      // Normalized slope and correction
      float slope_x = lVx / lVz;
      float parallCorr = slope_x * separation;

      if (debug) {
        printf("    [ParallaxDebug Module %u] slope_x=%.6f parallCorr=%.6f\n", moduleId, slope_x, parallCorr);
      }

      if (!alpaka::math::isfinite(acc, parallCorr)) {
        if (debug) {
          printf("    [ParallaxDebug Module %u] WARNING: parallCorr not finite, returning 0\n", moduleId);
        }
        return 0.0f;
      }

      return parallCorr;  // along LOWER-sensor local_x, transporting LOWER->UPPER along globalLowUpNorm
    }

    // Tilt-angle based parallax correction (alternative to the full vector-based version above)
    // IMPORTANT: The tiltAngle is defined using the PHYSICAL inner->outer direction,
    // not the topological lower->upper direction. This ensures consistent behavior
    // regardless of whether the module is flipped.
    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float computeParallaxCorrection(TAcc const& acc,
                                                                   float gx_inner,
                                                                   float gy_inner,
                                                                   float gz_inner,
                                                                   float tiltAngle,
                                                                   float separation,
                                                                   bool debug = false,
                                                                   uint32_t moduleId = 0) {
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

      if (debug) {
        printf("    [ParallaxDebug Module %u] Inputs: gPos=(%.4f, %.4f, %.4f) gR=%.4f tilt=%.6f sep=%.4f\n",
               moduleId,
               gx_inner,
               gy_inner,
               gz_inner,
               gR_inner,
               tiltAngle,
               separation);
        printf("    [ParallaxDebug Module %u] Trig: cos(theta)=%.6f sin(theta)=%.6f\n", moduleId, cosTheta, sinTheta);
        printf("    [ParallaxDebug Module %u] Local: lV_x=%.6f lV_z=%.6f\n", moduleId, lV_x, lV_z);
      }

      // Check for valid local z-component (must be positive - pointing from inner to outer sensor)
      if (alpaka::math::abs(acc, lV_z) < 1e-6f) {
        if (debug) {
          printf("    [ParallaxDebug Module %u] WARNING: lV_z too small, returning 0\n", moduleId);
        }
        return 0.0f;
      }

      // Compute normalized slope in local x-z plane
      // This represents how much the track deviates in x per unit of z
      float lV_norm_x = lV_x / lV_z;

      // Parallax correction: x-displacement over sensor separation distance
      float parallCorr = lV_norm_x * separation;

      if (debug) {
        printf("    [ParallaxDebug Module %u] lV_norm_x=%.6f parallCorr=%.6f\n", moduleId, lV_norm_x, parallCorr);
      }

      // Sanity check for numerical stability
      if (!alpaka::math::isfinite(acc, parallCorr)) {
        if (debug) {
          printf("    [ParallaxDebug Module %u] WARNING: parallCorr not finite, returning 0\n", moduleId);
        }
        return 0.0f;
      }

      return parallCorr;
    }

    // Counting kernel: Count stubs per module using VectorHits cuts
    class CountStubsVectorHitStyleKernel {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    ::reco::OTRecHitsConstView hits,
                                    ::reco::OTHitModuleConstView moduleView,
                                    ::reco::StackedModuleGeometryConstView geometry,
                                    float const* maxWidthBarrelFlat,
                                    float const* maxWidthBarrelTilted,
                                    float const* maxWidthEndcap,
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
          // Debug for first thread
          // if (iModule == 0) {
          //   printf("CountStubsVectorHitStyle kernel running: nModules=%u\n", nModules);
          // }

          uint32_t count = 0;

          // Get hit ranges
          uint32_t hitStart = moduleView[iModule].moduleStart();
          uint32_t hitEnd = moduleView[iModule + 1].moduleStart();
          uint32_t upperStart = moduleView[iModule].upperSensorStart();

          uint32_t nLower = upperStart - hitStart;
          uint32_t nUpper = hitEnd - upperStart;

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
              printf("STUB_DIAG_VH module=%u type=PS totalPHits=%u pHitOnlyEntries=%u "
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
          float separation = geometry[iModule].sensorSeparation();  // in mm
          bool isBarrel = geometry[iModule].isBarrel();
          bool isFlat = geometry[iModule].isFlat();
          [[maybe_unused]] bool isFlipped = geometry[iModule].isFlipped();
          float tiltAngle = geometry[iModule].tiltAngle();  // in radians
          uint8_t layer = geometry[iModule].layer();
          float globalLowUpNormX = geometry[iModule].globalLowUpNormX();
          float globalLowUpNormY = geometry[iModule].globalLowUpNormY();
          float globalLowUpNormZ = geometry[iModule].globalLowUpNormZ();
          float localXInGlobalX = geometry[iModule].localXInGlobalX();
          float localXInGlobalY = geometry[iModule].localXInGlobalY();
          float localXInGlobalZ = geometry[iModule].localXInGlobalZ();

          // Get layer-dependent cut (separate cuts for flat and tilted barrel modules)
          float cut = isBarrel ? (isFlat ? maxWidthBarrelFlat[layer] : maxWidthBarrelTilted[layer]) : maxWidthEndcap[layer];

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
          uint32_t pHitsLostWidthCut = 0;
          uint32_t maxStubsPerPHit = 0;
#endif

          // Debug: print cut value for modules with hits
          // if (iModule < 13000 && (nLower > 0 || nUpper > 0)) {
          //   printf(
          //       "CountStubsVectorHitStyle: Module %u: isBarrel=%d, layer=%u, cut=%.6f, tiltAngle=%.4f, nLower=%u, "
          //       "nUpper=%u\n",
          //       iModule,
          //       isBarrel,
          //       layer,
          //       cut,
          //       tiltAngle,
          //       nLower,
          //       nUpper);
          // }

          // Count valid combinations
          for (uint32_t iLower = hitStart; iLower < upperStart; ++iLower) {
            auto const& lowerHit = hits[iLower];

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

              // Use software lower sensor as reference (matches original VectorHit)
              // With tilt angle transformation, this correctly handles all module orientations
              float lx_lower = lowerHit.xLocal();
              float lx_upper = upperHit.xLocal();

              // Enable detailed debug for first few modules with hits
              // bool enableDebug = (iModule == 6158 || iModule == 6175) && (iLower == hitStart) && (iUpper == upperStart);
              // bool enableDebug = (iLower == hitStart) && (iUpper == upperStart);
              constexpr bool enableDebug = false;

              if (enableDebug) {
                printf("\n  [Module %u] === VectorHitStyle Parallax Calculation ===\n", iModule);
                printf("  [Module %u] Lower hit: local=(%.6f, %.6f) global=(%.4f, %.4f, %.4f)\n",
                       iModule,
                       lowerHit.xLocal(),
                       lowerHit.yLocal(),
                       lowerHit.xGlobal(),
                       lowerHit.yGlobal(),
                       lowerHit.zGlobal());
                printf("  [Module %u] Upper hit: local=(%.6f, %.6f) global=(%.4f, %.4f, %.4f)\n",
                       iModule,
                       upperHit.xLocal(),
                       upperHit.yLocal(),
                       upperHit.xGlobal(),
                       upperHit.yGlobal(),
                       upperHit.zGlobal());
                printf("  [Module %u] Geometry: separation=%.4f mm (%.4f cm), tiltAngle=%.6f rad (%.2f deg)\n",
                       iModule,
                       separation,
                       separation * 0.1f,
                       tiltAngle,
                       tiltAngle * 180.0f / M_PI);
              }

              // Parallax correction using proper local coordinate transformation
              // Replicates VectorHit logic: transform global position vector to local frame,
              // compute x/z slope, multiply by separation
              //
              // IMPORTANT: The globalLowUpNorm vectors are stored as the PHYSICAL inner->outer
              // direction. For the width calculation (VectorHit style), we need the TOPOLOGICAL
              // lower->upper direction, which is the local-z axis of the lower sensor.
              // For flipped modules: physical inner->outer = upper->lower (topological)
              // So we need to flip the sign to get lower->upper.
              float flipSign = isFlipped ? -1.0f : 1.0f;
              float topoLowUpNormX = flipSign * globalLowUpNormX;
              float topoLowUpNormY = flipSign * globalLowUpNormY;
              float topoLowUpNormZ = flipSign * globalLowUpNormZ;

              // separation is in mm, convert to cm for consistency with cut units
              float pC = computeParallaxCorrection(acc,
                                                   lowerHit.xGlobal(),
                                                   lowerHit.yGlobal(),
                                                   lowerHit.zGlobal(),
                                                   topoLowUpNormX,
                                                   topoLowUpNormY,
                                                   topoLowUpNormZ,
                                                   localXInGlobalX,
                                                   localXInGlobalY,
                                                   localXInGlobalZ,
                                                   separation * 0.1f,  // mm to cm
                                                   enableDebug,
                                                   iModule);

              // Apply parallax correction to local positions (matches original VectorHit logic)
              // The correction is applied to bring the upper cluster position into alignment
              // with where it would appear if projected back through the lower cluster
              float lpos_lower_corr = 0.0f;
              float lpos_upper_corr = 0.0f;

              if (lx_upper > lx_lower) {
                if (lx_upper > 0) {
                  lpos_lower_corr = lx_lower;
                  lpos_upper_corr = lx_upper - alpaka::math::abs(acc, pC);
                } else {
                  lpos_lower_corr = lx_lower + alpaka::math::abs(acc, pC);
                  lpos_upper_corr = lx_upper;
                }
              } else if (lx_upper < lx_lower) {
                if (lx_upper > 0) {
                  lpos_lower_corr = lx_lower - alpaka::math::abs(acc, pC);
                  lpos_upper_corr = lx_upper;
                } else {
                  lpos_lower_corr = lx_lower;
                  lpos_upper_corr = lx_upper + alpaka::math::abs(acc, pC);
                }
              } else {
                if (lx_upper > 0) {
                  lpos_lower_corr = lx_lower;
                  lpos_upper_corr = lx_upper - alpaka::math::abs(acc, pC);
                } else {
                  lpos_lower_corr = lx_lower;
                  lpos_upper_corr = lx_upper + alpaka::math::abs(acc, pC);
                }
              }

              // Width: difference between corrected positions (in local x coordinate, units: cm)
              float width = lpos_lower_corr - lpos_upper_corr;

              // Debug: print correction application and final result
              if (enableDebug) {
                printf("  [Module %u] Correction application:\n", iModule);
                printf("  [Module %u]   lx_lower=%.6f, lx_upper=%.6f\n", iModule, lx_lower, lx_upper);
                printf("  [Module %u]   Comparison: lx_upper %s lx_lower, lx_upper %s 0\n",
                       iModule,
                       (lx_upper > lx_lower) ? ">" : ((lx_upper < lx_lower) ? "<" : "=="),
                       (lx_upper > 0) ? ">" : ((lx_upper < 0) ? "<" : "=="));
                printf("  [Module %u]   pC=%.6f, |pC|=%.6f\n", iModule, pC, alpaka::math::abs(acc, pC));
                printf("  [Module %u]   lpos_lower_corr=%.6f, lpos_upper_corr=%.6f\n",
                       iModule,
                       lpos_lower_corr,
                       lpos_upper_corr);
                printf("  [Module %u] Final: width=%.6f cm, cut=%.6f cm, pass=%d\n",
                       iModule,
                       width,
                       cut,
                       (alpaka::math::abs(acc, width) < cut) ? 1 : 0);
                printf("  [Module %u] ==========================================\n\n", iModule);
              

              // Borderline stub diagnostic: detect stubs near the width cut boundary
              // These are the stubs most susceptible to FP precision differences between CPU and GPU
              
                float absWidth = alpaka::math::abs(acc, width);
                float margin = cut - absWidth;
                if (margin > 0.0f && margin < 1e-4f) {
                  printf("BORDERLINE_STUB module=%u iLower=%u iUpper=%u width=%.8f cut=%.8f margin=%.2e\n",
                         iModule, iLower, iUpper, width, cut, margin);
                }
              }

              if (alpaka::math::abs(acc, width) < cut) {
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
              // This lower hit (P-hit for PS) formed no stubs - all pairs failed width cut
              pHitsLostWidthCut++;
            }
#endif
          }

          // Write to offset position for exclusive scan
          stubCounts[iModule + 1] = count;

#ifdef STUB_DIAGNOSTIC_COUNTERS
          // Print per-module diagnostic for modules with lower hits (P-hits for PS)
          if (nLower > 0) {
            const char* modType = isPS ? "PS" : "SS";
            printf("STUB_DIAG_VH module=%u type=%s totalPHits=%u stubsFormed=%u pHitsWithStub=%u pHitsLost=%u "
                   "lostReason=widthCut maxStubsPerPHit=%u\n",
                   iModule, modType, nLower, count, pHitsWithStub, pHitsLostWidthCut, maxStubsPerPHit);
          }
#endif

          // Debug: print count for modules with stubs
          // if (count > 0 && iModule < 13000) {
          //   printf("CountStubsVectorHitStyle: Module %u counted %u stubs\n", iModule, count);
          // }
        }
      }
    };

    // Formation kernel for VectorHits style
    class FormStubsVectorHitStyleKernel {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    ::reco::OTRecHitsConstView hits,
                                    ::reco::OTHitModuleConstView moduleView,
                                    ::reco::StackedModuleGeometryConstView geometry,
                                    float const* maxWidthBarrelFlat,
                                    float const* maxWidthBarrelTilted,
                                    float const* maxWidthEndcap,
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
                                    ::ALPAKA_ACCELERATOR_NAMESPACE::otStubFormation::CAModuleOffsets caOffsets) const {
        // One thread per module
        for (auto iModule : cms::alpakatools::uniform_elements(acc, nModules)) {
          // Get hit ranges
          uint32_t hitStart = moduleView[iModule].moduleStart();
          uint32_t hitEnd = moduleView[iModule + 1].moduleStart();
          uint32_t upperStart = moduleView[iModule].upperSensorStart();

          uint32_t nLower = upperStart - hitStart;
          uint32_t nUpper = hitEnd - upperStart;

          // Get geometry parameters
          [[maybe_unused]] float separation = geometry[iModule].sensorSeparation();  // in mm
          uint8_t moduleType = geometry[iModule].moduleType();
          bool isPS = (moduleType == 0 || moduleType == 1);
          bool isBarrel = geometry[iModule].isBarrel();
          bool isFlat = geometry[iModule].isFlat();
          [[maybe_unused]] bool isFlipped = geometry[iModule].isFlipped();
          [[maybe_unused]] float tiltAngle = geometry[iModule].tiltAngle();  // in radians
          uint8_t layer = geometry[iModule].layer();
          float globalLowUpNormX = geometry[iModule].globalLowUpNormX();
          float globalLowUpNormY = geometry[iModule].globalLowUpNormY();
          float globalLowUpNormZ = geometry[iModule].globalLowUpNormZ();
          float localXInGlobalX = geometry[iModule].localXInGlobalX();
          float localXInGlobalY = geometry[iModule].localXInGlobalY();
          float localXInGlobalZ = geometry[iModule].localXInGlobalZ();

          float cut = isBarrel ? (isFlat ? maxWidthBarrelFlat[layer] : maxWidthBarrelTilted[layer]) : maxWidthEndcap[layer];

          // Per-layer cluster size cuts
          int32_t maxCSDiff = isBarrel ? (isFlat ? barrelFlatMaxCSDiff[layer] : barrelTiltedMaxCSDiff[layer])
                                       : endcapMaxCSDiff[layer];
          int32_t maxCS = isBarrel ? (isFlat ? barrelFlatMaxCS[layer] : barrelTiltedMaxCS[layer])
                                   : endcapMaxCS[layer];
          int32_t maxCSSum = isBarrel ? (isFlat ? barrelFlatMaxCSSum[layer] : barrelTiltedMaxCSSum[layer])
                                      : endcapMaxCSSum[layer];

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

          // Form stubs
          for (uint32_t iLower = hitStart; iLower < upperStart; ++iLower) {
            auto const& lowerHit = hits[iLower];

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

              // Determine physical inner/outer for stub properties (position, hit indices)
              // Note: We avoid using ternary operator with references (auto const& = cond ? a : b)
              // as this pattern can cause issues in GPU device code. Instead, we select indices
              // and access the hits directly when needed.
              uint32_t innerIdx = isFlipped ? iUpper : iLower;
              uint32_t outerIdx = isFlipped ? iLower : iUpper;

              // Use software lower/upper for width calculation (matches original VectorHit)
              float lx_lower = lowerHit.xLocal();
              float lx_upper = upperHit.xLocal();

              // Enable detailed debug for first stub of specific modules
              // bool enableDebug = (iModule == 6158 || iModule == 6175) && (iLower == hitStart) && (iUpper == upperStart);
              // bool enableDebug = (iLower == hitStart) && (iUpper == upperStart);
              constexpr bool enableDebug = false;

              // Parallax correction using proper local coordinate transformation
              // Replicates VectorHit logic: transform global position vector to local frame,
              // compute x/z slope, multiply by separation
              //
              // IMPORTANT: The globalLowUpNorm vectors are stored as the PHYSICAL inner->outer
              // direction. For the width calculation (VectorHit style), we need the TOPOLOGICAL
              // lower->upper direction, which is the local-z axis of the lower sensor.
              // For flipped modules: physical inner->outer = upper->lower (topological)
              // So we need to flip the sign to get lower->upper.
              float flipSign = isFlipped ? -1.0f : 1.0f;
              float topoLowUpNormX = flipSign * globalLowUpNormX;
              float topoLowUpNormY = flipSign * globalLowUpNormY;
              float topoLowUpNormZ = flipSign * globalLowUpNormZ;

              // separation is in mm, convert to cm for consistency with cut units
              float pC = computeParallaxCorrection(acc,
                                                   lowerHit.xGlobal(),
                                                   lowerHit.yGlobal(),
                                                   lowerHit.zGlobal(),
                                                   topoLowUpNormX,
                                                   topoLowUpNormY,
                                                   topoLowUpNormZ,
                                                   localXInGlobalX,
                                                   localXInGlobalY,
                                                   localXInGlobalZ,
                                                   separation * 0.1f,  // mm to cm
                                                   enableDebug,
                                                   iModule);

              // Apply parallax correction to local positions (matches original VectorHit logic)
              float lpos_lower_corr = 0.0f;
              float lpos_upper_corr = 0.0f;

              if (lx_upper > lx_lower) {
                if (lx_upper > 0) {
                  lpos_lower_corr = lx_lower;
                  lpos_upper_corr = lx_upper - alpaka::math::abs(acc, pC);
                } else {
                  lpos_lower_corr = lx_lower + alpaka::math::abs(acc, pC);
                  lpos_upper_corr = lx_upper;
                }
              } else if (lx_upper < lx_lower) {
                if (lx_upper > 0) {
                  lpos_lower_corr = lx_lower - alpaka::math::abs(acc, pC);
                  lpos_upper_corr = lx_upper;
                } else {
                  lpos_lower_corr = lx_lower;
                  lpos_upper_corr = lx_upper + alpaka::math::abs(acc, pC);
                }
              } else {
                if (lx_upper > 0) {
                  lpos_lower_corr = lx_lower;
                  lpos_upper_corr = lx_upper - alpaka::math::abs(acc, pC);
                } else {
                  lpos_lower_corr = lx_lower;
                  lpos_upper_corr = lx_upper + alpaka::math::abs(acc, pC);
                }
              }

              // Width: difference between corrected positions (in local x coordinate, units: cm)
              float width = lpos_lower_corr - lpos_upper_corr;

              if (alpaka::math::abs(acc, width) >= cut)
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

              // Valid stub
              // For PS modules, use the PIXEL hit position (better spatial precision)
              // In CMSSW convention, the pixel sensor is always the "lower" sensor in the
              // stack reference frame for PS modules, so lowerHit is the pixel hit.
              // For SS modules, use the physically-inner sensor (closer to the beam: smaller
              // r in barrel, smaller |z| in endcap; resolved via `innerIdx` and `isFlipped`).
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
                printf("PS_STUB_VH module=%u stubIdx=%u lowerHitIdx=%u: "
                       "x=%.6f y=%.6f z=%.6f r=%.4f "
                       "iphi_lower=%d xerrLocal=%.4e yerrLocal=%.4e sensorDetId=%u\n",
                       iModule, stubIdx, iLower,
                       xg, yg, zg, stub_r,
                       lowerHit.iphi(), lowerHit.xerrLocal(), lowerHit.yerrLocal(), lowerHit.sensorDetId());
#endif
              } else {
                xg = hits[innerIdx].xGlobal();
                yg = hits[innerIdx].yGlobal();
                zg = hits[innerIdx].zGlobal();
                stubs[stubIdx].xGlobal() = xg;
                stubs[stubIdx].yGlobal() = yg;
                stubs[stubIdx].zGlobal() = zg;
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

              // Apply parallax correction to dphi
              // Only apply for tilted barrel and endcap modules
              // For flat barrel modules, local-x is aligned with z, not phi, so no dphi correction
              //
              // IMPORTANT: For dPhiDr, we need to compute parallax correction using the PHYSICAL
              // inner hit position (which may be upperHit for flipped modules), not the topological
              // lowerHit used for width calculation. This is because dPhiDr is computed in the
              // physical inner->outer reference frame.
              float dphi;
              bool applyDphiParallax = !isFlat || !isBarrel;  // tilted barrel or endcap
              if (applyDphiParallax) {
                // Compute parallax correction using physical inner hit position
                // The globalLowUpNorm vectors now represent the physical inner->outer direction
                float zi = hits[innerIdx].zGlobal();
                float pC_dphi = computeParallaxCorrection(acc,
                                                          xi,
                                                          yi,
                                                          zi,
                                                          globalLowUpNormX,
                                                          globalLowUpNormY,
                                                          globalLowUpNormZ,
                                                          localXInGlobalX,
                                                          localXInGlobalY,
                                                          localXInGlobalZ,
                                                          separation * 0.1f,  // mm to cm
                                                          enableDebug,
                                                          iModule);
                float dphi_parallax = (ri > 1e-6f) ? pC_dphi / ri : 0.0f;
                dphi = dphi_raw - dphi_parallax;
              } else {
                // Flat barrel: no dphi parallax correction
                dphi = dphi_raw;
              }

              float dr_geometric = ro - ri;
              // Use stub position from local variables (already computed above)
              float r_stub = rg;
              float z_stub = zg;

              // Get tilt parameters for effective dr calculation
              float sinTilt = geometry[iModule].sinTilt();
              float cosTilt = geometry[iModule].cosTilt();
              float separation_cm = geometry[iModule].sensorSeparation() * 0.1f;  // mm to cm

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

              // Bend: angular deviation from straight track, computed from width for selection
              // (kept for compatibility, but dPhiDr is now computed from global positions)
              float bend = (r_stub > 0.0f) ? width / r_stub : 0.0f;
              stubs[stubIdx].bend() = bend;
              stubs[stubIdx].ptEst() = ptEst;
              // Detector index - map to CA module index
              // Since StackedModuleGeometry is sorted in CA order (barrel by layer, backward, forward),
              // the CA module index is simply nPixelModules + geometry index
              uint32_t caModuleIndex = caOffsets.nPixelModules + iModule;
              stubs[stubIdx].detectorIndex() = static_cast<uint16_t>(caModuleIndex);
              stubs[stubIdx].innerHitIdx() = innerIdx;
              stubs[stubIdx].outerHitIdx() = outerIdx;
              stubs[stubIdx].stubType() = isPS ? ::reco::StubType::PS : ::reco::StubType::SS;
              stubs[stubIdx].flags() = ::reco::StubFlags::makeFlags(isBarrel, isFlat, true, layer);

              // P-hit group ID: use unique stubIdx so each stub is treated independently
              // (set to iLower to re-enable grouping of stubs sharing the same P-hit)
              stubs[stubIdx].pHitGroupId() = stubIdx;

              if (enableDebug) {
                printf("  [FormStub Module %u] Created stub at index %u:\n", iModule, stubIdx);
                printf("    Position: global=(%.4f, %.4f, %.4f) r=%.4f local=(%.6f, %.6f)\n",
                       stubs[stubIdx].xGlobal(),
                       stubs[stubIdx].yGlobal(),
                       stubs[stubIdx].zGlobal(),
                       stubs[stubIdx].rGlobal(),
                       stubs[stubIdx].xLocal(),
                       stubs[stubIdx].yLocal());
                printf("    Width (selection): %.6f cm, bend=%.6f rad, dPhiDr=%.6f rad/cm, dPhiDrErr=%.6f\n",
                       width,
                       stubs[stubIdx].bend(),
                       stubs[stubIdx].dPhiDr(),
                       stubs[stubIdx].dPhiDrError());
                printf("    Hits: inner=%u outer=%u, type=%s, detIdx=%u\n",
                       stubs[stubIdx].innerHitIdx(),
                       stubs[stubIdx].outerHitIdx(),
                       isPS ? "PS" : "SS",
                       stubs[stubIdx].detectorIndex());
                printf("    isFlipped=%d -> innerIdx=%u outerIdx=%u\n", isFlipped, innerIdx, outerIdx);
              }

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
    // Diagnostic kernel: Compute event-level summary statistics for stub formation (VectorHit style)
    // This kernel runs after stub formation to aggregate statistics across all modules
    class DiagnosticSummaryKernel {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    ::reco::OTRecHitsConstView hits,
                                    ::reco::OTHitModuleConstView moduleView,
                                    ::reco::StackedModuleGeometryConstView geometry,
                                    float const* maxWidthBarrelFlat,
                                    float const* maxWidthBarrelTilted,
                                    float const* maxWidthEndcap,
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
          uint32_t pHitsLostWidthCut_PS = 0;
          uint32_t pHitsLostWidthCut_SS = 0;
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
            bool isBarrel = geometry[iModule].isBarrel();
            bool isFlipped = geometry[iModule].isFlipped();
            uint8_t layer = geometry[iModule].layer();
            float separation = geometry[iModule].sensorSeparation();
            float globalLowUpNormX = geometry[iModule].globalLowUpNormX();
            float globalLowUpNormY = geometry[iModule].globalLowUpNormY();
            float globalLowUpNormZ = geometry[iModule].globalLowUpNormZ();
            float localXInGlobalX = geometry[iModule].localXInGlobalX();
            float localXInGlobalY = geometry[iModule].localXInGlobalY();
            float localXInGlobalZ = geometry[iModule].localXInGlobalZ();
            bool isFlat = geometry[iModule].isFlat();
            float cut = isBarrel ? (isFlat ? maxWidthBarrelFlat[layer] : maxWidthBarrelTilted[layer]) : maxWidthEndcap[layer];

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
              uint32_t stubsForThisLowerHit = 0;

              for (uint32_t iUpper = upperStart; iUpper < hitEnd; ++iUpper) {
                auto const& upperHit = hits[iUpper];

                // Same-sign local-y cut: reject if hits are on opposite strip sides
                if (lowerHit.yLocal() * upperHit.yLocal() < 0.f)
                  continue;

                // VectorHit style width calculation (simplified for summary)
                float lx_lower = lowerHit.xLocal();
                float lx_upper = upperHit.xLocal();

                // Parallax correction
                float flipSign = isFlipped ? -1.0f : 1.0f;
                float topoLowUpNormX = flipSign * globalLowUpNormX;
                float topoLowUpNormY = flipSign * globalLowUpNormY;
                float topoLowUpNormZ = flipSign * globalLowUpNormZ;

                float pC = computeParallaxCorrection(acc,
                                                     lowerHit.xGlobal(),
                                                     lowerHit.yGlobal(),
                                                     lowerHit.zGlobal(),
                                                     topoLowUpNormX,
                                                     topoLowUpNormY,
                                                     topoLowUpNormZ,
                                                     localXInGlobalX,
                                                     localXInGlobalY,
                                                     localXInGlobalZ,
                                                     separation * 0.1f);

                // Apply parallax correction
                float lpos_lower_corr = 0.0f;
                float lpos_upper_corr = 0.0f;

                if (lx_upper > lx_lower) {
                  if (lx_upper > 0) {
                    lpos_lower_corr = lx_lower;
                    lpos_upper_corr = lx_upper - alpaka::math::abs(acc, pC);
                  } else {
                    lpos_lower_corr = lx_lower + alpaka::math::abs(acc, pC);
                    lpos_upper_corr = lx_upper;
                  }
                } else if (lx_upper < lx_lower) {
                  if (lx_upper > 0) {
                    lpos_lower_corr = lx_lower - alpaka::math::abs(acc, pC);
                    lpos_upper_corr = lx_upper;
                  } else {
                    lpos_lower_corr = lx_lower;
                    lpos_upper_corr = lx_upper + alpaka::math::abs(acc, pC);
                  }
                } else {
                  if (lx_upper > 0) {
                    lpos_lower_corr = lx_lower;
                    lpos_upper_corr = lx_upper - alpaka::math::abs(acc, pC);
                  } else {
                    lpos_lower_corr = lx_lower;
                    lpos_upper_corr = lx_upper + alpaka::math::abs(acc, pC);
                  }
                }

                float width = lpos_lower_corr - lpos_upper_corr;

                if (alpaka::math::abs(acc, width) < cut) {
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
                // Lost to width cut
                if (isPS) {
                  pHitsLostWidthCut_PS++;
                } else {
                  pHitsLostWidthCut_SS++;
                }
              }
            }
          }

          // Print event-level summary
          printf("STUB_DIAG_EVENT_VH type=PS totalPHits=%u pHitsWithStub=%u pHitsLost=%u "
                 "lostNoUpper=%u lostWidthCut=%u totalStubs=%u maxStubsPerPHit=%u pHitsMultiStub=%u\n",
                 totalPHits_PS, pHitsWithStub_PS, pHitsLostNoUpper_PS + pHitsLostWidthCut_PS,
                 pHitsLostNoUpper_PS, pHitsLostWidthCut_PS, totalStubs, maxStubsPerPHit_PS, pHitsMultiStub_PS);
          printf("STUB_DIAG_EVENT_VH type=SS totalPHits=%u pHitsWithStub=%u pHitsLost=%u "
                 "lostNoUpper=%u lostWidthCut=%u totalStubs=%u maxStubsPerPHit=%u pHitsMultiStub=%u\n",
                 totalPHits_SS, pHitsWithStub_SS, pHitsLostNoUpper_SS + pHitsLostWidthCut_SS,
                 pHitsLostNoUpper_SS, pHitsLostWidthCut_SS, totalStubs, maxStubsPerPHit_SS, pHitsMultiStub_SS);
          printf("STUB_DIAG_EVENT_VH type=ALL totalPHits=%u pHitsWithStub=%u pHitsLost=%u totalStubs=%u\n",
                 totalPHits_PS + totalPHits_SS,
                 pHitsWithStub_PS + pHitsWithStub_SS,
                 pHitsLostNoUpper_PS + pHitsLostWidthCut_PS + pHitsLostNoUpper_SS + pHitsLostWidthCut_SS,
                 totalStubs);
        }
      }
    };
#endif  // STUB_DIAGNOSTIC_COUNTERS

  }  // namespace otStubFormationVectorHitStyle

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_OTStubFormationVectorHitStyleKernels_h
