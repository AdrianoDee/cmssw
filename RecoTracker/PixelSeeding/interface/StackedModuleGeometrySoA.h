#ifndef RecoTracker_PixelSeeding_interface_StackedModuleGeometrySoA_h
#define RecoTracker_PixelSeeding_interface_StackedModuleGeometrySoA_h

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace reco {

  // Stacked Module Geometry SoA
  // Stores geometry information for Phase-2 OT stacked modules needed for stub formation
  // Each module consists of two closely-spaced sensors (inner and outer)
  //
  // Module types:
  //   - PS (Pixel-Strip): inner=macro-pixel, outer=strip, provides precise z
  //   - SS (Strip-Strip): both strips, no precise z
  //
  GENERATE_SOA_LAYOUT(
      StackedModuleGeometryLayout,
      // DetId as uint32_t
      SOA_COLUMN(uint32_t, detId),
      // Partner DetId as uint32_t
      SOA_COLUMN(uint32_t, partnerDetId),
      // Stacked DetId as uint32_t
      SOA_COLUMN(uint32_t, stackedDetId),
      // Original index in the TrackerGeometry detUnit vector
      SOA_COLUMN(uint32_t, geomIndex),
      // kind of module: 0-P, 1-SinPS, 2-SS
      SOA_COLUMN(uint8_t, moduleType),
      // Sensor separation: distance between inner and outer sensors (mm)
      // Typical values: 2.0-4.0mm for PS, 1.6-4.0mm for SS
      SOA_COLUMN(float, sensorSeparation),

      // Mean radius of the module (cm)
      SOA_COLUMN(float, meanRadius),

      // Tilt angle: angle of module axis from radial direction (radians)
      // IMPORTANT: Calculated using PHYSICAL inner→outer direction, NOT topological lower→upper:
      //   tiltAngle = atan2(dz_phys, dr_phys)
      //   For non-flipped: dz_phys = upper.z - lower.z, dr_phys = upper.r - lower.r
      //   For flipped:     dz_phys = lower.z - upper.z, dr_phys = lower.r - upper.r
      // This ensures dPhiDr has consistent sign regardless of flipped status.
      // Barrel flat: ~0 rad (physical inner→outer points radially outward)
      // Barrel tilted: non-zero (module tilted toward IP, sign matches z of module)
      // Endcap: ~±π/2 (module axis nearly parallel to z, sign matches z position)
      // Range: [-π/2, π/2]
      SOA_COLUMN(float, tiltAngle),

      // Precomputed sin and cos of tilt angle for effective dr calculation
      // Used in dr_effective = separation / (cosTilt + sinTilt * z/r)
      // Note: tiltAngle = atan2(dz, dr) measured from radial axis, so sinTilt > 0 for +z tilt
      SOA_COLUMN(float, sinTilt),
      SOA_COLUMN(float, cosTilt),

      // Module type: true if PS (pixel-strip), false if SS (strip-strip)
      SOA_COLUMN(bool, isPS),

      // Sensor orientation: true if "lower" sensor (in topology) is physically farther from beam
      // For barrel: true if lower sensor has larger r than upper sensor
      // For endcap: true if lower sensor has larger |z| than upper sensor
      // Used to determine which sensor is physically inner/outer for stub direction
      SOA_COLUMN(bool, isFlipped),

      // Detector location: true for barrel, false for endcap
      SOA_COLUMN(bool, isBarrel),

      // Module is flat (not tilted) - only meaningful for barrel
      // Flat means module axis is nearly radial (|cos(tiltAngle)| > cos(0.1) ≈ 0.995)
      // Corresponds to tilt angles near 0 (inner→outer nearly radial)
      SOA_COLUMN(bool, isFlat),

      // Forward endcap: true if z > 0 (forward), false if z < 0 (backward)
      // Only meaningful for endcap modules (isBarrel == false)
      SOA_COLUMN(bool, isFwdEndcap),

      // Layer number within OT (0-5, corresponding to TBPS/TBSS layers)
      SOA_COLUMN(uint8_t, layer),

      // Maximum allowed bend for pT threshold (radians)
      // Calculated as: maxBend = 0.3 * B * separation / minPt
      // Stubs with |bend| > maxBend are rejected
      SOA_COLUMN(float, maxBend),

      // Normalized vector from physical inner to physical outer sensor, in global coordinates
      // IMPORTANT: Uses PHYSICAL inner→outer direction, NOT topological lower→upper:
      //   For non-flipped: (upper - lower) / |upper - lower|
      //   For flipped:     (lower - upper) / |lower - upper|
      // This is consistent with tiltAngle and ensures sign-independent dPhiDr calculation.
      SOA_COLUMN(float, globalLowUpNormX),
      SOA_COLUMN(float, globalLowUpNormY),
      SOA_COLUMN(float, globalLowUpNormZ),

      // Let's also store the direction of the local x axis in global coordinates. Useful for parallax correction
      SOA_COLUMN(float, localXInGlobalX),
      SOA_COLUMN(float, localXInGlobalY),
      SOA_COLUMN(float, localXInGlobalZ))

  using StackedModuleGeometrySoA = StackedModuleGeometryLayout<>;
  using StackedModuleGeometryView = StackedModuleGeometrySoA::View;
  using StackedModuleGeometryConstView = StackedModuleGeometrySoA::ConstView;

}  // namespace reco

#endif  // RecoTracker_PixelSeeding_interface_StackedModuleGeometrySoA_h
