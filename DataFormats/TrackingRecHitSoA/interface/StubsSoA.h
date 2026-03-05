#ifndef DataFormats_TrackingRecHitSoA_interface_StubsSoA_h
#define DataFormats_TrackingRecHitSoA_interface_StubsSoA_h

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"

namespace reco {

  // Main stub SoA layout
  // Contains all information needed for tracking with stubs from stacked OT sensors
  //
  // Stubs are formed from hit pairs on stacked sensors (2-5mm separation).
  // They provide:
  //   - Position (from inner/P-side sensor for PS modules)
  //   - Direction measurement (dPhi/dr from the two sensors)
  //   - pT discrimination (bend cut rejects low-pT tracks)
  //
  GENERATE_SOA_LAYOUT(StubsLayout,
                      // Global position (from P-side for PS, averaged for SS)
                      SOA_COLUMN(float, xGlobal),
                      SOA_COLUMN(float, yGlobal),
                      SOA_COLUMN(float, zGlobal),
                      SOA_COLUMN(float, rGlobal),
                      SOA_COLUMN(int16_t, iphi),

                      // Local position on the inner (P-side for PS) sensor
                      SOA_COLUMN(float, xLocal),
                      SOA_COLUMN(float, yLocal),

                      // Position errors
                      SOA_COLUMN(float, xerrLocal),
                      SOA_COLUMN(float, yerrLocal),

                      // Stub-specific: direction measurement from stacked sensors
                      // dPhiDr = (phi_outer - phi_inner) / (r_outer - r_inner)
                      // This encodes the track curvature information for CA validation
                      SOA_COLUMN(float, dPhiDr),

                      // Error on direction measurement
                      SOA_COLUMN(float, dPhiDrError),

                      // Raw bend: delta phi between inner and outer sensor hits (radians)
                      // Used for pT threshold cut validation
                      SOA_COLUMN(float, bend),

                      // PT estimate
                      SOA_COLUMN(float, ptEst),

                      // Module index for CA layer assignment
                      // Pre-adjusted to be: modulesInPixel + stubModuleOffset
                      // This allows unified indexing with pixel hits in the CA
                      SOA_COLUMN(uint16_t, detectorIndex),

                      // Original hit indices for track fitting
                      // These point to the original OT RecHit collection
                      // After stub-based tracking, fits use the original hits
                      SOA_COLUMN(uint32_t, innerHitIdx),
                      SOA_COLUMN(uint32_t, outerHitIdx),

                      // Stub type: 0 = PS (pixel-strip, precise z), 1 = SS (strip-strip, no z precision)
                      SOA_COLUMN(uint8_t, stubType),

                      // Flags stored as a single byte for simplicity
                      // Bit 0: isBarrel (1) vs endcap (0)
                      // Bit 1: isFlat (1) vs tilted (0) - only meaningful for barrel
                      // Bit 2: isValid (1) - stub passed all quality cuts
                      // Bits 3-5: layer number within OT (0-5)
                      // Bits 6-7: reserved
                      SOA_COLUMN(uint8_t, flags),

                      // P-hit group ID: For PS modules, this is the index of the P-hit (lower hit)
                      // that was used to form this stub. Stubs sharing the same P-hit will have
                      // the same pHitGroupId. This is used for duplicate stub handling in the CA
                      // to ensure that only one stub per P-hit survives fishbone cleaning.
                      // For SS modules, this is set to the lower hit index as well.
                      // Value UINT32_MAX indicates unset/invalid (e.g., for pixel hits).
                      SOA_COLUMN(uint32_t, pHitGroupId),

                      // Scalars
                      // Offset to first stub in the unified hit collection (after pixel hits)
                      SOA_SCALAR(int32_t, offsetStubs));

  // Module start array layout - tracks cumulative stub count per module
  // Same pattern as HitModulesLayout for TrackingRecHits
  // Size is nModules + 1, where the last element holds the total stub count
  GENERATE_SOA_LAYOUT(StubModulesLayout, SOA_COLUMN(uint32_t, moduleStart));

  GENERATE_SOA_BLOCKS(StubBlocksLayout,
                      SOA_BLOCK(stubs, StubsLayout),
                      SOA_BLOCK(stubModules, StubModulesLayout))

  // Type aliases for convenient usage
  using StubsSoA = StubsLayout<>;
  using StubsView = StubsSoA::View;
  using StubsConstView = StubsSoA::ConstView;

  using StubModuleSoA = StubModulesLayout<>;
  using StubModuleView = StubModuleSoA::View;
  using StubModuleConstView = StubModuleSoA::ConstView;

  using StubBlocksSoA = StubBlocksLayout<>;
  using StubBlocksSoAView = StubBlocksSoA::View;
  using StubBlocksSoAConstView = StubBlocksSoA::ConstView;

  // Constants for stub types
  namespace StubType {
    constexpr uint8_t PS = 0;       // Pixel-Strip module (precise z from pixel sensor)
    constexpr uint8_t SS = 1;       // Strip-Strip module (no z precision)
    constexpr uint8_t PHitOnly = 2; // P-hit only (no matching S-hit, no bend measurement)
  }  // namespace StubType

  // Helper functions for flags manipulation
  // These are constexpr and can be used on both host and device
  namespace StubFlags {
    constexpr uint8_t isBarrelMask = 0x01;  // Bit 0
    constexpr uint8_t isFlatMask = 0x02;    // Bit 1
    constexpr uint8_t isValidMask = 0x04;   // Bit 2
    constexpr uint8_t layerMask = 0x38;     // Bits 3-5
    constexpr uint8_t layerShift = 3;

    inline constexpr bool isBarrel(uint8_t flags) { return (flags & isBarrelMask) != 0; }
    inline constexpr bool isFlat(uint8_t flags) { return (flags & isFlatMask) != 0; }
    inline constexpr bool isValid(uint8_t flags) { return (flags & isValidMask) != 0; }
    inline constexpr uint8_t layer(uint8_t flags) { return (flags & layerMask) >> layerShift; }

    inline constexpr uint8_t makeFlags(bool barrel, bool flat, bool valid, uint8_t layerNum) {
      return (barrel ? isBarrelMask : 0) | (flat ? isFlatMask : 0) | (valid ? isValidMask : 0) |
             ((layerNum << layerShift) & layerMask);
    }
  }  // namespace StubFlags

}  // namespace reco

#endif  // DataFormats_TrackingRecHitSoA_interface_StubsSoA_h
