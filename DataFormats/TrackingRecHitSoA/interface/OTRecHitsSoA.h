#ifndef DataFormats_TrackingRecHitSoA_interface_OTRecHitsSoA_h
#define DataFormats_TrackingRecHitSoA_interface_OTRecHitsSoA_h

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"

namespace reco {

  // OT RecHits SoA
  // Stores hits from Phase-2 Outer Tracker stacked modules for stub formation
  // Hits are organized by stacked module, with lower sensor hits followed by upper sensor hits
  //
  // Layout enables efficient stub formation:
  //   - Hits from same stack are consecutive in memory
  //   - Sensor metadata allows pairing lower and upper hits
  //   - Module start array provides quick access to each stack's hits
  //
  GENERATE_SOA_LAYOUT(OTRecHitsLayout,
                      // Local position (cm)
                      SOA_COLUMN(float, xLocal),
                      SOA_COLUMN(float, yLocal),
                      // Local position errors (cm)
                      SOA_COLUMN(float, xerrLocal),
                      SOA_COLUMN(float, yerrLocal),
                      // Global position (cm, beam-spot corrected)
                      SOA_COLUMN(float, xGlobal),
                      SOA_COLUMN(float, yGlobal),
                      SOA_COLUMN(float, zGlobal),
                      SOA_COLUMN(float, rGlobal),
                      // Global error covariance matrix (6 elements: xx, xy, yy, xz, yz, zz)
                      // Pre-computed from local errors and detector frame
                      SOA_COLUMN(float, xGlobalErr),   // sigma_xx
                      SOA_COLUMN(float, xyGlobalErr),  // sigma_xy
                      SOA_COLUMN(float, yGlobalErr),   // sigma_yy
                      SOA_COLUMN(float, xzGlobalErr),  // sigma_xz
                      SOA_COLUMN(float, yzGlobalErr),  // sigma_yz
                      SOA_COLUMN(float, zGlobalErr),   // sigma_zz
                      // Discretized phi (1024 bins over 2π)
                      SOA_COLUMN(int16_t, iphi),
                      // DetectorIndex: index into stacked module array
                      // Maps to StackedModuleGeometrySoA for geometry access
                      SOA_COLUMN(uint16_t, detectorIndex),
                      // Stacked module DetId (same for both sensors in a stack)
                      SOA_COLUMN(uint32_t, stackDetId),
                      // Individual sensor DetId (lower or upper)
                      SOA_COLUMN(uint32_t, sensorDetId),
                      // Sensor type: true if lower (inner) sensor, false if upper (outer)
                      SOA_COLUMN(bool, isLower),
                      // Index into the original Phase2TrackerRecHit1DCollectionNew (flat sequential order)
                      // Enables truth matching by retrieving the original RecHit for SimHit association
                      SOA_COLUMN(uint32_t, origRecHitIdx),
                      // Cluster size (number of strips/pixels along measurement direction)
                      // From Phase2TrackerCluster1D::size()
                      SOA_COLUMN(uint16_t, clusterSize))

  // Module start indices and sensor boundaries
  // For each stacked module:
  //   moduleStart[i]: index of first hit in stack i (lower sensor)
  //   upperSensorStart[i]: index where upper sensor hits begin in stack i
  //   moduleStart[i+1] - moduleStart[i] = total hits in stack i (lower + upper)
  GENERATE_SOA_LAYOUT(OTHitModulesLayout,
                      SOA_COLUMN(uint32_t, moduleStart),
                      SOA_COLUMN(uint32_t, upperSensorStart))

  GENERATE_SOA_BLOCKS(OTRecHitBlocksLayout,
                      SOA_BLOCK(otRecHits, OTRecHitsLayout),
                      SOA_BLOCK(otHitModules, OTHitModulesLayout))

  using OTRecHitsSoA = OTRecHitsLayout<>;
  using OTRecHitsView = OTRecHitsSoA::View;
  using OTRecHitsConstView = OTRecHitsSoA::ConstView;

  using OTHitModuleSoA = OTHitModulesLayout<>;
  using OTHitModuleView = OTHitModuleSoA::View;
  using OTHitModuleConstView = OTHitModuleSoA::ConstView;

  using OTRecHitBlocksSoA = OTRecHitBlocksLayout<>;
  using OTRecHitBlocksSoAView = OTRecHitBlocksSoA::View;
  using OTRecHitBlocksSoAConstView = OTRecHitBlocksSoA::ConstView;

}  // namespace reco

#endif  // DataFormats_TrackingRecHitSoA_interface_OTRecHitsSoA_h
