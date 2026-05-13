#ifndef DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsSoA_h
#define DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsSoA_h

#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/TrackingRecHitSoA/interface/SiPixelHitStatus.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"

namespace reco {

  GENERATE_SOA_LAYOUT(TrackingHitsLayout,
                      SOA_COLUMN(float, xLocal),
                      SOA_COLUMN(float, yLocal),
                      SOA_COLUMN(float, xerrLocal),
                      SOA_COLUMN(float, yerrLocal),
                      SOA_COLUMN(float, xGlobal),
                      SOA_COLUMN(float, yGlobal),
                      SOA_COLUMN(float, zGlobal),
                      SOA_COLUMN(float, rGlobal),
                      SOA_COLUMN(int16_t, iphi),
                      SOA_COLUMN(SiPixelHitStatusAndCharge, chargeAndStatus),
                      SOA_COLUMN(int16_t, clusterSizeX),
                      SOA_COLUMN(int16_t, clusterSizeY),
                      SOA_COLUMN(uint16_t, detectorIndex),
                      // Stub-specific fields for CA integration
                      // When isStub=true, this hit comes from an OT stub with direction info
                      SOA_COLUMN(bool, isStub),           // True if this is from a stub (not a regular pixel hit)
                      SOA_COLUMN(uint8_t, stubType),      // 0=PS (good z), 1=SS (poor z), ignored if isStub=false
                      SOA_COLUMN(float, dPhiDr),          // Stub direction (local track angle), 0 if isStub=false
                      SOA_COLUMN(float, dPhiDrError),     // Error on stub direction, 0 if isStub=false
                      SOA_COLUMN(float, ptEst),           // pT estimate from stub bend (GeV), 0 if isStub=false
                      // P-hit group ID: For stub hits, identifies which P-hit (lower sensor hit)
                      // was used to form this stub. Stubs sharing the same P-hit have the same ID.
                      // Used by CAFishbone to correctly handle duplicate stubs from the same P-hit.
                      // Value UINT32_MAX indicates unset/invalid (for regular pixel hits).
                      SOA_COLUMN(uint32_t, pHitGroupId),
                      // Packed stub flags from StubsSoA: isBarrel(bit0), isFlat(bit1), isValid(bit2), layer(bits3-5)
                      // Used by stub-stub pairwise compatibility cut in CA doublet formation.
                      // Zero for regular pixel hits. Decode with reco::StubFlags helpers.
                      SOA_COLUMN(uint8_t, stubFlags),
                      SOA_SCALAR(int32_t, offsetBPIX2),
                      SOA_SCALAR(uint32_t, offsetStubs)); // Offset where stub-derived hits start
                                                          // For stub hits: stubIndex = hitIndex - offsetStubs
                                                          // Stubs must be added in same order as StubsSoACollection

  GENERATE_SOA_LAYOUT(HitModulesLayout, SOA_COLUMN(uint32_t, moduleStart));

  GENERATE_SOA_BLOCKS(TrackingBlocksLayout,
                      SOA_BLOCK(trackingHits, TrackingHitsLayout),
                      SOA_BLOCK(hitModules, HitModulesLayout))

  // N.B. this layout is not really included by default in the hits SoA
  // This holds the needed parameters to activate (via ONLY_TRIPLETS_IN_HOLE) the
  // calculations to check if a triplet points to the disk hole
  // and then retain only those that fulfil this requirement.
  // At the moment this feature is not fully (re)implemented.

  GENERATE_SOA_LAYOUT(AverageGeometryLayout,
                      SOA_COLUMN(float, ladderZ),
                      SOA_COLUMN(float, ladderX),
                      SOA_COLUMN(float, ladderY),
                      SOA_COLUMN(float, ladderR),
                      SOA_COLUMN(float, ladderMinZ),
                      SOA_COLUMN(float, ladderMaxZ),
                      SOA_SCALAR(int32_t, endCapZPos),
                      SOA_SCALAR(int32_t, endCapZNeg))

  GENERATE_SOA_LAYOUT(TrackingRecHitsMaskingLayout, SOA_COLUMN(uint32_t, recHitMask));

  using TrackingRecHitSoA = TrackingHitsLayout<>;
  using TrackingRecHitView = TrackingRecHitSoA::View;
  using TrackingRecHitConstView = TrackingRecHitSoA::ConstView;

  using HitModuleSoA = HitModulesLayout<>;
  using HitModuleSoAView = HitModuleSoA::View;
  using HitModuleSoAConstView = HitModuleSoA::ConstView;

  using TrackingBlocksSoA = TrackingBlocksLayout<>;
  using TrackingBlocksSoAView = TrackingBlocksSoA::View;
  using TrackingBlocksSoAConstView = TrackingBlocksSoA::ConstView;

  using AverageGeometrySoA = AverageGeometryLayout<>;
  using AverageGeometryView = AverageGeometrySoA::View;
  using AverageGeometryConstView = AverageGeometrySoA::ConstView;

  using TrackingRecHitsMaskingSoA = TrackingRecHitsMaskingLayout<>;
  using TrackingRecHitsMaskingView = TrackingRecHitsMaskingSoA::View;
  using TrackingRecHitsMaskingConstView = TrackingRecHitsMaskingSoA::ConstView;

};  // namespace reco

#endif
