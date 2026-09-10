#ifndef DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsSoA_h
#define DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsSoA_h

#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/TrackingRecHitSoA/interface/SiPixelHitStatus.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"

namespace reco {

  GENERATE_SOA_LAYOUT(
      TrackingHitsLayout,
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
      // Stub columns. dPhiDrError doubles as the stub predicate, isStub == (dPhiDrError >= 0), so
      // every hit producer must write it: -1 for pixel/OT hits, the stub's error (>= 0) for stubs.
      SOA_COLUMN(float, dPhiDr),       // stub direction; 0 for non-stub hits
      SOA_COLUMN(float, dPhiDrError),  // stub direction error; -1 for non-stub hits
      // Same bend error from the precision (local-x) rows only; < 0 for non-stub hits.
      SOA_COLUMN(float, dPhiDrErrorPrec),
      // Lower-sensor (P-side) hit that formed a PS stub; stubs sharing it share the value, which the
      // fishbone uses to spot duplicates. UINT32_MAX = unset (pixel hits, 2S stubs).
      SOA_COLUMN(uint32_t, lowerHitIdx),
      // Packed stub flags (StubsSoA); zero for pixel hits. Decode with reco::StubFlags.
      SOA_COLUMN(uint8_t, stubFlags),
      SOA_SCALAR(int32_t, offsetBPIX2),
      SOA_SCALAR(
          uint32_t,
          offsetStubs));  // first stub hit; stubIndex = hitIndex - offsetStubs; must match StubsSoACollection order

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

  ALPAKA_FN_HOST_ACC inline bool isStub(const TrackingRecHitConstView &hits, int32_t i) {
    return hits[i].dPhiDrError() >= 0.f;
  }

};  // namespace reco

#endif
