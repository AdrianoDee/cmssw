#ifndef RecoTracker_PixelSeeding_interface_CAGeometry_h
#define RecoTracker_PixelSeeding_interface_CAGeometry_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/GeometrySurface/interface/SOARotation.h"

namespace reco {

  using GraphNode = std::array<uint32_t, 2>;
  using DetFrame = SOAFrame<float>;

  // Hard upper bound on the number of CA layers a layers block may describe: the OT-extension walk
  // keeps its per-layer state in compile-time-sized block shared memory and encodes the set of
  // covered layers in a uint64_t bitmask. Producers of the layers block check it and throw.
  inline constexpr int kMaxCALayers = 64;

  // CAModulesLayout: one row per CA module (pixel + OT stack).
  //   detFrame: the module's surface frame (pixel sensor, or the OT stack's lower-sensor frame).
  //   innerSensorFrame: the sensor whose local position/errors are written into a stub (one fit
  //     point per stub): =detFrame for pixel; PSP -> lower P-side; PSS -> upper P-side;
  //     SS -> physically-inner sensor (lower if !isFlipped, upper if isFlipped). Consumers propagate
  //     the local errors to the global covariance via SOAFrame::toGlobal.
  GENERATE_SOA_LAYOUT(CAModulesLayout, SOA_COLUMN(DetFrame, detFrame), SOA_COLUMN(DetFrame, innerSensorFrame))

  // CALayersLayout: per-CA-layer geometry only, built once per IOV and shared by every CA iteration
  // and the merger. Per-iteration per-layer configuration lives in CANtupletCutsLayout.
  GENERATE_SOA_LAYOUT(CALayersLayout,
                      // layer properties
                      SOA_COLUMN(uint32_t, layerStarts),
                      SOA_COLUMN(bool, isBarrel),
                      SOA_COLUMN(bool, isOT),
                      SOA_COLUMN(bool, isSS))

  GENERATE_SOA_LAYOUT(CAGraphLayout,
                      // layer-pair / graph properties
                      SOA_COLUMN(GraphNode, layerPair),
                      // effectively skipsLayers is a boolean but it's used as an index offset,
                      // therefore stored as same type as the layerPairId in CACell
                      SOA_COLUMN(int16_t, skipsLayers),
                      SOA_COLUMN(bool, startingPair))

  GENERATE_SOA_LAYOUT(CADoubletCutsLayout,
                      // doublet vector cuts
                      SOA_COLUMN(int16_t, maxDPhi),
                      SOA_COLUMN(float, minInner),
                      SOA_COLUMN(float, maxInner),
                      SOA_COLUMN(float, minOuter),
                      SOA_COLUMN(float, maxOuter),
                      SOA_COLUMN(float, maxDZ),
                      SOA_COLUMN(float, minDZ),
                      SOA_COLUMN(float, maxDR),
                      SOA_COLUMN(float, minPt),
                      SOA_COLUMN(float, maxZ0),
                      SOA_COLUMN(float, maxStubCurvSigma),
                      // scalar doublet cuts
                      SOA_SCALAR(float, dzdrFact),
                      SOA_SCALAR(int16_t, minInnerSizeB1),
                      SOA_SCALAR(int16_t, minInnerSizeB2),
                      SOA_SCALAR(int16_t, maxDSizeB1),
                      SOA_SCALAR(int16_t, maxDSize),
                      SOA_SCALAR(int16_t, maxDSizePred))

  GENERATE_SOA_LAYOUT(CATripletCutsLayout,
                      // triplet vector cuts
                      SOA_COLUMN(float, maxRZTolerance),
                      SOA_COLUMN(float, maxDCA),
                      SOA_COLUMN(float, floorDCA),
                      SOA_COLUMN(float, maxStubGeomCurvSigma),
                      SOA_COLUMN(float, maxStubInnerDoubletDCurv),
                      // scalars for triplet cuts
                      SOA_SCALAR(float, ptmin),
                      SOA_SCALAR(float, maxCurv),
                      SOA_SCALAR(float, maxPhiResid),
                      SOA_SCALAR(bool, sameDPhiSign))

  // CANtupletCutsLayout: one row per CA layer, not per layer pair.
  GENERATE_SOA_LAYOUT(CANtupletCutsLayout,
                      // N-tuplet cuts
                      SOA_COLUMN(float, startMaxInnerR),
                      // quadruplet cuts
                      SOA_COLUMN(float, maxDCurv),
                      SOA_COLUMN(float, floorDCurv),
                      // Fishbone: squared-cosine threshold between two doublets sharing an outer
                      // hit, indexed by that hit's CA layer.
                      SOA_COLUMN(float, fishboneCut))

  GENERATE_SOA_BLOCKS(CALayoutTemplate,
                      SOA_BLOCK(layers, CALayersLayout),
                      SOA_BLOCK(graph, CAGraphLayout),
                      SOA_BLOCK(doubletCuts, CADoubletCutsLayout),
                      SOA_BLOCK(tripletCuts, CATripletCutsLayout),
                      SOA_BLOCK(ntupletCuts, CANtupletCutsLayout),
                      SOA_BLOCK(modules, CAModulesLayout))

  using CALayersSoA = CALayersLayout<>;
  using CALayersSoAView = CALayersSoA::View;
  using CALayersSoAConstView = CALayersSoA::ConstView;

  using CAGraphSoA = CAGraphLayout<>;
  using CAGraphSoAView = CAGraphSoA::View;
  using CAGraphSoAConstView = CAGraphSoA::ConstView;

  using CAModulesSoA = CAModulesLayout<>;
  using CAModulesView = CAModulesSoA::View;
  using CAModulesConstView = CAModulesSoA::ConstView;

  using CADoubletCutsSoA = CADoubletCutsLayout<>;
  using CADoubletCutsSoAView = CADoubletCutsSoA::View;
  using CADoubletCutsSoAConstView = CADoubletCutsSoA::ConstView;

  using CATripletCutsSoA = CATripletCutsLayout<>;
  using CATripletCutsSoAView = CATripletCutsSoA::View;
  using CATripletCutsSoAConstView = CATripletCutsSoA::ConstView;

  using CANtupletCutsSoA = CANtupletCutsLayout<>;
  using CANtupletCutsSoAView = CANtupletCutsSoA::View;
  using CANtupletCutsSoAConstView = CANtupletCutsSoA::ConstView;

  using CALayout = CALayoutTemplate<>;
  using CALayoutView = CALayout::View;
  using CALayoutConstView = CALayout::ConstView;

}  // namespace reco
#endif  // RecoTracker_PixelSeeding_interface_CAGeometry_h
