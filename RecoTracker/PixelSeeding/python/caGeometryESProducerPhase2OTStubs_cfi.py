import FWCore.ParameterSet.Config as cms

# Geometry-only blocks of the CA geometry SoA for Phase2OTStubs: module surface frames and per-CA-layer
# classification, shared by both OT-stub CA iterations and the merger. Nothing here is per iteration.
caGeometryESProducerPhase2OTStubs = cms.ESProducer('CAGeometryESProducerPhase2OTStubs@alpaka',
    # pixelTopology::Phase2OTStubs::numberOfLayers, 28 pixel + 26 outer-tracker CA layers; must match
    # every CA producer sharing this geometry.
    nLayers = cms.uint32(54),
    appendToDataLabel = cms.string(''),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')  # Empty string = use default backend
    )
)
