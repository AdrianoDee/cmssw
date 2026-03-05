import FWCore.ParameterSet.Config as cms

# Legacy track converter for stub-based tracking
# Converts SoA tracks to legacy reco::Track with stub expansion
hltPhase2PixelTracksWithStubs = cms.EDProducer("PixelTrackProducerFromSoAAlpaka",
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    trackSrc = cms.InputTag("hltPhase2PixelTracksSoA"),
    pixelRecHitLegacySrc = cms.InputTag("hltSiPixelRecHits"),
    outerTrackerRecHitSrc = cms.InputTag("hltSiPhase2RecHits"),
    otRecHitsSoASrc = cms.InputTag("hltPixelSeedingOTRecHitsSoA"),
    stubsSoASrc = cms.InputTag("hltOTStubProducer"),
    minNumberOfHits = cms.int32(0),
    minQuality = cms.string('loose'),
    useOTExtension = cms.bool(True),
    expandStubs = cms.bool(True),
    requireQuadsFromConsecutiveLayers = cms.bool(True)
)
