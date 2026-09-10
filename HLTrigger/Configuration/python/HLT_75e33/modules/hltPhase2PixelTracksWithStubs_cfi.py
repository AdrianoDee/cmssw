import FWCore.ParameterSet.Config as cms

hltPhase2PixelTracksWithStubs = cms.EDProducer("PixelTrackProducerFromSoAAlpaka",
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    trackSrc = cms.InputTag("hltPhase2PixelTrackTorchHighPuritySelector"),
    pixelRecHitLegacySrc = cms.InputTag("hltSiPixelRecHits"),
    outerTrackerRecHitSrc = cms.InputTag("hltSiPhase2RecHits"),
    otRecHitsSoASrc = cms.InputTag("hltPixelSeedingOTRecHitsSoA"),
    stubsSoASrc = cms.InputTag("hltOTStubProducer"),
    minNumberOfHits = cms.int32(0),
    minQuality = cms.string('tight'),
    useOTExtension = cms.bool(True),
    expandStubs = cms.bool(True),
    requireQuadsFromConsecutiveLayers = cms.bool(False)
)

# Two-iteration stub chain (pixelTrackMask): trackSrc is the final high-purity selection, not the merger.
from Configuration.ProcessModifiers.pixelTrackMask_cff import pixelTrackMask
pixelTrackMask.toModify(hltPhase2PixelTracksWithStubs,
    trackSrc = cms.InputTag("hltPhase2PixelTrackHighPuritySelectorMerged"),
)
