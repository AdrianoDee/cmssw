import FWCore.ParameterSet.Config as cms

hltPhase2PixelTracksCAExtension = cms.EDProducer("PixelTrackProducerFromSoAAlpaka",
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    minNumberOfHits = cms.int32(0),
    minQuality = cms.string('tight'),
    pixelRecHitLegacySrc = cms.InputTag("hltSiPixelRecHits"),
    trackSrc = cms.InputTag("hltPhase2PixelTracksSoA"),
    outerTrackerRecHitSrc = cms.InputTag("hltSiPhase2RecHits"),
    outerTrackerRecHitSoAConverterSrc = cms.InputTag("hltPhase2OtRecHitsSoA"),
    useOTExtension = cms.bool(True),
    requireQuadsFromConsecutiveLayers = cms.bool(False)
)

from Configuration.ProcessModifiers.phase2CAStubs_cff import phase2CAStubs
from .hltPhase2PixelTracksWithStubs_cfi import hltPhase2PixelTracksWithStubs as _hltPhase2PixelTracksWithStubs
phase2CAStubs.toReplaceWith(hltPhase2PixelTracksCAExtension,
    _hltPhase2PixelTracksWithStubs.clone(minQuality='tight')
)
