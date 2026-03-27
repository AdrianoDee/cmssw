import FWCore.ParameterSet.Config as cms

# OT RecHits SoA Converter - wrapper for @alpaka backend selection
pixelSeedingOTRecHitsSoAConverter = cms.EDProducer('PixelSeedingOTRecHitsSoAConverter@alpaka',
    otRecHitSource = cms.InputTag("siPhase2RecHits"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    psOnly = cms.bool(False)
)

# Alias for backward compatibility
otRecHitsSoAConverter = pixelSeedingOTRecHitsSoAConverter.clone()
