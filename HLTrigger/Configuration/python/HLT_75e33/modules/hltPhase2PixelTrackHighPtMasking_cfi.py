import FWCore.ParameterSet.Config as cms

hltPhase2PixelTrackHighPtMasking = cms.EDProducer('PixelTracksMaskingSoA@alpaka',
    iterationIndex = cms.uint32(1),
    minQuality = cms.string('tight'),
    tracksSoASrc = cms.InputTag('hltPhase2PixelTrackTorchHighPuritySelector'),
    recHitsMaskSoASrc = cms.InputTag('hltPhase2PixelRecHitsStubsMerger'),
)