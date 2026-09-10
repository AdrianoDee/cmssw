import FWCore.ParameterSet.Config as cms

hltPhase2PixelTrackHighPtMasking = cms.EDProducer('PixelTracksMaskingSoA@alpaka',
    iterationIndex = cms.uint32(1),
    minQuality = cms.string('tight'),
    # Masking reads the pre-extension high-purity selector on purpose: masking with the
    # stub-extended tracks removes too many hits and starves the displaced iteration.
    tracksSoASrc = cms.InputTag('hltPhase2PixelTrackTorchHighPuritySelector'),
    recHitsMaskSoASrc = cms.InputTag('hltPhase2PixelRecHitsStubsMerger'),
)
