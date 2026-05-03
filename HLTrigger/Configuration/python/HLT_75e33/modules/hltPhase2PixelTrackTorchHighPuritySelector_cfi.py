import FWCore.ParameterSet.Config as cms

hltPhase2PixelTrackTorchHighPuritySelector = cms.EDProducer('PixelTrackTorchHighPuritySelector@alpaka',
    pixelRecHitSrc = cms.InputTag('hltPhase2PixelRecHitsExtendedSoA'),
    pixelTrackSrc = cms.InputTag('hltPhase2PixelTracksSoA'),
    maxNumberOfTracks = cms.int32(2*60*1024),
    maxPreselectedTracks = cms.int32(4_096),
    minNumberOfHits = cms.int32(0),
    avgHitsPerTrack = cms.int32(8),
    minimumTrackQuality = cms.string('tight'),
    model = cms.FileInPath('RecoTracker/FinalTrackSelectors/data/track_classifier_GPU_enriched.pt'),
    scoreThreshold = cms.double(0.19),
    batchSize = cms.int32(4_096)
)

from Configuration.ProcessModifiers.phase2CAStubs_cff import phase2CAStubs
phase2CAStubs.toModify(hltPhase2PixelTrackTorchHighPuritySelector,
   pixelRecHitSrc = cms.InputTag('hltPhase2PixelRecHitsStubsMerger'),
)