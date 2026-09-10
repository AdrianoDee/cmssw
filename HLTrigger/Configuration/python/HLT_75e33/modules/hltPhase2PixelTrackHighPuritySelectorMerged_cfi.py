import FWCore.ParameterSet.Config as cms

# Final high-purity selection of the two-iteration stub chain (phase2CAStubs & pixelTrackMask): it scores
# the merger output, a merged, OT-extended, refit and de-duplicated collection.
hltPhase2PixelTrackHighPuritySelectorMerged = cms.EDProducer('PixelTrackForestHighPuritySelector@alpaka',
    pixelTrackSrc = cms.InputTag('hltPhase2PixelTracksSoAMerger'),

    # Caps on the tracks kept and scored, sized for PU200 with ample headroom.
    maxNumberOfTracks = cms.int32(2*60*1024),
    maxPreselectedTracks = cms.int32(9_984),
    # The merger's attach walk adds OT hits on top of the CA hit content.
    avgHitsPerTrack = cms.int32(16),

    # Forest trained on the merged collection.
    model = cms.FileInPath('RecoTracker/FinalTrackSelectors/data/PixelTrackTorchHighPuritySelector/'
                           'merged_tree42_finalhpfrw_20260823.bin'),
    # The cut ramps from scoreThresholdLowDxy at |dxyBS| = 0 to scoreThreshold at |dxyBS| >= dxyRampKnee.
    scoreThreshold = cms.double(0.02),          # displaced arm
    scoreThresholdLowDxy = cms.double(0.0422),  # prompt-like arm
    dxyRampKnee = cms.double(1.0)               # cm
)
