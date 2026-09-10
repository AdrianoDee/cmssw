import FWCore.ParameterSet.Config as cms

# Stage-2 high-purity selector of the displaced iteration: a gradient-boosted decision forest.
hltPhase2PixelTrackHighPuritySelectorDisplaced = cms.EDProducer('PixelTrackForestHighPuritySelector@alpaka',
    pixelTrackSrc = cms.InputTag('hltPhase2PixelTracksSoADisplaced'),
    # Every track the displaced iteration of the stub chain can output is considered
    # (hltPhase2PixelTracksSoADisplacedWithStubs.maxNumberOfTuples).
    maxNumberOfTracks = cms.int32(15_872),
    # Hard cap on the number of tracks that get scored, sized for PU200 with headroom.
    maxPreselectedTracks = cms.int32(3_072),
    # Compact custom binary (not TorchScript), loaded once per process into a shared GlobalCache.
    model = cms.FileInPath('RecoTracker/FinalTrackSelectors/data/PixelTrackTorchHighPuritySelector/disp_tree31_wp_20260821.bin'),
    # Flat working point: the dxy-dependent threshold ramp is disabled (scoreThresholdLowDxy < 0).
    scoreThreshold = cms.double(0.1824)
)
