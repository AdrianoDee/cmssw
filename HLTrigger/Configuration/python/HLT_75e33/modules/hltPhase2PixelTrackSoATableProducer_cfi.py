import FWCore.ParameterSet.Config as cms

hltPhase2PixelTrackSoATableProducer = cms.EDProducer("HLTPixelTrackSoATableProducer",
    trackSrc = cms.InputTag("hltPhase2PixelTrackTorchHighPuritySelector"),
)

# The table source is the merger's SoA (merged, OT-extended, refit, de-duplicated), not the prompt arm alone.
from Configuration.ProcessModifiers.pixelTrackMask_cff import pixelTrackMask
pixelTrackMask.toModify(hltPhase2PixelTrackSoATableProducer, trackSrc = "hltPhase2PixelTracksSoAMerger")
