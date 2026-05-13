import FWCore.ParameterSet.Config as cms

hltPhase2PixelTracksSoAMerger = cms.EDProducer('PixelTracksSoAMerger@alpaka',
    inputTkSoAs = cms.VInputTag("hltPhase2PixelTrackTorchHighPuritySelector","hltPhase2PixelTrackTorchHighPuritySelectorLowPt"),
    minQuality = cms.string('tight'),
    matchFraction = cms.double(0.0),
)