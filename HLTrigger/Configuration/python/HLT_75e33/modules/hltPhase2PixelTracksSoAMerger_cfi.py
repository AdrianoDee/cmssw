import FWCore.ParameterSet.Config as cms

# Merges the high-purity tracks of both iterations, then runs the OT-hit attach walk, the GBL refit
# and the duplicate removal in one pass. The attach-walk tables are compiled in (ExtDerivedTables.h).
hltPhase2PixelTracksSoAMerger = cms.EDProducer('PixelTracksSoAMerger@alpaka',
    # Both iterations' high-purity selector outputs, still un-extended: the OT-hit extension runs here.
    inputTkSoAs = cms.VInputTag("hltPhase2PixelTrackTorchHighPuritySelector", "hltPhase2PixelTrackHighPuritySelectorDisplaced"),
    # Arm of each input collection: 0 = prompt-side, 1 = displaced-side.
    inputArms = cms.vuint32(0, 1),
    minQuality = cms.string('tight'),

    # Ceiling on attach scratch candidate capacity (effective bound = min(merged track capacity, this)).
    extRefitMaxCandidates = cms.uint32(3072),
)
