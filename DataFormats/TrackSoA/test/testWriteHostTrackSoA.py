import FWCore.ParameterSet.Config as cms

process = cms.Process("WRITE")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 10

process.collectionProducer = cms.EDProducer("TestWriteHostTrackSoA",
    trackSize = cms.uint32(2708)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testTrackSoAWriter.root')
)

process.path = cms.Path(process.collectionProducer)
process.endPath = cms.EndPath(process.out)