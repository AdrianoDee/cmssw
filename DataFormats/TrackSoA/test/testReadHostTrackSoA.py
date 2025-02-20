import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("READ")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:"+sys.argv[1]))

process.testReadHostTrackSoA = cms.EDAnalyzer("TestReadHostTrackSoA",
    collectionTag = cms.InputTag("input", "", "WRITE")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testTrackSoAReader.root'),
    fastCloning = cms.untracked.bool(False)
)

process.path = cms.Path(process.testReadHostTrackSoA)

process.endPath = cms.EndPath(process.out)