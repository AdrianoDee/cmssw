import FWCore.ParameterSet.Config as cms

from Geometry.CommonTopologies.globalTrackingGeometry_cfi import *
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *


process = cms.Process("RECOCC")

process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2024_realistic', '')

process.HelperSplitter = cms.EDProducer("HelperSplitter",
    Candidate = cms.InputTag("ak4CaloJets", "", "RECO"),
    siPixelClusters = cms.InputTag("siPixelClusters","","RECO"),
    ptMin = cms.double(0.5),
    tanLorentzAngle = cms.double(0.1),
    tanLorentzAngleBarrelLayer1 = cms.double(0.2),
    verbose = cms.bool(True)
)

# Define the execution path and schedule.
process.HelperSplitter_step = cms.Path(process.HelperSplitter)
process.schedule = cms.Schedule(process.HelperSplitter_step)

# Input: use step3.root which already contains ak4CaloJets.
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:step3.root')
)

# Output: write your output (even if HelperSplitter doesn't produce any new products, you can still run it).
process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:step_output.root')
)

# Process a limited number of events.
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

# Define an EndPath to write the output.
process.endpath = cms.EndPath(process.output)
process.schedule.append(process.endpath)
