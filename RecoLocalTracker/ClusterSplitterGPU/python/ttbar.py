import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

# Load message logger 
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 50

# Set the maximum number of events
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(1000))

# Define the source file
process.source = cms.Source("PoolSource",
    fileNames=cms.untracked.vstring("file:/data/user/tetto/12834.0_TTbar_14TeV+2024/step3.root")
)

# Define the trial EDProducer with additional inputs
process.trial = cms.EDProducer("trial",
    configString=cms.string("This is my configuration string"),
    nHits=cms.uint32(100),
    offset=cms.int32(10)
    #clusterTag=cms.InputTag("siPixelClusters"),  # Input for clusters
    #digiTag=cms.InputTag("siPixelDigis"),       # Input for digis
    #recHitTag=cms.InputTag("trackingRecHits")   # Input for tracking hits
)

process.p = cms.Path(process.trial)
