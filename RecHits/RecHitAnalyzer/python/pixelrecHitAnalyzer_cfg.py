import FWCore.ParameterSet.Config as cms

process = cms.Process("RecHitAnalyzerTest")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')
#-------------------------------------------------                                                                                                                                                         
# CALIBRATION                                                                                                                                                                                              
#-------------------------------------------------                                                                                                                                                         
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = "124X_dataRun3_v10"
process.GlobalTag.globaltag = "131X_mcRun3_2023_realistic_v3"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

process.source = cms.Source("PoolSource",
    # Input file(s)
    fileNames = cms.untracked.vstring("file:step3_SingleMuon.root")
)
process.myConverter = cms.EDProducer("SiStripRecHitConverter",
    ClusterProducer = cms.InputTag("siStripClusters"),
    rphiRecHits = cms.string("rphiRecHit"),
    stereoRecHits = cms.string("stereoRecHit"),
    matchedRecHits = cms.string("matchedRecHit"),
    doMatching = cms.bool(True)
)

process.RecHitAnalyzerHelp = cms.EDAnalyzer("RecHitAnalyzer",
    recHits = cms.InputTag("siStripClusters"),
    pixels = cms.InputTag("pixelTracks"),
    recHitCollectionMatched = cms.InputTag("myConverter", "matchedRecHit"),
    recHitCollection = cms.InputTag("myConverter", "rphiRecHit")
 #   recHitCollection = cms.InputTag("myConverter", "stereoRecHit")
)

process.p = cms.Path(process.myConverter * process.RecHitAnalyzerHelp)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))
