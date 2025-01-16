import FWCore.ParameterSet.Config as cms

process = cms.Process("RECOOOOOO")

# Standard services, geometry, magnetic field, and GlobalTag
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2024_realistic', '')

# Load beam spot (optional)
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

# Load Pixel Clusterizer
process.load("RecoLocalTracker.SiPixelClusterizer.siPixelClustersPreSplitting_cff")

# Define the JetCoreClusterSplitter EDProducer
process.jetCoreClusterSplitter = cms.EDProducer("JetCoreClusterSplitter",
    pixelClusters         = cms.InputTag('siPixelClusters'),  # Input pixel clusters
    vertices              = cms.InputTag('offlinePrimaryVertices'),
    pixelCPE              = cms.string("PixelCPEGeneric"),
    verbose               = cms.bool(False),
    ptMin                 = cms.double(100),
    cores                 = cms.InputTag("ak4CaloJets", "", "RECO"),
    chargeFractionMin     = cms.double(2.0),
    deltaRmax             = cms.double(0.05),
    forceXError           = cms.double(100),
    forceYError           = cms.double(150),
    fractionalWidth       = cms.double(0.4),
    chargePerUnit         = cms.double(2000),
    centralMIPCharge      = cms.double(26000)
)

# Define the process path
process.siPixelClustersPreSplitting_step = cms.Path(process.siPixelClustersPreSplitting)
process.jetCoreClusterSplitter_step = cms.Path(process.jetCoreClusterSplitter)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:step3ok.root')
)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

# Output module
process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier   = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:jetCoreClusterSplitter_output.root'),
    outputCommands = cms.untracked.vstring("keep *_*_*_*"),
    splitLevel = cms.untracked.int32(0)
)

process.out = cms.EndPath(process.RECOSIMoutput)

# Set the schedule
process.schedule = cms.Schedule(
    process.siPixelClustersPreSplitting_step,
    process.jetCoreClusterSplitter_step,
    process.out
)
