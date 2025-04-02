import FWCore.ParameterSet.Config as cms

# Load necessary producers
from RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi import calotowermaker
from RecoJets.JetProducers.ak4CaloJets_cfi import ak4CaloJets

# Define the HelperSplitter producer
HelperSplitter = cms.EDProducer(
    "HelperSplitter",
    Candidate = cms.InputTag("ak4CaloJets"),          # Input for HelperSplitter
    SiPixelClusters = cms.InputTag("SiPixelClusters"),
    ptMin = cms.double(0.5),                          # Default value
    tanLorentzAngle = cms.double(0.1),                # Default value
    tanLorentzAngleBarrelLayer1 = cms.double(0.2),    # Default value
    verbose = cms.bool(False)
)

# Create the sequence
HelperSplitterTask = cms.Task(calotowermaker, ak4CaloJets, HelperSplitter)
HelperSplitterSequence = cms.Sequence(HelperSplitterTask)
