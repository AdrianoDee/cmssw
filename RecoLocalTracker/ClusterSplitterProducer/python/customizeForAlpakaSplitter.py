import FWCore.ParameterSet.Config as cms

def customizeForAlpakaSplitter(process):

    print("Customizing process for HelperSplitter")

    # Load necessary producers
    process.load("RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi")  # Load Calo Towers producer
    process.load("RecoJets.JetProducers.ak4CaloJets_cfi")  # Load the ak4CaloJets producer

    # Define a Task to ensure dependencies are managed explicitly
    process.towerMakerTask = cms.Task(process.calotowermaker)
    process.ak4CaloJetsTask = cms.Task(process.ak4CaloJets)

    # Add Tasks to Paths
    process.towerMaker_step = cms.Path(process.towerMakerTask)
    process.ak4CaloJets_step = cms.Path(process.ak4CaloJetsTask)

    # Define the HelperSplitter producer
    process.HelperSplitter = cms.EDProducer("HelperSplitter",
        Candidate = cms.InputTag("ak4CaloJets", "", "RECO"),
        SiPixelClusters = cms.InputTag("SiPixelClusters"),
        ptMin = cms.double(0.5),                          # Default value
        tanLorentzAngle = cms.double(0.1),                # Default value
        tanLorentzAngleBarrelLayer1 = cms.double(0.2),    # Default value
        verbose = cms.bool(False)
    )

    # Create a Task for HelperSplitter
    process.HelperSplitterTask = cms.Task(process.HelperSplitter)
    process.HelperSplitter_step = cms.Path(process.HelperSplitterTask)

    process.schedule = cms.Schedule(
        process.towerMaker_step,  # Make sure Calo Towers are created first
        process.ak4CaloJets_step,  # Then produce ak4CaloJets
        process.HelperSplitter_step  # Finally, run HelperSplitter
    )

    return process
