import FWCore.ParameterSet.Config as cms

def customizeForAlpakaSplitterFull(process):

    # Load necessary producers
    process.load("RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi")  # Load Calo Towers producer
    process.load("RecoJets.JetProducers.ak4CaloJets_cfi")  # Load the ak4CaloJets producer
    process.load("FWCore.MessageService.MessageLogger_cfi")

    #process.MessageLogger.cout.enable = True
    #process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')
    #process.MessageLogger.cerr.FwkReport.reportEvery = 1
    #process.MessageLogger.debugModules = cms.untracked.vstring('HelperSplitter','trial')

    # Define a Task to ensure dependencies are managed explicitly
    process.towerMakerTask = cms.Task(process.calotowermaker)
    process.ak4CaloJetsTask = cms.Task(process.ak4CaloJets)

    # Add Tasks to Paths
    process.towerMaker_step = cms.Path(process.towerMakerTask)
    process.ak4CaloJets_step = cms.Path(process.ak4CaloJetsTask)

    # Define the HelperSplitter producer
    process.HelperSplitter = cms.EDProducer(
        "HelperSplitter",
        Candidate = cms.InputTag("hltAK4CaloJets", "", "HLT"),
        SiPixelClusters = cms.InputTag("SiPixelClusters"),
        ptMin = cms.double(0.5),                          # Default value
        tanLorentzAngle = cms.double(0.1),                # Default value
        tanLorentzAngleBarrelLayer1 = cms.double(0.2),    # Default value
        verbose = cms.bool(False)
    )

    process.consumer = cms.EDAnalyzer("GenericConsumer",
        eventProducts = cms.untracked.vstring("HelperSplitter")
    )
    process.consume_step = cms.EndPath(process.consumer)

    # Create a Task for HelperSplitter
    process.HelperSplitterTask = cms.Task(process.HelperSplitter)
    process.HelperSplitter_step = cms.Path(process.HelperSplitterTask)


    # Define the `trial` producer
    process.trial = cms.EDProducer(
        "trial",
        nHits=cms.uint32(100),
        offset=cms.int32(10),
        ptMin=cms.double(200),
        deltaR=cms.double(0.05),
        chargeFracMin=cms.double(2.0),
        tanLorentzAngle=cms.double(0.02),
        tanLorentzAngleBarrelLayer1=cms.double(0.015),
        expSizeXAtLorentzAngleIncidence=cms.double(0.1),
        expSizeXDeltaPerTanAlpha=cms.double(0.02),
        expSizeYAtNormalIncidence=cms.double(0.1),
        centralMIPCharge=cms.double(26000),
        chargePerUnit=cms.double(2000),
        forceXError=cms.double(100),
        forceYError=cms.double(150),
        fractionalWidth=cms.double(0.4),
        siPixelClusters=cms.InputTag("SiPixelClustersSoACollection"),   # Replace with correct InputTags
        siPixelDigis=cms.InputTag("SiPixelDigisSoACollection"),
        trackingRecHits=cms.InputTag("trackingRecHitsSoACollection"),
        candidateInput=cms.InputTag("candidateDataSoA"),     # Connect to HelperSplitter output
        zVertex=cms.InputTag("zVertex"),
        geometryInput=cms.InputTag("ClusterGeometrySoA"),
        verbose=cms.bool(False),
    )

    # Create a Task for `trial`
    process.trialTask = cms.Task(process.trial)
    process.trial_step = cms.Path(process.trialTask)


    process.schedule = cms.Schedule(
        process.towerMaker_step,
        process.ak4CaloJets_step,
        process.HelperSplitter_step,
        process.trial_step
    )

    # Enforce execution order in the schedule
    #process.schedule.extend([process.towerMaker_step, process.ak4CaloJets_step, process.HelperSplitter_step, process.trial_step,process.consume_step])

    return process

