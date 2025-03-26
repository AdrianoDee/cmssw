import FWCore.ParameterSet.Config as cms

def customizeForAlpakaSplitter(process):

    # Define the cluster splitter producer
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
        siPixelClusters=cms.InputTag("siPixelClustersSoACollection"),
        siPixelDigis=cms.InputTag("siPixelDigisSoACollection"),
        trackingRecHits=cms.InputTag("trackingRecHitsSoACollection"),
        candidateInput=cms.InputTag("candidateDataSoA"),
        zVertex=cms.InputTag("zVertex"),
        geometryInput=cms.InputTag("ClusterGeometrySoA"),
        verbose=cms.bool(False),
    )

    # Define the steps for the workflow
    process.first_step = cms.Path(process.trial)          # First producer
    process.trial_step = cms.Path(process.trial)          # Second producer

    # Add the paths to the schedule
    if hasattr(process, 'schedule'):
        process.schedule.extend([process.first_step, process.trial_step])
    else:
        process.schedule = cms.Schedule(process.first_step, process.trial_step)

    return process
