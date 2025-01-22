import FWCore.ParameterSet.Config as cms

def customizeForAlpakaSplitter(process):
    
    process.trial = cms.EDProducer("trial",  # Corrected path to the plugin
        configString=cms.string("This is my configuration string"),
        nHits=cms.uint32(100),   # Match uint32_t in C++
        offset=cms.int32(10),     # Match int32_t in C++
        centralMIPCharge = cms.double(26000),
        chargeFractionMin = cms.double(2.0),
        chargePerUnit = cms.double(2000),
        candidateInput = cms.InputTag("ak5CaloJets"),
        deltaRmax = cms.double(0.05),
        forceXError = cms.double(100),
        forceYError = cms.double(150),
        fractionalWidth = cms.double(0.4),
        pixelCPE = cms.string('PixelCPEGeneric'),
        siPixelClusters = cms.InputTag("siPixelClustersPreSplittingAlpaka"),
        SiPixelDigisSoA = cms.InputTag("siPixelClustersPreSplittingAlpaka"),
        ptMin = cms.double(200),
        verbose = cms.bool(False),
        vertices = cms.InputTag("pixelVerticesAlpaka")
    )


    process.trial_step = cms.Path(process.trial)    

    process.schedule.extend([process.trial_step])

    return process
