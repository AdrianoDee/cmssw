import FWCore.ParameterSet.Config as cms

# helper functions
from HLTrigger.Configuration.common import *

# add one customisation function per PR
# - put the PR number into the name of the function
# - add a short comment
# for example:

# CCCTF tuning
# def customiseFor12718(process):
#     for pset in process._Process__psets.values():
#         if hasattr(pset,'ComponentType'):
#             if (pset.ComponentType == 'CkfBaseTrajectoryFilter'):
#                 if not hasattr(pset,'minGoodStripCharge'):
#                     pset.minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('HLTSiStripClusterChargeCutNone'))
#     return process



def customiseForOffline(process):
    # For running HLT offline on Run-3 Data, use "(OnlineBeamSpotESProducer).timeThreshold = 1e6",
    # in order to pick the beamspot that was actually used by the HLT (instead of a "fake" beamspot).
    # These same settings can be used offline for Run-3 Data and Run-3 MC alike.
    # Note: the products of the OnlineBeamSpotESProducer are used only
    #       if the configuration uses "(BeamSpotOnlineProducer).useTransientRecord = True".
    # See CMSHLT-2271 and CMSHLT-2300 for further details.
    for prod in esproducers_by_type(process, 'OnlineBeamSpotESProducer'):
        prod.timeThreshold = int(1e6)

    # For running HLT offline and relieve the strain on Frontier so it will no longer inject a
    # transaction id which tells Frontier to add a unique "&freshkey" to many query URLs.
    # That was intended as a feature to only be used by the Online HLT, to guarantee that fresh conditions
    # from the database were loaded at each Lumi section
    # Seee CMSHLT-3123 for further details
    if hasattr(process, 'GlobalTag'):
        # Set ReconnectEachRun and RefreshEachRun to False
        process.GlobalTag.ReconnectEachRun = cms.untracked.bool(False)
        process.GlobalTag.RefreshEachRun = cms.untracked.bool(False)

        if hasattr(process.GlobalTag, 'toGet'):
            # Filter out PSet objects containing only 'record' and 'refreshTime'
            process.GlobalTag.toGet = [
                pset for pset in process.GlobalTag.toGet
                if set(pset.parameterNames_()) != {'record', 'refreshTime'}
            ]

    return process

def customizeHLTfor46935(process):
    """Changes parameter names of EcalUncalibRecHitSoAToLegacy producer"""
    for prod in producers_by_type(process, 'EcalUncalibRecHitSoAToLegacy'):
        if hasattr(prod, 'uncalibRecHitsPortableEB'):
            prod.inputCollectionEB = prod.uncalibRecHitsPortableEB
            delattr(prod, 'uncalibRecHitsPortableEB')
        if hasattr(prod, 'uncalibRecHitsPortableEE'):
            prod.inputCollectionEE = prod.uncalibRecHitsPortableEE
            delattr(prod, 'uncalibRecHitsPortableEE')
        if hasattr(prod, 'recHitsLabelCPUEB'):
            prod.outputLabelEB = prod.recHitsLabelCPUEB
            delattr(prod, 'recHitsLabelCPUEB')
        if hasattr(prod, 'recHitsLabelCPUEE'):
            prod.outputLabelEE = prod.recHitsLabelCPUEE
            delattr(prod, 'recHitsLabelCPUEE')

    return process

def customiseHLTFor46647(process):
    for prod in producers_by_type(process, 'CtfSpecialSeedGenerator'):
        if hasattr(prod, "DontCountDetsAboveNClusters"):
            value = prod.DontCountDetsAboveNClusters.value()
            delattr(prod, "DontCountDetsAboveNClusters")
            # Replace it with cms.uint32
            prod.DontCountDetsAboveNClusters = cms.uint32(value)


def customizeHLTfor47017(process):
    """Remove unneeded parameters from the HLT menu"""
    for prod in producers_by_type(process, 'MaskedMeasurementTrackerEventProducer'):
        if hasattr(prod, 'OnDemand'):
            delattr(prod, 'OnDemand')

    for prod in producers_by_type(process, 'HcalHaloDataProducer'):
        if hasattr(prod, 'HcalMaxMatchingRadiusParam'):
            delattr(prod, 'HcalMaxMatchingRadiusParam')
        if hasattr(prod, 'HcalMinMatchingRadiusParam'):
            delattr(prod, 'HcalMinMatchingRadiusParam')

    for prod in producers_by_type(process, 'SiPixelRecHitConverter'):
        if hasattr(prod, 'VerboseLevel'):
            delattr(prod, 'VerboseLevel')

def customizeHLTforXXX(process):
    

    process.frameSoAESProducerPhase1 = cms.ESProducer('FrameSoAESProducerPhase1@alpaka',
      ComponentName = cms.string('FrameSoAPhase1'),
      appendToDataLabel = cms.string(''),
      alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
      )
    )

    for producer in producers_by_type(process, "CAHitNtupletAlpakaPhase1@alpaka"):
        #print("entered the producers loop")
        if hasattr(producer, "CPE"):
            print("found CPE stuff")
            delattr(producer, "CPE")
        if not hasattr(producer, 'frameSoA'):
            setattr(producer, 'frameSoA', cms.string('FrameSoAPhase1'))

    for producer in producers_by_type(process, "alpaka_serial_sync::CAHitNtupletAlpakaPhase1"):
        #print("entered the producers loop")
        if hasattr(producer, "CPE"):
            print("found CPE stuff")
            delattr(producer, "CPE")
        if not hasattr(producer, 'frameSoA'):
            setattr(producer, 'frameSoA', cms.string('FrameSoAPhase1'))
    
def customizeHLTfor47017(process):
    """Remove unneeded parameters from the HLT menu"""
    for prod in producers_by_type(process, 'MaskedMeasurementTrackerEventProducer'):
        if hasattr(prod, 'OnDemand'):
            delattr(prod, 'OnDemand')

    for prod in producers_by_type(process, 'HcalHaloDataProducer'):
        if hasattr(prod, 'HcalMaxMatchingRadiusParam'):
            delattr(prod, 'HcalMaxMatchingRadiusParam')
        if hasattr(prod, 'HcalMinMatchingRadiusParam'):
            delattr(prod, 'HcalMinMatchingRadiusParam')

    for prod in producers_by_type(process, 'SiPixelRecHitConverter'):
        if hasattr(prod, 'VerboseLevel'):
            delattr(prod, 'VerboseLevel')

    return process


def customizeHLTfor47079(process):
    """Remove unneeded parameters from the HLT menu"""
    for filt in filters_by_type(process, 'PrimaryVertexObjectFilter'):
        if hasattr(filt, 'filterParams') and hasattr(filt.filterParams, 'pvSrc'):
            del filt.filterParams.pvSrc  # Remove the pvSrc parameter

    for prod in producers_by_type(process, 'HcalHitReconstructor'):
        # Remove useless parameters
        if hasattr(prod,'setHSCPFlags'):
            delattr(prod,'setHSCPFlags')

        if hasattr(prod,'setPulseShapeFlags'):
            delattr(prod,'setPulseShapeFlags')
                    
    return process

def customizeHLTfor47047(process):
    """Migrates many ESProducers to MoveToDeviceCache"""
    import copy
    if hasattr(process, "ecalMultifitParametersSource"):
        del process.ecalMultifitParametersSource
    esProducer = None
    for prod in esproducers_by_type(process, "EcalMultifitParametersHostESProducer@alpaka"):
        if esProducer is not None:
            raise Exception("Assumption of only one EcalMultifitParametersHostESProducer@alpaka in a process broken")
        esProducer = prod
    if esProducer is not None:
        for prod in producers_by_type(process, "EcalUncalibRecHitProducerPortable@alpaka", "alpaka_serial_sync::EcalUncalibRecHitProducerPortable"):
            for attr in ["EBtimeFitParameters", "EEtimeFitParameters", "EBamplitudeFitParameters", "EEamplitudeFitParameters"]:
                setattr(prod, attr, copy.deepcopy(getattr(esProducer, attr)))
        delattr(process, esProducer.label())

    for prod in producers_by_type(process, "HBHERecHitProducerPortable@alpaka", "alpaka_serial_sync::HBHERecHitProducerPortable"):
        if not hasattr(prod, 'mahiPulseOffSets'):
            continue
        pulseOffsetLabel = prod.mahiPulseOffSets.getModuleLabel()
        if hasattr(process, pulseOffsetLabel):
            esProducer = getattr(process, pulseOffsetLabel)
            prod.pulseOffsets = copy.deepcopy(esProducer.pulseOffsets)
        del prod.mahiPulseOffSets
    for prod in list(esproducers_by_type(process, "HcalMahiPulseOffsetsESProducer@alpaka")):
        delattr(process, prod.label())

    for prod in producers_by_type(process, "PFClusterSoAProducer@alpaka", "alpaka_serial_sync::PFClusterSoAProducer"):
        if not hasattr(prod, 'pfClusterParams'):
            continue
        clusterParamsLabel = prod.pfClusterParams.getModuleLabel()
        if hasattr(process, clusterParamsLabel):
            esProducer = getattr(process, clusterParamsLabel)
            for attr in ["seedFinder", "initialClusteringStep", "pfClusterBuilder"]:
                setattr(prod, attr, copy.deepcopy(getattr(esProducer, attr).copy()))
        del prod.pfClusterParams
    for prod in list(esproducers_by_type(process, "PFClusterParamsESProducer@alpaka")):
        delattr(process, prod.label())

    if hasattr(process, "hltESSJobConfigurationGPURecord"):
        del process.hltESSJobConfigurationGPURecord

    return process
        
def customizeHLTfor47107(process):
    """Remove unneeded parameters from the HLT menu"""

    for prod in producers_by_type(process, 'TrackProducer'):
        if hasattr(prod, 'alias'):
            delattr(prod, 'alias')

    for prod in producers_by_type(process, 'GsfTrackProducer'):
        if hasattr(prod, 'producer'):
            delattr(prod, 'producer')

    return process


def customizeHLTfor47191(process):
    for esprod in esproducers_by_type(process, "PromptTrackCountingESProducer"):
        if hasattr(esprod, 'minimumImpactParameter'):
            delattr(esprod, 'minimumImpactParameter')

        if hasattr(esprod, 'useSignedImpactParameterSig'):
            delattr(esprod, 'useSignedImpactParameterSig')
            
    return process

def customizeHLTforXYZ(process):
    """ Add CAGeometry ESProducer"""
    
    if not hasattr(process, 'HLTRecoPixelTracksSequence'):
        return process
        
    ca_producers = ['CAHitNtupletAlpakaPhase1@alpaka','alpaka_serial_sync::CAHitNtupletAlpakaPhase1']

    ca_parameters = [ 'CAThetaCutBarrel', 'CAThetaCutForward', 
        'dcaCutInnerTriplet', 'dcaCutOuterTriplet', 
        'doPtCut', 'doZ0Cut', 'idealConditions', 
        'includeJumpingForwardDoublets', 'phiCuts'] 

    process.hltCAGeometry = cms.ESProducer('CAGeometryESProducer@alpaka',
        caDCACuts = cms.vdouble(
            0.15000000596046448, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25
        ),
        caThetaCuts = cms.vdouble(
            0.0020000000949949026,
            0.0020000000949949026,
            0.0020000000949949026,
            0.0020000000949949026,
            0.0030000000260770321,
            0.0030000000260770321,
            0.0030000000260770321,
            0.0030000000260770321,
            0.0030000000260770321,
            0.0030000000260770321
        ),
        startingPairs = cms.vint32( [i for i in range(8)] + [13, 14, 15, 16, 17, 18, 19]),
        pairGraph = cms.vint32( 0, 1, 0, 4, 0,
            7, 1, 2, 1, 4,
            1, 7, 4, 5, 7,
            8, 2, 3, 2, 4,
            2, 7, 5, 6, 8,
            9, 0, 2, 1, 3,
            0, 5, 0, 8, 
            4, 6, 7, 9 
        ),
        phiCuts = cms.vint32( 522, 730, 730, 522, 626,
            626, 522, 522, 626, 626,
            626, 522, 522, 522, 522,
            522, 522, 522, 522
        ),
        minZ = cms.vdouble(
                -20., 0., -30., -22., 10., 
                -30., -70., -70., -22., 15., 
                -30, -70., -70., -20., -22., 
                0, -30., -70., -70.
        ),
        maxZ = cms.vdouble( 20., 30., 0., 22., 30., 
            -10., 70., 70., 22., 30., 
            -15., 70., 70., 20., 22., 
            30., 0., 70., 70.),
        maxR = cms.vdouble(20., 9., 9., 20., 7., 
            7., 5., 5., 20., 6., 
            6., 5., 5., 20., 20., 
            9., 9., 9., 9.),
        appendToDataLabel = cms.string('hltCAGeometry'),
        alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(''),
        synchronize = cms.optional.untracked.bool
        )
    )

    for ca_producer in ca_producers:
        for prod in producers_by_type(process, ca_producer):
            
            if hasattr(prod, 'CPE'):
                delattr(prod, 'CPE')

            if not hasattr(prod, 'caGeometry'):
                setattr(prod, 'caGeometry', cms.string('hltCAGeometry'))
            
            if hasattr(prod, 'maxNumberOfDoublets'):
                v = getattr(prod, 'maxNumberOfDoublets')
                delattr(prod, 'maxNumberOfDoublets')
                setattr(prod, 'maxNumberOfDoublets', cms.string(str(v.value())))
            
            if hasattr(prod, 'maxNumberOfTuples'):
                v = getattr(prod, 'maxNumberOfTuples')
                delattr(prod, 'maxNumberOfTuples')
                setattr(prod, 'maxNumberOfTuples', cms.string(str(v.value())))
            
            if not hasattr(prod, 'avgCellsPerCell'):
                setattr(prod, 'avgCellsPerCell', cms.double(0.071))
            
            if not hasattr(prod, 'avgCellsPerHit'):
                setattr(prod, 'avgCellsPerHit', cms.double(27))
            
            if not hasattr(prod, 'avgHitsPerTrack'):
                setattr(prod, 'avgHitsPerTrack', cms.double(4.5))
            
            if not hasattr(prod, 'avgTracksPerCell'):
                setattr(prod, 'avgTracksPerCell', cms.double(0.127))
            
            for par in ca_parameters:
                if hasattr(prod, par):
                    delattr(prod,par)

    for prod in producers_by_type(process, 'PixelTrackProducerFromSoAAlpakaPhase1'):
        print(prod)

    return process

# CMSSW version specific customizations
def customizeHLTforCMSSW(process, menuType="GRun"):

    process = customiseForOffline(process)

    # add call to action function in proper order: newest last!
    # process = customiseFor12718(process)
    
    process = customizeHLTforXYZ(process)

    process = customizeHLTfor46935(process)
    process = customizeHLTfor47017(process)
    process = customizeHLTfor47079(process)
    process = customizeHLTfor47047(process)
    process = customizeHLTfor47107(process)
    process = customizeHLTfor47191(process)
    
    return process
