import FWCore.ParameterSet.Config as cms
# validation & dqm modules 
from HLTrigger.Configuration.phase2TrackingValidation_cff import *

def addSimTruth(process):

    process.simHitTPAssocProducer = simHitTPAssociation.clone()
    # Sim Track/Tracking particle associator to clusters
    process.tpClusterProducer = tpClusterProducer.clone()
    # Utility to associate the number of layers to TPs
    process.trackingParticleNumberOfLayersProducer = trackingParticleNumberOfLayersProducer.clone()
    # TP to Track association
    # The associator itself
    process.quickTrackAssociatorByHits = quickTrackAssociatorByHits.clone()
    # The association to pixel tracks
    process.trackingParticlePixelTrackAssociation = trackingParticlePixelTrackAssociation.clone()
    # The association to general track
    process.trackingParticleGeneralTrackAssociation = trackingParticleGeneralTrackAssociation.clone()

    #The validators
    process.trackValidatorPixelTrackingOnly = trackValidatorPixelTrackingOnly.clone()
    process.trackValidatorGeneralTrackingOnly = trackValidatorGeneralTrackingOnly.clone()
    
    #Associating the vertices to the sim PVs
    process.VertexAssociatorByPositionAndTracks = VertexAssociatorByPositionAndTracks.clone()
    # ... and for pixel vertices
    process.VertexAssociatorByPositionAndTracksPixel = VertexAssociatorByPositionAndTracksPixel.clone()

    #The vertex validator
    process.vertexAnalysis = vertexAnalysis.clone()

    process.tracksValidationTruth = cms.Task(process.simHitTPAssocProducer,
                                             process.VertexAssociatorByPositionAndTracksPixel,
                                             process.VertexAssociatorByPositionAndTracks, 
                                             process.quickTrackAssociatorByHits, 
                                             process.tpClusterProducer, 
                                             process.trackingParticleNumberOfLayersProducer, 
                                             process.trackingParticleGeneralTrackAssociation,
                                             process.trackingParticlePixelTrackAssociation)

    return process

def addTrackingValidation(process):

    # process.TrackMon_gentk  = TrackMon_gentk.clone()
    # process.TrackSplitMonitor = TrackSplitMonitor.clone()
    # process.TrackerCollisionSelectedTrackMonCommongeneralTracks = TrackerCollisionSelectedTrackMonCommongeneralTracks.clone()
    # process.dqmInfoTracking = dqmInfoTracking.clone()
    # process.pvMonitor = pvMonitor.clone()

    import DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi
    process.TrackMon_ckf 					   = DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi.TrackEffMon.clone()
    process.TrackMon_ckf.TKTrackCollection                     = 'generalTracks'#ctfWithMaterialTracksBeamHaloMuon'#rsWithMaterialTracksP5'#muons'#globalCosmicMuons'#ctfWithMaterialTracksP5'
    process.TrackMon_ckf.AlgoName                              = 'CKFTk'
    process.TrackMon_ckf.FolderName                            = 'Tracking/TrackParameters/TrackEfficiency'

    process.TrackMonintor = TrackMon.clone()
    process = addSimTruth(process)

    from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
    process.dqmInfoTracking = DQMEDAnalyzer('DQMEventInfo',
        subSystemFolder = cms.untracked.string('Tracking')
    )

    #DQM
    #process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    #    dataset = cms.untracked.PSet(
    #        dataTier = cms.untracked.string('DQMIO'),
    #        filterName = cms.untracked.string('')
    #    ),
    #    fileName = cms.untracked.string('file:Phase2HLT_DQM.root'),
    #    outputCommands = process.DQMEventContent.outputCommands,
    #    splitLevel = cms.untracked.int32(0)
   #)
   #process.DQMoutput_step = cms.EndPath(process.DQMoutput)
    process.dqm = cms.Sequence(process.TrackMonintor + process.dqmInfoTracking)
    # process.dqm = cms.Task()#process.TrackMonintor)
    process.dqm_step = cms.EndPath(process.dqm)

    #Validation
    process.tracksValidation = cms.Sequence(process.trackValidatorGeneralTrackingOnly + process.trackValidatorPixelTrackingOnly)# + process.vertexAnalysis)

    process.validation = cms.Sequence(process.tracksValidation,process.tracksValidationTruth)
    process.validation_step = cms.EndPath(process.validation)

    process.schedule.extend([process.validation_step,
        process.dqm_step,
        process.DQMoutput_step])
    
    return process

def customisePhase2HLTForTrackingOnly(process):
    
    #Baseline tracking path
    process.HLTTrackingV61Path = cms.Path(process.HLTTrackingV61Sequence)
 
    process.localTask = cms.Task(process.RawToDigiTask, process.calolocalrecoTask)
    process.localSeq = cms.Sequence(process.localTask) #For the moment no MTD,process.mtdRecoTask)
    process.localPath = cms.Path(process.localSeq)

    process.vertexRecoTask = cms.Task(process.ak4CaloJetsForTrk, process.initialStepPVTask, process.offlinePrimaryVertices, process.trackRefsForJetsBeforeSorting, process.trackWithVertexRefSelectorBeforeSorting, process.unsortedOfflinePrimaryVertices,process.goodOfflinePrimaryVertices)

    process.vertexRecoSeq = cms.Sequence(process.vertexRecoTask) ## No MTD : ,process.vertex4DrecoTask)
    process.vertexRecoPath = cms.Path(process.vertexRecoSeq)    

    ##Local Reco
    process.localPath = cms.Path(process.RawToDigiTask,process.localrecoTask)

    process.schedule = cms.Schedule(*[
        process.localPath,
        process.HLTTrackingV61Path,
        process.vertexRecoPath,
        ])
        
    return process


def customisePhase2HLTForPatatrack(process):

    from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
    process.load("Configuration.StandardSequences.Accelerators_cff")

    if not hasattr(process, "CUDAService"):
        from HeterogeneousCore.CUDAServices.CUDAService_cfi import CUDAService
        process.add_(CUDAService)

    from RecoVertex.BeamSpotProducer.offlineBeamSpotToCUDA_cfi import offlineBeamSpotToCUDA as _offlineBeamSpotToCUDA
    process.onlineBeamSpotToCUDA = _offlineBeamSpotToCUDA.clone(src = cms.InputTag('hltOnlineBeamSpot'))

    from RecoLocalTracker.SiPixelRecHits.pixelCPEFastESProducerPhase2_cfi import pixelCPEFastESProducerPhase2
    process.PixelCPEFastESProducerPhase2 = pixelCPEFastESProducerPhase2.clone()
    ### SiPixelClusters on GPU

    process.siPixelClustersLegacy = process.siPixelClusters.clone()

    from RecoLocalTracker.SiPixelClusterizer.siPixelPhase2DigiToClusterCUDA_cfi import siPixelPhase2DigiToClusterCUDA as _siPixelPhase2DigiToClusterCUDA
    process.siPixelClustersCUDA = _siPixelPhase2DigiToClusterCUDA.clone()
    
    from EventFilter.SiPixelRawToDigi.siPixelDigisSoAFromCUDA_cfi import siPixelDigisSoAFromCUDA as _siPixelDigisSoAFromCUDA
    process.siPixelDigisPhase2SoA = _siPixelDigisSoAFromCUDA.clone(
        src = "siPixelClustersCUDA"
    )

    from RecoLocalTracker.SiPixelClusterizer.siPixelDigisClustersFromSoAPhase2_cfi import siPixelDigisClustersFromSoAPhase2 as _siPixelDigisClustersFromSoAPhase2

    process.siPixelClusters = SwitchProducerCUDA(
        cpu = cms.EDAlias(
            siPixelClustersLegacy = cms.VPSet(cms.PSet(
                type = cms.string('SiPixelClusteredmNewDetSetVector')
            ))
            ),
        cuda = _siPixelDigisClustersFromSoAPhase2.clone(
            clusterThreshold_layer1 = 4000,
            clusterThreshold_otherLayers = 4000,
            src = "siPixelDigisPhase2SoA",
            produceDigis = False
            )
    )

    process.siPixelClustersTask = cms.Task(
                            process.onlineBeamSpotToCUDA,
                            process.siPixelClustersLegacy,
                            process.siPixelClustersCUDA,
                            process.siPixelDigisPhase2SoA,
                            process.siPixelClusters)
    
    ### SiPixel Hits

    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitCUDAPhase2_cfi import siPixelRecHitCUDAPhase2 as _siPixelRecHitCUDAPhase2
    process.siPixelRecHitsCUDA = _siPixelRecHitCUDAPhase2.clone(
        src = cms.InputTag('siPixelClustersCUDA'),
        beamSpot = "onlineBeamSpotToCUDA"
    )
    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromLegacyPhase2_cfi import siPixelRecHitSoAFromLegacyPhase2 as _siPixelRecHitsSoAPhase2
    process.siPixelRecHitsCPU = _siPixelRecHitsSoAPhase2.clone(
        convertToLegacy=True, 
        src = 'siPixelClusters',
        CPE = 'PixelCPEFastPhase2',
        beamSpot = "hltOnlineBeamSpot")

    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromCUDAPhase2_cfi import siPixelRecHitSoAFromCUDAPhase2 as _siPixelRecHitSoAFromCUDAPhase2
    process.siPixelRecHitsSoA = SwitchProducerCUDA(
        cpu = cms.EDAlias(
            siPixelRecHitsCPU = cms.VPSet(
                 cms.PSet(type = cms.string("pixelTopologyPhase2TrackingRecHitSoAHost")),
                 cms.PSet(type = cms.string("uintAsHostProduct"))
             )),
        cuda = _siPixelRecHitSoAFromCUDAPhase2.clone()

    )

    
    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromCUDAPhase2_cfi import siPixelRecHitFromCUDAPhase2 as _siPixelRecHitFromCUDAPhase2

    _siPixelRecHits = SwitchProducerCUDA(
        cpu = cms.EDAlias(
            siPixelRecHitsCPU = cms.VPSet(
                 cms.PSet(type = cms.string("SiPixelRecHitedmNewDetSetVector")),
                 cms.PSet(type = cms.string("uintAsHostProduct"))
             )),
        cuda = _siPixelRecHitFromCUDAPhase2.clone(
            pixelRecHitSrc = cms.InputTag('siPixelRecHitsCUDA'),
            src = cms.InputTag('siPixelClusters'),
        )
    )

    process.siPixelRecHits = _siPixelRecHits.clone()
    process.siPixelRecHitsTask = cms.Task(
        process.siPixelRecHitsCUDA,
        process.siPixelRecHitsCPU,
        process.siPixelRecHits,
        process.siPixelRecHitsSoA
        )

    ### Pixeltracks

    from RecoTracker.PixelSeeding.caHitNtupletCUDAPhase2_cfi import caHitNtupletCUDAPhase2 as _pixelTracksCUDAPhase2
    # process.pixelTracksCUDA = _pixelTracksCUDAPhase2.clone(
    #     pixelRecHitSrc = "siPixelRecHitsCUDA",
    #     idealConditions = False,
    #     onGPU = True,
    #     includeJumpingForwardDoublets = True,
    #     minHitsPerNtuplet = 4,
    #     z0Cut = 10,
    #     hardCurvCut = 0.012,
    #     ptCut = 0.95,
    #     dupPassThrough = False
        
    # )

    process.pixelTracksCUDA = cms.EDProducer("CAHitNtupletCUDAPhase2",

    # CAThetaCutBarrel = cms.double(0.006),
    # CAThetaCutForward = cms.double(0.001363453691948839),
    # dcaCutInnerTriplet = cms.double(0.016623433752720233),
    # dcaCutOuterTriplet = cms.double(0.20183371949043288),
    # doClusterCut = cms.bool(True),
    # doPtCut = cms.bool(True),
    # doSharedHitCut = cms.bool(True),
    # doZ0Cut = cms.bool(True),
    # dupPassThrough = cms.bool(False),
    # earlyFishbone = cms.bool(True),
    # fillStatistics = cms.bool(False),
    # fitNas4 = cms.bool(False),
    # hardCurvCut = cms.double(0.6212705662183171),
    # idealConditions = cms.bool(False),
    # includeFarForwards = cms.bool(True),
    # includeJumpingForwardDoublets = cms.bool(True),
    # lateFishbone = cms.bool(False),
    # maxNumberOfDoublets = cms.uint32(2621440),
    # mightGet = cms.optional.untracked.vstring,
    # minHitsForSharingCut = cms.uint32(10),
    # minHitsPerNtuplet = cms.uint32(3),
    # onGPU = cms.bool(True),
    # phiCuts = cms.vint32(
    #     903, 722, 996, 692, 726,
    #     992, 486, 815, 1000, 400,
    #     561, 757, 760, 466, 1000,
    #     1000, 489, 751, 542, 721,
    #     1000, 1000, 400, 763, 917,
    #     400, 1000, 906, 400, 941,
    #     592, 603, 1000, 1000, 772,
    #     1000, 516, 674, 400, 859,
    #     965, 1000, 570, 1000, 437,
    #     599, 662, 668, 686, 400,
    #     428, 910, 776, 499, 916
    # ),
    # pixelRecHitSrc = cms.InputTag("siPixelRecHitsCUDA"),
    # ptCut = cms.double(0.8500000238418579),
    # ptmin = cms.double(0.8999999761581421),
    # trackQualityCuts = cms.PSet(
    #     maxChi2 = cms.double(5),
    #     maxTip = cms.double(0.3),
    #     maxZip = cms.double(12),
    #     minPt = cms.double(0.5)
    # ),
    # useRiemannFit = cms.bool(False),
    # useSimpleTripletCleaner = cms.bool(True),
    # z0Cut = cms.double(12.707861959253352)

    ## no jump
    # CAThetaCutBarrel = cms.double(0.002853725128146409),
    # CAThetaCutForward = cms.double(0.016065619347025797),
    # dcaCutInnerTriplet = cms.double(0.012782204227053838),
    # dcaCutOuterTriplet = cms.double(0.12081458184336838),
    # doClusterCut = cms.bool(True),
    # doPtCut = cms.bool(True),
    # doSharedHitCut = cms.bool(True),
    # doZ0Cut = cms.bool(True),
    # dupPassThrough = cms.bool(False),
    # earlyFishbone = cms.bool(True),
    # fillStatistics = cms.bool(False),
    # fitNas4 = cms.bool(False),
    # hardCurvCut = cms.double(0.3423589999174777),
    # idealConditions = cms.bool(False),
    # includeFarForwards = cms.bool(True),
    # includeJumpingForwardDoublets = cms.bool(True),
    # lateFishbone = cms.bool(False),
    # maxNumberOfDoublets = cms.uint32(2621440),
    # mightGet = cms.optional.untracked.vstring,
    # minHitsForSharingCut = cms.uint32(10),
    # minHitsPerNtuplet = cms.uint32(3),
    # onGPU = cms.bool(True),
    # phiCuts = cms.vint32(
    #     674, 593, 843, 844, 501,
    #     751, 688, 911, 670, 400,
    #     641, 839, 777, 661, 791,
    #     854, 930, 634, 412, 588,
    #     726, 755, 733, 512, 901,
    #     400, 809, 465, 654, 573,
    #     573, 777, 623, 682, 528,
    #     400, 406, 746, 975, 906,
    #     755, 477, 764, 872, 642,
    #     621, 853, 718, 965, 775,
    #     707, 607, 843, 768, 634
    # ),
    # pixelRecHitSrc = cms.InputTag("siPixelRecHitsCUDA"),
    # ptCut = cms.double(0.8500000238418579),
    # ptmin = cms.double(0.8999999761581421),
    # trackQualityCuts = cms.PSet(
    #     maxChi2 = cms.double(5),
    #     maxTip = cms.double(0.3),
    #     maxZip = cms.double(12),
    #     minPt = cms.double(0.5)
    # ),
    # useRiemannFit = cms.bool(False),
    # useSimpleTripletCleaner = cms.bool(True),
    # z0Cut = cms.double(9.958220068831185)

    ## with jump
    CAThetaCutBarrel = cms.double(0.002518561301574317),
    CAThetaCutForward = cms.double(0.009649516058900914),
    dcaCutInnerTriplet = cms.double(0.007099381977144485),
    dcaCutOuterTriplet = cms.double(0.14969121682516986),
    doClusterCut = cms.bool(True),
    doPtCut = cms.bool(True),
    doSharedHitCut = cms.bool(True),
    doZ0Cut = cms.bool(True),
    dupPassThrough = cms.bool(False),
    earlyFishbone = cms.bool(True),
    fillStatistics = cms.bool(False),
    fitNas4 = cms.bool(False),
    hardCurvCut = cms.double(0.707609612658449),
    idealConditions = cms.bool(False),
    includeFarForwards = cms.bool(True),
    includeJumpingForwardDoublets = cms.bool(True),
    lateFishbone = cms.bool(False),
    maxNumberOfDoublets = cms.uint32(2621440),
    mightGet = cms.optional.untracked.vstring,
    minHitsForSharingCut = cms.uint32(10),
    minHitsPerNtuplet = cms.uint32(3),
    onGPU = cms.bool(True),
    phiCuts = cms.vint32(
        716, 841, 736, 467, 999,
        423, 909, 862, 983, 950,
        735, 479, 542, 420, 974,
        987, 465, 423, 629, 822,
        449, 462, 851, 922, 848,
        422, 743, 963, 966, 561,
        572, 415, 628, 438, 945,
        437, 934, 743, 653, 412,
        773, 790, 952, 932, 459,
        516, 472, 928, 655, 479,
        971, 983, 788, 982, 497
    ),
    pixelRecHitSrc = cms.InputTag("siPixelRecHitsCUDA"),
    ptCut = cms.double(0.8500000238418579),
    ptmin = cms.double(0.8999999761581421),
    trackQualityCuts = cms.PSet(
        maxChi2 = cms.double(5),
        maxTip = cms.double(0.3),
        maxZip = cms.double(12),
        minPt = cms.double(0.5)
    ),
    useRiemannFit = cms.bool(False),
    useSimpleTripletCleaner = cms.bool(True),
    z0Cut = cms.double(12.851884460665062)

)

    from RecoTracker.PixelTrackFitting.pixelTrackSoAFromCUDAPhase2_cfi import pixelTrackSoAFromCUDAPhase2 as _pixelTracksSoAPhase2
    process.pixelTracksSoA = SwitchProducerCUDA(
        # build pixel ntuplets and pixel tracks in SoA format on the CPU
        cpu = _pixelTracksCUDAPhase2.clone(
            pixelRecHitSrc = "siPixelRecHitsCPU",
            idealConditions = False,
            onGPU = False,
            includeJumpingForwardDoublets = True,
            dupPassThrough = False,
	        minHitsPerNtuplet = 4,
            z0Cut = 9.5,
            hardCurvCut = 0.012,
            ptCut = 1,
        ),
        cuda = _pixelTracksSoAPhase2.clone()
    )

    from RecoTracker.PixelTrackFitting.pixelTrackProducerFromSoAPhase2_cfi import pixelTrackProducerFromSoAPhase2 as _pixelTrackProducerFromSoAPhase2
    process.hltPhase2PixelTracks = _pixelTrackProducerFromSoAPhase2.clone(
        pixelRecHitLegacySrc = "siPixelRecHits",
        beamSpot = "hltOnlineBeamSpot"
    )

    process.pixelTracksTask = cms.Task(
        process.pixelTracksCUDA,
        process.pixelTracksSoA,
        process.hltPhase2PixelTracks
    )

    process.HLTTrackingV61Task = cms.Task(process.MeasurementTrackerEvent, 
                                          process.generalTracks, 
                                          process.highPtTripletStepClusters, 
                                          process.highPtTripletStepHitDoublets, 
                                          process.highPtTripletStepHitTriplets, 
                                          process.highPtTripletStepSeedLayers, 
                                          process.highPtTripletStepSeeds, 
                                          process.highPtTripletStepTrackCandidates, 
                                          process.highPtTripletStepTrackCutClassifier, 
                                          process.highPtTripletStepTrackSelectionHighPurity, 
                                          process.hltPhase2PixelTracksAndHighPtStepTrackingRegions,
                                          process.highPtTripletStepTracks, 
                                          process.initialStepSeeds, 
                                          process.initialStepTrackCandidates, 
                                          process.initialStepTrackCutClassifier, 
                                          process.initialStepTrackSelectionHighPurity, 
                                          process.initialStepTracks, 
                                          process.hltPhase2PixelVertices, ## for the moment leaving it as it was
                                          )

    process.trackerClusterCheckTask = cms.Task(process.trackerClusterCheck,
                                               process.siPhase2Clusters, 
                                               process.siPixelClusterShapeCache)
    process.HLTTrackingV61Sequence = cms.Sequence(process.trackerClusterCheckTask,
                                                  process.siPixelClustersTask,
                                                  process.siPixelRecHitsTask,
                                                  process.pixelTracksTask,
                                                  process.HLTTrackingV61Task)
    
    return process

def customisePhase2HLTForPatatrackTwoIters(process):

    process = customisePhase2HLTForPatatrack(process)
    
    process.pixelTracksCUDA.minHitsPerNtuplet = 3
    process.pixelTracksCUDA.includeFarForwards = True
    process.pixelTracksCUDA.includeJumpingForwardDoublets = False
    process.pixelTracksCUDA.dupPassThrough = False
    # process.pixelTracksCUDA.doClusterCut = True
    process.pixelTracksCUDA.minHitsForSharingCut = 6
    process.pixelTracksCUDA.lateFishbone = True
    process.pixelTracksCUDA.earlyFishbone = True
    process.pixelTracksCUDA.doSharedHitCut = True
    process.pixelTracksCUDA.useSimpleTripletCleaner = True
    process.pixelTracksCUDA.fillStatistics = True
    process.pixelTracksCUDA.z0Cut = 11
    process.pixelTracksCUDA.hardCurvCut = 0.012
    process.pixelTracksCUDA.ptCut = 0.95

    from CommonTools.RecoAlgos.TrackWithVertexSelector_cfi import trackWithVertexSelector

    process.pixelTracksQuads = trackWithVertexSelector.clone(
        src = "pixelTracks",
        ptMin = 0.9,
        quality = "loose",
        useVtx = False,
        numberOfValidPixelHits = 4,
        ptErrorCut = 5.0,
        vertexTag = '' # just to avoid circular dependencies from offlinePrimaryVertices
    )
    #     etaMin = cms.double( 0.0 ),
    #     cms.EDProducer( "TrackWithVertexSelector",
        
    #     etaMax = cms.double( 5.0 ),
        
    #     ptMax = cms.double( 500.0 ),
    #     d0Max = cms.double( 999.0 ),
    #     dzMax = cms.double( 999.0 ),
    #     normalizedChi2 = cms.double( 999999.0 ),
    #     numberOfValidHits = cms.uint32( 0 ),
    #     numberOfLostHits = cms.uint32( 999 ),
    #     maxNumberOfValidPixelHits = cms.uint32( 999 ),
    #     ptErrorCut = cms.double( 5.0 ),
        
        
    #     vertexTag = cms.InputTag( "hltPhase2PixelVertices" ),
    #     timesTag = cms.InputTag( "" ),
    #     timeResosTag = cms.InputTag( "" ),
    #     nVertices = cms.uint32( 200 ),
    #     vtxFallback = cms.bool( True ),
    #     zetaVtx = cms.double( 0.3 ),
    #     rhoVtx = cms.double( 0.2 ),
    #     nSigmaDtVertex = cms.double( 0.0 ),
    #     copyExtras = cms.untracked.bool( False ),
    #     copyTrajectories = cms.untracked.bool( False )
    # )

    process.pixelTracksTrips = process.pixelTracksQuads.clone(maxNumberOfValidPixelHits = 3, 
                                                              numberOfValidPixelHits = 0)
    process.initialStepSeeds.InputCollection = "pixelTracksQuads"
    process.highPtTripletStepSeeds = process.initialStepSeeds.clone(InputCollection = "pixelTracksTrips")
    process.pixelTracksTask.add(process.pixelTracksQuads,process.pixelTracksTrips)
    
    return process

def customisePhase2HLTForPatatrackOneIter(process):

    process = customisePhase2HLTForPatatrack(process)

    process.pixelTracksSoA.cpu.minHitsPerNtuplet = 3
    process.pixelTracksSoA.cpu.includeFarForwards = True
    process.pixelTracksSoA.cpu.includeJumpingForwardDoublets = True
    process.pixelTracksSoA.cpu.doClusterCut = True
    process.pixelTracksSoA.cpu.earlyFishbone = False
    process.pixelTracksSoA.cpu.lateFishbone = False
    process.pixelTracksSoA.cpu.doSharedHitCut = False

    process.pixelTracksCUDA.minHitsPerNtuplet = 3
    process.pixelTracksCUDA.includeFarForwards = True
    process.pixelTracksCUDA.includeJumpingForwardDoublets = False
    process.pixelTracksCUDA.dupPassThrough = False
    # process.pixelTracksCUDA.doClusterCut = True
    process.pixelTracksCUDA.minHitsForSharingCut = 10
    process.pixelTracksCUDA.lateFishbone = True
    process.pixelTracksCUDA.doSharedHitCut = True
    process.pixelTracksCUDA.useSimpleTripletCleaner = True
    # process.pixelTracksCUDA.fillStatistics = True
    #process.pixelTracksCUDA.z0Cut = 9
    process.pixelTracksCUDA.hardCurvCut = 0.012
    process.pixelTracksCUDA.ptCut = 0.95

    process.pixelTracksClean = cms.EDProducer( "TrackWithVertexSelector",
        src = cms.InputTag( "pixelTracks" ),
        etaMin = cms.double( 0.0 ),
        etaMax = cms.double( 5.0 ),
        ptMin = cms.double( 0.9 ),
        ptMax = cms.double( 500.0 ),
        d0Max = cms.double( 999.0 ),
        dzMax = cms.double( 999.0 ),
        maxNumberOfValidPixelHits = cms.uint32( 999 ),
        normalizedChi2 = cms.double( 999999.0 ),
        numberOfValidHits = cms.uint32( 0 ),
        numberOfLostHits = cms.uint32( 999 ),
        numberOfValidPixelHits = cms.uint32( 3 ),
        ptErrorCut = cms.double( 1.0 ),
        quality = cms.string( "loose" ),
        useVtx = cms.bool( False ),
        vertexTag = cms.InputTag( "hltPhase2PixelVertices" ),
        timesTag = cms.InputTag( "" ),
        timeResosTag = cms.InputTag( "" ),
        nVertices = cms.uint32( 200 ),
        vtxFallback = cms.bool( True ),
        zetaVtx = cms.double( 0.3 ),
        rhoVtx = cms.double( 0.2 ),
        nSigmaDtVertex = cms.double( 0.0 ),
        copyExtras = cms.untracked.bool( False ),
        copyTrajectories = cms.untracked.bool( False )
    )
    
    process.pixelTracksQuads = process.pixelTracksClean.clone(numberOfValidPixelHits = cms.uint32( 4 ))
    # process.trackValidatorPixelTrackingOnly.label = cms.VInputTag(["pixelTracks","pixelTracksClean","pixelTracksQuads"])

    process.initialStepSeeds.InputCollection = "pixelTracksClean"
    # process.pixelTracksCUDA.earlyFishbone = False
    # process.pixelTracksCUDA.lateFishbone = False
    # process.pixelTracksCUDA.doSharedHitCut = False

    # process.generalTracks = process.initialStepTracks.clone()
    # process.firstStepPrimaryVerticesUnsorted.TrackLabel = "generalTracks"
    # process.initialStepSeeds.useProtoTrackKinematics = True
    # process.initialStepSeeds.usePV = False
    # process.InputVertexCollection = "hltPhase2PixelVertices"
    
    process.generalTracks.TrackProducers = ["initialStepTrackSelectionHighPurity"]
    process.generalTracks.indivShareFrac = [1.0]
    process.generalTracks.hasSelector = [0]
    process.generalTracks.selectedTrackQuals = ["initialStepTrackSelectionHighPurity"]
    process.generalTracks.setsToMerge.pQual = True
    process.generalTracks.setsToMerge.tLists = [0]

    #process.initialStepTrackCandidates.cleanTrajectoryAfterInOut =  False 
    #process.initialStepTrackCandidates.doSeedingRegionRebuilding =  True 
    #process.initialStepTrackCandidates.onlyPixelHitsForSeedCleaner =  False 
    #process.initialStepTrackCandidates.reverseTrajectories =  False 
    #process.initialStepTrackCandidates.useHitsSplitting =  False 
    #process.initialStepTrackCandidates.RedundantSeedCleaner = "none"
    
    process.HLTTrackingV61Task = cms.Task(process.MeasurementTrackerEvent, 
                                          process.pixelTracksQuads,
                                          process.pixelTracksClean,
                                          process.generalTracks, 
                                          process.initialStepSeeds, 
                                          process.initialStepTrackCandidates, 
                                          process.initialStepTrackCutClassifier, 
                                          process.initialStepTrackSelectionHighPurity, 
                                          process.initialStepTracks, 
                                          process.hltPhase2PixelVertices, ## for the moment leaving it as it was
                                          )
    
    if hasattr(process, "trackingNtuple"):
        
        _seedsProducers = ["initialStepSeeds"]
        process.trackingNtuple.trackCandidates = [i.replace("seedTracks", "").replace("Seeds", "TrackCandidates") for i in _seedsProducers]

        process.seedValidationSequence = cms.Sequence(process.seedValidation)
        process.validation_step = cms.EndPath(process.trackingNtupleSequence + process.seedValidationSequence)

        process.trackingNtupleSeedSelectors = cms.Task(process.seedTracksInitialStep)
        # process.trackingNtupleTask.add(process.trackingNtupleSeedSelectors,process.trackingParticlesIntime)
        process.trackingNtuple.seedTracks = ["seedTracksInitialStep"]

        process.trackingNtupleTaskWithTruth = cms.Task(process.tracksValidationTruth, process.trackingNtupleTask)
        process.trackingNtupleSequence = cms.Sequence(process.trackingNtuple, process.trackingNtupleTaskWithTruth)
        
        process.ttrhbwor.PixelCPE = 'PixelCPEFastPhase2'

        process.trackingNtuple.clusterMasks = cms.untracked.VPSet()

    return process
    

def customisePhase2HLTForWeightedVertexing(process):

    process.unsortedOfflinePrimaryVertices.vertexCollections = cms.VPSet(
                           [cms.PSet(label=cms.string(""),
                                     algorithm=cms.string("WeightedMeanFitter"),
                                     chi2cutoff = cms.double(2.5),
                                     minNdof=cms.double(0.0),
                                     useBeamConstraint = cms.bool(False),
                                     maxDistanceToBeam = cms.double(1.0)
                           ),
                           cms.PSet(label=cms.string("WithBS"),
                                     algorithm = cms.string('WeightedMeanFitter'),
                                     minNdof=cms.double(0.0),
                                     chi2cutoff = cms.double(2.5),
                                     useBeamConstraint = cms.bool(True),
                                     maxDistanceToBeam = cms.double(1.0)
                                     )
                           ]
                           )

    return process                        
   

def addTrackingNtuple(process):

    from Validation.RecoTrack.trackingNtuple_cfi import trackingNtuple
    # import Validation.RecoTrack.TrackValidation_cff as _TrackValidation_cff

    process.TFileService = cms.Service("TFileService",
        fileName = cms.string('trackingNtuple.root')
    )

    process.seedValidation = cms.EDAnalyzer("SeedValidation",
                                            src = cms.InputTag("initialStepSeeds"),
                                            tpMap = cms.InputTag("tpClusterProducer"),
                                            beamSpot = cms.InputTag("hltOnlineBeamSpot"),
                                            stops =  cms.InputTag("initialStepTrackCandidates"))
    
    process.seedValidationTrips = cms.EDAnalyzer("SeedValidation",
                                            src = cms.InputTag("highPtTripletStepSeeds"),
                                            tpMap = cms.InputTag("tpClusterProducer"),
                                            beamSpot = cms.InputTag("hltOnlineBeamSpot"),
                                            stops =  cms.InputTag("highPtTripletStepTrackCandidates"))
    process.trackingNtupleTask = cms.Task()

    from CommonTools.RecoAlgos.trackingParticleRefSelector_cfi import trackingParticleRefSelector as _trackingParticleRefSelector
    process.trackingParticlesIntime = _trackingParticleRefSelector.clone(
        signalOnly = False,
        intimeOnly = True,
        chargedOnly = False,
        tip = 1e5,
        lip = 1e5,
        minRapidity = -10,
        maxRapidity = 10,
        ptMin = 0,
    )
    process.trackingNtuple = trackingNtuple.clone()
    process.trackingNtuple.trackingParticles = "trackingParticlesIntime"
    process.trackingNtuple.trackingParticlesRef = True
    process.trackingNtuple.includeAllHits = False
    process.trackingNtuple.includeSeeds = True
    process.trackingNtuple.includeMVA = False
    process.trackingNtuple.includeTrackingParticles = True
    
    _seedsProducers = ["initialStepSeeds","highPtTripletStepSeeds"]
    process.trackingNtuple.trackCandidates = [i.replace("seedTracks", "").replace("Seeds", "TrackCandidates") for i in _seedsProducers]

    # Phase2
    process.trackingNtuple.pixelDigiSimLink = cms.untracked.InputTag('simSiPixelDigis', "Pixel")
    process.trackingNtuple.stripDigiSimLink = cms.untracked.InputTag('')
    process.trackingNtuple.phase2OTSimLink = cms.untracked.InputTag('simSiPixelDigis', "Tracker")

    from Validation.RecoTrack.trajectorySeedTracks_cfi import trajectorySeedTracks
    process.seedTracksInitialStep = trajectorySeedTracks.clone(src = cms.InputTag("initialStepSeeds"))
    process.seedTracksHighPtTripletStep = trajectorySeedTracks.clone(src = cms.InputTag("highPtTripletStepSeeds"))

    process.trackingNtupleSeedSelectors = cms.Task()
    process.trackingNtupleTask = cms.Task(process.seedTracksInitialStep,process.seedTracksHighPtTripletStep, process.trackingParticlesIntime)
    process.trackingNtuple.seedTracks = ["seedTracksInitialStep","seedTracksHighPtTripletStep"]

    process.trackingNtuple.clusterMasks = cms.untracked.VPSet(
        cms.PSet( 
            index = cms.untracked.uint32(22),
            src = cms.untracked.InputTag("highPtTripletStepClusters")
        )
    )

    process = addSimTruth(process)

    process.trackingNtupleTaskWithTruth = cms.Task(process.tracksValidationTruth, process.trackingNtupleTask)
    process.trackingNtupleSequence = cms.Sequence(process.trackingNtuple, process.trackingNtupleTaskWithTruth)
    process.seedValidationSequence = cms.Sequence(process.seedValidation + process.seedValidationTrips)
    process.validation_step = cms.EndPath(process.trackingNtupleSequence + process.seedValidationSequence)

    from RecoTracker.TransientTrackingRecHit.tkTransientTrackingRecHitBuilderESProducer_cfi import tkTransientTrackingRecHitBuilderESProducer
    process.ttrhbwor =  tkTransientTrackingRecHitBuilderESProducer.clone(StripCPE = 'FakeStripCPE',
                                                                Phase2StripCPE = 'Phase2StripCPE',
                                                                ComponentName = 'WithoutRefit',
                                                                PixelCPE = 'PixelCPEGeneric', #PixelCPEFastPhase2
                                                                Matcher = 'Fake',
                                                                ComputeCoarseLocalPositionFromDisk = False)

    process.schedule.extend([process.validation_step])
    
    return process

    
