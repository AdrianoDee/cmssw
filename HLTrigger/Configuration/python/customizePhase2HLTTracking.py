import FWCore.ParameterSet.Config as cms

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
    process.pixelTracksCUDA = cms.EDProducer("CAHitNtupletCUDAPhase2",
    CAThetaCutBarrel = cms.double(0.002853725128146409),
    CAThetaCutForward = cms.double(0.016065619347025797),
    dcaCutInnerTriplet = cms.double(0.012782204227053838),
    dcaCutOuterTriplet = cms.double(0.12081458184336838),
    doClusterCut = cms.bool(True),
    doPtCut = cms.bool(True),
    doSharedHitCut = cms.bool(True),
    doZ0Cut = cms.bool(True),
    dupPassThrough = cms.bool(False),
    earlyFishbone = cms.bool(True),
    fillStatistics = cms.bool(False),
    fitNas4 = cms.bool(False),
    hardCurvCut = cms.double(0.3423589999174777),
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
        674, 593, 843, 844, 501,
        751, 688, 911, 670, 400,
        641, 839, 777, 661, 791,
        854, 930, 634, 412, 588,
        726, 755, 733, 512, 901,
        400, 809, 465, 654, 573,
        573, 777, 623, 682, 528,
        400, 406, 746, 975, 906,
        755, 477, 764, 872, 642,
        621, 853, 718, 965, 775,
        707, 607, 843, 768, 634
    ),
    pixelRecHitSrc = cms.InputTag("siPixelRecHitsCUDA"),
    ptCut = cms.double(0.85),
    ptmin = cms.double(0.9),
    trackQualityCuts = cms.PSet(
        maxChi2 = cms.double(5),
        maxTip = cms.double(0.3),
        maxZip = cms.double(12),
        minPt = cms.double(0.5)
    ),
    useRiemannFit = cms.bool(False),
    useSimpleTripletCleaner = cms.bool(True),
    z0Cut = cms.double(9.958220068831185)
)

    from RecoTracker.PixelTrackFitting.pixelTrackSoAFromCUDAPhase2_cfi import pixelTrackSoAFromCUDAPhase2 as _pixelTracksSoAPhase2
    process.pixelTracksSoA = SwitchProducerCUDA(
        # build pixel ntuplets and pixel tracks in SoA format on the CPU
        cpu = _pixelTracksCUDAPhase2.clone(
            pixelRecHitSrc = "siPixelRecHitsCPU",
            idealConditions = False,
            onGPU = False,
            includeJumpingForwardDoublets = True,
            dupPassThrough = False 
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
    process.pixelTracksCUDA.minHitsForSharingCut = 10
    process.pixelTracksCUDA.lateFishbone = True
    process.pixelTracksCUDA.doSharedHitCut = True
    process.pixelTracksCUDA.useSimpleTripletCleaner = True
   
    process.pixelTracksClean = cms.EDProducer( "TrackWithVertexSelector",
        src = cms.InputTag( "hltPhase2PixelTracks" ),
        etaMin = cms.double( 0.0 ),
        etaMax = cms.double( 5.0 ),
        ptMin = cms.double( 0.85 ),
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
    
  
    process.initialStepSeeds.InputCollection = "pixelTracksClean"
   
    process.generalTracks.TrackProducers = ["initialStepTrackSelectionHighPurity"]
    process.generalTracks.indivShareFrac = [1.0]
    process.generalTracks.hasSelector = [0]
    process.generalTracks.selectedTrackQuals = ["initialStepTrackSelectionHighPurity"]
    process.generalTracks.setsToMerge.pQual = True
    process.generalTracks.setsToMerge.tLists = [0]
  
    process.HLTTrackingV61Task = cms.Task(process.MeasurementTrackerEvent, 
                     
                                          process.pixelTracksClean,
                                          process.generalTracks, 
                                          process.initialStepSeeds, 
                                          process.initialStepTrackCandidates, 
                                          process.initialStepTrackCutClassifier, 
                                          process.initialStepTrackSelectionHighPurity, 
                                          process.initialStepTracks, 
                                          process.hltPhase2PixelVertices # legacy gap pixel vertices
                                          )  

    return process
