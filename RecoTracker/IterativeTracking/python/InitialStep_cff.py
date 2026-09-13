import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016 as _tracker_apv_vfp30_2016
from Configuration.Eras.Modifier_fastSim_cff import fastSim

# for dnn classifier
from Configuration.ProcessModifiers.trackdnn_cff import trackdnn
from RecoTracker.IterativeTracking.dnnQualityCuts import qualityCutDictionary

# for no-loopers
from Configuration.ProcessModifiers.trackingNoLoopers_cff import trackingNoLoopers

### STEP 0 ###

# hit building
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *
from RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff import *

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
import RecoTracker.TkSeedingLayers.PixelLayerQuadruplets_cfi
initialStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone()
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(initialStepSeedLayers,
    layerList = RecoTracker.TkSeedingLayers.PixelLayerQuadruplets_cfi.PixelLayerQuadruplets.layerList.value()
)
trackingPhase2PU140.toModify(initialStepSeedLayers,
    layerList = RecoTracker.TkSeedingLayers.PixelLayerQuadruplets_cfi.PixelLayerQuadruplets.layerList.value()
)

# TrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpot_cfi import globalTrackingRegionFromBeamSpot as _globalTrackingRegionFromBeamSpot
initialStepTrackingRegions = _globalTrackingRegionFromBeamSpot.clone(RegionPSet = dict(
    ptMin        = 0.6,
    originRadius = 0.02,
    nSigmaZ      = 4.0
))
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase1.toModify(initialStepTrackingRegions, RegionPSet = dict(ptMin = 0.5))
from Configuration.Eras.Modifier_highBetaStar_cff import highBetaStar
highBetaStar.toModify(initialStepTrackingRegions,RegionPSet = dict(
     ptMin        = 0.05,
     originRadius = 0.2
))
trackingPhase2PU140.toModify(initialStepTrackingRegions, RegionPSet = dict(ptMin = 0.6,originRadius = 0.03))

# seeding
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
initialStepHitDoublets = _hitPairEDProducer.clone(
    seedingLayers   = 'initialStepSeedLayers',
    trackingRegions = 'initialStepTrackingRegions',
    maxElement      = 50000000,
    produceIntermediateHitDoublets = True,
)
from RecoTracker.PixelSeeding.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoTracker.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoTracker.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
initialStepHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets              = 'initialStepHitDoublets',
    produceSeedingHitSets = True,
    SeedComparitorPSet = RecoTracker.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone()
)
from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer_cff import seedCreatorFromRegionConsecutiveHitsEDProducer as _seedCreatorFromRegionConsecutiveHitsEDProducer
initialStepSeeds = _seedCreatorFromRegionConsecutiveHitsEDProducer.clone(
    seedingHitSets = 'initialStepHitTriplets',
)
from RecoTracker.PixelSeeding.caHitQuadrupletEDProducer_cfi import caHitQuadrupletEDProducer as _caHitQuadrupletEDProducer
_initialStepCAHitQuadruplets = _caHitQuadrupletEDProducer.clone(
    doublets = 'initialStepHitDoublets',
    extraHitRPhitolerance = initialStepHitTriplets.extraHitRPhitolerance,
    SeedComparitorPSet = initialStepHitTriplets.SeedComparitorPSet,
    maxChi2 = dict(
        pt1    = 0.7, pt2    = 2,
        value1 = 200, value2 = 50,
    ),
    useBendingCorrection = True,
    fitFastCircle        = True,
    fitFastCircleChi2Cut = True,
    CAThetaCut           = 0.0012,
    CAPhiCut             = 0.2,
)
highBetaStar.toModify(_initialStepCAHitQuadruplets,
    CAThetaCut = 0.0024,
    CAPhiCut   = 0.4
)
initialStepHitQuadruplets = _initialStepCAHitQuadruplets.clone()

trackingPhase1.toModify(initialStepHitDoublets, layerPairs = [0,1,2]) # layer pairs (0,1), (1,2), (2,3)

trackingPhase2PU140.toModify(initialStepHitDoublets, layerPairs = [0,1,2]) # layer pairs (0,1), (1,2), (2,3)
trackingPhase2PU140.toModify(initialStepHitQuadruplets,
    CAThetaCut = 0.0010,
    CAPhiCut   = 0.175,
)

from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer_cff import seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer as _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer
_initialStepSeedsConsecutiveHitsTripletOnly = _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer.clone(
    seedingHitSets     = 'initialStepHitTriplets',
    SeedComparitorPSet = dict(# FIXME: is this defined in any cfi that could be imported instead of copy-paste?
        ComponentName = 'PixelClusterShapeSeedComparitor',
        FilterAtHelixStage = cms.bool(False),
        FilterPixelHits = cms.bool(True),
        FilterStripHits = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
        ClusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCache')
    ),
)
trackingPhase1.toReplaceWith(initialStepSeeds, _initialStepSeedsConsecutiveHitsTripletOnly.clone(
        seedingHitSets = 'initialStepHitQuadruplets'
))
trackingPhase2PU140.toReplaceWith(initialStepSeeds, _initialStepSeedsConsecutiveHitsTripletOnly.clone(
        seedingHitSets = 'initialStepHitQuadruplets'
))
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
from FastSimulation.Tracking.SeedingMigration import _hitSetProducerToFactoryPSet
_fastSim_initialStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    trackingRegions = 'initialStepTrackingRegions',
    seedFinderSelector = dict( pixelTripletGeneratorFactory = _hitSetProducerToFactoryPSet(initialStepHitTriplets).clone(SeedComparitorPSet = dict(ComponentName = 'none')),
                               layerList = initialStepSeedLayers.layerList.value()
                             )
)
#new for phase1
#adding phase2 also
(trackingPhase1|trackingPhase2PU140).toModify(_fastSim_initialStepSeeds, seedFinderSelector = dict(
        pixelTripletGeneratorFactory = None,
        CAHitQuadrupletGeneratorFactory = _hitSetProducerToFactoryPSet(initialStepHitQuadruplets).clone(SeedComparitorPSet = dict(ComponentName = 'none')),
        #new parameters required for phase1 seeding
        BPix = dict(
            TTRHBuilder = 'WithoutRefit',
            HitProducer = 'TrackingRecHitProducer',
            ),
        FPix = dict(
            TTRHBuilder = 'WithoutRefit',
            HitProducer = 'TrackingRecHitProducer',
            ),
        layerPairs = initialStepHitDoublets.layerPairs.value()
        )
)

fastSim.toReplaceWith(initialStepSeeds,_fastSim_initialStepSeeds)

# building
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
_initialStepTrajectoryFilterBase = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt               = 0.2,
)
initialStepTrajectoryFilterBase = _initialStepTrajectoryFilterBase.clone(
    maxCCCLostHits = 0,
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
)
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016
_tracker_apv_vfp30_2016.toModify(initialStepTrajectoryFilterBase, maxCCCLostHits = 2)

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
(pp_on_XeXe_2017 | pp_on_AA).toModify(initialStepTrajectoryFilterBase, minPt=0.6)
highBetaStar.toModify(initialStepTrajectoryFilterBase, minPt = 0.05)

initialStepTrajectoryFilterInOut = initialStepTrajectoryFilterBase.clone(
    minimumNumberOfHits = 4,
    seedExtension       = 1,
    strictSeedExtension = True, # don't allow inactive
    pixelSeedExtension  = True,
)
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toReplaceWith(initialStepTrajectoryFilterBase, _initialStepTrajectoryFilterBase)
trackingPhase2PU140.toReplaceWith(initialStepTrajectoryFilterBase, _initialStepTrajectoryFilterBase)

import RecoTracker.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi
initialStepTrajectoryFilterShape = RecoTracker.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi.StripSubClusterShapeTrajectoryFilterTIX12.clone()
initialStepTrajectoryFilter = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet(
        cms.PSet( refToPSet_ = cms.string('initialStepTrajectoryFilterBase')),
    #    cms.PSet( refToPSet_ = cms.string('initialStepTrajectoryFilterShape'))
    ),
)

trackingPhase2PU140.toReplaceWith(initialStepTrajectoryFilter, TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt               = 0.2
))
import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
initialStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = 'initialStepChi2Est',
    nSigma        = 3.0,
    MaxChi2       = 30.0,
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose')),
    pTChargeCutThreshold = 15.
)
_tracker_apv_vfp30_2016.toModify(initialStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTiny')
)
trackingPhase2PU140.toModify(initialStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutNone'),
)


import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
initialStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilderIterativeDefault.clone(
    trajectoryFilter = dict(refToPSet_ = 'initialStepTrajectoryFilter'),
    alwaysUseInvalidHits = True,
    maxCand = 3,
    estimator = 'initialStepChi2Est',
    maxDPhiForLooperReconstruction = 2.0,
    maxPtForLooperReconstruction = 0.7,
)
trackingNoLoopers.toModify(initialStepTrajectoryBuilder,
                           maxPtForLooperReconstruction = 0.0)
trackingLowPU.toModify(initialStepTrajectoryBuilder, maxCand = 5)
trackingPhase1.toModify(initialStepTrajectoryBuilder,
    minNrOfHitsForRebuild = 1,
    keepOriginalIfRebuildFails = True,
)
trackingPhase2PU140.toModify(initialStepTrajectoryBuilder,
    minNrOfHitsForRebuild = 1,
    keepOriginalIfRebuildFails = True,
)

import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
# Give handle for CKF for HI
_initialStepTrackCandidatesCkf = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidatesIterativeDefault.clone(
    src = 'initialStepSeeds',
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = 50,
    onlyPixelHitsForSeedCleaner = True,
    TrajectoryBuilderPSet = dict(refToPSet_ = 'initialStepTrajectoryBuilder'),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True,
)
initialStepTrackCandidates = _initialStepTrackCandidatesCkf.clone()

from Configuration.ProcessModifiers.trackingMkFitInitialStep_cff import trackingMkFitInitialStep
from RecoTracker.MkFit.mkFitGeometryESProducer_cfi import mkFitGeometryESProducer
import RecoTracker.MkFit.mkFitSiPixelHitConverter_cfi as mkFitSiPixelHitConverter_cfi
import RecoTracker.MkFit.mkFitSiStripHitConverter_cfi as mkFitSiStripHitConverter_cfi
import RecoTracker.MkFit.mkFitPhase2HitConverter_cfi as mkFitPhase2HitConverter_cfi
import RecoTracker.MkFit.mkFitEventOfHitsProducer_cfi as mkFitEventOfHitsProducer_cfi
import RecoTracker.MkFit.mkFitSeedConverter_cfi as mkFitSeedConverter_cfi
import RecoTracker.MkFit.mkFitIterationConfigESProducer_cfi as mkFitIterationConfigESProducer_cfi
import RecoTracker.MkFit.mkFitProducer_cfi as mkFitProducer_cfi
import RecoTracker.MkFit.mkFitOutputConverter_cfi as mkFitOutputConverter_cfi
mkFitSiPixelHits = mkFitSiPixelHitConverter_cfi.mkFitSiPixelHitConverter.clone() # TODO: figure out better place for this module?
mkFitSiPhase2Hits = mkFitPhase2HitConverter_cfi.mkFitPhase2HitConverter.clone()
mkFitEventOfHits = mkFitEventOfHitsProducer_cfi.mkFitEventOfHitsProducer.clone() # TODO: figure out better place for this module?
initialStepTrackCandidatesMkFitSeeds = mkFitSeedConverter_cfi.mkFitSeedConverter.clone(
    seeds = 'initialStepSeeds',
)
initialStepTrackCandidatesMkFitConfig = mkFitIterationConfigESProducer_cfi.mkFitIterationConfigESProducer.clone(
    ComponentName = 'initialStepTrackCandidatesMkFitConfig',
    config = 'RecoTracker/MkFit/data/mkfit-phase1-initialStep.json',
)
initialStepTrackCandidatesMkFit = mkFitProducer_cfi.mkFitProducer.clone(
    seeds = 'initialStepTrackCandidatesMkFitSeeds',
    config = ('', 'initialStepTrackCandidatesMkFitConfig'),
)
trackingMkFitInitialStep.toReplaceWith(initialStepTrackCandidates, mkFitOutputConverter_cfi.mkFitOutputConverter.clone(
    seeds = 'initialStepSeeds',
    mkFitSeeds = 'initialStepTrackCandidatesMkFitSeeds',
    tracks = 'initialStepTrackCandidatesMkFit',
))
(pp_on_XeXe_2017 | pp_on_AA).toModify(initialStepTrackCandidatesMkFitConfig, minPt=0.6)

import FastSimulation.Tracking.TrackCandidateProducer_cfi
fastSim.toReplaceWith(initialStepTrackCandidates,
                      FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
        src = 'initialStepSeeds',
        MinNumberOfCrossedLayers = 3
))


# fitting
import RecoTracker.TrackProducer.TrackProducerIterativeDefault_cfi
initialStepTracks = RecoTracker.TrackProducer.TrackProducerIterativeDefault_cfi.TrackProducerIterativeDefault.clone(
    src           = 'initialStepTrackCandidates',
    AlgorithmName = 'initialStep',
    Fitter        = 'FlexibleKFFittingSmoother'
)
fastSim.toModify(initialStepTracks, TTRHBuilder = 'WithoutRefit')

from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
phase2_timing_layer.toModify(initialStepTracks, TrajectoryInEvent = True)

#vertices
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices as _offlinePrimaryVertices
firstStepPrimaryVerticesUnsorted = _offlinePrimaryVertices.clone(
    TrackLabel = 'initialStepTracks',
    vertexCollections = [_offlinePrimaryVertices.vertexCollections[0].clone()]
)
(pp_on_XeXe_2017 | pp_on_AA).toModify(firstStepPrimaryVerticesUnsorted,
    TkFilterParameters = dict(
        trackQuality = 'any',
        maxNumTracksThreshold = 2**31-1
    ) 
)

# we need a replacment for the firstStepPrimaryVerticesUnsorted
# that includes tracker information of signal and pile up
# after mixing there is no such thing as initialStepTracks,
# so we replace the input collection for firstStepPrimaryVerticesUnsorted with generalTracks
firstStepPrimaryVerticesBeforeMixing =  firstStepPrimaryVerticesUnsorted.clone()
fastSim.toModify(firstStepPrimaryVerticesUnsorted, TrackLabel = 'generalTracks')


from RecoJets.JetProducers.TracksForJets_cff import trackRefsForJets
initialStepTrackRefsForJets = trackRefsForJets.clone(
    src = 'initialStepTracks'
)
fastSim.toModify(initialStepTrackRefsForJets, src = 'generalTracks')
from RecoJets.JetProducers.caloJetsForTrk_cff import *
from CommonTools.RecoAlgos.sortedPrimaryVertices_cfi import sortedPrimaryVertices as _sortedPrimaryVertices
firstStepPrimaryVertices = _sortedPrimaryVertices.clone(
    vertices  = 'firstStepPrimaryVerticesUnsorted',
    particles = 'initialStepTrackRefsForJets',
)


# Final selection
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *

initialStepClassifier1 = TrackMVAClassifierPrompt.clone(
    src         = 'initialStepTracks',
    mva         = dict(GBRForestLabel = 'MVASelectorIter0_13TeV'),
    qualityCuts = [-0.9,-0.8,-0.7]
)
fastSim.toModify(initialStepClassifier1,vertices = 'firstStepPrimaryVerticesBeforeMixing')

from RecoTracker.IterativeTracking.DetachedTripletStep_cff import detachedTripletStepClassifier1
from RecoTracker.IterativeTracking.LowPtTripletStep_cff import lowPtTripletStep
initialStepClassifier2 = detachedTripletStepClassifier1.clone(
    src = 'initialStepTracks'
)
fastSim.toModify(initialStepClassifier2,vertices = 'firstStepPrimaryVerticesBeforeMixing')
initialStepClassifier3 = lowPtTripletStep.clone(
    src = 'initialStepTracks'
)
fastSim.toModify(initialStepClassifier3,vertices = 'firstStepPrimaryVerticesBeforeMixing')

from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
initialStep = ClassifierMerger.clone(
    inputClassifiers=['initialStepClassifier1','initialStepClassifier2','initialStepClassifier3']
)
trackingPhase1.toReplaceWith(initialStep, initialStepClassifier1.clone(
     mva         = dict(GBRForestLabel = 'MVASelectorInitialStep_Phase1'),
     qualityCuts = [-0.95,-0.85,-0.75]
))
pp_on_AA.toModify(initialStep, 
        mva         = dict(GBRForestLabel = 'HIMVASelectorInitialStep_Phase1'),
        qualityCuts = [-0.9, -0.5, 0.2],
)

from RecoTracker.FinalTrackSelectors.trackTfClassifier_cfi import *
from RecoTracker.FinalTrackSelectors.trackSelectionTf_cfi import *
from RecoTracker.FinalTrackSelectors.trackSelectionTf_CKF_cfi import *
trackdnn.toReplaceWith(initialStep, trackTfClassifier.clone(
        src         = 'initialStepTracks',
        qualityCuts = qualityCutDictionary.InitialStep.value()
))

(trackdnn & fastSim).toModify(initialStep,vertices = 'firstStepPrimaryVerticesBeforeMixing')

(pp_on_AA & trackdnn).toModify(initialStep, qualityCuts = [0.35, 0.69, 0.88] )

# For LowPU and Phase2PU140
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
initialStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src = 'initialStepTracks',
    useAnyMVA = cms.bool(False),
    GBRForestLabel = cms.string('MVASelectorIter0'),
    trackSelectors = [
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'initialStepLoose',
        ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'initialStepTight',
            preFilterName = 'initialStepLoose',
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'QualityMasks',
            preFilterName = 'initialStepTight',
        ),
    ] #end of vpset
) #end of clone
trackingPhase2PU140.toModify(initialStepSelector,
    useAnyMVA = None,
    GBRForestLabel = None,
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'initialStepLoose',
            chi2n_par = 2.0,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 3,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.8, 4.0 ),
            dz_par1 = ( 0.9, 4.0 ),
            d0_par2 = ( 0.6, 4.0 ),
            dz_par2 = ( 0.8, 4.0 )
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'initialStepTight',
            preFilterName = 'initialStepLoose',
            chi2n_par = 1.4,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.7, 4.0 ),
            dz_par1 = ( 0.8, 4.0 ),
            d0_par2 = ( 0.5, 4.0 ),
            dz_par2 = ( 0.7, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'initialStep',
            preFilterName = 'initialStepTight',
            min_eta = -4.1,
            max_eta = 4.1,            
            chi2n_par = 1.2,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.6, 4.0 ),
            dz_par1 = ( 0.7, 4.0 ),
            d0_par2 = ( 0.45, 4.0 ),
            dz_par2 = ( 0.55, 4.0 )
            ),
        ), #end of vpset
) #end of clone

fastSim.toModify(initialStepSelector,vertices = "firstStepPrimaryVerticesBeforeMixing")

# Final sequence
InitialStepTask = cms.Task(initialStepSeedLayers,
                           initialStepTrackingRegions,
                           initialStepHitDoublets,
                           initialStepHitTriplets,
                           initialStepSeeds,
                           initialStepTrackCandidates,
                           initialStepTracks,
                           firstStepPrimaryVerticesUnsorted,
                           initialStepTrackRefsForJets,
                           firstStepPrimaryVertices,
                           initialStepClassifier1,initialStepClassifier2,initialStepClassifier3,
                           initialStep,caloJetsForTrkTask)
InitialStep = cms.Sequence(InitialStepTask)

from Configuration.ProcessModifiers.trackingGPUOffline_cff import trackingGPUOffline
from RecoTracker.PixelSeeding.caHitNtupletAlpakaPhase2OT_cfi import caHitNtupletAlpakaPhase2OT as _pixelTracksAlpakaPhase2Extended
from RecoLocalTracker.Phase2TrackerRecHits.phase2OTRecHitsSoAConverter_cfi import phase2OTRecHitsSoAConverter as _phase2OTRecHitsSoAConverter
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitExtendedAlpaka_cfi import siPixelRecHitExtendedAlpaka as _siPixelRecHitExtendedAlpaka
from  RecoTracker.PixelTrackFitting.pixelTrackProducerFromSoAAlpaka_cfi import pixelTrackProducerFromSoAAlpaka as _pixelTrackProducerFromSoAAlpaka

# _pixelTracksAlpakaPostDNN = cms.EDProducer('PixelTrackTorchHighPuritySelector@alpaka',
#     pixelTrackSrc = cms.InputTag('pixelTracksAlpakaPreDNN'),
#     maxNumberOfTracks = cms.int32(2*60*1024),
#     maxPreselectedTracks = cms.int32(9_984),
#     minNumberOfHits = cms.int32(0),
#     avgHitsPerTrack = cms.int32(8),
#     minimumTrackQuality = cms.string('tight'),
#     model = cms.FileInPath('RecoTracker/FinalTrackSelectors/data/PixelTrackTorchHighPuritySelector/pixel_track_classifier_FP16.pt'),
#     scoreThreshold = cms.double(0.4),
#     batchSize = cms.int32(4_992)
# )

### CA Tracks HighPt
initialStepCATracksHighPt = _pixelTracksAlpakaPhase2Extended.clone(
    hitMask = "siPixelOTRecHitSoA",
    pixelRecHitSrc = "siPixelOTRecHitSoA",
    iterationName = "promptHighPt",
)
## CA Mask HighPt
from RecoTracker.PixelSeeding.caMasking_cfi import caMasking as _caMasking
initialStepCATracksHighPtMask = _caMasking.clone(
    iterationIndex = 1,
    minQuality = "tight",
    tracksSoASrc = "initialStepCATracksHighPt",
    recHitsMaskSoASrc = "siPixelOTRecHitSoA"
)

## CA Tracks LowPt
lowPtPtMinCut = 0.35 # 0.45 works, but 0.40 starts showing too many tracks with "zero" eta and phi
                     # Maybe there is another cell cut that balances this, but need to check
initialStepCATracksLowPt = _pixelTracksAlpakaPhase2Extended.clone(
    hitMask = "initialStepCATracksHighPtMask",
    pixelRecHitSrc = "siPixelOTRecHitSoA",
    ptmin = lowPtPtMinCut + 0.05,
    maxNumberOfDoublets = str(12400000),
    maxNumberOfTuples   = str(32 * 32 * 1024),
    hardCurvCut = cms.double(0.035),
    iterationName = "promptLowPt",
)

initialStepCATracksLowPt.trackQualityCuts.minPt = cms.double(lowPtPtMinCut + 0.05)
initialStepCATracksLowPt.geometry.ptCuts = cms.vdouble(73 * [lowPtPtMinCut ])

# CA Tracks SoA Merger
from RecoTracker.FinalTrackSelectors.tracksSoAMerger_cfi import tracksSoAMerger as _tracksSoAMerger

initialStepCATracksSoA = _tracksSoAMerger.clone(
    inputTkSoAs = cms.VInputTag("initialStepCATracksHighPt","initialStepCATracksLowPt"),
    minQuality = cms.string('tight'),
    matchFraction = cms.double(0.5),
    dupNSigma2 = 3.0,
    dupMaxDeltaR2 = 0.001,
)

initialStepCATracks = _pixelTrackProducerFromSoAAlpaka.clone(
    pixelRecHitLegacySrc = "siPixelRecHits",
    beamSpot = cms.InputTag("offlineBeamSpot"),
    minNumberOfHits = cms.int32(0),
    minQuality = cms.string('tight'),
    trackSrc = cms.InputTag("initialStepCATracksSoA"),
    outerTrackerRecHitSrc = cms.InputTag("siPhase2RecHits"),
    outerTrackerRecHitSoAConverterSrc = cms.InputTag("siOTRecHitSoA"),
    useOTExtension = cms.bool(True),
    requireQuadsFromConsecutiveLayers = cms.bool(True)
)

from RecoTracker.TkSeedGenerator.SeedGeneratorFromProtoTracksEDProducer_cfi import SeedGeneratorFromProtoTracksEDProducer as _seedProducerFromTrack

initialSeedFromProtoTrack = cms.PSet(
    ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
    MinOneOverPtError = cms.double(1.0),
    OriginTransverseErrorMultiplier = cms.double(1.0),
    SeedMomentumForBOFF = cms.double(5.0),
    TTRHBuilder = cms.string('WithTrackAngle'),
    forceKinematicWithRegionDirection = cms.bool(False),
    magneticField = cms.string(''),
    propagator = cms.string('PropagatorWithMaterial')
)

initialStepCASeeds = _seedProducerFromTrack.clone(
    InputCollection = cms.InputTag("initialStepCATracks"),
    InputVertexCollection = cms.InputTag(""),
    useProtoTrackKinematics = cms.bool(True),
        SeedCreatorPSet = cms.PSet(
        refToPSet_ = cms.string('initialSeedFromProtoTrack')
    ),
    includeFourthHit = cms.bool(False),
    useEventsWithNoVertex = cms.bool(True),
    usePV = cms.bool(False),
)

siOTRecHitSoA = _phase2OTRecHitsSoAConverter.clone(
    beamSpot = "offlineBeamSpot",
    otRecHitSource = "siPhase2RecHits",
    pixelRecHitSoASource = "siPixelRecHitsSoA"
)

siPixelOTRecHitSoA = _siPixelRecHitExtendedAlpaka.clone(
    pixelRecHitsSoA = "siPixelRecHitsSoA",
    trackerRecHitsSoA = "siOTRecHitSoA"
)

from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromSoAAlpaka_cfi import siPixelRecHitFromSoAAlpaka as _siPixelRecHitFromSoAAlpaka
siPixelOTRecHits =_siPixelRecHitFromSoAAlpaka.clone(
            pixelRecHitSrc = cms.InputTag('siPixelOTRecHitSoA'),
            src = cms.InputTag('siPixelClusters')
)

InitialStepTask.copyAndExclude([initialStepSeedLayers, initialStepTrackingRegions, initialStepHitDoublets, initialStepHitTriplets])
trackingGPUOffline.toReplaceWith(initialStepSeeds, initialStepCASeeds)
## TODO! For ecalDrivenElectronSeeds we need to pass seeds with at least 2 pixel hits!!!

from RecoLocalTracker.Phase2TrackerRecHits.Phase2TrackerRecHits_cfi import siPhase2RecHits
from Configuration.ProcessModifiers.trackingMkFitCommon_cff import trackingMkFitCommon
_InitialStepTask_trackingMkFitCommon = InitialStepTask.copy()
_InitialStepTask_trackingMkFitCommon_Phase2 = InitialStepTask.copy()
_InitialStepTask_trackingMkFitCommon.add(mkFitSiPixelHits, mkFitEventOfHits, mkFitGeometryESProducer)
_InitialStepTask_trackingMkFitCommon_Phase2.add(siPhase2RecHits, mkFitSiPixelHits, mkFitSiPhase2Hits, mkFitEventOfHits, mkFitGeometryESProducer)
(trackingMkFitCommon & (~trackingPhase2PU140)).toReplaceWith(InitialStepTask, _InitialStepTask_trackingMkFitCommon)
(trackingMkFitCommon & trackingPhase2PU140).toReplaceWith(InitialStepTask, _InitialStepTask_trackingMkFitCommon_Phase2)

_InitialStepTask_trackingPhase2_GPUOffline = cms.Task(
                           siPhase2RecHits,
                           siOTRecHitSoA,
                           siPixelOTRecHitSoA,
                           siPixelOTRecHits,
                           initialStepCATracksHighPt,
                           initialStepCATracksHighPtMask,
                           initialStepCATracksLowPt,
                           initialStepCATracksSoA,
                           initialStepCATracks,
                           initialStepSeeds,
                           initialStepTrackCandidates,
                           initialStepTracks,
                           firstStepPrimaryVerticesUnsorted,
                           initialStepTrackRefsForJets,
                           firstStepPrimaryVertices,
                           initialStepClassifier1,initialStepClassifier2,initialStepClassifier3,
                           initialStep,caloJetsForTrkTask
)

(trackingPhase2PU140 & trackingGPUOffline).toReplaceWith(initialStepSeeds, initialStepCASeeds)
(trackingPhase2PU140 & trackingGPUOffline).toReplaceWith(InitialStepTask, _InitialStepTask_trackingPhase2_GPUOffline)
_InitialStepTask_trackingMkFitCommon_Phase2GPU = _InitialStepTask_trackingPhase2_GPUOffline.copy()
_InitialStepTask_trackingMkFitCommon_Phase2GPU.add(siPhase2RecHits, mkFitSiPixelHits, mkFitSiPhase2Hits, mkFitEventOfHits, mkFitGeometryESProducer)
(trackingMkFitCommon & trackingPhase2PU140 & trackingGPUOffline).toReplaceWith(InitialStepTask,_InitialStepTask_trackingMkFitCommon_Phase2GPU)

_InitialStepTask_trackingMkFit = InitialStepTask.copy()
_InitialStepTask_trackingMkFit.add(initialStepTrackCandidatesMkFitSeeds, initialStepTrackCandidatesMkFit, initialStepTrackCandidatesMkFitConfig)
trackingMkFitInitialStep.toReplaceWith(InitialStepTask, _InitialStepTask_trackingMkFit)

_InitialStepTask_LowPU = InitialStepTask.copyAndExclude([firstStepPrimaryVerticesUnsorted, initialStepTrackRefsForJets, caloJetsForTrkTask, firstStepPrimaryVertices, initialStepClassifier1, initialStepClassifier2, initialStepClassifier3])
_InitialStepTask_LowPU.replace(initialStep, initialStepSelector)
trackingLowPU.toReplaceWith(InitialStepTask, _InitialStepTask_LowPU)

_InitialStepTask_Phase1 = InitialStepTask.copyAndExclude([initialStepClassifier2, initialStepClassifier3])
_InitialStepTask_Phase1.replace(initialStepHitTriplets, initialStepHitQuadruplets)
trackingPhase1.toReplaceWith(InitialStepTask, _InitialStepTask_Phase1)

_InitialStepTask_trackingPhase2 = InitialStepTask.copyAndExclude([initialStepClassifier1, initialStepClassifier2, initialStepClassifier3])
_InitialStepTask_trackingPhase2.replace(initialStepHitTriplets, initialStepHitQuadruplets)
_InitialStepTask_trackingPhase2.replace(initialStep, initialStepSelector)
trackingPhase2PU140.toReplaceWith(InitialStepTask, _InitialStepTask_trackingPhase2)


from Configuration.ProcessModifiers.seedingLST_cff import seedingLST
from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
(trackingPhase2PU140 & (seedingLST | trackingLST)).toModify(firstStepPrimaryVerticesUnsorted, TrackLabel = 'highPtTripletStepTracks')
(trackingPhase2PU140 & (seedingLST | trackingLST)).toModify(initialStepTrackRefsForJets, src = 'highPtTripletStepTracks')

_InitialStepTask_trackingPhase2_LST = InitialStepTask.copyAndExclude([initialStepTrackCandidatesMkFitSeeds, initialStepTrackCandidatesMkFit, initialStepTrackCandidatesMkFitConfig, initialStepTrackCandidates, initialStepTracks, initialStepSelector])
(trackingPhase2PU140 & (seedingLST | trackingLST)).toReplaceWith(InitialStepTask, _InitialStepTask_trackingPhase2_LST)

(trackingMkFitCommon & trackingPhase2PU140).toModify(mkFitEventOfHits, stripHits=cms.InputTag('mkFitSiPhase2Hits'), useStripStripQualityDB=cms.bool(False))
(trackingMkFitInitialStep & trackingPhase2PU140).toModify(initialStepTrackCandidatesMkFit, stripHits=cms.InputTag('mkFitSiPhase2Hits'))
(trackingMkFitInitialStep & trackingPhase2PU140).toModify(initialStepTrackCandidates, mkFitStripHits=cms.InputTag('mkFitSiPhase2Hits'))
(trackingMkFitInitialStep & trackingPhase2PU140).toModify(initialStepTrackCandidatesMkFitConfig, config='RecoTracker/MkFit/data/mkfit-phase2-initialStep.json')


from Configuration.Eras.Modifier_fastSim_cff import fastSim
_InitialStepTask_fastSim = cms.Task(initialStepTrackingRegions
                           ,initialStepSeeds
                           ,initialStepTrackCandidates
                           ,initialStepTracks
                           ,firstStepPrimaryVerticesBeforeMixing
                           ,initialStepClassifier1,initialStepClassifier2,initialStepClassifier3
                           ,initialStep
                           )
_InitialStepTask_fastSim_Phase2 = _InitialStepTask_fastSim.copyAndExclude([initialStepClassifier1, initialStepClassifier2, initialStepClassifier3])
_InitialStepTask_fastSim_Phase2.replace(initialStep, initialStepSelector)
fastSim.toReplaceWith(InitialStepTask, _InitialStepTask_fastSim)
(fastSim & trackingPhase2PU140).toReplaceWith(InitialStepTask, _InitialStepTask_fastSim_Phase2)

##
## Modify for the tau embedding methods reco sim step
##
from Configuration.ProcessModifiers.tau_embedding_sim_cff import tau_embedding_sim
from TauAnalysis.MCEmbeddingTools.Simulation_RECO_cff import tau_embedding_correct_hlt_vertices
tau_embedding_sim.toReplaceWith(firstStepPrimaryVerticesUnsorted, tau_embedding_correct_hlt_vertices)
