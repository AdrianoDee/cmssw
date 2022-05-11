import FWCore.ParameterSet.Config as cms

from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
from RecoPixelVertexing.PixelLowPtUtilities.trackCleaner_cfi import *
from RecoPixelVertexing.PixelTrackFitting.pixelFitterByConformalMappingAndLine_cfi import *
from RecoHI.HiTracking.HIPixelTrackFilter_cff import *
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *

from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

# Hit ntuplets
hiConformalPixelTracksHitDoublets = _hitPairEDProducer.clone(
    clusterCheck    = "",
    seedingLayers   = "PixelLayerTriplets",
    trackingRegions = "hiTrackingRegionWithVertex",
    maxElement      = 50000000,
    produceIntermediateHitDoublets = True,
)

hiConformalPixelTracksHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets   = "hiConformalPixelTracksHitDoublets",
    maxElement = 5000000, # increase threshold for triplets in generation step (default: 100000)
    produceSeedingHitSets = True,
)

import RecoPixelVertexing.PixelTrackFitting.pixelTracks_cfi as _mod
# Pixel tracks
hiConformalPixelTracks = _mod.pixelTracks.clone(
    #passLabel  = 'Pixel triplet low-pt tracks with vertex constraint',
    # Ordered Hits
    SeedingHitSets = "hiConformalPixelTracksHitTriplets",
    # Fitter
    Fitter = 'pixelFitterByConformalMappingAndLine',
    # Filter
    Filter = "hiConformalPixelFilter",
    # Cleaner
    Cleaner = "trackCleaner"
)

###Pixel Tracking -  PhaseI geometry

#Tracking regions - use PV from pp tracking
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cfi import globalTrackingRegionWithVertices
hiConformalPixelTracksPhase1TrackingRegions = globalTrackingRegionWithVertices.clone(
    RegionPSet = dict(
	precise = True,
	useMultipleScattering = False,
	useFakeVertices  = False,
	beamSpot         = "offlineBeamSpot",
	useFixedError    = True,
	nSigmaZ          = 3.0,
	sigmaZVertex     = 3.0,
	fixedError       = 0.2,
	VertexCollection = "offlinePrimaryVertices",
	ptMin            = 0.3,
	useFoundVertices = True,
	originRadius     = 0.2
    )
)

# SEEDING LAYERS
# Using 4 layers layerlist
from RecoTracker.IterativeTracking.LowPtQuadStep_cff import lowPtQuadStepSeedLayers
hiConformalPixelTracksPhase1SeedLayers = lowPtQuadStepSeedLayers.clone(
    BPix = cms.PSet(
	HitProducer = cms.string('siPixelRecHits'),
        TTRHBuilder = cms.string('WithTrackAngle'),
    ),
    FPix = cms.PSet(
        HitProducer = cms.string('siPixelRecHits'),
        TTRHBuilder = cms.string('WithTrackAngle'),
    )
)


# Hit ntuplets
from RecoTracker.IterativeTracking.LowPtQuadStep_cff import lowPtQuadStepHitDoublets
hiConformalPixelTracksPhase1HitDoubletsCA = lowPtQuadStepHitDoublets.clone(
    seedingLayers   = "hiConformalPixelTracksPhase1SeedLayers",
    trackingRegions = "hiConformalPixelTracksPhase1TrackingRegions"
)


from RecoTracker.IterativeTracking.LowPtQuadStep_cff import lowPtQuadStepHitQuadruplets
hiConformalPixelTracksPhase1HitQuadrupletsCA = lowPtQuadStepHitQuadruplets.clone(
    doublets   = "hiConformalPixelTracksPhase1HitDoubletsCA",
    CAPhiCut   = 0.2,
    CAThetaCut = 0.0012,
    SeedComparitorPSet = dict(
       ComponentName = 'none'
    ),
    extraHitRPhitolerance = 0.032,
    maxChi2 = dict(
       enabled = True,
       pt1     = 0.7,
       pt2     = 2,
       value1  = 200,
       value2  = 50
    )
)

#Filter
hiConformalPixelTracksPhase1Filter = hiConformalPixelFilter.clone(
    VertexCollection = "offlinePrimaryVertices",
    chi2   = 30.0,
    lipMax = 999.0,
    nSigmaLipMaxTolerance = 3.0,
    nSigmaTipMaxTolerance = 3.0,
    ptMax  = 999999,
    ptMin  = 0.30,
    tipMax = 999.0
)

from RecoPixelVertexing.PixelTrackFitting.pixelNtupletsFitter_cfi import pixelNtupletsFitter

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(hiConformalPixelTracks,
    Cleaner = 'pixelTrackCleanerBySharedHits',
    Filter  = "hiConformalPixelTracksPhase1Filter",
    Fitter  = "pixelNtupletsFitter",
    SeedingHitSets = "hiConformalPixelTracksPhase1HitQuadrupletsCA",
)

hiConformalPixelTracksTask = cms.Task(
    hiTrackingRegionWithVertex ,
    hiConformalPixelTracksHitDoublets ,
    hiConformalPixelTracksHitTriplets ,
    pixelFitterByConformalMappingAndLine ,
    hiConformalPixelFilter ,
    hiConformalPixelTracks
)

from Configuration.ProcessModifiers.gpu_cff import gpu
from RecoPixelVertexing.PixelTrackFitting.pixelTracksSoA_cfi import pixelTracksSoA as _pixelTracksSoA
from RecoPixelVertexing.PixelTriplets.pixelTracksCUDA_cfi import pixelTracksCUDA as _pixelTracksCUDA
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitCUDA_cfi import siPixelRecHitCUDA as _siPixelRecHitCUDA
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import siPixelClusters as _siPixelClusters #legacy clusters

from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

# pixelTracksSoAHIon = _pixelTracksSoA.clone(src = 'pixelTracksCUDAHIon')
pixelTracksCUDAHIon = _pixelTracksCUDA.clone(pixelRecHitSrc="siPixelRecHitsPreSplittingCUDA")

#Pixel tracks in SoA format on the CPU
pixelTracksHIonCPU = _pixelTracksCUDA.clone(
    pixelRecHitSrc = "siPixelRecHitsPreSplitting",
    idealConditions = False,
    onGPU = False
)

# SwitchProducer providing the pixel tracks in SoA format on the CPU
pixelTracksSoAHIon = SwitchProducerCUDA(
    # build pixel ntuplets and pixel tracks in SoA format on the CPU
    cpu = pixelTracksHIonCPU
)

gpu.toModify(pixelTracksSoAHIon,
    # transfer the pixel tracks in SoA format to the host
    cuda = _pixelTracksSoA.clone(src="pixelTracksCUDAHIon")
)

# pixelTracksSoA = SwitchProducerCUDA(
#     # build pixel ntuplets and pixel tracks in SoA format on the CPU
#     cpu = _pixelTracksCUDA.clone(
#         pixelRecHitSrc = "siPixelRecHitsPreSplitting",
#         idealConditions = False,
#         onGPU = False
#     )
# )
#
# gpu.toModify(pixelTracksSoA,
#     # transfer the pixel tracks in SoA format to the host
#     cuda = _pixelTracksSoA.clone()
# )

# siPixelRecHitsPreSplittingCUDAHIon = _siPixelRecHitCUDA.clone(
#     beamSpot = "offlineBeamSpotToCUDA",
#     src = 'siPixelClustersPreSplittingCUDAHIon',
# )

from RecoPixelVertexing.PixelTrackFitting.pixelTrackProducerFromSoA_cfi import pixelTrackProducerFromSoA as _pixelTrackProducerFromSoA
gpu.toReplaceWith(hiConformalPixelTracks,_pixelTrackProducerFromSoA.clone(
    pixelRecHitLegacySrc = "siPixelRecHitsPreSplitting",
    trackSrc = "pixelTracksSoAHIon",

))

from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromCUDA_cfi import siPixelRecHitFromCUDA as _siPixelRecHitFromCUDA

from RecoLocalTracker.SiPixelClusterizer.siPixelRawToClusterCUDA_cfi import siPixelRawToClusterCUDA as _siPixelRawToClusterCUDA
siPixelClustersPreSplittingCUDAHIon = _siPixelRawToClusterCUDA.clone(MaxFEDWords = 500000)

from RecoLocalTracker.SiPixelClusterizer.siPixelDigisClustersFromSoA_cfi import siPixelDigisClustersFromSoA as _siPixelDigisClustersFromSoA
siPixelDigisClustersPreSplittingHIon = _siPixelDigisClustersFromSoA.clone(src = "siPixelDigisSoAHIon")

from EventFilter.SiPixelRawToDigi.siPixelDigisSoAFromCUDA_cfi import siPixelDigisSoAFromCUDA as _siPixelDigisSoAFromCUDA
# siPixelDigisSoAHIon = _siPixelDigisSoAFromCUDA.clone(
#     src = "siPixelClustersPreSplittingCUDAHIon"
# )
#
# siPixelRecHitFromCUDAHIon = _siPixelRecHitFromCUDA.clone(src='siPixelDigisClustersPreSplittingHIon', pixelRecHitSrc = 'siPixelRecHitsPreSplittingCUDAHIon')
hiConformalPixelTracksTaskPhase1 = cms.Task(
    hiConformalPixelTracksPhase1TrackingRegions ,
    hiConformalPixelTracksPhase1SeedLayers ,
    hiConformalPixelTracksPhase1HitDoubletsCA ,
    hiConformalPixelTracksPhase1HitQuadrupletsCA ,
    pixelNtupletsFitter ,
    hiConformalPixelTracksPhase1Filter ,
    hiConformalPixelTracks
)


gpu.toReplaceWith(hiConformalPixelTracksTaskPhase1, cms.Task(
    #pixelTracksTrackingRegions,
    #siPixelClustersPreSplittingCUDAHIon,
    #siPixelDigisSoAHIon,
    #siPixelDigisClustersPreSplittingHIon,
    #siPixelRecHitsPreSplittingCUDAHIon,
    #siPixelRecHitFromCUDAHIon,
    # build the pixel ntuplets and the pixel tracks in SoA format on the GPU
    pixelTracksCUDAHIon,
    pixelTracksSoAHIon,
    # convert the pixel tracks from SoA to legacy format
    hiConformalPixelTracks
))

hiConformalPixelTracksSequencePhase1 = cms.Sequence(hiConformalPixelTracksTaskPhase1)
