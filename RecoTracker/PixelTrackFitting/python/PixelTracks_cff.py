import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import *

from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
import RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi
myTTRHBuilderWithoutAngle = RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi.ttrhbwr.clone(
    StripCPE = 'Fake',
    ComponentName = 'PixelTTRHBuilderWithoutAngle'
)
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
from RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4PixelTriplets_cfi import *
from RecoTracker.PixelTrackFitting.pixelFitterByHelixProjections_cfi import pixelFitterByHelixProjections
from RecoTracker.PixelTrackFitting.pixelNtupletsFitter_cfi import pixelNtupletsFitter
from RecoTracker.PixelTrackFitting.pixelTrackFilterByKinematics_cfi import pixelTrackFilterByKinematics
from RecoTracker.PixelTrackFitting.pixelTrackCleanerBySharedHits_cfi import pixelTrackCleanerBySharedHits
from RecoTracker.PixelTrackFitting.pixelTracks_cfi import pixelTracks as _pixelTracks
from RecoTracker.TkTrackingRegions.globalTrackingRegion_cfi import globalTrackingRegion as _globalTrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpot_cfi import globalTrackingRegionFromBeamSpot as _globalTrackingRegionFromBeamSpot
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
from RecoTracker.PixelSeeding.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoTracker.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoTracker.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder

# Eras
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
from Configuration.Eras.Modifier_run3_common_cff import run3_common

# seeding layers
from RecoTracker.IterativeTracking.InitialStep_cff import initialStepSeedLayers, initialStepHitDoublets, _initialStepCAHitQuadruplets

# TrackingRegion
pixelTracksTrackingRegions = _globalTrackingRegion.clone()
trackingLowPU.toReplaceWith(pixelTracksTrackingRegions, _globalTrackingRegionFromBeamSpot.clone())


# Pixel quadruplets tracking
pixelTracksSeedLayers = initialStepSeedLayers.clone(
    BPix = dict(HitProducer = "siPixelRecHitsPreSplitting"),
    FPix = dict(HitProducer = "siPixelRecHitsPreSplitting")
)

pixelTracksHitDoublets = initialStepHitDoublets.clone(
    clusterCheck = "",
    seedingLayers = "pixelTracksSeedLayers",
    trackingRegions = "pixelTracksTrackingRegions"
)

pixelTracksHitQuadruplets = _initialStepCAHitQuadruplets.clone(
    doublets = "pixelTracksHitDoublets",
    SeedComparitorPSet = dict(clusterShapeCacheSrc = 'siPixelClusterShapeCachePreSplitting')
)

pixelTracks = _pixelTracks.clone(
    SeedingHitSets = "pixelTracksHitQuadruplets"
)

pixelTracksTask = cms.Task(
    pixelTracksTrackingRegions,
    pixelFitterByHelixProjections,
    pixelTrackFilterByKinematics,
    pixelTracksSeedLayers,
    pixelTracksHitDoublets,
    pixelTracksHitQuadruplets,
    pixelTracks
)

pixelTracksSequence = cms.Sequence(pixelTracksTask)


# Pixel triplets for trackingLowPU
pixelTracksHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets = "pixelTracksHitDoublets",
    produceSeedingHitSets = True,
    SeedComparitorPSet = RecoTracker.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone(
        clusterShapeCacheSrc = "siPixelClusterShapeCachePreSplitting"
    )
)

trackingLowPU.toModify(pixelTracks,
    SeedingHitSets = "pixelTracksHitTriplets"
)

_pixelTracksTask_lowPU = pixelTracksTask.copy()
_pixelTracksTask_lowPU.replace(pixelTracksHitQuadruplets, pixelTracksHitTriplets)
trackingLowPU.toReplaceWith(pixelTracksTask, _pixelTracksTask_lowPU)

# Phase 2 modifier
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
# HIon modifiers
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA

######################################################################

### Alpaka Pixel Track Reco

from Configuration.ProcessModifiers.alpaka_cff import alpaka
from Configuration.ProcessModifiers.stripNtupletFit_cff import stripNtupletFit
#from RecoTracker.PixelSeeding.caGeometryESProducer_cfi import caGeometryESProducer as _caGeometryESProducer
def _addCAGeometryESProducer(process):
    process.load("RecoTracker.PixelSeeding.CAGeometryESProducer_cfi")
    # process.caGeometryESProducer.appendToDataLabel = cms.string("caGeometry")

    # phase2_tracker.toModify(process.caGeometryESProducer,
    #     pairGraph = [ 0,  1,  0,  4,  0,  16,  ##BPIX1 (3)
    #   1,  2,  1,  4,  1,  16,  ##BPIX2 (6)
    #   2,  3,  2,  4,  2,  16,  ##BPIX3 & Forward (9)

    #   4,  5,  5,  6,  6,  7,  7,  8,  8,  9,  9,  10, 10, 11,  ##POS (16)
    #   16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23,  ##NEG (23)

    #   0,  2,  0,  5,  0,  17, 0,  6,  0,  18,  ##BPIX1 Jump (28)
    #   1,  3,  1,  5,  1,  17, 1,  6,  1,  18,  ##BPIX2 Jump (33)

    #   11, 12, 12, 13, 13, 14, 14, 15,  ##Late POS (37)
    #   23, 24, 24, 25, 25, 26, 26, 27,  ##Late NEG (41)

    #   4,  6,  5,  7,  6,  8,  7,  9,  8,  10, 9,  11, 10, 12,  ##POS Jump (48)
    #   16, 18, 17, 19, 18, 20, 19, 21, 20, 22, 21, 23, 22, 24 ],

    #     maxZ = [ 17.0, 22.0,  -4.0,  17.0,  22.0,  -6.0,  18.0,  22.0,  -11.0,  28.0,   35.0,   44.0,   55.0,   70.0,
    #   87.0, 113.0, -23.0, -30.0, -39.0, -50.0, -65.0, -82.0, -109.0, 17.0,   22.0,   -7.0,   22.0,   -10.0,
    #   17.0, 22.0,  -9.0,  22.0,  -13.0, 142.0, 177.0, 203.0, 233.0,  -137.0, -173.0, -199.0, -229.0, 28.0,
    #   35.0, 44.0,  55.0,  70.0,  87.0,  113.0, -23.0, -30.0, -39.0,  -50.0,  -65.0,  -82.0,  -109.0 ],
    #     phiCuts = [ 522, 522, 522, 626, 730,
    #     730, 626, 730, 730, 522,
    #     522, 522, 522, 522, 522,
    #     522, 522, 522, 522, 522,
    #     522, 522, 522, 522, 522,
    #     522, 522, 522, 522, 730,
    #     730, 730, 730, 730, 730,
    #     730, 730, 730, 730, 730,
    #     730, 730, 730, 730, 730,
    #     730, 730, 522, 522, 522,
    #     522, 522, 522, 522, 522],
    #     caDCACuts = [0.15 ] + [0.25] * 27,
    #     caThetaCuts = [0.002,0.002,0.002,0.002] + [0.003] * 24,
    #     startingPairs = [f for f in range(33)],
    #     minZ = [-16.0, 4.0,   -22.0, -17.0, 6.0,   -22.0, -18.0, 11.0,  -22.0,  23.0,   30.0,   39.0,   50.0,   65.0,
    #   82.0,  109.0, -28.0, -35.0, -44.0, -55.0, -70.0, -87.0, -113.0, -16.,   7.0,    -22.0,  11.0,   -22.0,
    #   -17.0, 9.0,   -22.0, 13.0,  -22.0, 137.0, 173.0, 199.0, 229.0,  -142.0, -177.0, -203.0, -233.0, 23.0,
    #   30.0,  39.0,  50.0,  65.0,  82.0,  109.0, -28.0, -35.0, -44.0,  -55.0,  -70.0,  -87.0,  -113.0 ],

    #     maxR = [5.0, 5.0, 5.0, 7.0, 8.0, 8.0,  7.0, 7.0, 7.0, 6.0, 6.0, 6.0, 6.0, 5.0,
    #                                          6.0, 5.0, 6.0, 6.0, 6.0, 6.0,  5.0, 6.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
    #                                          5.0, 8.0, 8.0, 8.0, 8.0, 6.0,  5.0, 5.0, 5.0, 6.0, 5.0, 5.0, 5.0, 9.0,
    #                                          9.0, 9.0, 8.0, 8.0, 8.0, 11.0, 9.0, 9.0, 9.0, 8.0, 8.0, 8.0, 11.0]
    #   )

    # stripNtupletFit.toModify(process.caGeometryESProducer,
    #     pairGraph = list(getattr(process.caGeometryESProducer, "pairGraph")) + [0, 11],
    #     maxZ = list(getattr(process.caGeometryESProducer, "maxZ")) + [1000.0],
    #     caDCACuts = list(getattr(process.caGeometryESProducer, "caDCACuts")) + [0.3],
    #     caThetaCuts = list(getattr(process.caGeometryESProducer, "caThetaCuts")) + [0.25],
    #     minZ =  list(getattr(process.caGeometryESProducer, "minZ")) + [-1000.0],
    #     maxR =  list(getattr(process.caGeometryESProducer, "maxR")) + [10000.0],
    #     phiCuts = list(getattr(process.caGeometryESProducer, "phiCuts")) + [1000]
    #   )

modifyConfigurationForAlpakaCAGeometry_ = alpaka.makeProcessModifier(_addCAGeometryESProducer)


# pixel tracks SoA producer on the device
from RecoTracker.PixelSeeding.caHitNtupletAlpakaPhase1_cfi import caHitNtupletAlpakaPhase1 as _pixelTracksAlpakaPhase1
from RecoTracker.PixelSeeding.caHitNtupletAlpakaPhase2_cfi import caHitNtupletAlpakaPhase2 as _pixelTracksAlpakaPhase2
from RecoTracker.PixelSeeding.caHitNtupletAlpakaHIonPhase1_cfi import caHitNtupletAlpakaHIonPhase1 as _pixelTracksAlpakaHIonPhase1

pixelTracksAlpaka = _pixelTracksAlpakaPhase1.clone()
phase2_tracker.toReplaceWith(pixelTracksAlpaka,_pixelTracksAlpakaPhase2.clone())
phase2_tracker.toModify(pixelTracksAlpaka,
    maxNumberOfDoublets = str(1500000),
    maxNumberOfTuples = str(256 * 1024),
    avgHitsPerTrack = 9.0,
    avgCellsPerHit = 50.0,
    avgCellsPerCell = 50.0,
    avgTracksPerCell = 2.0,
    cellPtCut = 0.85,
    cellZ0Cut = 7.5,
    minYsizeB1 = 25,
    minYsizeB2 = 15,
    maxDYsize12 = 12,
    maxDYsize = 10,
    maxDYPred = 20,
)

(pp_on_AA & ~phase2_tracker).toReplaceWith(pixelTracksAlpaka, _pixelTracksAlpakaHIonPhase1.clone())

# pixel tracks SoA producer on the cpu, for validation
pixelTracksAlpakaSerial = makeSerialClone(pixelTracksAlpaka,
    pixelRecHitSrc = 'siPixelRecHitsPreSplittingAlpakaSerial'
)

# legacy pixel tracks from SoA
from  RecoTracker.PixelTrackFitting.pixelTrackProducerFromSoAAlpaka_cfi import pixelTrackProducerFromSoAAlpaka as _pixelTrackProducerFromSoAAlpaka

(alpaka).toReplaceWith(pixelTracks, _pixelTrackProducerFromSoAAlpaka.clone(
    pixelRecHitLegacySrc = "siPixelRecHitsPreSplitting",
))

# (alpaka & stripNtupletFit).toModify(pixelTracks, 
#         useStripHits = True)

alpaka.toReplaceWith(pixelTracksTask, cms.Task(
    # Build the pixel ntuplets and the pixel tracks in SoA format with alpaka on the device
    pixelTracksAlpaka,
    # Build the pixel ntuplets and the pixel tracks in SoA format with alpaka on the cpu (if requested by the validation)
    pixelTracksAlpakaSerial,
    # Convert the pixel tracks from SoA to legacy format
    pixelTracks)
)
