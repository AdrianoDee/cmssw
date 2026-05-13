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

# pixel tracks SoA producer on the device
from RecoTracker.PixelSeeding.caHitNtupletAlpakaPhase1_cfi import caHitNtupletAlpakaPhase1 as _pixelTracksAlpakaPhase1
from RecoTracker.PixelSeeding.caHitNtupletAlpakaPhase2_cfi import caHitNtupletAlpakaPhase2 as _pixelTracksAlpakaPhase2
from RecoTracker.PixelSeeding.caHitNtupletAlpakaHIonPhase1_cfi import caHitNtupletAlpakaHIonPhase1 as _pixelTracksAlpakaHIonPhase1
from RecoTracker.PixelSeeding.caHitNtupletAlpakaPhase2OT_cfi import caHitNtupletAlpakaPhase2OT as _pixelTracksAlpakaPhase2Extended

pixelTracksAlpaka = _pixelTracksAlpakaPhase1.clone(
    avgHitsPerTrack    = 4.6,      
    avgCellsPerHit     = 13,
    avgCellsPerCell    = 0.0268, 
    avgTracksPerCell   = 0.0123, 
    maxNumberOfDoublets = str(512*1024),    # could be lowered to 315k, keeping the same for a fair comparison with master
    maxNumberOfTuples   = str(32 * 1024),   # this couul be much lower (2.1k, these are quads)
)

phase2_tracker.toReplaceWith(pixelTracksAlpaka,_pixelTracksAlpakaPhase2.clone())

def _modifyForPPonAAandNotPhase2(producer):
    nPairs = int(len(producer.geometry.pairGraph) / 2)
    producer.maxNumberOfDoublets = str(6 * 512 *1024)    # this could be 2.3M
    producer.maxNumberOfTuples = str(256 * 1024)         # this could be 4.7
    producer.avgHitsPerTrack = 5.0
    producer.avgCellsPerHit = 40
    producer.avgCellsPerCell = 0.07                      # with maxNumberOfDoublets ~= 3.14M; 0.02  for HLT HI on 2024 HI Data 
    producer.avgTracksPerCell = 0.03                     # with maxNumberOfDoublets ~= 3.14M; 0.005 for HLT HI on 2024 HI Data
    producer.cellZ0Cut = 8.0                             # setup currenlty used @ HLT (was 10.0) 
    producer.geometry.ptCuts = [0.5] * nPairs            # setup currenlty used @ HLT (was 0.0) 

(pp_on_AA & ~phase2_tracker).toModify(pixelTracksAlpaka, _modifyForPPonAAandNotPhase2)

from Configuration.ProcessModifiers.phase2CAExtension_cff import phase2CAExtension
phase2CAExtension.toReplaceWith(pixelTracksAlpaka,_pixelTracksAlpakaPhase2Extended.clone(
    pixelRecHitSrc = "siPixelRecHitsExtendedPreSplittingAlpaka",
    hitMask = "siPixelRecHitsExtendedPreSplittingAlpaka",
))

# pixel tracks SoA producer on the cpu, for validation
pixelTracksAlpakaSerial = makeSerialClone(pixelTracksAlpaka,
    pixelRecHitSrc = 'siPixelRecHitsPreSplittingAlpakaSerial'
)

phase2CAExtension.toModify(pixelTracksAlpakaSerial,
                           pixelRecHitSrc = 'siPixelRecHitsExtendedPreSplittingAlpakaSerial'
                           )

# legacy pixel tracks from SoA
from  RecoTracker.PixelTrackFitting.pixelTrackProducerFromSoAAlpaka_cfi import pixelTrackProducerFromSoAAlpaka as _pixelTrackProducerFromSoAAlpaka

(alpaka & ~phase2CAExtension).toReplaceWith(pixelTracks, _pixelTrackProducerFromSoAAlpaka.clone(
    pixelRecHitLegacySrc = "siPixelRecHitsPreSplitting",
))

phase2CAExtension.toReplaceWith(pixelTracks, _pixelTrackProducerFromSoAAlpaka.clone(
    pixelRecHitLegacySrc = "siPixelRecHitsPreSplitting",
    beamSpot = cms.InputTag("offlineBeamSpot"),
    minNumberOfHits = cms.int32(0),
    minQuality = cms.string('tight'),
    trackSrc = cms.InputTag("pixelTracksAlpaka"),
    outerTrackerRecHitSrc = cms.InputTag("siPhase2RecHits"),
    outerTrackerRecHitSoAConverterSrc = cms.InputTag("phase2OTRecHitsSoAConverter"),
    useOTExtension = cms.bool(True),
    requireQuadsFromConsecutiveLayers = cms.bool(True)
))

pixelTracksHighPt = pixelTracks.clone()
pixelTracksLowPt = pixelTracks.clone()

alpaka.toReplaceWith(pixelTracksTask, cms.Task(
    # Build the pixel ntuplets and the pixel tracks in SoA format with alpaka on the device
    pixelTracksAlpaka,
    # Build the pixel ntuplets and the pixel tracks in SoA format with alpaka on the cpu (if requested by the validation)
    pixelTracksAlpakaSerial,
    # Just to validate
    pixelTracksHighPt,
    # Just to validate
    pixelTracksLowPt,
    # Convert the pixel tracks from SoA to legacy format
    pixelTracks)
)

pixelTracksHighPtAlpaka = _pixelTracksAlpakaPhase1.clone(
    avgHitsPerTrack    = 4.6,      
    avgCellsPerHit     = 13,
    avgCellsPerCell    = 0.0268, 
    avgTracksPerCell   = 0.0123, 
    maxNumberOfDoublets = str(12400000),    # could be lowered to 315k, keeping the same for a fair comparison with master
    maxNumberOfTuples   = str(32 * 32 * 1024),   # this couul be much lower (2.1k, these are quads)
)

from Configuration.ProcessModifiers.pixelTrackMask_cff import pixelTrackMask
highPtItMinPtCut = 1.9
(pixelTrackMask & phase2CAExtension).toReplaceWith(pixelTracksHighPtAlpaka,_pixelTracksAlpakaPhase2Extended.clone(
    hitMask = "siPixelRecHitsExtendedPreSplittingAlpaka",
    pixelRecHitSrc = "siPixelRecHitsExtendedPreSplittingAlpaka",
    maxNumberOfDoublets = str(12400000),    # could be lowered to 315k, keeping the same for a fair comparison with master
    maxNumberOfTuples   = str(32 * 32 * 1024),   # this couul be much lower (2.1k, these are quads)
    tripletCuts = cms.PSet(
        floorDCA = cms.optional.vdouble,
        maxCurv = cms.double(0.010),
        maxDCA = cms.vdouble(
            0.15000000596046448, 0.25, 0.20000000298023224, 0.20000000298023224, 0.25,
            0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.10000000149011612, 0.10000000149011612,
            0.10000000149011612, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0
        ),
        maxPhiResid = cms.double(-1),
        maxRZTolerance = cms.vdouble(
            0.0020000000949949026, 0.0020000000949949026, 0.0020000000949949026, 0.0020000000949949026, 0.003000000026077032,
            0.003000000026077032, 0.003000000026077032, 0.003000000026077032, 0.003000000026077032, 0.003000000026077032,
            0.003000000026077032, 0.003000000026077032, 0.003000000026077032, 0.003000000026077032, 0.003000000026077032,
            0.003000000026077032, 0.003000000026077032, 0.003000000026077032, 0.003000000026077032, 0.003000000026077032,
            0.003000000026077032, 0.003000000026077032, 0.003000000026077032, 0.003000000026077032, 0.003000000026077032,
            0.003000000026077032, 0.003000000026077032, 0.003000000026077032, 0.003000000026077032, 0.003000000026077032,
            0.003000000026077032, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0
        ),
        maxStubGeomCurvSigma = cms.optional.vdouble,
        maxStubInnerDoubletDCurv = cms.optional.vdouble,
        ptmin = cms.double(highPtItMinPtCut + 0.1),
        sameDPhiSign = cms.bool(False)
    ),
    trackQualityCuts = cms.PSet(
        maxChi2 = cms.double(5),
        maxChi2Quintuplets = cms.double(3),
        maxChi2TripletsOrQuadruplets = cms.double(1),
        maxTip = cms.double(0.3),
        maxZip = cms.double(12),
        minPt = cms.double(highPtItMinPtCut + 0.1)
    ),
    iterationName = "promptHighPt",

    doubletCuts = cms.PSet(
        dzdrFact = cms.double(15.199999809265137),
        maxDPhi = cms.vint32(
            int(0.8*350), int(0.8*600), int(0.8*450), int(0.8*522), int(0.8*450),
            int(0.8*522), int(0.8*400), int(0.8*650), int(0.8*500), int(0.8*730),
            int(0.8*500), int(0.8*730), int(0.8*350), int(0.8*400), int(0.8*400),
            int(0.8*300), int(0.8*522), int(0.8*300), int(0.8*522), int(0.8*250),
            int(0.8*522), int(0.8*250), int(0.8*522), int(0.8*250), int(0.8*522),
            int(0.8*300), int(0.8*522), int(0.8*240), int(0.8*650), int(0.8*300),
            int(0.8*200), int(0.8*220), int(0.8*250), int(0.8*250), int(0.8*250),
            int(0.8*250), int(0.8*300), int(0.8*522), int(0.8*300), int(0.8*522),
            int(0.8*250), int(0.8*522), int(0.8*250), int(0.8*522), int(0.8*250),
            int(0.8*522), int(0.8*300), int(0.8*522), int(0.8*240), int(0.8*650),
            int(0.8*300), int(0.8*200), int(0.8*220), int(0.8*250), int(0.8*250),
            int(0.8*250), int(0.8*250), int(0.8*1200), int(0.8*1200), int(0.8*1200),
            int(0.8*1000), int(0.8*1000), int(0.8*1000), int(0.8*1000), int(0.8*1000),
            int(0.8*850), int(0.8*1000), int(0.8*1000), int(0.8*1000), int(0.8*1000),
            int(0.8*1000), int(0.8*1100), int(0.8*1250)
        ),
        maxDR = cms.vdouble(
            5, 10, 8, 5, 8,
            5, 7, 10, 8, 10,
            8, 10, 7, 7, 7,
            4.5, 9, 4.5, 9, 4.5,
            9, 4.5, 8, 4, 8,
            4.5, 8, 4, 10, 5,
            3, 3, 4, 4, 4,
            3.5, 4.5, 9, 4.5, 9,
            4.5, 9, 4.5, 8, 4,
            8, 4.5, 8, 4, 10,
            5, 3, 3, 4, 4,
            4, 3.5, 10000, 10000, 10000,
            10000, 16, 16, 16, 16,
            14, 16, 16, 16, 16,
            14, 10000, 10000
        ),
        maxDSize = cms.int32(10),
        maxDSizeB1 = cms.int32(12),
        maxDSizePred = cms.int32(24),
        maxDZ = cms.vdouble(
            16, 16, 25, 25, 0,
            0, 13, 15, 19, 21,
            0, 0, 9, 13, 0,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 15, -10, 35,
            22, 32.5, 50, 50, 70,
            70, -5, -10, -5, -15,
            -25, 50, 40
        ),
        maxInner = cms.vdouble(
            17, 14, 10000, 10000, -4,
            -7, 17, 15, 10000, 10000,
            -6, -9, 18, 10000, -11,
            14, 14, 13, 13, 13,
            13, 13, 13, 13, 13,
            13, 13, 13, 16, 16,
            6, 4, 6, 22, 22,
            22, 14, 14, 13, 13,
            13, 13, 13, 13, 13,
            13, 13, 13, 13, 16,
            16, 6, 4, 6, 22,
            22, 22, 10, -10, 20,
            20, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 1200, 1200
        ),
        maxOuter = cms.vdouble(
            10000, 10000, 10, 10000, 10,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 21,
            7, 7, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            21, 7, 7, 10000, 10000,
            10000, 10000, 30, -25, 50,
            45, 57, 80, 95, 110,
            10000, -30, -40, -55, -70,
            -80, 10000, 10000
        ),
        maxStubCurvSigma = cms.optional.vdouble,
        maxZ0 = cms.vdouble(
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5
        ),
        minDZ = cms.vdouble(
            -16, -16, 0, 0, -25,
            -25, -13, -15, 0, 0,
            -19, -21, -9, 0, -13,
            -10000, -10000, -10000, -10000, -10000,
            -10000, -10000, -10000, -10000, -10000,
            -10000, -10000, -10000, -10000, -10000,
            -10000, -10000, -10000, -10000, -10000,
            -10000, -10000, -10000, -10000, -10000,
            -10000, -10000, -10000, -10000, -10000,
            -10000, -10000, -10000, -10000, -10000,
            -10000, -10000, -10000, -10000, -10000,
            -10000, -10000, -15, -35, 10,
            -22, 5, -10, 5, 15,
            25, -32.5, -50, -50, -70,
            -70, -50, -40
        ),
        minInner = cms.vdouble(
            -17, -14, 4, 7, -10000,
            -10000, -17, -15, 6, 9,
            -10000, -10000, -18, 11, -10000,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 12, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 12,
            0, 0, 0, 0, 0,
            0, 0, -10, -20, 10,
            -20, 11, 11, 11, 11,
            0, 11, 11, 11, 11,
            0, -1200, -1200
        ),
        minInnerSizeB1 = cms.int32(20),
        minInnerSizeB2 = cms.int32(18),
        minOuter = cms.vdouble(
            -10000, -10000, 0, 0, 0,
            0, -10000, -10000, 6, 6,
            6, 6, -10000, 11, 11,
            3, 3, 3, 3, 3,
            3, 3, 3, 3, 3,
            4, 4, 3, 20, 6,
            0, 0, 0, 7, 7,
            7, 3, 3, 3, 3,
            3, 3, 3, 3, 3,
            3, 4, 4, 3, 20,
            6, 0, 0, 0, 7,
            7, 7, -30, -50, 25,
            -45, 30, 40, 55, 70,
            80, -57, -70, -95, -110,
            -10000, -10000, -10000
        ),

        minPt = cms.vdouble(
            highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut,
            highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut,
            highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut,
            highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut,
            highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut,
            highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut,
            highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut,
            highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut,
            highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut,
            highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut,
            highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut,
            highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut,
            highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut,
            highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut,
            highPtItMinPtCut, highPtItMinPtCut, highPtItMinPtCut
        )
    ),
))

# pixel tracks SoA producer on the cpu, for validation
pixelTracksHighPtAlpakaSerial = makeSerialClone(pixelTracksHighPtAlpaka,
    pixelRecHitSrc = 'siPixelRecHitsPreSplittingAlpakaSerial'
)

(pixelTrackMask & phase2CAExtension).toModify(pixelTracksHighPtAlpakaSerial,
                           pixelRecHitSrc = 'siPixelRecHitsExtendedPreSplittingAlpakaSerial',
                           maxNumberOfDoublets = str(12400000),    # could be lowered to 315k, keeping the same for a fair comparison with master
                           maxNumberOfTuples   = str(32 * 32 * 1024),   # this couul be much lower (2.1k, these are quads)
                           )

# pixel tracks SoA merger
from RecoTracker.PixelSeeding.pixelTracksMaskingSoA_cfi import pixelTracksMaskingSoA as _pixelTracksMaskingSoA

pixelTracksHighPtMaskingSoA = _pixelTracksMaskingSoA.clone(
    iterationIndex = 1,
    minQuality = "tight",
    tracksSoASrc = "pixelTracksHighPtAlpaka",
)

pixelTracksLowPtAlpaka = _pixelTracksAlpakaPhase1.clone(
    avgHitsPerTrack    = 4.6,
    avgCellsPerHit     = 13,
    avgCellsPerCell    = 0.0268,
    avgTracksPerCell   = 0.0123,
    maxNumberOfDoublets = str(12400000),    # could be lowered to 315k, keeping the same for a fair comparison with master
    maxNumberOfTuples   = str(32 * 32 * 1024),   # this couul be much lower (2.1k, these are quads)
)

lowPtItMinPtCut = 0.45 # 0.45 works, but 0.40 starts showing too many tracks with "zero" eta and phi
                     # Maybe there is another cell cut that balances this, but need to check

(pixelTrackMask & phase2CAExtension).toReplaceWith(pixelTracksLowPtAlpaka,_pixelTracksAlpakaPhase2Extended.clone(
    hitMask = "pixelTracksHighPtMaskingSoA",
    pixelRecHitSrc = "siPixelRecHitsExtendedPreSplittingAlpaka",
    maxNumberOfDoublets = str(12400000),    # could be lowered to 315k, keeping the same for a fair comparison with master
    maxNumberOfTuples   = str(32 * 32 * 1024),   # this couul be much lower (2.1k, these are quads)
    tripletCuts = cms.PSet(
        floorDCA = cms.optional.vdouble,
        maxCurv = cms.double(0.035),
        maxDCA = cms.vdouble(
            0.15000000596046448, 0.25, 0.20000000298023224, 0.20000000298023224, 0.25,
            0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.10000000149011612, 0.10000000149011612,
            0.10000000149011612, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0
        ),
        maxPhiResid = cms.double(-1),
        maxRZTolerance = cms.vdouble(
            0.0020000000949949026, 0.0020000000949949026, 0.0020000000949949026, 0.0020000000949949026, 0.003000000026077032,
            0.003000000026077032, 0.003000000026077032, 0.003000000026077032, 0.003000000026077032, 0.003000000026077032,
            0.003000000026077032, 0.003000000026077032, 0.003000000026077032, 0.003000000026077032, 0.003000000026077032,
            0.003000000026077032, 0.003000000026077032, 0.003000000026077032, 0.003000000026077032, 0.003000000026077032,
            0.003000000026077032, 0.003000000026077032, 0.003000000026077032, 0.003000000026077032, 0.003000000026077032,
            0.003000000026077032, 0.003000000026077032, 0.003000000026077032, 0.003000000026077032, 0.003000000026077032,
            0.003000000026077032, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0
        ),
        maxStubGeomCurvSigma = cms.optional.vdouble,
        maxStubInnerDoubletDCurv = cms.optional.vdouble,
        ptmin = cms.double(lowPtItMinPtCut + 0.05),
        sameDPhiSign = cms.bool(False)
    ),
    trackQualityCuts = cms.PSet(
        maxChi2 = cms.double(5),
        maxChi2Quintuplets = cms.double(3),
        maxChi2TripletsOrQuadruplets = cms.double(1),
        maxTip = cms.double(0.3),
        maxZip = cms.double(12),
        minPt = cms.double(lowPtItMinPtCut + 0.05)
    ),
    iterationName = "promptLowPt",
    doubletCuts = cms.PSet(
        dzdrFact = cms.double(15.199999809265137),
        maxDPhi = cms.vint32(
            int(1.0*350), int(1.0*600), int(1.0*450), int(1.0*522), int(1.0*450),
            int(1.0*522), int(1.0*400), int(1.0*650), int(1.0*500), int(1.0*730),
            int(1.0*500), int(1.0*730), int(1.0*350), int(1.0*400), int(1.0*400),
            int(1.0*300), int(1.0*522), int(1.0*300), int(1.0*522), int(1.0*250),
            int(1.0*522), int(1.0*250), int(1.0*522), int(1.0*250), int(1.0*522),
            int(1.0*300), int(1.0*522), int(1.0*240), int(1.0*650), int(1.0*300),
            int(1.0*200), int(1.0*220), int(1.0*250), int(1.0*250), int(1.0*250),
            int(1.0*250), int(1.0*300), int(1.0*522), int(1.0*300), int(1.0*522),
            int(1.0*250), int(1.0*522), int(1.0*250), int(1.0*522), int(1.0*250),
            int(1.0*522), int(1.0*300), int(1.0*522), int(1.0*240), int(1.0*650),
            int(1.0*300), int(1.0*200), int(1.0*220), int(1.0*250), int(1.0*250),
            int(1.0*250), int(1.0*250), int(1.0*1200), int(1.0*1200), int(1.0*1200),
            int(1.0*1000), int(1.0*1000), int(1.0*1000), int(1.0*1000), int(1.0*1000),
            int(1.0*850), int(1.0*1000), int(1.0*1000), int(1.0*1000), int(1.0*1000),
            int(1.0*1000), int(1.0*1100), int(1.0*1250)
        ),
        maxDR = cms.vdouble(
            5, 10, 8, 5, 8,
            5, 7, 10, 8, 10,
            8, 10, 7, 7, 7,
            4.5, 9, 4.5, 9, 4.5,
            9, 4.5, 8, 4, 8,
            4.5, 8, 4, 10, 5,
            3, 3, 4, 4, 4,
            3.5, 4.5, 9, 4.5, 9,
            4.5, 9, 4.5, 8, 4,
            8, 4.5, 8, 4, 10,
            5, 3, 3, 4, 4,
            4, 3.5, 10000, 10000, 10000,
            10000, 16, 16, 16, 16,
            14, 16, 16, 16, 16,
            14, 10000, 10000
        ),
        maxDSize = cms.int32(10),
        maxDSizeB1 = cms.int32(12),
        maxDSizePred = cms.int32(24),
        maxDZ = cms.vdouble(
            16, 16, 25, 25, 0,
            0, 13, 15, 19, 21,
            0, 0, 9, 13, 0,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 15, -10, 35,
            22, 32.5, 50, 50, 70,
            70, -5, -10, -5, -15,
            -25, 50, 40
        ),
        maxInner = cms.vdouble(
            17, 14, 10000, 10000, -4,
            -7, 17, 15, 10000, 10000,
            -6, -9, 18, 10000, -11,
            14, 14, 13, 13, 13,
            13, 13, 13, 13, 13,
            13, 13, 13, 16, 16,
            6, 4, 6, 22, 22,
            22, 14, 14, 13, 13,
            13, 13, 13, 13, 13,
            13, 13, 13, 13, 16,
            16, 6, 4, 6, 22,
            22, 22, 10, -10, 20,
            20, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 1200, 1200
        ),
        maxOuter = cms.vdouble(
            10000, 10000, 10, 10000, 10,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 21,
            7, 7, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            10000, 10000, 10000, 10000, 10000,
            21, 7, 7, 10000, 10000,
            10000, 10000, 30, -25, 50,
            45, 57, 80, 95, 110,
            10000, -30, -40, -55, -70,
            -80, 10000, 10000
        ),
        maxStubCurvSigma = cms.optional.vdouble,
        maxZ0 = cms.vdouble(
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5
        ),
        minDZ = cms.vdouble(
            -16, -16, 0, 0, -25,
            -25, -13, -15, 0, 0,
            -19, -21, -9, 0, -13,
            -10000, -10000, -10000, -10000, -10000,
            -10000, -10000, -10000, -10000, -10000,
            -10000, -10000, -10000, -10000, -10000,
            -10000, -10000, -10000, -10000, -10000,
            -10000, -10000, -10000, -10000, -10000,
            -10000, -10000, -10000, -10000, -10000,
            -10000, -10000, -10000, -10000, -10000,
            -10000, -10000, -10000, -10000, -10000,
            -10000, -10000, -15, -35, 10,
            -22, 5, -10, 5, 15,
            25, -32.5, -50, -50, -70,
            -70, -50, -40
        ),
        minInner = cms.vdouble(
            -17, -14, 4, 7, -10000,
            -10000, -17, -15, 6, 9,
            -10000, -10000, -18, 11, -10000,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 12, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 12,
            0, 0, 0, 0, 0,
            0, 0, -10, -20, 10,
            -20, 11, 11, 11, 11,
            0, 11, 11, 11, 11,
            0, -1200, -1200
        ),
        minInnerSizeB1 = cms.int32(20),
        minInnerSizeB2 = cms.int32(18),
        minOuter = cms.vdouble(
            -10000, -10000, 0, 0, 0,
            0, -10000, -10000, 6, 6,
            6, 6, -10000, 11, 11,
            3, 3, 3, 3, 3,
            3, 3, 3, 3, 3,
            4, 4, 3, 20, 6,
            0, 0, 0, 7, 7,
            7, 3, 3, 3, 3,
            3, 3, 3, 3, 3,
            3, 4, 4, 3, 20,
            6, 0, 0, 0, 7,
            7, 7, -30, -50, 25,
            -45, 30, 40, 55, 70,
            80, -57, -70, -95, -110,
            -10000, -10000, -10000
        ),

        minPt = cms.vdouble(
            lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut,
            lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut,
            lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut,
            lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut,
            lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut,
            lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut,
            lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut,
            lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut,
            lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut,
            lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut,
            lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut,
            lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut,
            lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut,
            lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut,
            lowPtItMinPtCut, lowPtItMinPtCut, lowPtItMinPtCut
        )
    ),
))

# pixel tracks SoA producer on the cpu, for validation
pixelTracksLowPtAlpakaSerial = makeSerialClone(pixelTracksLowPtAlpaka,
    pixelRecHitSrc = 'siPixelRecHitsPreSplittingAlpakaSerial'
)

(pixelTrackMask & phase2CAExtension).toModify(pixelTracksLowPtAlpakaSerial,
                            hitMask = "pixelTracksHighPtMaskingSoA",
                            pixelRecHitSrc = "siPixelRecHitsExtendedPreSplittingAlpaka",
                            trackQualityCuts = cms.PSet(
                                maxChi2 = cms.double(5),
                                maxChi2Quintuplets = cms.double(3),
                                maxChi2TripletsOrQuadruplets = cms.double(1),
                                maxTip = cms.double(0.3),
                                maxZip = cms.double(12),
                                minPt = cms.double(0.3)
                            ),
                            maxNumberOfDoublets = str(12400000),    # could be lowered to 315k, keeping the same for a fair comparison with master
                            maxNumberOfTuples   = str(32 * 32 * 1024),   # this couul be much lower (2.1k, these are quads)
                           )

# legacy pixel tracks from SoA
from  RecoTracker.PixelTrackFitting.pixelTrackProducerFromSoAAlpaka_cfi import pixelTrackProducerFromSoAAlpaka as _pixelTrackProducerFromSoAAlpaka

(alpaka & ~phase2CAExtension).toReplaceWith(pixelTracks, _pixelTrackProducerFromSoAAlpaka.clone(
    pixelRecHitLegacySrc = "siPixelRecHitsPreSplitting",
))

# pixel tracks SoA merger
from RecoTracker.PixelSeeding.pixelTracksSoAMerger_cfi import pixelTracksSoAMerger as _pixelTracksSoAMerger

pixelTracksLowPtMaskingSoA = _pixelTracksMaskingSoA.clone(
    iterationIndex = 2,
    minQuality = "tight",
    recHitsMaskSoASrc = "pixelTracksHighPtMaskingSoA",
    tracksSoASrc = "pixelTracksLowPtAlpaka",
)

pixelTracksDisplHighPtAlpaka = _pixelTracksAlpakaPhase1.clone(
    avgHitsPerTrack    = 4.6,
    avgCellsPerHit     = 13,
    avgCellsPerCell    = 0.0268,
    avgTracksPerCell   = 0.0123,
    maxNumberOfDoublets = str(12400000),    # could be lowered to 315k, keeping the same for a fair comparison with master
    maxNumberOfTuples   = str(32 * 32 * 1024),   # this couul be much lower (2.1k, these are quads)
)

(pixelTrackMask & phase2CAExtension).toReplaceWith(pixelTracksDisplHighPtAlpaka,_pixelTracksAlpakaPhase2Extended.clone(
    hitMask = "pixelTracksLowPtMaskingSoA",
    pixelRecHitSrc = "siPixelRecHitsExtendedPreSplittingAlpaka",
    trackQualityCuts = cms.PSet(
        maxChi2 = cms.double(5),
        maxChi2Quintuplets = cms.double(3),
        maxChi2TripletsOrQuadruplets = cms.double(1),
        maxTip = cms.double(10.0),
        maxZip = cms.double(12),
        minPt = cms.double(0.9)
    ),
    maxNumberOfDoublets = str(12400000),    # could be lowered to 315k, keeping the same for a fair comparison with master
    maxNumberOfTuples   = str(32 * 32 * 1024),   # this couul be much lower (2.1k, these are quads)
))

pixelTracksDisplHighPtAlpakaSerial = makeSerialClone(pixelTracksDisplHighPtAlpaka,
    pixelRecHitSrc = 'siPixelRecHitsPreSplittingAlpakaSerial'
)

(pixelTrackMask & phase2CAExtension).toModify(pixelTracksDisplHighPtAlpakaSerial,
                           hitMask = "pixelTracksLowPtMaskingSoA",
                           pixelRecHitSrc = 'siPixelRecHitsExtendedPreSplittingAlpakaSerial',
                           trackQualityCuts = cms.PSet(
                               maxChi2 = cms.double(5),
                               maxChi2Quintuplets = cms.double(3),
                               maxChi2TripletsOrQuadruplets = cms.double(1),
                               maxTip = cms.double(10.0),
                               maxZip = cms.double(12),
                               minPt = cms.double(0.9)
                           ),
                           maxNumberOfDoublets = str(12400000),    # could be lowered to 315k, keeping the same for a fair comparison with master
                           maxNumberOfTuples   = str(32 * 32 * 1024),   # this couul be much lower (2.1k, these are quads)
                           )

pixelTracksDisplHighPtMaskingSoA = _pixelTracksMaskingSoA.clone(
    minQuality = "tight",
    recHitsMaskSoASrc = "pixelTracksLowPtMaskingSoA",
    tracksSoASrc = "pixelTracksDisplHighPtAlpaka",
)



pixelTracksDisplLowPtAlpaka = _pixelTracksAlpakaPhase1.clone(
    avgHitsPerTrack    = 4.6,
    avgCellsPerHit     = 13,
    avgCellsPerCell    = 0.0268,
    avgTracksPerCell   = 0.0123,
    maxNumberOfDoublets = str(12400000),    # could be lowered to 315k, keeping the same for a fair comparison with master
    maxNumberOfTuples   = str(32 * 32 * 1024),   # this couul be much lower (2.1k, these are quads)
)

(pixelTrackMask & phase2CAExtension).toReplaceWith(pixelTracksDisplLowPtAlpaka,_pixelTracksAlpakaPhase2Extended.clone(
    hitMask = "pixelTracksDisplHighPtMaskingSoA",
    pixelRecHitSrc = "siPixelRecHitsExtendedPreSplittingAlpaka",
    trackQualityCuts = cms.PSet(
        maxChi2 = cms.double(5),
        maxChi2Quintuplets = cms.double(3),
        maxChi2TripletsOrQuadruplets = cms.double(1),
        maxTip = cms.double(10.0),
        maxZip = cms.double(12),
        minPt = cms.double(0.3)
    ),
    maxNumberOfDoublets = str(12400000),    # could be lowered to 315k, keeping the same for a fair comparison with master
    maxNumberOfTuples   = str(32 * 32 * 1024),   # this couul be much lower (2.1k, these are quads)
))

pixelTracksDisplLowPtAlpakaSerial = makeSerialClone(pixelTracksDisplLowPtAlpaka,
    pixelRecHitSrc = 'siPixelRecHitsPreSplittingAlpakaSerial'
)

(pixelTrackMask & phase2CAExtension).toModify(pixelTracksDisplLowPtAlpakaSerial,
                           hitMask = "pixelTracksDisplHighPtMaskingSoA",
                           pixelRecHitSrc = 'siPixelRecHitsExtendedPreSplittingAlpakaSerial',
                           trackQualityCuts = cms.PSet(
                               maxChi2 = cms.double(5),
                               maxChi2Quintuplets = cms.double(3),
                               maxChi2TripletsOrQuadruplets = cms.double(1),
                               maxTip = cms.double(10.0),
                               maxZip = cms.double(12),
                               minPt = cms.double(0.3)
                           ),
                           maxNumberOfDoublets = str(12400000),    # could be lowered to 315k, keeping the same for a fair comparison with master
                           maxNumberOfTuples   = str(32 * 32 * 1024),   # this couul be much lower (2.1k, these are quads)
                           )

(pixelTrackMask & phase2CAExtension).toReplaceWith(pixelTracksAlpaka, _pixelTracksSoAMerger.clone(
    inputTkSoAs = cms.VInputTag("pixelTracksHighPtAlpaka","pixelTracksLowPtAlpaka"),
    # inputTkSoAs = cms.VInputTag("pixelTracksHighPtAlpaka","pixelTracksLowPtAlpaka","pixelTracksDisplHighPtAlpaka"),
    minQuality = cms.string('tight'),
    matchFraction = cms.double(0.0),
))

(pixelTrackMask & phase2CAExtension).toReplaceWith(pixelTracksHighPt, _pixelTrackProducerFromSoAAlpaka.clone(
    pixelRecHitLegacySrc = "siPixelRecHitsPreSplitting",
    beamSpot = cms.InputTag("offlineBeamSpot"),
    minNumberOfHits = cms.int32(0),
    minQuality = cms.string('tight'),
    trackSrc = cms.InputTag("pixelTracksHighPtAlpaka"),
    outerTrackerRecHitSrc = cms.InputTag("siPhase2RecHits"),
    outerTrackerRecHitSoAConverterSrc = cms.InputTag("phase2OTRecHitsSoAConverter"),
    useOTExtension = cms.bool(True),
    requireQuadsFromConsecutiveLayers = cms.bool(True)
))

(pixelTrackMask & phase2CAExtension).toReplaceWith(pixelTracksLowPt, _pixelTrackProducerFromSoAAlpaka.clone(
    pixelRecHitLegacySrc = "siPixelRecHitsPreSplitting",
    beamSpot = cms.InputTag("offlineBeamSpot"),
    minNumberOfHits = cms.int32(0),
    minQuality = cms.string('tight'),
    trackSrc = cms.InputTag("pixelTracksLowPtAlpaka"),
    outerTrackerRecHitSrc = cms.InputTag("siPhase2RecHits"),
    outerTrackerRecHitSoAConverterSrc = cms.InputTag("phase2OTRecHitsSoAConverter"),
    useOTExtension = cms.bool(True),
    requireQuadsFromConsecutiveLayers = cms.bool(True)
))

(pixelTrackMask & phase2CAExtension).toReplaceWith(pixelTracks, _pixelTrackProducerFromSoAAlpaka.clone(
    pixelRecHitLegacySrc = "siPixelRecHitsPreSplitting",
    beamSpot = cms.InputTag("offlineBeamSpot"),
    minNumberOfHits = cms.int32(0),
    minQuality = cms.string('tight'),
    trackSrc = cms.InputTag("pixelTracksAlpaka"),
    outerTrackerRecHitSrc = cms.InputTag("siPhase2RecHits"),
    outerTrackerRecHitSoAConverterSrc = cms.InputTag("phase2OTRecHitsSoAConverter"),
    useOTExtension = cms.bool(True),
    requireQuadsFromConsecutiveLayers = cms.bool(True)
))

# (pixelTrackMask & phase2CAExtension).toReplaceWith(pixelTracksTask, cms.Task(
#     # Build the highPt pixel ntuplets and the pixel tracks in SoA format with alpaka on the device
#     pixelTracksHighPtAlpaka,
#     # Build the highPt pixel ntuplets and the pixel tracks in SoA format with alpaka on the cpu (if requested by the validation)
#     pixelTracksHighPtAlpakaSerial,
#     # Updates the TrackingRecHitsMasking collection for next iteration
#     pixelTracksHighPtMaskingSoA,

#     # Build the lowPt pixel ntuplets and the pixel tracks in SoA format with alpaka on the device
#     pixelTracksLowPtAlpaka,
#     # Build the lowPt pixel ntuplets and the pixel tracks in SoA format with alpaka on the cpu (if requested by the validation)
#     pixelTracksLowPtAlpakaSerial,
#     # Updates the TrackingRecHitsMasking collection for next iteration
#     pixelTracksLowPtMaskingSoA,

#     # Build the lowPt pixel ntuplets and the pixel tracks in SoA format with alpaka on the device
#     pixelTracksDisplHighPtAlpaka,
#     # Build the lowPt pixel ntuplets and the pixel tracks in SoA format with alpaka on the cpu (if requested by the validation)
#     pixelTracksDisplHighPtAlpakaSerial,
#     # Updates the TrackingRecHitsMasking collection for next iteration
#     pixelTracksDisplHighPtMaskingSoA,

#     # Build the lowPt pixel ntuplets and the pixel tracks in SoA format with alpaka on the device
#     pixelTracksDisplLowPtAlpaka,
#     # Build the lowPt pixel ntuplets and the pixel tracks in SoA format with alpaka on the cpu (if requested by the validation)
#     pixelTracksDisplLowPtAlpakaSerial,
#     # No need for masking after the last iteration

#     # Merge the produced SoAs directly
#     pixelTracksAlpaka,
#     # Convert the pixel tracks from SoA to legacy format
#     pixelTracks)
# )

# Just used to check only 2 iterations; more iterations (above) will probably be better
(pixelTrackMask & phase2CAExtension).toReplaceWith(pixelTracksTask, cms.Task(
    # Build the highPt pixel ntuplets and the pixel tracks in SoA format with alpaka on the device
    pixelTracksHighPtAlpaka,
    # Build the highPt pixel ntuplets and the pixel tracks in SoA format with alpaka on the cpu (if requested by the validation)
    pixelTracksHighPtAlpakaSerial,
    # Updates the TrackingRecHitsMasking collection for next iteration
    pixelTracksHighPtMaskingSoA,
    # Convert the highPt pixel tracks from SoA to legacy format for validation
    pixelTracksHighPt,

    # Build the lowPt pixel ntuplets and the pixel tracks in SoA format with alpaka on the device
    pixelTracksLowPtAlpaka,
    # Build the lowPt pixel ntuplets and the pixel tracks in SoA format with alpaka on the cpu (if requested by the validation)
    pixelTracksLowPtAlpakaSerial,
    # Convert the lowPt pixel tracks from SoA to legacy format for validation
    pixelTracksLowPt,

    # Merge the produced SoAs directly
    pixelTracksAlpaka,
    # Convert the pixel tracks from SoA to legacy format
    pixelTracks)
)
