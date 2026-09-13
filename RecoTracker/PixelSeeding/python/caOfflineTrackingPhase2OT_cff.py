import FWCore.ParameterSet.Config as cms

from RecoTracker.PixelSeeding.caHitNtupletAlpakaPhase2OT_cfi import caHitNtupletAlpakaPhase2OT as _caHitNtupletAlpakaPhase2OT
from RecoTracker.PixelSeeding.caMasking_cfi import caMasking as _caMasking
from RecoTracker.PixelSeeding.pixelTracksSoAMerger_cfi import pixelTracksSoAMerger as _pixelTracksSoAMerger
from RecoTracker.PixelTrackFitting.pixelTrackProducerFromSoAAlpaka_cfi import pixelTrackProducerFromSoAAlpaka as _pixelTrackProducerFromSoAAlpaka


# Standalone Phase-2 CA-offline sketch. This file is intentionally not wired
# into the standard tracking sequences yet; a future caOfflineTracking
# processModifier can replace the Phase-2 iterative-tracking seed source with
# caOfflinePhase2OTTracks once the final SoA-to-CKF boundary is settled.

_phase2OTLayerNames = (
    "BPix1",
    "BPix2",
    "BPix3",
    "BPix4",
    "FPix+1",
    "FPix+2",
    "FPix+3",
    "FPix+4",
    "FPix+5",
    "FPix+6",
    "FPix+7",
    "FPix+8",
    "FPix+9",
    "FPix+10",
    "FPix+11",
    "FPix+12",
    "FPix-1",
    "FPix-2",
    "FPix-3",
    "FPix-4",
    "FPix-5",
    "FPix-6",
    "FPix-7",
    "FPix-8",
    "FPix-9",
    "FPix-10",
    "FPix-11",
    "FPix-12",
    "TOBPSP1",
    "TOBPSP2",
    "TOBPSP3",
)

_phase2OTDefaultPairGraph = (
    (0, 1),
    (0, 2),
    (0, 4),
    (0, 5),
    (0, 16),
    (0, 17),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 16),
    (1, 17),
    (2, 3),
    (2, 4),
    (2, 16),
    (4, 5),
    (4, 6),
    (5, 6),
    (5, 7),
    (6, 7),
    (6, 8),
    (7, 8),
    (7, 9),
    (8, 9),
    (8, 10),
    (9, 10),
    (9, 11),
    (10, 11),
    (10, 12),
    (11, 12),
    (11, 13),
    (12, 13),
    (12, 14),
    (13, 14),
    (13, 15),
    (14, 15),
    (16, 17),
    (16, 18),
    (17, 18),
    (17, 19),
    (18, 19),
    (18, 20),
    (19, 20),
    (19, 21),
    (20, 21),
    (20, 22),
    (21, 22),
    (21, 23),
    (22, 23),
    (22, 24),
    (23, 24),
    (23, 25),
    (24, 25),
    (24, 26),
    (25, 26),
    (25, 27),
    (26, 27),
    (2, 28),
    (2, 28),
    (2, 28),
    (3, 28),
    (4, 28),
    (5, 28),
    (6, 28),
    (7, 28),
    (8, 28),
    (16, 28),
    (17, 28),
    (18, 28),
    (19, 28),
    (20, 28),
    (28, 29),
    (29, 30),
)

_fullPhase2OTPairIds = tuple(range(len(_phase2OTDefaultPairGraph)))
_pixelAndOTStartPairIds = tuple(i for i, pair in enumerate(_phase2OTDefaultPairGraph) if pair[0] in (0, 1, 2, 4, 16))
_reducedForwardAndOTPairIds = tuple(
    i
    for i, pair in enumerate(_phase2OTDefaultPairGraph)
    if pair[0] <= 11 or 16 <= pair[0] <= 23 or pair[0] >= 28 or pair[1] >= 28
)

_pairIndexedGeometryParams = (
    "skipsLayers",
    "phiCuts",
    "ptCuts",
    "minInner",
    "maxInner",
    "minOuter",
    "maxOuter",
    "maxDR",
    "minDZ",
    "maxDZ",
)


def _flatten_pair_graph(pair_ids):
    return cms.vuint32(*(layer for pair_id in pair_ids for layer in _phase2OTDefaultPairGraph[pair_id]))


def _slice_cms_vector(values, pair_ids):
    return type(values)(*[values[pair_id] for pair_id in pair_ids])


def _phase2OTGeometry(pair_ids=_fullPhase2OTPairIds, *, ptCut=None, startingPairIds=None):
    geometry = _caHitNtupletAlpakaPhase2OT.geometry.clone()
    geometry.pairGraph = _flatten_pair_graph(pair_ids)

    for name in _pairIndexedGeometryParams:
        value = getattr(geometry, name)
        setattr(geometry, name, _slice_cms_vector(value, pair_ids))

    if ptCut is not None:
        geometry.ptCuts = cms.vdouble([ptCut] * len(pair_ids))

    if startingPairIds is None:
        oldStartingPairs = set(_caHitNtupletAlpakaPhase2OT.geometry.startingPairs)
        startingPairIds = tuple(pair_id for pair_id in pair_ids if pair_id in oldStartingPairs)

    oldToNewPairId = {pair_id: newPairId for newPairId, pair_id in enumerate(pair_ids)}
    geometry.startingPairs = cms.vuint32(*(oldToNewPairId[pair_id] for pair_id in startingPairIds if pair_id in oldToNewPairId))
    return geometry


def _phase2OTCAIteration(
    name,
    *,
    iterationName,
    hitMask,
    minHitsPerNtuplet,
    ptmin,
    trackMinPt,
    maxChi2,
    maxChi2TripletsOrQuadruplets,
    maxChi2Quintuplets,
    pairIds=_fullPhase2OTPairIds,
    pairPtCut=None,
    maxNumberOfDoublets="0.00022*pow(x,2) + 0.53*x + 10000",
    maxNumberOfTuples="0.00006*pow(x,2) + 0.18*x + 10000",
    hardCurvCut=None,
):
    producer = _caHitNtupletAlpakaPhase2OT.clone(
        hitMask=hitMask,
        pixelRecHitSrc="siPixelRecHitsExtendedPreSplittingAlpaka",
        iterationName=iterationName,
        minHitsPerNtuplet=minHitsPerNtuplet,
        ptmin=ptmin,
        geometry=_phase2OTGeometry(pairIds, ptCut=pairPtCut),
        maxNumberOfDoublets=maxNumberOfDoublets,
        maxNumberOfTuples=maxNumberOfTuples,
    )
    producer.trackQualityCuts.minPt = cms.double(trackMinPt)
    producer.trackQualityCuts.maxChi2 = cms.double(maxChi2)
    producer.trackQualityCuts.maxChi2TripletsOrQuadruplets = cms.double(maxChi2TripletsOrQuadruplets)
    producer.trackQualityCuts.maxChi2Quintuplets = cms.double(maxChi2Quintuplets)
    if hardCurvCut is not None:
        producer.hardCurvCut = cms.double(hardCurvCut)
    return producer


caOfflinePhase2OTInitialStep = _phase2OTCAIteration(
    "caOfflinePhase2OTInitialStep",
    iterationName="initialStep",
    hitMask="siPixelRecHitsExtendedPreSplittingAlpaka",
    minHitsPerNtuplet=4,
    ptmin=0.65,
    trackMinPt=0.65,
    maxChi2=5.0,
    maxChi2TripletsOrQuadruplets=1.0,
    maxChi2Quintuplets=3.0,
    pairIds=_fullPhase2OTPairIds,
    pairPtCut=0.60,
)

caOfflinePhase2OTInitialStepMask = _caMasking.clone(
    recHitsMaskSoASrc="siPixelRecHitsExtendedPreSplittingAlpaka",
    tracksSoASrc="caOfflinePhase2OTInitialStep",
    minQuality="tight",
    iterationIndex=0,
)

caOfflinePhase2OTHighPtTripletStep = _phase2OTCAIteration(
    "caOfflinePhase2OTHighPtTripletStep",
    iterationName="highPtTripletStep",
    hitMask="caOfflinePhase2OTInitialStepMask",
    minHitsPerNtuplet=3,
    ptmin=0.75,
    trackMinPt=0.75,
    maxChi2=5.0,
    maxChi2TripletsOrQuadruplets=1.0,
    maxChi2Quintuplets=3.0,
    pairIds=_pixelAndOTStartPairIds,
    pairPtCut=0.70,
)

caOfflinePhase2OTHighPtTripletStepMask = _caMasking.clone(
    recHitsMaskSoASrc="caOfflinePhase2OTInitialStepMask",
    tracksSoASrc="caOfflinePhase2OTHighPtTripletStep",
    minQuality="tight",
    iterationIndex=1,
)

caOfflinePhase2OTLowPtQuadStep = _phase2OTCAIteration(
    "caOfflinePhase2OTLowPtQuadStep",
    iterationName="lowPtQuadStep",
    hitMask="caOfflinePhase2OTHighPtTripletStepMask",
    minHitsPerNtuplet=4,
    ptmin=0.45,
    trackMinPt=0.45,
    maxChi2=5.0,
    maxChi2TripletsOrQuadruplets=1.0,
    maxChi2Quintuplets=3.0,
    pairIds=_fullPhase2OTPairIds,
    pairPtCut=0.35,
    maxNumberOfDoublets=str(12400000),
    maxNumberOfTuples=str(32 * 32 * 1024),
    hardCurvCut=0.035,
)

caOfflinePhase2OTLowPtQuadStepMask = _caMasking.clone(
    recHitsMaskSoASrc="caOfflinePhase2OTHighPtTripletStepMask",
    tracksSoASrc="caOfflinePhase2OTLowPtQuadStep",
    minQuality="tight",
    iterationIndex=2,
)

caOfflinePhase2OTLowPtTripletStep = _phase2OTCAIteration(
    "caOfflinePhase2OTLowPtTripletStep",
    iterationName="lowPtTripletStep",
    hitMask="caOfflinePhase2OTLowPtQuadStepMask",
    minHitsPerNtuplet=3,
    ptmin=0.45,
    trackMinPt=0.45,
    maxChi2=5.0,
    maxChi2TripletsOrQuadruplets=1.0,
    maxChi2Quintuplets=3.0,
    pairIds=_reducedForwardAndOTPairIds,
    pairPtCut=0.40,
    maxNumberOfDoublets=str(12400000),
    maxNumberOfTuples=str(32 * 32 * 1024),
    hardCurvCut=0.035,
)

caOfflinePhase2OTLowPtTripletStepMask = _caMasking.clone(
    recHitsMaskSoASrc="caOfflinePhase2OTLowPtQuadStepMask",
    tracksSoASrc="caOfflinePhase2OTLowPtTripletStep",
    minQuality="tight",
    iterationIndex=3,
)

caOfflinePhase2OTDetachedQuadStep = _phase2OTCAIteration(
    "caOfflinePhase2OTDetachedQuadStep",
    iterationName="detachedQuadStep",
    hitMask="caOfflinePhase2OTLowPtTripletStepMask",
    minHitsPerNtuplet=4,
    ptmin=0.50,
    trackMinPt=0.50,
    maxChi2=8.0,
    maxChi2TripletsOrQuadruplets=3.0,
    maxChi2Quintuplets=5.0,
    pairIds=_fullPhase2OTPairIds,
    pairPtCut=0.45,
    maxNumberOfDoublets=str(12400000),
    maxNumberOfTuples=str(32 * 32 * 1024),
    hardCurvCut=0.030,
)

caOfflinePhase2OTTracksSoA = _pixelTracksSoAMerger.clone(
    inputTkSoAs=cms.VInputTag(
        "caOfflinePhase2OTInitialStep",
        "caOfflinePhase2OTHighPtTripletStep",
        "caOfflinePhase2OTLowPtQuadStep",
        "caOfflinePhase2OTLowPtTripletStep",
        "caOfflinePhase2OTDetachedQuadStep",
    ),
    minQuality="tight",
    matchFraction=0.0,
    minHitsForDuplicate=3,
)

caOfflinePhase2OTTracks = _pixelTrackProducerFromSoAAlpaka.clone(
    pixelRecHitLegacySrc="siPixelRecHitsPreSplitting",
    beamSpot=cms.InputTag("offlineBeamSpot"),
    minNumberOfHits=cms.int32(0),
    minQuality=cms.string("tight"),
    trackSrc=cms.InputTag("caOfflinePhase2OTTracksSoA"),
    outerTrackerRecHitSrc=cms.InputTag("siPhase2RecHits"),
    outerTrackerRecHitSoAConverterSrc=cms.InputTag("phase2OTRecHitsSoAConverter"),
    useOTExtension=cms.bool(True),
    requireQuadsFromConsecutiveLayers=cms.bool(True),
)

caOfflinePhase2OTIterationSummaries = cms.VPSet(
    cms.PSet(
        name=cms.string("InitialStep"),
        role=cms.string("First high-efficiency quadruplet-like CA pass from an unmasked Phase-2 pixel+OT rec-hit mask."),
        baselineSeeding=cms.string("PixelLayerQuadruplets with layerPairs [0, 1, 2], ptMin about 0.6, originRadius about 0.03."),
        caSketch=cms.string("Run the full Phase2OT CA graph, require at least 4 hits, then mask tight tracks for later CA passes."),
    ),
    cms.PSet(
        name=cms.string("HighPtTripletStep"),
        role=cms.string("Early high-pT recovery pass after InitialStep masking."),
        baselineSeeding=cms.string("Explicit Phase-2 triplet list with ptMin about 0.7 and tighter phi/theta cell cuts."),
        caSketch=cms.string("Use a reduced high-pT CA graph seeded from early pixel and OT-start pairs, require at least 3 hits."),
    ),
    cms.PSet(
        name=cms.string("LowPtQuadStep"),
        role=cms.string("Lower-pT quadruplet-like recovery after high-pT tracks are masked."),
        baselineSeeding=cms.string("PixelLayerQuadruplets with ptMin about 0.35 and looser CA chi2/phi acceptance."),
        caSketch=cms.string("Use the full Phase2OT graph with lower pair pT cuts and larger doublet/tuple capacity formulas."),
    ),
    cms.PSet(
        name=cms.string("LowPtTripletStep"),
        role=cms.string("Reduced low-pT triplet-like recovery after LowPtQuadStep masking."),
        baselineSeeding=cms.string("Reduced explicit Phase-2 triplet list up to the lower forward disks, ptMin about 0.40."),
        caSketch=cms.string("Use a reduced forward+OT CA graph, require at least 3 hits, and preserve the serial mask chain."),
    ),
    cms.PSet(
        name=cms.string("DetachedQuadStep"),
        role=cms.string("Detached high-impact-parameter quadruplet-like recovery as the fifth CA-only pass."),
        baselineSeeding=cms.string("PixelLayerQuadruplets with enlarged originRadius about 0.9 and nSigmaZ about 5."),
        caSketch=cms.string("Reuse the full Phase2OT graph with looser quality cuts; impact-parameter behavior still needs final CA-side tuning."),
    ),
)

caOfflinePhase2OTTask = cms.Task(
    caOfflinePhase2OTInitialStep,
    caOfflinePhase2OTInitialStepMask,
    caOfflinePhase2OTHighPtTripletStep,
    caOfflinePhase2OTHighPtTripletStepMask,
    caOfflinePhase2OTLowPtQuadStep,
    caOfflinePhase2OTLowPtQuadStepMask,
    caOfflinePhase2OTLowPtTripletStep,
    caOfflinePhase2OTLowPtTripletStepMask,
    caOfflinePhase2OTDetachedQuadStep,
    caOfflinePhase2OTTracksSoA,
    caOfflinePhase2OTTracks,
)

caOfflinePhase2OTSequence = cms.Sequence(caOfflinePhase2OTTask)
