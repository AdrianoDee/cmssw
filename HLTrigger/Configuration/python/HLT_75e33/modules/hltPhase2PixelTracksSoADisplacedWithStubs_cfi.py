import FWCore.ParameterSet.Config as cms

# Displaced OT-only iteration: a second CA pass over the hits the prompt iteration left free.
# It mirrors the prompt stub arm's geometry, capped as below, with the triplet and track DNN gates on.
STUB_SIGMA_CAP = 3.0   # cap applied to the two stub-curvature-significance cuts
FISHBONE_CAP = 0.9999  # fishbone merge-threshold knee

displacedItMinPtCut = 0.85

# The 54 CA layers (28 pixel + 6 OT barrel + 20 OT disk) are the prompt stub arm's, verbatim.
from .hltPhase2PixelTracksSoAWithStubs_cfi import layers

# 41 OT-only layer pairs, prompt arm's column layout mapped to geometry.* members.
layerPairsDisplaced = [
    #  0,   1,      2,      3,       4,         5,         6,         7,         8,        9,        10,        11,       12,     13,       14,       15,     16,        17,     18,       19
    #  i,   o,  start,   skip,  phiCut,     minIn,     maxIn,    minOut,    maxOut,    maxDR,     minDZ,     maxDZ,   ptCuts, z0Cuts, stubSigma, geomKappa, innerDCurv, caTheta, caDCA,    floor
    [28, 29,  True, False,   1100,     -1200,      1200,    -10000,     10000,    10000,       -50,        50,  0.74989,    13,       3,       3,   0.1,     0.005,    15,       0],
    [28, 34, False, False,   1600,        40,       130,        20,        40,    10000,    -10000,     10000,  0.79433,    13,       3,       3,   0.1, 0.0095586,    15, 2.000e-04],
    [28, 44, False, False,   2500,      -130,       -40,        20,        40,    10000,    -10000,     10000,  0.79433,    13,       3,       3,   0.1, 0.0095586,    15, 2.000e-04],
    [29, 30,  True, False,   1140,     -1200,      1200,    -10000,     10000,    10000,       -40,        40,  0.74989,    13,       3,     2.8,   0.1,   0.11979,    15,       0],
    [29, 34, False, False,   1700,        80,       130,        30,        60,    10000,    -10000,     10000,  0.79433,    13,       3,       3,   0.1, 0.0095586,    15, 2.000e-04],
    [29, 44, False, False,   2500,      -130,       -80,        30,        60,    10000,    -10000,     10000,  0.59566,    13,       3,       3,   0.1,  0.017985,    15, 2.000e-04],
    [30, 31,  True, False,   1300,    -10000,     10000,    -10000,     10000,    10000,       -30,        30,  0.44668,    18,       3,       3,   0.1,         3,    15,       0],
    [30, 34, False, False,   2000,        99,       120,        50,        80,   12.667,    -10000,     10000,  0.56234,    13,       3,     2.6,   0.1,   0.79792,    15, 2.000e-04],
    [30, 35, False, False,   1640,        79,       108,        50,        80,    10000,    -10000,     10000,  0.79433,    23,       3,       3,   0.1,  0.046416,    15, 2.000e-04],
    [30, 44, False, False,    820,      -120,       -97,        50,        80,   12.667,    -10000,     10000,  0.31623,    14,       3,     2.7,   0.1,    1.0945,    15, 2.000e-04],
    [30, 45, False, False,   1680,      -108,       -79,        50,        80,    10000,    -10000,     10000,  0.79433,    23,       3,       3,   0.1,  0.046416,    15, 2.000e-04],
    [31, 32,  True, False,   1460,    -10000,     10000,    -10000,     10000,    10000,       -30,        30,  0.56234,    27,       3,       3,   0.1,         3,    15,       0],
    [31, 35, False, False,   1660,        86,       116,        60,       110,    10000,    -10000,     10000,  0.70795,    35,       3,       3,   0.1,   0.11979,    15, 2.000e-04],
    [31, 45, False, False,   1660,      -116,       -85,        60,       110,    10000,    -10000,     10000,  0.74989,    35,       3,       3,   0.1,  0.087333,    15, 2.000e-04],
    [32, 33, False, False,   1860,    -10000,     10000,    -10000,     10000,    10000,       -29,        29,  0.59566,    30,       3,       3,   0.1,         3,    15,       0],
    [32, 35, False, False,   1480,        76,       116,        80,       110,    10000,    -10000,     10000,  0.50119,    46,       3,       3,   0.1,   0.22539,    15, 2.000e-04],
    [32, 45, False, False,   1540,      -116,       -95,        80,       110,    10000,    -10000,     10000,  0.53088,    40,       3,       3,   0.1,   0.22539,    15, 2.000e-04],
    [34, 36,  True, False,   2500,        20,       115,        20,       115,       60,        15,        50,  0.74989,    18,       3,       3,   0.1,      0.02,    15, 2.000e-04],
    [34, 37, False, False,   2500,        20,       115,        20,       115,       60,        15,        50,  0.79433,    43,       3,       3,   0.1,  0.063668,    15, 2.000e-04],
    [35, 37, False, False,   2500,        20,       115,        20,       115,       60,        15,        50,  0.63096,    47,       3,       3,   0.1,   0.16432,    15, 2.000e-04],
    [36, 38,  True, False,   2500,        20,       115,        20,       115,       60,        15,        50,  0.56234,    18,       3,       3,   0.1,      0.02,    15, 2.000e-04],
    [36, 39, False, False,   2500,        20,       115,        20,       115,       60,        15,        50,  0.79433,    47,       3,       3,   0.1,  0.063668,    15, 2.000e-04],
    [37, 39, False, False,   2500,        20,       115,        20,       115,       60,        15,        50,  0.59566,    47,       3,     2.7,   0.1,   0.16432,    15, 2.000e-04],
    [38, 40, False, False,   2500,        20,       115,        20,       115,       60,        15,        50,  0.37584,    15,       3,       3,   0.1,   0.16432,    15, 2.000e-04],
    [38, 41, False, False,   2500,        20,       115,        20,       115,       60,        15,        50,  0.74989,    47,       3,       3,   0.1,  0.063668,    15, 2.000e-04],
    [39, 41, False, False,   2500,        20,       115,        20,       115,       60,        15,        50,  0.56234,    45,       3,       3,   0.1,   0.22539,    15, 2.000e-04],
    [40, 42, False, False,   2500,        20,       115,        20,       115,       60,        15,        50,  0.31623,    18,       3,       3,   0.1,   0.79792,    15, 2.000e-04],
    [40, 43, False, False,   2500,        20,       115,        20,       115,       60,        15,        50,  0.70795,    48,       3,       3,   0.1,  0.063668,    15, 2.000e-04],
    [41, 43, False, False,   2500,        20,       115,        20,       115,   18.333,        15,        50,  0.53088,    47,       3,       3,   0.1,   0.58171,    15, 2.000e-04],
    [44, 46,  True, False,   2500,        20,       115,        20,       115,       60,       -50,       -15,  0.74989,    18,       3,       3,   0.1,      0.02,    15, 2.000e-04],
    [44, 47, False, False,   2500,        20,       115,        20,       115,       60,       -50,       -15,  0.79433,    42,       3,       3,   0.1,  0.063668,    15, 2.000e-04],
    [45, 47, False, False,   2500,        20,       115,        20,       115,       60,       -50,       -15,  0.59566,    47,       3,       3,   0.1,   0.22539,    15, 2.000e-04],
    [46, 48,  True, False,   2500,        20,       115,        20,       115,       60,       -50,       -15,  0.56234,    18,       3,       3,   0.1,      0.02,    15, 2.000e-04],
    [46, 49, False, False,   2500,        20,       115,        20,       115,       60,       -50,       -15,  0.74989,    47,       3,       3,   0.1,  0.087333,    15, 2.000e-04],
    [47, 49, False, False,   2500,        20,       115,        20,       115,       60,       -50,       -15,  0.56234,    47,       3,       3,   0.1,   0.16432,    15, 2.000e-04],
    [48, 50, False, False,   2500,        20,       115,        20,       115,       60,       -50,       -15,  0.35481,    16,       3,       3,   0.1,  0.087333,    15, 2.000e-04],
    [48, 51, False, False,   2500,        20,       115,        20,       115,       60,       -50,       -15,  0.74989,    46,       3,       3,   0.1,  0.063668,    15, 2.000e-04],
    [49, 51, False, False,   2500,        20,       115,        20,       115,       60,       -50,       -15,  0.56234,    45,       3,       3,   0.1,   0.58171,    15, 2.000e-04],
    [50, 52, False, False,   2500,        20,       115,        20,       115,       60,       -50,       -15,  0.29854,    17,       3,       3,   0.1,   0.79792,    15, 2.000e-04],
    [50, 53, False, False,   2500,        20,       115,        20,       115,       60,       -50,       -15,  0.70795,    47,       3,       3,   0.1,  0.063668,    15, 2.000e-04],
    [51, 53, False, False,   1220,        20,       115,        20,       115,   18.333,       -50,       -15,  0.50119,    47,       3,       3,   0.1,   0.58171,    15, 2.000e-04],
]

hltPhase2PixelTracksSoADisplacedWithStubs = cms.EDProducer('CAHitNtupletAlpakaPhase2OTStubs@alpaka',
    pixelRecHitSrc = cms.InputTag('hltPhase2PixelRecHitsStubsMerger'),
    hitMask = cms.InputTag('hltPhase2PixelTrackHighPtMasking'),
    iterationName = cms.string('displaced'),
    # In-kernel triplet and track classifiers; the working points are baked into the weight headers.
    useTripletDNN = cms.bool(True),
    useTrackDNN = cms.bool(True),
    # Generic cut scalars that differ from the Phase2OTStubs defaults; everything else stays at its default.
    hardCurvCut = cms.double(0.0320),
    minYsizeB1 = cms.int32(15),
    minYsizeB2 = cms.int32(14),
    maxDYsize12 = cms.int32(15),
    maxDYsize = cms.int32(20),
    # Max |phi residual| per connection during the chain extension [rad].
    maxPhiResid = cms.double(0.0224),
    # Container sizes: same policy as the prompt arm (see hltPhase2PixelTracksSoAWithStubs_cfi.py).
    minNumberOfDoublets = cms.uint32(22528),  # quiet-event floors; this arm runs on masked hits, so the
    minNumberOfTuples = cms.uint32(8448),     # formulas rule at collision occupancy and both are inert
    avgHitsPerTrack = cms.double(8.0),        # above the average, so the hit container never binds first
    avgCellsPerCell = cms.double(0.1314),
    avgTracksPerCell = cms.double(0.1790),
    # avgCellsPerHit is not set: hit->cell storage holds one entry per cell and is sized from the cell bound.
    maxNumberOfDoublets = cms.string("max(5.444689e-06*x*x-2.004765e-01*x,0)"),
    # Flat tuple cap: on masked hits the tuple demand does not scale with the total hit count.
    maxNumberOfTuples = cms.string("15872"),
    # No OT rechit or OT stub input: the OT-hit extension and the final refit run once, in the merger.
    trackQualityCuts = cms.PSet(
        minPt = cms.double(displacedItMinPtCut + 0.05),
        # Ntuplet-wide stub-curvature consistency: max reduced-chi2 of the stub curvatures on a
        # track around their weighted mean; demotes combinatorial chains below tight (-1 = disabled).
        maxNtupletStubChi2 = cms.double(10.0),
    ),
    # CA graph and cuts; caDCACuts / caThetaCuts are replaced on every pair by the per-pair overrides below.
    geometry = cms.PSet(
        maxDCurv    = cms.vdouble([l[3] for l in layers]),
        floorDCurv  = cms.vdouble([l[4] for l in layers]),
        # Fishbone knee: min() so that no layer is loosened with respect to the prompt arm.
        fishboneCuts = cms.vdouble([min(l[5], FISHBONE_CAP) for l in layers]),
        pairGraph     = cms.vuint32(sum([[lp[0], lp[1]] for lp in layerPairsDisplaced], [])),
        startingPairs = cms.vuint32([i for i, lp in enumerate(layerPairsDisplaced) if lp[2]]),
        skipsLayers   = cms.vuint32([int(lp[3]) for lp in layerPairsDisplaced]),
        phiCuts  = cms.vint32( [lp[ 4] for lp in layerPairsDisplaced]),
        minInner = cms.vdouble([lp[ 5] for lp in layerPairsDisplaced]),
        maxInner = cms.vdouble([lp[ 6] for lp in layerPairsDisplaced]),
        minOuter = cms.vdouble([lp[ 7] for lp in layerPairsDisplaced]),
        maxOuter = cms.vdouble([lp[ 8] for lp in layerPairsDisplaced]),
        maxDR    = cms.vdouble([lp[ 9] for lp in layerPairsDisplaced]),
        minDZ    = cms.vdouble([lp[10] for lp in layerPairsDisplaced]),
        maxDZ    = cms.vdouble([lp[11] for lp in layerPairsDisplaced]),
        ptCuts   = cms.vdouble([lp[12] for lp in layerPairsDisplaced]),
        # OT-stub extension: per-pair overrides and stub-only cuts (columns 14/15 already capped).
        cellZ0CutPerPair         = cms.vdouble([lp[13] for lp in layerPairsDisplaced]),
        maxStubCurvSigma         = cms.vdouble([lp[14] for lp in layerPairsDisplaced]),
        maxStubGeomCurvSigma     = cms.vdouble([lp[15] for lp in layerPairsDisplaced]),
        maxStubInnerDoubletDCurv = cms.vdouble([lp[16] for lp in layerPairsDisplaced]),
        caThetaCutsPerPair       = cms.vdouble([lp[17] for lp in layerPairsDisplaced]),
        caDCACutsPerPair         = cms.vdouble([lp[18] for lp in layerPairsDisplaced]),
        floorDCACuts             = cms.vdouble([lp[19] for lp in layerPairsDisplaced]),
    ),
    mightGet = cms.optional.untracked.vstring,
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)
