import FWCore.ParameterSet.Config as cms

# list of layers to exclude from the CA (empty list doesn't exclude)
layersToExclude = []

# layers for Phase2 with OT stubs
# CA layers: 28 pixel + 6 OT barrel + 10 OT disks = 44 total
# Inside-out ordering (required by CA algorithm):
# Layers 28-33: OT barrel layers 1-6
# Layers 34-38: OT endcap backward disks 1-5
# Layers 39-43: OT endcap forward disks 1-5
layers = [
    #     0,        1,     2,       3,         4,           5
    # index, isBarrel, caDCA, caTheta, geomKappa, caPhiMiddle
    # Pixel barrel layers
    [     0,     True,  0.15,   0.002,      -1.0,       -1.0],
    [     1,     True,  0.25,   0.002,      -1.0,       -1.0],
    [     2,     True,  0.20,   0.002,      -1.0,       -1.0],
    [     3,     True,  0.20,   0.002,      -1.0,       -1.0],
    # Pixel endcap layers (forward)
    [     4,    False,  0.25,   0.003,      -1.0,       -1.0],
    [     5,    False,  0.25,   0.003,      -1.0,       -1.0],
    [     6,    False,  0.25,   0.003,      -1.0,       -1.0],
    [     7,    False,  0.25,   0.003,      -1.0,       -1.0],
    [     8,    False,  0.25,   0.003,      -1.0,       -1.0],
    [     9,    False,  0.25,   0.003,      -1.0,       -1.0],
    [    10,    False,  0.25,   0.003,      -1.0,       -1.0],
    [    11,    False,  0.25,   0.003,      -1.0,       -1.0],
    [    12,    False,  0.25,   0.003,      -1.0,       -1.0],
    [    13,    False,  0.25,   0.003,      -1.0,       -1.0],
    [    14,    False,  0.25,   0.003,      -1.0,       -1.0],
    [    15,    False,  0.25,   0.003,      -1.0,       -1.0],
    # Pixel endcap layers (backward)
    [    16,    False,  0.25,   0.003,      -1.0,       -1.0],
    [    17,    False,  0.25,   0.003,      -1.0,       -1.0],
    [    18,    False,  0.25,   0.003,      -1.0,       -1.0],
    [    19,    False,  0.25,   0.003,      -1.0,       -1.0],
    [    20,    False,  0.25,   0.003,      -1.0,       -1.0],
    [    21,    False,  0.25,   0.003,      -1.0,       -1.0],
    [    22,    False,  0.25,   0.003,      -1.0,       -1.0],
    [    23,    False,  0.25,   0.003,      -1.0,       -1.0],
    [    24,    False,  0.25,   0.003,      -1.0,       -1.0],
    [    25,    False,  0.25,   0.003,      -1.0,       -1.0],
    [    26,    False,  0.25,   0.003,      -1.0,       -1.0],
    [    27,    False,  0.25,   0.003,      -1.0,       -1.0],
    # OT barrel layers 1-6 (stubs)
    # dcaCut increased for layers 28-32 to handle barrel-to-endcap triplet transitions
    # (the fitted circle DCA is larger when transitioning from barrel to endcap geometry)
    # geomKappaSigmaCut: geometric-vs-stub kappa significance (placeholder 5.0)
    [    28,     True,  0.30,   0.003,       5.0,       -1.0],
    [    29,     True,  0.30,   0.003,       5.0,       -1.0],
    [    30,     True,  0.30,   0.003,       5.0,       -1.0],
    [    31,     True,  0.30,   0.007,       5.0,       -1.0],
    [    32,     True,  0.30,   0.007,       5.0,       -1.0],
    [    33,     True,  0.30,   0.007,       5.0,       -1.0],
    # OT endcap backward disks 1-5 (stubs)
    # dcaCut increased to 0.15 to handle endcap triplet geometries
    # caPhiMiddle: phi residual at middle hit [rad] (0.1 = initial value, to be tuned)
    [    34,    False,  0.30,   0.007,       5.0,        0.1],
    [    35,    False,  0.30,   0.008,       5.0,        0.1],
    [    36,    False,  0.30,   0.007,       5.0,        0.1],
    [    37,    False,  0.30,   0.007,       5.0,        0.1],
    [    38,    False,  0.30,   0.007,       5.0,        0.1],
    # OT endcap forward disks 1-5 (stubs)
    # dcaCut increased to 0.15 to handle endcap triplet geometries
    # caPhiMiddle: phi residual at middle hit [rad] (0.1 = initial value, to be tuned)
    [    39,    False,  0.30,   0.007,       5.0,        0.1],
    [    40,    False,  0.30,   0.008,       5.0,        0.1],
    [    41,    False,  0.30,   0.007,       5.0,        0.1],
    [    42,    False,  0.30,   0.007,       5.0,        0.1],
    [    43,    False,  0.30,   0.007,       5.0,        0.1],
]

# layerPairs for doublet building including pair-specific cut values
# Extended to include connections to/from OT stub layers
#
# stubSigmaCut (column 12): Stub-stub pairwise compatibility cut in units of sigma.
#   Negative value = disabled. All stub-stub pairs use unified kappa-corrected significance:
#   kappa = dPhiDr / sqrt(1 + r^2 * dPhiDr^2). This works for all module types (flat barrel,
#   tilted barrel, endcap, and mixed transitions) because dr_effective normalization makes
#   dPhiDr a universal curvature proxy. Pixel hits and PHitOnly stubs are automatically skipped.
layerPairs = [
    #  0,  1,     2,      3,      4,      5,       6,       7,     8,      9,     10,     11,      12
    #  i,  o, start, phiCut,  minIn,  maxIn,  minOut,  maxOut, maxDR,  minDZ,  maxDZ, ptCuts, stubSigma
    # Pixel-only connections (same as Phase2OT)
    [  0,  1,  True,    350,  -17.0,   17.0,  -10000,   10000,   5.0,  -16.0,   16.0,  0.85,  -1.0],
    [  0,  2,  True,    600,  -14.0,   14.0,  -10000,   10000,  10.0,  -16.0,   16.0,  0.85,  -1.0],
    [  0,  4,  True,    450,    4.0,  10000,       0,    10.0,   8.0,    0.0,   25.0,  0.85,  -1.0],
    [  0,  5,  True,    522,    7.0,  10000,       0,   10000,   5.0,    0.0,   25.0,  0.85,  -1.0],
    [  0, 16,  True,    450, -10000,   -4.0,       0,    10.0,   8.0,  -25.0,    0.0,  0.85,  -1.0],
    [  0, 17,  True,    522, -10000,   -7.0,       0,   10000,   5.0,  -25.0,    0.0,  0.85,  -1.0],
    [  1,  2,  True,    400,  -17.0,   17.0,  -10000,   10000,   7.0,  -13.0,   13.0,  0.85,  -1.0],
    [  1,  3, False,    650,  -15.0,   15.0,  -10000,   10000,  10.0,  -15.0,   15.0,  0.85,  -1.0],
    [  1,  4,  True,    500,    6.0,  10000,     6.5,   10000,   8.0,    0.0,   19.0,  0.85,  -1.0],
    [  1,  5, False,    730,    9.0,  10000,     6.5,   10000,  10.0,    0.0,   21.0,  0.85,  -1.0],
    [  1, 16,  True,    500, -10000,   -6.0,     6.5,   10000,   8.0,  -19.0,    0.0,  0.85,  -1.0],
    [  1, 17, False,    730, -10000,   -9.0,     6.5,   10000,  10.0,  -21.0,    0.0,  0.85,  -1.0],
    [  2,  3,  True,    350,  -18.0,   18.0,  -10000,   10000,   7.0,   -9.0,    9.0,  0.85,  -1.0],
    [  2,  4, False,    400,   11.0,  10000,    11.7,   10000,   7.0,    0.0,   13.0,  0.85,  -1.0],
    [  2, 16, False,    400, -10000,  -11.0,    11.7,   10000,   7.0,  -13.0,    0.0,  0.85,  -1.0],
    # Pixel barrel L3 (CA 2) to OT barrel L1 (CA 28) - from HLT CA extension
    [  2, 28, False,   1200,    -10,     10,   -30.0,    30.0,  16.0,  -15.0,   15.0,   2.0,   0.0],  # central
    [  2, 28, False,   1200,    -20,    -10,   -50.0,   -25.0, 10000,  -35.0,  -10.0,  0.85,   0.0],  # backward
    [  2, 28, False,   1200,     10,     20,    25.0,    50.0, 10000,   10.0,   35.0,  0.85,   0.0],  # forward
    # Pixel barrel L4 (CA 3) to OT barrel L1 (CA 28) - from HLT CA extension
    [  3, 28, False,   2000,    -20,     20,   -45.0,    45.0, 10000,  -22.0,   22.0,  0.85,   0.0],
    # Pixel forward endcap disks to OT barrel L1 (CA 28) - from HLT CA extension
    [  4, 28, False,   2400,   11.6,  10000,    30.0,    57.5,  16.0,    5.0,  100.0,  0.85,   0.0],
    [  5, 28, False,   2100,   11.6,  10000,    40.0,    80.0,  16.0,  -10.0,   90.0,  0.85,   0.0],
    [  6, 28, False,   2000,   11.6,  10000,    55.0,    95.0,  16.0,    5.0,   80.0,  0.85,   0.0],
    [  7, 28, False,   2000,   11.8,  10000,    70.0,   110.0,  16.0,   15.0,   70.0,  0.85,   0.0],
    [  8, 28, False,   2000,      0,  10000,    80.0,   10000,  14.0,   25.0,   70.0,  0.85,   0.0],
    # Pixel backward endcap disks to OT barrel L1 (CA 28) - from HLT CA extension
    [ 16, 28, False,   2400,   11.6,  10000,   -57.5,   -30.0,  16.0, -100.0,   -5.0,  0.85,   0.0],
    [ 17, 28, False,   2100,   11.6,  10000,   -70.0,   -40.0,  16.0,  -90.0,   10.0,  0.85,   0.0],
    [ 18, 28, False,   2000,   11.6,  10000,   -95.0,   -55.0,  16.0,  -80.0,   -5.0,  0.85,   0.0],
    [ 19, 28, False,   2000,   11.8,  10000,  -110.0,   -70.0,  16.0,  -70.0,  -15.0,  0.85,   0.0],
    [ 20, 28, False,   2000,      0,  10000,  -10000,   -80.0,  14.0,  -70.0,  -25.0,  0.85,   0.0],
    [  4,  5,  True,    300,      0,   14.0,     3.5,   10000,   4.5, -10000,  10000,  0.85,  -1.0],
    [  4,  6, False,    522,      0,   14.0,     3.5,   10000,   9.0, -10000,  10000,  0.85,  -1.0],
    [  5,  6,  True,    300,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85,  -1.0],
    [  5,  7, False,    522,      0,   13.0,     3.5,   10000,   9.0, -10000,  10000,  0.85,  -1.0],
    [  6,  7,  True,    250,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85,  -1.0],
    [  6,  8, False,    522,      0,   13.0,     3.5,   10000,   9.0, -10000,  10000,  0.85,  -1.0],
    [  7,  8,  True,    250,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85,  -1.0],
    [  7,  9, False,    522,      0,   13.0,     3.5,   10000,   8.0, -10000,  10000,  0.85,  -1.0],
    [  8,  9,  True,    250,      0,   13.0,     3.5,   10000,   4.0, -10000,  10000,  0.85,  -1.0],
    [  8, 10, False,    522,      0,   13.0,     3.5,   10000,   8.0, -10000,  10000,  0.85,  -1.0],
    [  9, 10,  True,    300,      0,   13.0,     4.0,   10000,   4.5, -10000,  10000,  0.85,  -1.0],
    [  9, 11, False,    522,      0,   13.0,     4.0,   10000,   8.0, -10000,  10000,  0.85,  -1.0],
    [ 10, 11,  True,    240,      0,   13.0,     3.5,   10000,   4.0, -10000,  10000,  0.85,  -1.0],
    [ 10, 12, False,    650,   12.5,   16.5,    20.0,   10000,  10.0, -10000,  10000,  0.85,  -1.0],
    [ 11, 12, False,    300,      0,   16.5,     6.0,    21.0,   5.0, -10000,  10000,  0.85,  -1.0],
    [ 11, 13, False,    200,      0,    6.0,       0,     7.5,   3.0, -10000,  10000,  0.85,  -1.0],
    [ 11, 14, False,    220,      0,    4.6,       0,     7.5,   3.0, -10000,  10000,  0.85,  -1.0],
    [ 11, 15, False,    250,      0,    6.0,       0,   10000,   4.0, -10000,  10000,  0.85,  -1.0],
    [ 12, 13, False,    250,      0,   22.5,     7.0,   10000,   4.0, -10000,  10000,  0.85,  -1.0],
    [ 13, 14, False,    250,      0,   22.5,     7.0,   10000,   4.0, -10000,  10000,  0.85,  -1.0],
    [ 14, 15, False,    250,      0,   22.5,     7.0,   10000,   3.5, -10000,  10000,  0.85,  -1.0],
    # Pixel backward endcap connections
    [ 16, 17,  True,    300,      0,   14.0,     3.5,   10000,   4.5, -10000,  10000,  0.85,  -1.0],
    [ 16, 18, False,    522,      0,   14.0,     3.5,   10000,   9.0, -10000,  10000,  0.85,  -1.0],
    [ 17, 18,  True,    300,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85,  -1.0],
    [ 17, 19, False,    522,      0,   13.0,     3.5,   10000,   9.0, -10000,  10000,  0.85,  -1.0],
    [ 18, 19,  True,    250,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85,  -1.0],
    [ 18, 20, False,    522,      0,   13.0,     3.5,   10000,   9.0, -10000,  10000,  0.85,  -1.0],
    [ 19, 20,  True,    250,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85,  -1.0],
    [ 19, 21, False,    522,      0,   13.0,     3.5,   10000,   8.0, -10000,  10000,  0.85,  -1.0],
    [ 20, 21,  True,    250,      0,   13.0,     3.5,   10000,   4.0, -10000,  10000,  0.85,  -1.0],
    [ 20, 22, False,    522,      0,   13.0,     3.5,   10000,   8.0, -10000,  10000,  0.85,  -1.0],
    [ 21, 22,  True,    300,      0,   13.0,     4.0,   10000,   4.5, -10000,  10000,  0.85,  -1.0],
    [ 21, 23, False,    522,      0,   13.0,     4.0,   10000,   8.0, -10000,  10000,  0.85,  -1.0],
    [ 22, 23,  True,    240,      0,   13.0,     3.5,   10000,   4.0, -10000,  10000,  0.85,  -1.0],
    [ 22, 24, False,    650,   12.5,   16.5,    20.0,   10000,  10.0, -10000,  10000,  0.85,  -1.0],
    [ 23, 24, False,    300,      0,   16.5,     6.0,    21.0,   5.0, -10000,  10000,  0.85,  -1.0],
    [ 23, 25, False,    200,      0,    6.0,       0,     7.5,   3.0, -10000,  10000,  0.85,  -1.0],
    [ 23, 26, False,    220,      0,    4.6,       0,     7.5,   3.0, -10000,  10000,  0.85,  -1.0],
    [ 23, 27, False,    250,      0,    6.0,       0,   10000,   4.0, -10000,  10000,  0.85,  -1.0],
    [ 24, 25, False,    250,      0,   22.5,     7.0,   10000,   4.0, -10000,  10000,  0.85,  -1.0],
    [ 25, 26, False,    250,      0,   22.5,     7.0,   10000,   4.0, -10000,  10000,  0.85,  -1.0],
    [ 26, 27, False,    250,      0,   22.5,     7.0,   10000,   3.5, -10000,  10000,  0.85,  -1.0],
    # OT barrel to barrel connections (layers 28-33) - from HLT CA extension for 28-29, 29-30
    # stubSigmaCut=5.0: kappa-corrected pairwise significance for flat-flat barrel stubs
    [ 28, 29, False,   2300,  -1200,   1200,  -10000,   10000, 10000,  -50.0,   50.0,  0.85,   5.0],
    [ 29, 30, False,   2300,  -1200,   1200,  -10000,   10000, 10000,  -40.0,   40.0,  0.85,   5.0],
    [ 30, 31, False,   1250, -10000,  10000,  -10000,   10000, 10000,  -30.0,   30.0,  0.85,   5.0],
    [ 31, 32, False,   2000, -10000,  10000,  -10000,   10000, 10000,  -30.0,   30.0,  0.85,   5.0],
    [ 32, 33, False,    850, -10000,  10000,  -10000,   10000, 10000,  -25.0,   25.0,  0.85,   5.0],
    # OT barrel layers to first backward disk (34) - kappa-corrected significance
    [ 28, 34, False,   1600,  -1300,    -80,      20,      40, 10000,  -10000,  10000,  0.85,   5.0],
    [ 29, 34, False,   1700,  -1300,    -80,      30,      60, 10000,  -10000,  10000,  0.85,   5.0],
    [ 30, 34, False,   2000,  -1300,    -80,      50,      80, 10000,  -10000,  10000,  0.85,   5.0],
    [ 31, 34, False,   2000,  -1300,    -80,      60,     110, 10000,  -10000,  10000,  0.85,   5.0],
    [ 32, 34, False,   3300,  -1300,    -80,      80,     110, 10000,  -10000,  10000,  0.85,   5.0],
    # OT barrel layers to first forward disk (39) - kappa-corrected significance
    [ 28, 39, False,   1600,     80,  10000,      20,      40, 10000,  -10000,  10000,  0.85,   5.0],
    [ 29, 39, False,   1700,     80,  10000,      30,      60, 10000,  -10000,  10000,  0.85,   5.0],
    [ 30, 39, False,   2000,     80,  10000,      50,      80, 10000,  -10000,  10000,  0.85,   5.0],
    [ 31, 39, False,   2000,     80,  10000,      60,     110, 10000,  -10000,  10000,  0.85,   5.0],
    [ 32, 39, False,   3300,     80,  10000,      80,     110, 10000,  -10000,  10000,  0.85,   5.0],
    # Backward disk to disk consecutive connections (layers 34-38)
    # For backward disks: dz < 0 (moving to more negative z)
    # r-range: 20-115 cm (~15% margin over actual 23-110 cm disk extent)
    # maxDR=60: covers full disk r-extent with margin
    # Signed dz: enforces correct z-direction AND constrains to actual disk separations
    [ 34, 35, False,   1500,     20,    115,      20,     115,  60.0,  -35.0,  -10.0,  0.85,   6.5],
    [ 35, 36, False,   1500,     20,    115,      20,     115,  60.0,  -55.0,  -20.0,  0.85,   5.6],
    [ 36, 37, False,   1500,     20,    115,      20,     115,  60.0,  -55.0,  -20.0,  0.85,   5.4],
    [ 37, 38, False,   1500,     20,    115,      20,     115,  60.0,  -70.0,  -25.0,  0.85,   6.0],
    # Forward disk to disk consecutive connections (layers 39-43)
    # For forward disks: dz > 0 (moving to more positive z)
    [ 39, 40, False,   1500,     20,    115,      20,     115,  60.0,   10.0,   35.0,  0.85,   6.5],
    [ 40, 41, False,   1500,     20,    115,      20,     115,  60.0,   20.0,   55.0,  0.85,   5.6],
    [ 41, 42, False,   1500,     20,    115,      20,     115,  60.0,   20.0,   55.0,  0.85,   5.4],
    [ 42, 43, False,   1500,     20,    115,      20,     115,  60.0,   25.0,   70.0,  0.85,   6.0],
]

# find the layerPairs that contain a layer that is excluded
excludeLayerPair = [any([(lp[0] == l) or (lp[1] == l) for l in layersToExclude]) for lp in layerPairs]

# exclude those layerPairs
layerPairsStubs = []
for i, lp in enumerate(layerPairs):
    if not excludeLayerPair[i]:
        layerPairsStubs.append(lp)

# get startingPairs for Ntuplet building
startingPairsStubs = []
for i, lp in enumerate(layerPairsStubs):
    if lp[2]:
        startingPairsStubs.append(i)

hltPhase2PixelTracksSoAWithStubs = cms.EDProducer('CAHitNtupletAlpakaPhase2OTStubs@alpaka',
    pixelRecHitSrc = cms.InputTag('hltPhase2PixelRecHitsStubsMerger'),
    otRecHitsSrc = cms.InputTag('hltPixelSeedingOTRecHitsSoA'),
    stubsSrc = cms.InputTag('hltOTStubProducer'),

    # Tracking parameters
    ptmin = cms.double(0.9),
    hardCurvCut = cms.double(0.01425),
    earlyFishbone = cms.bool(True),
    lateFishbone = cms.bool(False),
    fillStatistics = cms.bool(True),  # Enabled to diagnose tuple overflow
    minHitsPerNtuplet = cms.uint32(4),  # Require at least 1 OT hit for barrel testing
    maxNumberOfDoublets = cms.string(str(8*1024*1024)),   # 8.4M (~13% margin over 7.4M peak from 1k ttbar PU200)
    maxNumberOfTuples = cms.string(str(160*1024)),        # 164K (~12% margin over 146K peak from 1k ttbar PU200)
    cellZ0Cut = cms.double(15.0),  # Half the beam spot width - same as Phase2OT

    # Pixel cluster cuts
    dzdrFact = cms.double(15.2),
    minYsizeB1 = cms.int32(20),
    minYsizeB2 = cms.int32(18),
    maxDYsize12 = cms.int32(12),
    maxDYsize = cms.int32(10),
    maxDYPred = cms.int32(24),

    # Container size parameters
    avgHitsPerTrack = cms.double(10.0),   # pixel+OT tracks avg ~7 hits, max ~14
    avgCellsPerHit = cms.double(23),     # ~12% margin over peak ratio 20.6 (from 1k ttbar PU200)
    avgCellsPerCell = cms.double(0.3),   # ~32% margin over peak ratio 0.23
    avgTracksPerCell = cms.double(0.2),  # ~46% margin over peak ratio 0.14
    minHitsForSharingCut = cms.uint32(10),

    # Flags
    fitNas4 = cms.bool(False),
    useRiemannFit = cms.bool(False),
    doSharedHitCut = cms.bool(True),
    dupPassThrough = cms.bool(False),
    useSimpleTripletCleaner = cms.bool(True),

    # Reachability filter (kills L28 cells whose neighbor chains don't reach far enough)
    reachTargetLayer = cms.uint32(28),  # OT barrel L1
    reachMinHops = cms.uint32(0),       # must reach 2 distinct outer layers

    # Orphan chain recovery (disabled by default)
    doOrphanRecovery = cms.bool(False),
    minHitsOrphanNtuplet = cms.uint32(5),  # require 5+ hits to suppress fakes

    # Chain kappa consistency: reject chain extensions where adjacent connection
    # kappas differ by more than this threshold [cm^-1]. Negative = disabled.
    chainKappaCut = cms.double(0.001),  # ~10x stub measurement resolution

    # Track quality cuts (Phase2-specific)
    trackQualityCuts = cms.PSet(
        maxChi2 = cms.double(15.0),
        maxChi2TripletsOrQuadruplets = cms.double(1.0),
        maxChi2Quintuplets = cms.double(3.0),
        minPt = cms.double(0.9),
        maxTip = cms.double(0.3),
        maxZip = cms.double(15.0),
    ),

    # Geometry parameters extracted from layers and layerPairs tables
    geometry = cms.PSet(
        caDCACuts   = cms.vdouble([l[2] for l in layers]),
        caThetaCuts = cms.vdouble([l[3] for l in layers]),
        geomKappaSigmaCuts = cms.vdouble([l[4] for l in layers]),
        caPhiMiddleCuts = cms.vdouble([l[5] for l in layers]),
        startingPairs = cms.vuint32(startingPairsStubs),
        pairGraph = cms.vuint32(sum([[lp[0], lp[1]] for lp in layerPairsStubs], [])),
        phiCuts   = cms.vint32( [lp[ 3] for lp in layerPairsStubs]),
        minInner  = cms.vdouble([lp[ 4] for lp in layerPairsStubs]),
        maxInner  = cms.vdouble([lp[ 5] for lp in layerPairsStubs]),
        minOuter  = cms.vdouble([lp[ 6] for lp in layerPairsStubs]),
        maxOuter  = cms.vdouble([lp[ 7] for lp in layerPairsStubs]),
        maxDR     = cms.vdouble([lp[ 8] for lp in layerPairsStubs]),
        minDZ     = cms.vdouble([lp[ 9] for lp in layerPairsStubs]),
        maxDZ     = cms.vdouble([lp[10] for lp in layerPairsStubs]),
        ptCuts    = cms.vdouble([lp[11] for lp in layerPairsStubs]),
        stubSigmaCuts = cms.vdouble([lp[12] for lp in layerPairsStubs]),
    ),

    mightGet = cms.optional.untracked.vstring,
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)
