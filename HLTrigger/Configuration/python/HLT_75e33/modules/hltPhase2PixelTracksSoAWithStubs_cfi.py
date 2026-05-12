import FWCore.ParameterSet.Config as cms

# list of layers to exclude from the CA (empty list doesn't exclude)
layersToExclude = [35, 37, 39, 41, 43, 45, 47, 49, 51, 53]

# layers for Phase2 with OT stubs
# CA layers: 28 pixel + 6 OT barrel + 10 OT disks = 44 total
# Inside-out ordering (required by CA algorithm):
# Layers 28-33: OT barrel layers 1-6
# Layers 34-38: OT endcap backward disks 1-5
# Layers 39-43: OT endcap forward disks 1-5
# Triplets cuts
# - caTheta     cut on layer i is applied in the combination i -> i+1 -> i+2 (innermost)
# - caDCA       cut on layer i is applied in the combination i -> i+1 -> i+2 (innermost)
# - geomKappa   cut on layer i is applied in the combination i-1 -> i -> i+1 (middle)
# - caPhiMiddle cut on layer i is applied in the combination i-1 -> i -> i+1 (middle)
layers = [
    #     0,        1,     2,         3,        4,        5
    # index, isBarrel, startR,    DCurv,   DCurv0,  fishCut
    # Pixel barrel layers
    [     0,     True,   99.0,     99.0,     99.0, 0.99999],
    [     1,     True,   99.0,     99.0,     99.0, 0.99999],
    [     2,     True,   99.0,     99.0,     99.0, 0.99999],
    [     3,     True,   99.0, 4.90e-02, 8.54e-04, 0.99999],
    # Pixel endcap layers (forward)
    [     4,    False,   99.0, 6.68e-02, 9.13e-04, 0.99999],
    [     5,    False,   99.0, 8.73e-02, 1.67e-03, 0.99999],
    [     6,    False,   99.0, 1.02e-01, 3.65e-03, 0.99999],
    [     7,    False,   99.0, 1.06e-01, 4.16e-03, 0.99999],
    [     8,    False,   99.0, 5.24e-02, 4.87e-03, 0.999999],
    [     9,    False,   99.0, 1.18e-01, 4.04e-03, 0.999999],
    [    10,    False,   99.0, 1.12e-01, 4.01e-03, 0.999999],
    [    11,    False,   99.0, 8.79e-02, 4.15e-03, 0.999999],
    [    12,    False,   99.0, 8.79e-02, 4.15e-03, 0.999999],
    [    13,    False,   99.0, 1.05e-01, 4.33e-03, 0.999999],
    [    14,    False,   99.0, 1.05e-01, 4.33e-03, 0.999999],
    [    15,    False,   99.0, 1.05e-01, 4.33e-03, 0.999999],
    # Pixel endcap layers (backward)
    [    16,    False,   99.0, 6.68e-02, 9.13e-04, 0.99999],
    [    17,    False,   99.0, 8.73e-02, 1.67e-03, 0.99999],
    [    18,    False,   99.0, 1.02e-01, 3.65e-03, 0.99999],
    [    19,    False,   99.0, 1.06e-01, 4.16e-03, 0.99999],
    [    20,    False,   99.0, 5.24e-02, 4.87e-03, 0.999999],
    [    21,    False,   99.0, 1.18e-01, 4.04e-03, 0.999999],
    [    22,    False,   99.0, 1.12e-01, 4.01e-03, 0.999999],
    [    23,    False,   99.0, 8.79e-02, 4.15e-03, 0.999999],
    [    24,    False,   99.0, 8.79e-02, 4.15e-03, 0.999999],
    [    25,    False,   99.0, 1.05e-01, 4.33e-03, 0.999999],
    [    26,    False,   99.0, 1.05e-01, 4.33e-03, 0.999999],
    [    27,    False,   99.0, 1.05e-01, 4.33e-03, 0.999999],
    # OT barrel layers 1-6 (stubs)
    # dcaCut increased for layers 28-32 to handle barrel-to-endcap triplet transitions
    # (the fitted circle DCA is larger when transitioning from barrel to endcap geometry)
    # geomKappaSigmaCut: geometric-vs-stub kappa significance (placeholder 5.0)
    [    28,     True,   99.0, 8.16e-02, 8.32e-04, 0.99999],
    [    29,     True,   99.0, 4.82e-02, 4.27e-04, 0.99999],
    [    30,     True,   99.0, 3.79e-02, 2.76e-04, 0.99999],
    [    31,     True,   99.0,     99.0,     99.0, 0.99999],
    [    32,     True,   99.0,     99.0,     99.0, 0.99999],
    [    33,     True,   99.0,     99.0,     99.0, 0.99999],
    # OT endcap backward disks 1-5 (stubs)
    # caTheta (col 3): PS-only cut (0.03). caTheta1SS (col 6): 1 SS stub cut.
    # caTheta2SS (col 7): 2+ SS stubs, nearly disabled (99.0 = rely on other cuts).
    [    34,    False,   99.0,     99.0,      99.0, 0.99999],
    [    35,    False,   99.0,     99.0,      99.0, 0.99999],
    [    36,    False,   99.0,     99.0,      99.0, 0.99999],
    [    37,    False,   99.0,     99.0,      99.0, 0.99999],
    [    38,    False,   99.0,     99.0,      99.0, 0.99999],
    [    39,    False,   99.0,     99.0,      99.0, 0.99999],
    [    40,    False,   99.0,     99.0,      99.0, 0.99999],
    [    41,    False,   99.0,     99.0,      99.0, 0.99999],
    [    42,    False,   99.0,     99.0,      99.0, 0.99999],
    [    43,    False,   99.0,     99.0,      99.0, 0.99999],
    # OT endcap forward disks 1-5 (stubs)
    # caTheta (col 3): PS-only cut (0.03). caTheta1SS (col 6): 1 SS stub cut.
    # caTheta2SS (col 7): 2+ SS stubs, nearly disabled (99.0 = rely on other cuts).
    [    44,    False,   99.0,     99.0,      99.0, 0.99999],
    [    45,    False,   99.0,     99.0,      99.0, 0.99999],
    [    46,    False,   99.0,     99.0,      99.0, 0.99999],
    [    47,    False,   99.0,     99.0,      99.0, 0.99999],
    [    48,    False,   99.0,     99.0,      99.0, 0.99999],
    [    49,    False,   99.0,     99.0,      99.0, 0.99999],
    [    50,    False,   99.0,     99.0,      99.0, 0.99999],
    [    51,    False,   99.0,     99.0,      99.0, 0.99999],
    [    52,    False,   99.0,     99.0,      99.0, 0.99999],
    [    53,    False,   99.0,     99.0,      99.0, 0.99999],
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
    #  0,  1,     2,     3,      4,      5,      6,       7,       8,     9,     10,    11,     12,      13,        14,        15,     16,      17,    18,    19
    #  i,  o, start,  skip, phiCut,  minIn,  maxIn,  minOut,  maxOut, maxDR,  minDZ,  maxDZ, ptCuts, z0Cuts, stubSigma, geomKappa, phiMid, caTheta, caDCA, floor
    # Pixel-only connections (same as Phase2OT)
    [  0,  1,  True, False,    350,  -17.0,   17.0,  -10000,   10000,   5.0,  -16.0,   16.0,    0.7,   13.0,      -1.0,      -1.0,   -1.0,   0.002,  0.25, 0.000],
    [  0,  2,  True,  True,    600,  -14.0,   14.0,  -10000,   10000,  10.0,  -16.0,   16.0,    0.8,   13.0,      -1.0,      -1.0,   -1.0,   0.002,  0.25, 0.000],
    [  0,  4,  True, False,    450,    3.0,  10000,       0,    12.0,   8.5,    0.0,   25.0,    0.6,   13.0,      -1.0,      -1.0,   -1.0,   0.002,  0.25, 0.000],
    [  0,  5,  True,  True,    522,    7.0,  10000,       0,   10000,   5.0,    0.0,   25.0,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.002,  0.25, 0.000],
    [  0, 16,  True, False,    450, -10000,   -3.0,       0,    12.0,   8.5,  -25.0,    0.0,    0.6,   13.0,      -1.0,      -1.0,   -1.0,   0.002,  0.25, 0.000],
    [  0, 17,  True,  True,    522, -10000,   -7.0,       0,   10000,   5.0,  -25.0,    0.0,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.002,  0.25, 0.000],
    [  1,  2,  True, False,    400,  -17.0,   17.0,  -10000,   10000,   7.0,  -13.0,   13.0,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.002,  0.15, 0.000],
    [  1,  3, False,  True,    650,  -15.0,   15.0,  -10000,   10000,  10.0,  -17.0,   17.0,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.002,  0.25, 0.000],
    [  1,  4,  True, False,    500,    6.0,  10000,     6.5,   10000,   8.5,    0.0,   19.0,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.002,  0.25, 0.000],
    [  1,  5, False,  True,    730,    9.0,  10000,     6.5,   10000,  10.0,    0.0,   21.0,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.002,  0.25, 0.000],
    [  1, 16,  True, False,    500, -10000,   -6.0,     6.5,   10000,   8.5,  -19.0,    0.0,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.002,  0.25, 0.000],
    [  1, 17, False,  True,    730, -10000,   -9.0,     6.5,   10000,  10.0,  -21.0,    0.0,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.002,  0.25, 0.000],
    [  2,  3, False, False,    350,  -18.0,   18.0,  -10000,   10000,   7.0,   -9.0,    9.0,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.002,  0.25, 0.000],
    [  2,  4, False, False,    400,   11.0,  10000,    11.7,   10000,   7.0,    0.0,   13.0,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.002,  0.15, 0.000],
    [  2, 16, False, False,    400, -10000,  -11.0,    11.7,   10000,   7.0,  -13.0,    0.0,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.002,  0.15, 0.000],
    # Pixel barrel L3 (CA 2) to OT barrel L1 (CA 28) - from HLT CA extension
    [  2, 28, False,  True,   1200,    -10,     10,   -30.0,    30.0, 10000,  -15.0,   15.0,    2.0,   13.0,       4.0,      -1.0,   -1.0,   0.002,  0.15, 0.000],  # central
    [  2, 28, False, False,   1200,    -20,    -10,   -50.0,   -25.0, 10000,  -35.0,  -10.0,   0.85,   13.0,       4.0,      -1.0,   -1.0,   0.002,  0.15, 0.000],  # backward
    [  2, 28, False, False,   1200,     10,     20,    25.0,    50.0, 10000,   10.0,   35.0,   0.85,   13.0,       4.0,      -1.0,   -1.0,   0.002,  0.15, 0.000],  # forward
    # Pixel barrel L4 (CA 3) to OT barrel L1 (CA 28) - from HLT CA extension
    [  3, 28, False, False,   1000,    -20,     20,   -45.0,    45.0, 10000,  -22.0,   22.0,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.002,  0.20, 0.000],
    # Pixel forward endcap disks to OT barrel L1 (CA 28) - from HLT CA extension
    [  4, 28, False, False,   1000,   11.6,  10000,    30.0,    57.5,  16.0,    5.0,   32.5,   0.85,   13.0,       5.0,       5.0,   -1.0,   0.003,  0.25, 0.000],
    [  5, 28, False, False,   1000,   11.6,  10000,    40.0,    70.0,  16.0,    5.0,   50.0,   0.85,   13.0,       5.0,       5.0,   -1.0,   0.003,  0.25, 0.000],
    [  6, 28, False, False,   1000,   11.6,  10000,    55.0,    95.0,  16.0,    5.0,   50.0,   0.85,   13.0,       8.0,       5.0,   -1.0,   0.003,  0.25, 0.000],
    [  7, 28, False, False,   1000,   11.8,  10000,    70.0,   110.0,  16.0,   15.0,   70.0,   0.85,   13.0,       8.0,       5.0,   -1.0,   0.003,  0.25, 0.000],
    [  8, 28, False, False,   1000,      0,  10000,    80.0,   10000,  14.0,   25.0,   70.0,   0.85,   13.0,       8.0,       5.0,   -1.0,   0.003,  0.25, 0.000],
    # Pixel backward endcap disks to OT barrel L1 (CA 28) - from HLT CA extension
    [ 16, 28, False, False,   1000,   11.6,  10000,   -57.5,   -30.0,  16.0,  -32.5,   -5.0,   0.85,   13.0,       5.0,       5.0,   -1.0,   0.003,  0.25, 0.000],
    [ 17, 28, False, False,   1000,   11.6,  10000,   -80.0,   -40.0,  16.0,  -50.0,   -5.0,   0.85,   13.0,       5.0,       5.0,   -1.0,   0.003,  0.25, 0.000],
    [ 18, 28, False, False,   1000,   11.6,  10000,   -95.0,   -55.0,  16.0,  -50.0,   -5.0,   0.85,   13.0,       8.0,       5.0,   -1.0,   0.003,  0.25, 0.000],
    [ 19, 28, False, False,   1000,   11.8,  10000,  -110.0,   -70.0,  16.0,  -70.0,  -15.0,   0.85,   13.0,       8.0,       5.0,   -1.0,   0.003,  0.25, 0.000],
    [ 20, 28, False, False,   1000,      0,  10000,  -10000,   -80.0,  14.0,  -70.0,  -25.0,   0.85,   13.0,       8.0,       5.0,   -1.0,   0.003,  0.25, 0.000],
    # Pixel forward endcap connections
    [  4,  5,  True, False,    300,      0,   14.0,     3.5,   10000,   4.5, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [  4,  6, False,  True,    522,      0,   14.0,     3.5,   10000,   9.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [  5,  6,  True, False,    300,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [  5,  7, False,  True,    522,      0,   13.0,     3.5,   10000,   9.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [  6,  7,  True, False,    250,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [  6,  8, False,  True,    522,      0,   13.0,     3.5,   10000,   9.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [  7,  8,  True, False,    250,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [  7,  9, False,  True,    522,      0,   13.0,     3.5,   10000,   8.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [  8,  9,  True, False,    250,      0,   13.0,     3.5,   10000,   4.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [  8, 10, False,  True,    522,      0,   13.0,     3.5,   10000,   8.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [  9, 10,  True, False,    300,      0,   13.0,     4.0,   10000,   4.5, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [  9, 11, False,  True,    522,      0,   13.0,     4.0,   10000,   8.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 10, 11,  True, False,    240,      0,   13.0,     3.5,   10000,   4.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 10, 12, False,  True,    650,   12.5,   16.5,    20.0,   10000,  10.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 11, 12, False, False,    300,      0,   16.5,     6.0,    21.0,   5.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 11, 13, False, False,    200,      0,    6.0,       0,     7.5,   3.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 11, 14, False, False,    220,      0,    4.6,       0,     7.5,   3.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 11, 15, False, False,    250,      0,    6.0,       0,   10000,   4.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 12, 13, False, False,    250,      0,   22.5,     7.0,   10000,   4.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 13, 14, False, False,    250,      0,   22.5,     7.0,   10000,   4.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.40, 0.000],
    [ 14, 15, False, False,    250,      0,   22.5,     7.0,   10000,   3.5, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.40, 0.000],
    # Pixel backward endcap connections
    [ 16, 17,  True, False,    300,      0,   14.0,     3.5,   10000,   4.5, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 16, 18, False,  True,    522,      0,   14.0,     3.5,   10000,   9.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 17, 18,  True, False,    300,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 17, 19, False,  True,    522,      0,   13.0,     3.5,   10000,   9.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 18, 19,  True, False,    250,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 18, 20, False,  True,    522,      0,   13.0,     3.5,   10000,   9.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 19, 20,  True, False,    250,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 19, 21, False,  True,    522,      0,   13.0,     3.5,   10000,   8.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 20, 21,  True, False,    250,      0,   13.0,     3.5,   10000,   4.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 20, 22, False,  True,    522,      0,   13.0,     3.5,   10000,   8.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 21, 22,  True, False,    300,      0,   13.0,     4.0,   10000,   4.5, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 21, 23, False,  True,    522,      0,   13.0,     4.0,   10000,   8.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 22, 23,  True, False,    240,      0,   13.0,     3.5,   10000,   4.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 22, 24, False,  True,    650,   12.5,   16.5,    20.0,   10000,  10.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 23, 24, False, False,    300,      0,   16.5,     6.0,    21.0,   5.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 23, 25, False, False,    200,      0,    6.0,       0,     7.5,   3.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 23, 26, False, False,    220,      0,    4.6,       0,     7.5,   3.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 23, 27, False, False,    250,      0,    6.0,       0,   10000,   4.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 24, 25, False, False,    250,      0,   22.5,     7.0,   10000,   4.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.25, 0.000],
    [ 25, 26, False, False,    250,      0,   22.5,     7.0,   10000,   4.0, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.40, 0.000],
    [ 26, 27, False, False,    250,      0,   22.5,     7.0,   10000,   3.5, -10000,  10000,   0.85,   13.0,      -1.0,      -1.0,   -1.0,   0.003,  0.40, 0.000],
    # OT barrel to barrel connections (layers 28-33) - from HLT CA extension for 28-29, 29-30
    # stubSigmaCut=5.0: kappa-corrected pairwise significance for flat-flat barrel stubs
    [ 28, 29, False, False,   1100,  -1200,   1200,  -10000,   10000, 10000,  -50.0,   50.0,   0.85,   13.0,       5.0,       5.0,    0.1,   0.005,  0.50, 0.000],
    [ 29, 30, False, False,   1250,  -1200,   1200,  -10000,   10000, 10000,  -40.0,   40.0,   0.85,   13.0,       5.0,       5.0,    0.1,   0.005,  0.50, 0.000],
    [ 30, 31, False, False,   1250, -10000,  10000,  -10000,   10000, 10000,  -30.0,   30.0,   0.85,   24.0,       5.0,       5.0,    0.1,   3.000,  0.50, 0.000],
    [ 31, 32, False, False,   2000, -10000,  10000,  -10000,   10000, 10000,  -30.0,   30.0,   0.85,   30.0,       5.0,       5.0,    0.1,   3.000,  0.50, 0.000],
    [ 32, 33, False, False,   2000, -10000,  10000,  -10000,   10000, 10000,  -25.0,   25.0,   0.85,   30.0,       5.0,       5.0,    0.1,   3.000,  0.50, 0.000],
    # OT barrel layers to first backward disk (34) - kappa-corrected significance
    [ 28, 34, False, False,   1600,     40,    130,      20,      40, 10000, -10000,  10000,   0.85,   13.0,       5.0,       5.0,    0.1,   0.005,  0.50, 0.000],
    [ 29, 34, False, False,   1700,     80,    130,      30,      60, 10000, -10000,  10000,   0.85,   13.0,       5.0,       5.0,    0.1,   0.005,  0.50, 0.000],
    [ 30, 34, False, False,   2000,    100,    130,      50,      80, 10000, -10000,  10000,   0.85,   13.0,       5.0,       5.0,    0.1,   0.005,  0.50, 0.000],
    [ 31, 35, False, False,   2000,  -1300,    -80,      60,     110, 10000, -10000,  10000,   0.85,   13.0,       5.0,       5.0,    0.1,   0.005,  0.50, 0.000],
    [ 32, 35, False, False,   3300,  -1300,    -80,      80,     110, 10000, -10000,  10000,   0.85,   13.0,       5.0,       5.0,    0.1,   0.005,  0.50, 0.000],
    # OT barrel layers to first forward disk (39) - kappa-corrected significance
    [ 28, 44, False, False,   2500,   -130,  -40,      20,      40, 10000, -10000,  10000,   0.85,   13.0,       5.0,       5.0,    0.1,   0.005,  0.50, 0.000],
    [ 29, 44, False, False,   2500,   -130,  -80,      30,      60, 10000, -10000,  10000,   0.85,   13.0,       5.0,       5.0,    0.1,   0.005,  0.50, 0.000],
    [ 30, 44, False, False,   2000,   -130,  -100,     50,      80, 10000, -10000,  10000,   0.85,   13.0,       5.0,       5.0,    0.1,   0.005,  0.50, 0.000],
    [ 31, 45, False, False,   2000,    80,  10000,      60,     110, 10000, -10000,  10000,   0.85,   13.0,       5.0,       5.0,    0.1,   0.005,  0.50, 0.000],
    [ 32, 45, False, False,   3300,    80,  10000,      80,     110, 10000, -10000,  10000,   0.85,   13.0,       5.0,       5.0,    0.1,   0.005,  0.50, 0.000],
    # Backward disk to disk consecutive connections (layers 34-38)
    # For backward disks: dz < 0 (moving to more negative z)
    # r-range: 20-115 cm (~15% margin over actual 23-110 cm disk extent)
    # maxDR=60: covers full disk r-extent with margin
    # Signed dz: enforces correct z-direction AND constrains to actual disk separations
    [ 34, 36, False, False,   2500,     20,    115,      20,     115,  60.0,  15.0,   50.0,   0.85,   18.0,       6.5,       5.0,    0.1,   0.005,  0.50, 0.000],
    [ 36, 38, False, False,   2500,     20,    115,      20,     115,  60.0,  15.0,   50.0,   0.85,   18.0,       5.6,       5.0,    0.1,   0.005,  0.50, 0.000],
    [ 38, 40, False, False,   2500,     20,    115,      20,     115,  60.0,  15.0,   50.0,   0.85,   18.0,       5.4,       5.0,    0.1,   0.005,  0.50, 0.000],
    [ 40, 42, False, False,   2500,     20,    115,      20,     115,  60.0,  15.0,   50.0,   0.85,   18.0,       6.0,       5.0,    0.1,   0.005,  0.50, 0.000],
    # Forward disk to disk consecutive connections (layers 39-43)
    # For forward disks: dz > 0 (moving to more positive z)
    [ 44, 46, False, False,   2500,     20,    115,      20,     115,  60.0,  -50.0,  -15.0,   0.85,   18.0,       6.5,       5.0,    0.1,   0.005,  0.50, 0.000],
    [ 46, 48, False, False,   2500,     20,    115,      20,     115,  60.0,  -50.0,  -15.0,   0.85,   18.0,       5.6,       5.0,    0.1,   0.005,  0.50, 0.000],
    [ 48, 50, False, False,   2500,     20,    115,      20,     115,  60.0,  -50.0,  -15.0,   0.85,   18.0,       5.4,       5.0,    0.1,   0.005,  0.50, 0.000],
    [ 50, 52, False, False,   2500,     20,    115,      20,     115,  60.0,  -50.0,  -15.0,   0.85,   18.0,       6.0,       5.0,    0.1,   0.005,  0.50, 0.000],
]

# find the layerPairs that contain a layer that is excluded
excludeLayerPair = [any([(lp[0] == l) or (lp[1] == l) for l in layersToExclude]) for lp in layerPairs]

# exclude those layerPairs
layerPairsStubs = []
for i, lp in enumerate(layerPairs):
    if not excludeLayerPair[i]:
        layerPairsStubs.append(lp)

hltPhase2PixelTracksSoAWithStubs = cms.EDProducer('CAHitNtupletAlpakaPhase2OTStubs@alpaka',
    pixelRecHitSrc = cms.InputTag('hltPhase2PixelRecHitsStubsMerger'),
    otRecHitsSrc = cms.InputTag('hltPixelSeedingOTRecHitsSoA'),
    stubsSrc = cms.InputTag('hltOTStubProducer'),

    # Container size parameters
    avgHitsPerTrack = cms.double(10.0),   # pixel+OT tracks avg ~7 hits, max ~14
    avgCellsPerHit = cms.double(23),     # ~12% margin over peak ratio 20.6 (from 1k ttbar PU200)
    avgCellsPerCell = cms.double(0.3),   # ~32% margin over peak ratio 0.23
    avgTracksPerCell = cms.double(0.6),  # ~46% margin over peak ratio 0.14
    maxNumberOfDoublets = cms.string(str(6e6)),
    maxNumberOfTuples = cms.string(str(200e3)),

    # Flags
    fitNas4 = cms.bool(False),
    useRiemannFit = cms.bool(False),
    useSimpleTripletCleaner = cms.bool(True),
    onlySameLayersFishbone = cms.bool(True),
    fillStatistics = cms.bool(False),  # Enabled to diagnose tuple overflow
    # Flags for duplicate removals
    dupPassThrough = cms.bool(False),
    earlyFishbone = cms.bool(True),
    doEarlyDuplicateRemoval = cms.bool(True),
    lateFishbone = cms.bool(False),
    disableFastDuplicateRemover = cms.bool(False),
    doSharedHitCut = cms.bool(True),
    disableTripletCleaner = cms.bool(False),

    # Reachability filter (kills L28 cells whose neighbor chains don't reach far enough)
    reachTargetLayer = cms.uint32(28),  # OT barrel L1
    reachMinHops = cms.uint32(0),       # must reach 2 distinct outer layers

    # Orphan chain recovery (disabled by default)
    doOrphanRecovery = cms.bool(False),
    minHitsOrphanNtuplet = cms.uint32(5),  # require 5+ hits to suppress fakes


    # CA parameters (individual scalars)
    minLayersPerNtuplet = cms.uint32(4),  # Require at least 1 OT hit for barrel testing
    minHitsForSharingCut = cms.uint32(1),  # disables nLayers-based sharing cut in favor of nHits-based cut

    # CA parameters
    fishboneCuts = cms.vdouble([l[5] for l in layers]),

    graph = cms.PSet(
        layerPairs   = cms.vuint32(sum([[lp[0], lp[1]] for lp in layerPairsStubs], [])),
        startingPair = cms.vuint32([int(lp[2]) for lp in layerPairsStubs]),
        skipsLayers  = cms.vuint32([int(lp[3]) for lp in layerPairsStubs]),
    ),

    doubletCuts = cms.PSet(
        maxDPhi          = cms.vint32( [lp[ 4] for lp in layerPairsStubs]),
        minInner         = cms.vdouble([lp[ 5] for lp in layerPairsStubs]),
        maxInner         = cms.vdouble([lp[ 6] for lp in layerPairsStubs]),
        minOuter         = cms.vdouble([lp[ 7] for lp in layerPairsStubs]),
        maxOuter         = cms.vdouble([lp[ 8] for lp in layerPairsStubs]),
        maxDR            = cms.vdouble([lp[ 9] for lp in layerPairsStubs]),
        minDZ            = cms.vdouble([lp[10] for lp in layerPairsStubs]),
        maxDZ            = cms.vdouble([lp[11] for lp in layerPairsStubs]),
        minPt            = cms.vdouble([lp[12] for lp in layerPairsStubs]),
        maxZ0            = cms.vdouble([lp[13] for lp in layerPairsStubs]),
        maxStubCurvSigma = cms.vdouble([lp[14] for lp in layerPairsStubs]),
        
        dzdrFact = cms.double(15.2),
        minInnerSizeB1  = cms.int32(15),
        minInnerSizeB2  = cms.int32(14),
        maxDSizeB1      = cms.int32(15),
        maxDSize        = cms.int32(20),
        maxDSizePred    = cms.int32(24),
    ),

    tripletCuts = cms.PSet(
        maxRZTolerance           = cms.vdouble([lp[17] for lp in layerPairsStubs]),
        maxDCA                   = cms.vdouble([lp[18] for lp in layerPairsStubs]),
        floorDCA                 = cms.vdouble([lp[19] for lp in layerPairsStubs]),
        maxStubGeomCurvSigma     = cms.vdouble([lp[15] for lp in layerPairsStubs]),
        maxStubInnerDoubletDCurv = cms.vdouble([lp[16] for lp in layerPairsStubs]),
        
        ptmin       = cms.double(0.9),
        maxCurv     = cms.double(0.02),
        maxPhiResid = cms.double(0.004),
        sameDPhiSign = cms.bool(True),
    ),

    ntupletCuts = cms.PSet(
        startMaxInnerR = cms.vdouble([l[2] for l in layers]),
        maxDCurv       = cms.vdouble([l[3] for l in layers]),
        floorDCurv     = cms.vdouble([l[4] for l in layers]),
    ),

    trackQualityCuts = cms.PSet(
        maxChi2 = cms.double(7.0),
        maxChi2TripletsOrQuadruplets = cms.double(7.0),
        maxChi2Quintuplets = cms.double(7.0),
        minPt = cms.double(0.9),
        maxTip = cms.double(0.3),
        maxZip = cms.double(12.0),
    ),

    mightGet = cms.optional.untracked.vstring,
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)
