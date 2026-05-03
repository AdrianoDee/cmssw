import FWCore.ParameterSet.Config as cms

# list of layers to exclude from the CA (empty list doesn't exclude; [28, 29, 30] excludes the OT)
layersToExclude = []

# layers
layers = [
    #     0,        1,      2,     3,      4,       5
    # index, isBarrel, startR, DCurv, DCurv0, fishCut
    [     0,     True,   99.0,  99.0,   99.0, 0.99999],
    [     1,     True,   99.0,  99.0,   99.0, 0.99999],
    [     2,     True,   99.0,  99.0,   99.0, 0.99999],
    [     3,     True,   99.0,  99.0,   99.0, 0.99999],
    [     4,    False,   99.0,  99.0,   99.0, 0.99999],
    [     5,    False,   99.0,  99.0,   99.0, 0.99999],
    [     6,    False,   99.0,  99.0,   99.0, 0.99999],
    [     7,    False,   99.0,  99.0,   99.0, 0.99999],
    [     8,    False,   99.0,  99.0,   99.0, 0.99999],
    [     9,    False,   99.0,  99.0,   99.0, 0.99999],
    [    10,    False,   99.0,  99.0,   99.0, 0.99999],
    [    11,    False,   99.0,  99.0,   99.0, 0.99999],
    [    12,    False,   99.0,  99.0,   99.0, 0.99999],
    [    13,    False,   99.0,  99.0,   99.0, 0.99999],
    [    14,    False,   99.0,  99.0,   99.0, 0.99999],
    [    15,    False,   99.0,  99.0,   99.0, 0.99999],
    [    16,    False,   99.0,  99.0,   99.0, 0.99999],
    [    17,    False,   99.0,  99.0,   99.0, 0.99999],
    [    18,    False,   99.0,  99.0,   99.0, 0.99999],
    [    19,    False,   99.0,  99.0,   99.0, 0.99999],
    [    20,    False,   99.0,  99.0,   99.0, 0.99999],
    [    21,    False,   99.0,  99.0,   99.0, 0.99999],
    [    22,    False,   99.0,  99.0,   99.0, 0.99999],
    [    23,    False,   99.0,  99.0,   99.0, 0.99999],
    [    24,    False,   99.0,  99.0,   99.0, 0.99999],
    [    25,    False,   99.0,  99.0,   99.0, 0.99999],
    [    26,    False,   99.0,  99.0,   99.0, 0.99999],
    [    27,    False,   99.0,  99.0,   99.0, 0.99999],
    [    28,     True,   99.0,  99.0,   99.0, 0.99999],
    [    29,     True,   99.0,  99.0,   99.0, 0.99999],
    [    30,     True,   99.0,  99.0,   99.0, 0.99999],
]

# layerPairs for doublet building including pair-specific cut values
layerPairs = [
    #  0,  1,     2,     3,      4,      5,      6,       7,       8,     9,     10,     11,     12,     13,      14,    15,    16
    #  i,  o, start,  skip, phiCut,  minIn,  maxIn,  minOut,  maxOut, maxDR,  minDZ,  maxDZ, ptCuts, z0Cuts, caTheta, caDCA, floor
    [  0,  1,  True, False,    350,  -17.0,   17.0,  -10000,   10000,   5.0,  -16.0,   16.0,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  0,  2,  True, False,    600,  -14.0,   14.0,  -10000,   10000,  10.0,  -16.0,   16.0,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  0,  4,  True, False,    450,    4.0,  10000,       0,    10.0,   8.0,    0.0,   25.0,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  0,  5,  True, False,    522,    7.0,  10000,       0,   10000,   5.0,    0.0,   25.0,  0.85,    12.5,   0.002,  0.25, 0.005],
  # [  0,  6, False, False,    522,   11.0,  10000,       0,   10000,   5.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  0, 16,  True, False,    450, -10000,   -4.0,       0,    10.0,   8.0,  -25.0,    0.0,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  0, 17,  True, False,    522, -10000,   -7.0,       0,   10000,   5.0,  -25.0,    0.0,  0.85,    12.5,   0.002,  0.25, 0.005],
  # [  0, 18, False, False,    522, -10000,  -10.0,       0,   10000,   5.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  1,  2,  True, False,    400,  -17.0,   17.0,  -10000,   10000,   7.0,  -13.0,   13.0,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  1,  3, False, False,    650,  -15.0,   15.0,  -10000,   10000,  10.0,  -15.0,   15.0,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  1,  4,  True, False,    500,    6.0,  10000,     6.5,   10000,   8.0,    0.0,   19.0,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  1,  5, False, False,    730,    9.0,  10000,     6.5,   10000,  10.0,    0.0,   21.0,  0.85,    12.5,   0.002,  0.25, 0.005],
  # [  1,  6, False, False,    730,   13.0,  10000,     6.5,   10000,   8.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  1, 16,  True, False,    500, -10000,   -6.0,     6.5,   10000,   8.0,  -19.0,    0.0,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  1, 17, False, False,    730, -10000,   -9.0,     6.5,   10000,  10.0,  -21.0,    0.0,  0.85,    12.5,   0.002,  0.25, 0.005],
  # [  1, 18, False, False,    730, -10000,  -13.0,     6.5,   10000,   8.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
  # [  1, 28, False, False,   1300,    7.0,  10000,    30.0,    40.0, 10000,   19.0,   32.0,   1.0,    12.5,   0.002,  0.25, 0.005],
  # [  1, 28, False, False,   1300, -10000,   -7.0,   -40.0,   -30.0, 10000,  -32.0,  -19.0,   1.0,    12.5,   0.002,  0.25, 0.005],
    [  2,  3,  True, False,    350,  -18.0,   18.0,  -10000,   10000,   7.0,   -9.0,    9.0,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  2,  4, False, False,    400,   11.0,  10000,    11.7,   10000,   7.0,    0.0,   13.0,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  2, 16, False, False,    400, -10000,  -11.0,    11.7,   10000,   7.0,  -13.0,    0.0,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  2, 28, False, False,   1200,    -10,     10,   -30.0,    30.0, 10000,  -15.0,   15.0,   2.0,    12.5,   0.002,  0.25, 0.005],
    [  2, 28, False, False,   1200,    -20,    -10,   -50.0,   -25.0, 10000,  -35.0,  -10.0,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  2, 28, False, False,   1200,     10,     20,    25.0,    50.0, 10000,   10.0,   35.0,  0.85,    12.5,   0.002,  0.25, 0.005],
  # [  2, 28, False, False,   1200,    -20,     20,   -50.0,    50.0, 10000,  -35.0,   35.0,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  3, 28, False, False,   1000,    -20,     20,   -45.0,    45.0, 10000,  -22.0,   22.0,  0.85,    12.5,   0.002,  0.25, 0.005],
  # [  3, 29, False, False,   1500,    -40,     40,  -10000,   10000, 10000, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  4,  5,  True, False,    300,      0,   14.0,     3.5,   10000,   4.5, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  4,  6, False, False,    522,      0,   14.0,     3.5,   10000,   9.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  4, 28, False, False,   1000,   11.6,  10000,    30.0,    57.5,  16.0,    5.0,   32.5,  0.85,    12.5,   0.002,  0.25, 0.005],
  # [  4, 29, False, False,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  5,  6,  True, False,    300,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  5,  7, False, False,    522,      0,   13.0,     3.5,   10000,   9.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  5, 28, False, False,   1000,   11.6,  10000,    40.0,    80.0,  16.0,  -10.0,   50.0,  0.85,    12.5,   0.002,  0.25, 0.005],
  # [  5, 29, False, False,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  6,  7,  True, False,    250,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  6,  8, False, False,    522,      0,   13.0,     3.5,   10000,   9.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  6, 28, False, False,   1000,   11.6,  10000,    55.0,    95.0,  16.0,    5.0,   50.0,  0.85,    12.5,   0.002,  0.25, 0.005],
  # [  6, 29, False, False,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  7,  8,  True, False,    250,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  7,  9, False, False,    522,      0,   13.0,     3.5,   10000,   8.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  7, 28, False, False,   1000,   11.8,  10000,    70.0,   110.0,  16.0,   15.0,   70.0,  0.85,    12.5,   0.002,  0.25, 0.005],
  # [  7, 29, False, False,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  8,  9,  True, False,    250,      0,   13.0,     3.5,   10000,   4.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  8, 10, False, False,    522,      0,   13.0,     3.5,   10000,   8.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  8, 28, False, False,    850,      0,  10000,    80.0,   10000,  14.0,   25.0,   70.0,  0.85,    12.5,   0.002,  0.25, 0.005],
  # [  8, 29, False, False,   1000,      0,  10000,       0,   10000, 10000, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  9, 10,  True, False,    300,      0,   13.0,     4.0,   10000,   4.5, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [  9, 11, False, False,    522,      0,   13.0,     4.0,   10000,   8.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
  # [  9, 28, False, False,   1000,      0,  10000,       0,   10000, 10000, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 10, 11,  True, False,    240,      0,   13.0,     3.5,   10000,   4.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 10, 12, False, False,    650,   12.5,   16.5,    20.0,   10000,  10.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 11, 12, False, False,    300,      0,   16.5,     6.0,    21.0,   5.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 11, 13, False, False,    200,      0,    6.0,       0,     7.5,   3.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 11, 14, False, False,    220,      0,    4.6,       0,     7.5,   3.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 11, 15, False, False,    250,      0,    6.0,       0,   10000,   4.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 12, 13, False, False,    250,      0,   22.5,     7.0,   10000,   4.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 13, 14, False, False,    250,      0,   22.5,     7.0,   10000,   4.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 14, 15, False, False,    250,      0,   22.5,     7.0,   10000,   3.5, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 16, 17,  True, False,    300,      0,   14.0,     3.5,   10000,   4.5, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 16, 18, False, False,    522,      0,   14.0,     3.5,   10000,   9.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 16, 28, False, False,   1000,   11.6,  10000,   -57.5,   -30.0,  16.0,  -32.5,   -5.0,  0.85,    12.5,   0.002,  0.25, 0.005],
  # [ 16, 29, False, False,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 17, 18,  True, False,    300,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 17, 19, False, False,    522,      0,   13.0,     3.5,   10000,   9.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 17, 28, False, False,   1000,   11.6,  10000,   -70.0,   -40.0,  16.0,  -50.0,  -10.0,  0.85,    12.5,   0.002,  0.25, 0.005],
  # [ 17, 29, False, False,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 18, 19,  True, False,    250,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 18, 20, False, False,    522,      0,   13.0,     3.5,   10000,   9.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 18, 28, False, False,   1000,   11.6,  10000,   -95.0,   -55.0,  16.0,  -50.0,   -5.0,  0.85,    12.5,   0.002,  0.25, 0.005],
  # [ 18, 29, False, False,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 19, 20,  True, False,    250,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 19, 21, False, False,    522,      0,   13.0,     3.5,   10000,   8.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 19, 28, False, False,   1000,   11.8,  10000,  -110.0,   -70.0,  16.0,  -70.0,  -15.0,  0.85,    12.5,   0.002,  0.25, 0.005],
  # [ 19, 29, False, False,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 20, 21,  True, False,    250,      0,   13.0,     3.5,   10000,   4.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 20, 22, False, False,    522,      0,   13.0,     3.5,   10000,   8.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 20, 28, False, False,   1000,      0,  10000,  -10000,   -80.0,  14.0,  -70.0,  -25.0,  0.85,    12.5,   0.002,  0.25, 0.005],
  # [ 20, 29, False, False,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 21, 22,  True, False,    300,      0,   13.0,     4.0,   10000,   4.5, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 21, 23, False, False,    522,      0,   13.0,     4.0,   10000,   8.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
  # [ 21, 28, False, False,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 22, 23,  True, False,    240,      0,   13.0,     3.5,   10000,   4.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 22, 24, False, False,    650,   12.5,   16.5,    20.0,   10000,  10.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 23, 24, False, False,    300,      0,   16.5,     6.0,    21.0,   5.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 23, 25, False, False,    200,      0,    6.0,       0,     7.5,   3.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 23, 26, False, False,    220,      0,    4.6,       0,     7.5,   3.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 23, 27, False, False,    250,      0,    6.0,       0,   10000,   4.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 24, 25, False, False,    250,      0,   22.5,     7.0,   10000,   4.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 25, 26, False, False,    250,      0,   22.5,     7.0,   10000,   4.0, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 26, 27, False, False,    250,      0,   22.5,     7.0,   10000,   3.5, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 28, 29, False, False,   1100,  -1200,   1200,  -10000,   10000, 10000,  -50.0,   50.0,  0.85,    12.5,   0.002,  0.25, 0.005],
  # [ 28, 30, False, False,   2000,    -40,     40,  -10000,   10000, 10000, -10000,  10000,  0.85,    12.5,   0.002,  0.25, 0.005],
    [ 29, 30, False, False,   1250,  -1200,   1200,  -10000,   10000, 10000,  -40.0,   40.0,  0.85,    12.5,   0.002,  0.25, 0.005],
]

# find the layerPairs that contain a layer that is excluded
excludeLayerPair = [any([(lp[0] == l) or (lp[1] == l) for l in layersToExclude]) for lp in layerPairs]
excludeCAExtension = [any([(lp[0] == l) or (lp[1] == l) for l in [28, 29, 30]]) for lp in layerPairs]

# exclude those layerPairs
layerPairsAlpaka = []
layerPairsCAExtension = []
for i, lp in enumerate(layerPairs):
    if (not excludeLayerPair[i]) and (not excludeCAExtension[i]):
        layerPairsAlpaka.append(lp)
    if not excludeLayerPair[i]:
        layerPairsCAExtension.append(lp)

hltPhase2PixelTracksSoA = cms.EDProducer('CAHitNtupletAlpakaPhase2OT@alpaka',
    pixelRecHitSrc = cms.InputTag('hltPhase2PixelRecHitsExtendedSoA'),
    ptmin = cms.double(0.9),
    hardCurvCut = cms.double(0.01425), # corresponds to 800 MeV in 3.8T.
    earlyFishbone = cms.bool(True),
    lateFishbone = cms.bool(False),
    onlySameLayersFishbone = cms.bool(False),
    fillStatistics = cms.bool(False),
    minLayersPerNtuplet = cms.uint32(4),
    maxNumberOfDoublets = cms.string(str(6e6)), #12*512*1024
    maxNumberOfTuples = cms.string(str(2*60*1024)),
    minYsizeB1 = cms.int32(20),
    minYsizeB2 = cms.int32(18),
    maxDYsize12 = cms.int32(12),
    maxDYsize = cms.int32(10),
    maxDYPred = cms.int32(24),
    avgHitsPerTrack = cms.double(8.0),
    avgCellsPerHit = cms.double(17),
    avgCellsPerCell = cms.double(0.5),
    avgTracksPerCell = cms.double(0.09),
    minHitsForSharingCut = cms.uint32(10),
    fitNas4 = cms.bool(False),
    useRiemannFit = cms.bool(False),
    doSharedHitCut = cms.bool(True),
    dupPassThrough = cms.bool(False),
    useSimpleTripletCleaner = cms.bool(True),
    disableTripletCleaner = cms.bool(False),
    disableFastDuplicateRemover = cms.bool(False),
    doEarlyDuplicateRemoval = cms.bool(True),
    trackQualityCuts = cms.PSet(
        maxChi2TripletsOrQuadruplets = cms.double(1.0),
        maxChi2Quintuplets = cms.double(3.0),
        maxChi2 = cms.double(5.0),
        minPt   = cms.double(0.9),
        maxTip  = cms.double(0.3),
        maxZip  = cms.double(12),
    ),
    # CA parameters
    fishboneCuts = cms.vdouble([l[5] for l in layers]),
    graph = cms.PSet(
        layerPairs   = cms.vuint32(sum([[lp[0], lp[1]] for lp in layerPairsCAExtension], [])),
        startingPair = cms.vuint32([int(lp[2]) for lp in layerPairsCAExtension]),
        skipsLayers  = cms.vuint32([int(lp[3]) for lp in layerPairsCAExtension]),
    ),
    doubletCuts = cms.PSet(
        maxDPhi          = cms.vint32( [lp[ 4] for lp in layerPairsCAExtension]),
        minInner         = cms.vdouble([lp[ 5] for lp in layerPairsCAExtension]),
        maxInner         = cms.vdouble([lp[ 6] for lp in layerPairsCAExtension]),
        minOuter         = cms.vdouble([lp[ 7] for lp in layerPairsCAExtension]),
        maxOuter         = cms.vdouble([lp[ 8] for lp in layerPairsCAExtension]),
        maxDR            = cms.vdouble([lp[ 9] for lp in layerPairsCAExtension]),
        minDZ            = cms.vdouble([lp[10] for lp in layerPairsCAExtension]),
        maxDZ            = cms.vdouble([lp[11] for lp in layerPairsCAExtension]),
        minPt            = cms.vdouble([lp[12] for lp in layerPairsCAExtension]),
        maxZ0            = cms.vdouble([lp[13] for lp in layerPairsCAExtension]),
        dzdrFact = cms.double(15.2),
        minInnerSizeB1  = cms.int32(15),
        minInnerSizeB2  = cms.int32(14),
        maxDSizeB1      = cms.int32(15),
        maxDSize        = cms.int32(20),
        maxDSizePred    = cms.int32(24),
    ),
    tripletCuts = cms.PSet(
        maxRZTolerance = cms.vdouble([lp[14] for lp in layerPairsCAExtension]),
        maxDCA         = cms.vdouble([lp[15] for lp in layerPairsCAExtension]),
        floorDCA       = cms.vdouble([lp[16] for lp in layerPairsCAExtension]),
        ptmin   = cms.double(0.9),
        maxCurv = cms.double(0.02),
    ),
    ntupletCuts = cms.PSet(
        startMaxInnerR = cms.vdouble([l[2] for l in layers]),
        maxDCurv       = cms.vdouble([l[3] for l in layers]),
        floorDCurv     = cms.vdouble([l[4] for l in layers]),
    ),
    # autoselect the alpaka backend
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)


_hltPhase2PixelTracksSoANonCAExtended = cms.EDProducer('CAHitNtupletAlpakaPhase2@alpaka',
    pixelRecHitSrc = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
    ptmin = cms.double(0.9),
    hardCurvCut = cms.double(0.01425), # corresponds to 800 MeV in 3.8T.
    earlyFishbone = cms.bool(True),
    lateFishbone = cms.bool(False),
    onlySameLayersFishbone = cms.bool(False),
    fillStatistics = cms.bool(False),
    minLayersPerNtuplet = cms.uint32(4),
    maxNumberOfDoublets = cms.string(str(6*512*1024)),
    maxNumberOfTuples = cms.string(str(60*1024)),
    minYsizeB1 = cms.int32(20),
    minYsizeB2 = cms.int32(18),
    maxDYsize12 = cms.int32(12),
    maxDYsize = cms.int32(10),
    maxDYPred = cms.int32(24),
    avgHitsPerTrack = cms.double(7.0),
    avgCellsPerHit = cms.double(12),
    avgCellsPerCell = cms.double(0.151),
    avgTracksPerCell = cms.double(0.040),
    minHitsForSharingCut = cms.uint32(10),
    fitNas4 = cms.bool(False),
    useRiemannFit = cms.bool(False),
    doSharedHitCut = cms.bool(True),
    dupPassThrough = cms.bool(False),
    useSimpleTripletCleaner = cms.bool(True),
    disableTripletCleaner = cms.bool(False),
    disableFastDuplicateRemover = cms.bool(False),
    doEarlyDuplicateRemoval = cms.bool(True),
    trackQualityCuts = cms.PSet(
        maxChi2TripletsOrQuadruplets = cms.double(5.0),
        maxChi2Quintuplets = cms.double(5.0),
        maxChi2 = cms.double(5.0),
        minPt   = cms.double(0.9),
        maxTip  = cms.double(0.3),
        maxZip  = cms.double(12),
    ),
    # CA parameters
    fishboneCuts = cms.vdouble([l[5] for l in layers[:28]]),
    graph = cms.PSet(
        layerPairs   = cms.vuint32(sum([[lp[0], lp[1]] for lp in layerPairsAlpaka], [])),
        startingPair = cms.vuint32([int(lp[2]) for lp in layerPairsAlpaka]),
        skipsLayers  = cms.vuint32([int(lp[3]) for lp in layerPairsAlpaka]),
    ),
    doubletCuts = cms.PSet(
        maxDPhi          = cms.vint32( [lp[ 4] for lp in layerPairsAlpaka]),
        minInner         = cms.vdouble([lp[ 5] for lp in layerPairsAlpaka]),
        maxInner         = cms.vdouble([lp[ 6] for lp in layerPairsAlpaka]),
        minOuter         = cms.vdouble([lp[ 7] for lp in layerPairsAlpaka]),
        maxOuter         = cms.vdouble([lp[ 8] for lp in layerPairsAlpaka]),
        maxDR            = cms.vdouble([lp[ 9] for lp in layerPairsAlpaka]),
        minDZ            = cms.vdouble([lp[10] for lp in layerPairsAlpaka]),
        maxDZ            = cms.vdouble([lp[11] for lp in layerPairsAlpaka]),
        minPt            = cms.vdouble([lp[12] for lp in layerPairsAlpaka]),
        maxZ0            = cms.vdouble([lp[13] for lp in layerPairsAlpaka]),
        dzdrFact = cms.double(15.2),
        minInnerSizeB1  = cms.int32(15),
        minInnerSizeB2  = cms.int32(14),
        maxDSizeB1      = cms.int32(15),
        maxDSize        = cms.int32(20),
        maxDSizePred    = cms.int32(24),
    ),
    tripletCuts = cms.PSet(
        maxRZTolerance = cms.vdouble([lp[14] for lp in layerPairsAlpaka]),
        maxDCA         = cms.vdouble([lp[15] for lp in layerPairsAlpaka]),
        floorDCA       = cms.vdouble([lp[16] for lp in layerPairsAlpaka]),
        ptmin   = cms.double(0.9),
        maxCurv = cms.double(0.02),
    ),
    ntupletCuts = cms.PSet(
        startMaxInnerR = cms.vdouble([l[2] for l in layers[:28]]),
        maxDCurv       = cms.vdouble([l[3] for l in layers[:28]]),
        floorDCurv     = cms.vdouble([l[4] for l in layers[:28]]),
    ),
    # autoselect the alpaka backend
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)

from Configuration.ProcessModifiers.hltPhase2LegacyTracking_cff import hltPhase2LegacyTracking
from Configuration.ProcessModifiers.hltPhase2LegacyTrackingPatatrackQuadsChain_cff import hltPhase2LegacyTrackingPatatrackQuads
(hltPhase2LegacyTracking & hltPhase2LegacyTrackingPatatrackQuads).toReplaceWith(hltPhase2PixelTracksSoA, _hltPhase2PixelTracksSoANonCAExtended)


def _exclude_OT_layers(hltPhase2PixelTracksSoA, layers_to_exclude = [28, 29, 30]):
    keep_indices = []
    num_pairs = len(hltPhase2PixelTracksSoA.geometry.pairGraph) // 2
    for i in range(num_pairs):
        a = hltPhase2PixelTracksSoA.geometry.pairGraph[2*i]
        b = hltPhase2PixelTracksSoA.geometry.pairGraph[2*i + 1]
        if a not in layers_to_exclude and b not in layers_to_exclude:
            keep_indices.append(i)
    # Now update in place
    # For pairGraph, build the new flat list from kept pairs
    new_pairGraph = []
    for i in keep_indices:
        new_pairGraph.extend([hltPhase2PixelTracksSoA.geometry.pairGraph[2*i], hltPhase2PixelTracksSoA.geometry.pairGraph[2*i+1]])

    hltPhase2PixelTracksSoA.geometry.pairGraph[:] = new_pairGraph
    # Update all other lists in place
    hltPhase2PixelTracksSoA.geometry.skipsLayers[:] = [hltPhase2PixelTracksSoA.geometry.skipsLayers[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.phiCuts[:] = [hltPhase2PixelTracksSoA.geometry.phiCuts[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.minInnerR[:] = [hltPhase2PixelTracksSoA.geometry.minInnerR[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.maxInnerR[:] = [hltPhase2PixelTracksSoA.geometry.maxInnerR[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.minOuterR[:] = [hltPhase2PixelTracksSoA.geometry.minOuterR[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.maxOuterR[:] = [hltPhase2PixelTracksSoA.geometry.maxOuterR[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.maxDR[:] = [hltPhase2PixelTracksSoA.geometry.maxDR[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.minInnerZ[:] = [hltPhase2PixelTracksSoA.geometry.minInnerZ[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.maxInnerZ[:] = [hltPhase2PixelTracksSoA.geometry.maxInnerZ[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.minOuterZ[:] = [hltPhase2PixelTracksSoA.geometry.minOuterZ[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.maxOuterZ[:] = [hltPhase2PixelTracksSoA.geometry.maxOuterZ[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.minDZ[:] = [hltPhase2PixelTracksSoA.geometry.minDZ[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.maxDZ[:] = [hltPhase2PixelTracksSoA.geometry.maxDZ[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.ptCuts[:] = [hltPhase2PixelTracksSoA.geometry.ptCuts[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.z0Cuts[:] = [hltPhase2PixelTracksSoA.geometry.z0Cuts[i] for i in keep_indices]

#print("Using {} pair connections: {}".format(len(hltPhase2PixelTracksSoA.geometry.pairGraph) // 2, hltPhase2PixelTracksSoA.geometry.pairGraph))

# Stub-based tracking with OT stubs
from Configuration.ProcessModifiers.phase2CAStubs_cff import phase2CAStubs
from .hltPhase2PixelTracksSoAWithStubs_cfi import hltPhase2PixelTracksSoAWithStubs as _hltPhase2PixelTracksSoAWithStubs
phase2CAStubs.toReplaceWith(hltPhase2PixelTracksSoA, _hltPhase2PixelTracksSoAWithStubs)
