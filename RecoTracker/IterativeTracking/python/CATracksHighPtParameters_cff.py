# Parameters of the high-pT sub-iterations of the CA seeding of the initialStep
# (see CATracksIterations_cff), lowered to a pT of ~0.7 GeV.
#
# The ptCuts of the layer pairs are the reference ones scaled by ptMin / 0.9.
#
# The phiCuts are the windows of the reference layer pairs
# (RecoTracker/PixelTrackFitting/python/pixelTracksExtended_cfg.py, tuned at the
# pair ptCut there) scaled to the new pair ptCut by the ratio of dphi/dr of a
# track from the beamline at the outer radius r of the pair:
#   phiCut = phiCut_ref * sqrt(4 R_ref^2 - r^2) / sqrt(4 R^2 - r^2),  R = ptCut * 87.78 cm
# with r the mid radius of the outer layer (approximate Phase-2 geometry, within
# the pair limits) and ptCut floored so that r / 2R <= 0.9. The phiCuts are in
# units of 2pi / 2^16 rad.

ptMin = 0.7

# per layer pair, in the order of the pairGraph of pixelTracksHighPtAlpaka
layerPairs = [
    #  i,  o, phiCut, ptCut
    [  0,  1,    451, 0.544],
    [  0,  2,    773, 0.622],
    [  0,  4,    580, 0.467],
    [  0,  5,    673, 0.661],
    [  0, 16,    580, 0.467],
    [  0, 17,    673, 0.661],
    [  1,  2,    516, 0.661],
    [  1,  3,    839, 0.661],
    [  1,  4,    645, 0.661],
    [  1,  5,    941, 0.661],
    [  1, 16,    645, 0.661],
    [  1, 17,    941, 0.661],
    [  2,  3,    452, 0.661],
    [  2,  4,    516, 0.661],
    [  2, 16,    516, 0.661],
    [  2, 28,   1545, 1.556],
    [  2, 28,   1555, 0.661],
    [  2, 28,   1555, 0.661],
    [  3, 28,   1296, 0.661],
    [  4,  5,    387, 0.661],
    [  4,  6,    673, 0.661],
    [  4, 28,   1296, 0.661],
    [  5,  6,    387, 0.661],
    [  5,  7,    673, 0.661],
    [  5, 28,   1296, 0.661],
    [  6,  7,    322, 0.661],
    [  6,  8,    673, 0.661],
    [  6, 28,   1296, 0.661],
    [  7,  8,    322, 0.661],
    [  7,  9,    673, 0.661],
    [  7, 28,   1296, 0.661],
    [  8,  9,    322, 0.661],
    [  8, 10,    673, 0.661],
    [  8, 28,   1102, 0.661],
    [  9, 10,    387, 0.661],
    [  9, 11,    673, 0.661],
    [ 10, 11,    310, 0.661],
    [ 10, 12,    843, 0.661],
    [ 11, 12,    387, 0.661],
    [ 11, 13,    258, 0.661],
    [ 11, 14,    284, 0.661],
    [ 11, 15,    323, 0.661],
    [ 12, 13,    323, 0.661],
    [ 13, 14,    323, 0.661],
    [ 14, 15,    323, 0.661],
    [ 16, 17,    387, 0.661],
    [ 16, 18,    673, 0.661],
    [ 16, 28,   1296, 0.661],
    [ 17, 18,    387, 0.661],
    [ 17, 19,    673, 0.661],
    [ 17, 28,   1296, 0.661],
    [ 18, 19,    322, 0.661],
    [ 18, 20,    673, 0.661],
    [ 18, 28,   1296, 0.661],
    [ 19, 20,    322, 0.661],
    [ 19, 21,    673, 0.661],
    [ 19, 28,   1296, 0.661],
    [ 20, 21,    322, 0.661],
    [ 20, 22,    673, 0.661],
    [ 20, 28,   1296, 0.661],
    [ 21, 22,    387, 0.661],
    [ 21, 23,    673, 0.661],
    [ 22, 23,    310, 0.661],
    [ 22, 24,    843, 0.661],
    [ 23, 24,    387, 0.661],
    [ 23, 25,    258, 0.661],
    [ 23, 26,    284, 0.661],
    [ 23, 27,    323, 0.661],
    [ 24, 25,    323, 0.661],
    [ 25, 26,    323, 0.661],
    [ 26, 27,    323, 0.661],
    [ 28, 29,   1444, 0.661],
    [ 29, 30,   1681, 0.661],
]

# container sizes, from the occupancies measured vs the number of hits x
# (TTbar PU200, 700 events, 164k-313k hits; fillStatistics + CA_STATS):
# linear fits with 10% more slope, shifted to cover every event with a 5% margin
# measured: nCells = 37.4*x - 4.3M (max 7.6M), nTracks = 1.36*x - 0.21M (max 252k),
# cells per outer hit <= 26, triplets per cell <= 0.35, cell->track links per cell <= 0.11
containerSizes = dict(
    maxNumberOfDoublets = "max(41*x - 4.29e6, 50000)",
    maxNumberOfTuples = "max(1.5*x - 1.9e5, 50000)",
    avgHitsPerTrack = 10.0,
    avgCellsPerHit = 30.0,
    avgCellsPerCell = 0.4,
    avgTracksPerCell = 0.15
)

caTracksHighPtParameters = dict(
    ptmin = ptMin,                              # scales the caThetaCuts
    hardCurvCut = 1. / (0.8 * ptMin * 87.78),   # max triplet curvature (1/cm), 80% of ptMin
    trackQualityCuts = dict(
        minPt = ptMin
    ),
    geometry = dict(
        phiCuts = [lp[2] for lp in layerPairs],
        ptCuts  = [lp[3] for lp in layerPairs]
    ),
    **containerSizes
)
