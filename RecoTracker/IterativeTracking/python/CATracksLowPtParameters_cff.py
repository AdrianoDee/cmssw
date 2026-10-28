# Parameters of the low-pT sub-iterations of the CA seeding of the initialStep
# (see CATracksIterations_cff), lowered to a pT of ~0.2 GeV.
#
# At 0.2 GeV the track radius is ~18 cm: the phi windows grow fast with r near
# the turning point, so they are capped at 0.2 rad (2086). The pairs to the
# outer tracker are effectively used only by tracks above ~0.25-0.3 GeV.
# (v1 at 0.1 GeV with a 0.5 rad cap built 8-11M tuples per event: unusable.)
#
# The phiCuts are the windows of the reference layer pairs
# (RecoTracker/PixelTrackFitting/python/pixelTracksExtended_cfg.py, tuned at the
# pair ptCut there) scaled to the new pair ptCut by the ratio of dphi/dr of a
# track from the beamline at the outer radius r of the pair:
#   phiCut = phiCut_ref * sqrt(4 R_ref^2 - r^2) / sqrt(4 R^2 - r^2),  R = ptCut * 87.78 cm
# with r the mid radius of the outer layer (approximate Phase-2 geometry, within
# the pair limits) and ptCut floored so that r / 2R <= 0.9. The phiCuts are in
# units of 2pi / 2^16 rad.

ptMin = 0.2

# per layer pair, in the order of the pairGraph of pixelTracksLowPtAlpakaPhase2Extended
layerPairs = [
    #  i,  o, phiCut, ptCut
    [  0,  1,   1244, 0.200],
    [  0,  2,   2086, 0.200],
    [  0,  4,   1385, 0.200],
    [  0,  5,   2086, 0.200],
    [  0, 16,   1385, 0.200],
    [  0, 17,   2086, 0.200],
    [  1,  2,   1780, 0.200],
    [  1,  3,   2086, 0.200],
    [  1,  4,   2086, 0.200],
    [  1,  5,   2086, 0.200],
    [  1, 16,   2086, 0.200],
    [  1, 17,   2086, 0.200],
    [  2,  3,   1635, 0.200],
    [  2,  4,   1843, 0.200],
    [  2, 16,   1843, 0.200],
    [  4,  5,   1330, 0.200],
    [  4,  6,   2086, 0.200],
    [  5,  6,   1330, 0.200],
    [  5,  7,   2086, 0.200],
    [  6,  7,   1109, 0.200],
    [  6,  8,   2086, 0.200],
    [  7,  8,   1109, 0.200],
    [  7,  9,   2086, 0.200],
    [  8,  9,   1109, 0.200],
    [  8, 10,   2086, 0.200],
    [  9, 10,   1330, 0.200],
    [  9, 11,   2086, 0.200],
    [ 10, 11,   1064, 0.200],
    [ 10, 12,   2086, 0.200],
    [ 11, 12,   1380, 0.200],
    [ 11, 13,    867, 0.200],
    [ 11, 14,    954, 0.200],
    [ 11, 15,   1187, 0.200],
    [ 12, 13,   1192, 0.200],
    [ 13, 14,   1192, 0.200],
    [ 14, 15,   1192, 0.200],
    [ 16, 17,   1330, 0.200],
    [ 16, 18,   2086, 0.200],
    [ 17, 18,   1330, 0.200],
    [ 17, 19,   2086, 0.200],
    [ 18, 19,   1109, 0.200],
    [ 18, 20,   2086, 0.200],
    [ 19, 20,   1109, 0.200],
    [ 19, 21,   2086, 0.200],
    [ 20, 21,   1109, 0.200],
    [ 20, 22,   2086, 0.200],
    [ 21, 22,   1330, 0.200],
    [ 21, 23,   2086, 0.200],
    [ 22, 23,   1064, 0.200],
    [ 22, 24,   2086, 0.200],
    [ 23, 24,   1380, 0.200],
    [ 23, 25,    867, 0.200],
    [ 23, 26,    954, 0.200],
    [ 23, 27,   1187, 0.200],
    [ 24, 25,   1192, 0.200],
    [ 25, 26,   1192, 0.200],
    [ 26, 27,   1192, 0.200],
    [  2, 28,   2086, 0.200],
    [  2, 28,   2086, 0.200],
    [  2, 28,   2086, 0.200],
    [  3, 28,   2086, 0.200],
    [  4, 28,   2086, 0.200],
    [  5, 28,   2086, 0.200],
    [  6, 28,   2086, 0.200],
    [  7, 28,   2086, 0.200],
    [  8, 28,   2086, 0.200],
    [ 16, 28,   2086, 0.200],
    [ 17, 28,   2086, 0.200],
    [ 18, 28,   2086, 0.200],
    [ 19, 28,   2086, 0.200],
    [ 20, 28,   2086, 0.200],
    [ 28, 29,   2086, 0.200],
    [ 29, 30,   2086, 0.200],
]

# container sizes, from the occupancies measured vs the number of hits x
# (TTbar PU200, 1000 events, 156k-317k hits; fillStatistics + CA_STATS):
# linear fits with ~10% more slope, covering every event with a 5% margin
# measured: nCells = 32.2*x - 1.47M (max 8.5M), nTracks = 6.83*x - 0.75M (max 1.43M),
# cells per outer hit <= 29.1, triplets per cell <= 0.74, cell->track links per cell <= 0.55
containerSizes = dict(
    maxNumberOfDoublets = "max(35*x - 1.42e6, 50000)",
    maxNumberOfTuples = "max(7.5*x - 7.4e5, 50000)",
    avgHitsPerTrack = 10.0,
    avgCellsPerHit = 35.0,
    avgCellsPerCell = 0.9,
    avgTracksPerCell = 0.7
)

caTracksLowPtParameters = dict(
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
