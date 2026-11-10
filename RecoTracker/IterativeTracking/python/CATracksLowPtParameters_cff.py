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

# theta cuts (per inner layer) tightened to the legacy lowPtQuadStep equivalent: the cut is on
# ~caThetaCut/ptmin, legacy 0.0015/0.35 = 0.0043 vs 0.002-0.003/0.2 = 0.010-0.015 of the template,
# so the template values are scaled by (0.0015/0.35) * 0.2/0.002 = 0.4286, keeping their 2:3 ratio
caThetaCuts = [0.000857] * 4 + [0.001286] * 27

# container sizes, from the occupancies measured vs the number of hits x (all hits, masked included)
# with the mask built from the tracks kept by the highPt track selector (see CATracksIterations_cff)
# (TTbar PU200, 700 events, 164k-313k hits; fillStatistics + CA_STATS, v7 default):
# 2nd degree fits scaled by one factor to cover every event with a 5% margin
# measured: nCells = 1.181e-4*x^2 + 11.63*x - 1.03M (mean 8.3M, max 14.1M, rms 0.19M, scaled x1.116),
# nTracks = 2.98e-5*x^2 - 8.40*x + 0.72M (mean 0.41M, max 1.03M, rms 0.018M, scaled x1.181),
# cells per outer hit <= 47.8, triplets per cell <= 0.43, cell->track links per cell <= 0.23.
# A linear envelope overflowed the doublets on the lightest event (164k hits); the tuple parabola has its
# minimum at 141k hits and overestimates below it. Above ~335k hits the cells leave the 512 MiB allocator bin
# (the fits of the earlier versions, 32.2*x - 1.47M cells, held only with all the tight highPt tracks masked;
# maxNumberOfDoublets above ~21-25M crashes the CA, reached at ~370k hits)
containerSizes = dict(
    maxNumberOfDoublets = "max(1.318e-4*pow(x,2) + 12.97*x - 1.153e6, 100000)",
    maxNumberOfTuples = "max(3.521e-5*pow(x,2) - 9.923*x + 8.51e5, 50000)",
    avgHitsPerTrack = 10.0,
    avgCellsPerHit = 55.0,
    avgCellsPerCell = 0.5,
    avgTracksPerCell = 0.3
)

caTracksLowPtParameters = dict(
    ptmin = ptMin,                              # scales the caThetaCuts
    hardCurvCut = 1. / (0.8 * ptMin * 87.78),   # max triplet curvature (1/cm), 80% of ptMin
    trackQualityCuts = dict(
        minPt = ptMin
    ),
    geometry = dict(
        phiCuts = [lp[2] for lp in layerPairs],
        ptCuts  = [lp[3] for lp in layerPairs],
        caThetaCuts = caThetaCuts
    ),
    **containerSizes
)
