import FWCore.ParameterSet.Config as cms
from RecoTracker.PixelSeeding.caGeometryESProducer_cfi import caGeometryESProducer as _caGeometryESProducer
_caGeometryESProducer.appendToDataLabel = cms.string("caGeometry")

phase2_tracker.toModify(_caGeometryESProducer,
    pairGraph = [ 0,  1,  0,  4,  0,  16,  ##BPIX1 (3)
      1,  2,  1,  4,  1,  16,  ##BPIX2 (6)
      2,  3,  2,  4,  2,  16,  ##BPIX3 & Forward (9)

      4,  5,  5,  6,  6,  7,  7,  8,  8,  9,  9,  10, 10, 11,  ##POS (16)
      16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23,  ##NEG (23)

      0,  2,  0,  5,  0,  17, 0,  6,  0,  18,  ##BPIX1 Jump (28)
      1,  3,  1,  5,  1,  17, 1,  6,  1,  18,  ##BPIX2 Jump (33)

      11, 12, 12, 13, 13, 14, 14, 15,  ##Late POS (37)
      23, 24, 24, 25, 25, 26, 26, 27,  ##Late NEG (41)

      4,  6,  5,  7,  6,  8,  7,  9,  8,  10, 9,  11, 10, 12,  ##POS Jump (48)
      16, 18, 17, 19, 18, 20, 19, 21, 20, 22, 21, 23, 22, 24 ],

    maxZ = [ 17.0, 22.0,  -4.0,  17.0,  22.0,  -6.0,  18.0,  22.0,  -11.0,  28.0,   35.0,   44.0,   55.0,   70.0,
      87.0, 113.0, -23.0, -30.0, -39.0, -50.0, -65.0, -82.0, -109.0, 17.0,   22.0,   -7.0,   22.0,   -10.0,
      17.0, 22.0,  -9.0,  22.0,  -13.0, 142.0, 177.0, 203.0, 233.0,  -137.0, -173.0, -199.0, -229.0, 28.0,
      35.0, 44.0,  55.0,  70.0,  87.0,  113.0, -23.0, -30.0, -39.0,  -50.0,  -65.0,  -82.0,  -109.0 ],
    
    phiCuts = [ 522, 522, 522, 626, 730,
        730, 626, 730, 730, 522,
        522, 522, 522, 522, 522,
        522, 522, 522, 522, 522,
        522, 522, 522, 522, 522,
        522, 522, 522, 522, 730,
        730, 730, 730, 730, 730,
        730, 730, 730, 730, 730,
        730, 730, 730, 730, 730,
        730, 730, 522, 522, 522,
        522, 522, 522, 522, 522],
        caDCACuts = [0.15 ] + [0.25] * 27,
        caThetaCuts = [0.002,0.002,0.002,0.002] + [0.003] * 24,
        startingPairs = [f for f in range(33)],
        minZ = [-16.0, 4.0,   -22.0, -17.0, 6.0,   -22.0, -18.0, 11.0,  -22.0,  23.0,   30.0,   39.0,   50.0,   65.0,
      82.0,  109.0, -28.0, -35.0, -44.0, -55.0, -70.0, -87.0, -113.0, -16.,   7.0,    -22.0,  11.0,   -22.0,
      -17.0, 9.0,   -22.0, 13.0,  -22.0, 137.0, 173.0, 199.0, 229.0,  -142.0, -177.0, -203.0, -233.0, 23.0,
      30.0,  39.0,  50.0,  65.0,  82.0,  109.0, -28.0, -35.0, -44.0,  -55.0,  -70.0,  -87.0,  -113.0 ],

        maxR = [5.0, 5.0, 5.0, 7.0, 8.0, 8.0,  7.0, 7.0, 7.0, 6.0, 6.0, 6.0, 6.0, 5.0,
                                             6.0, 5.0, 6.0, 6.0, 6.0, 6.0,  5.0, 6.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                             5.0, 8.0, 8.0, 8.0, 8.0, 6.0,  5.0, 5.0, 5.0, 6.0, 5.0, 5.0, 5.0, 9.0,
                                             9.0, 9.0, 8.0, 8.0, 8.0, 11.0, 9.0, 9.0, 9.0, 8.0, 8.0, 8.0, 11.0]
      )

stripNtupletFit.toModify(_caGeometryESProducer,
    pairGraph = list(getattr(_caGeometryESProducer, "pairGraph")) + [0, 11],
    maxZ = list(getattr(_caGeometryESProducer, "maxZ")) + [1000.0],
    caDCACuts = list(getattr(_caGeometryESProducer, "caDCACuts")) + [0.3],
    caThetaCuts = list(getattr(_caGeometryESProducer, "caThetaCuts")) + [0.25],
    minZ =  list(getattr(_caGeometryESProducer, "minZ")) + [-1000.0],
    maxR =  list(getattr(_caGeometryESProducer, "maxR")) + [10000.0],
    phiCuts = list(getattr(_caGeometryESProducer, "phiCuts")) + [1000]
)