import FWCore.ParameterSet.Config as cms

def customizePixelTracksForTriplets(process):
   
   from HLTrigger.Configuration.common import producers_by_type, esproducers_by_type
   names = ['CAHitNtupletAlpakaPhase1@alpaka','CAHitNtupletAlpakaPhase2@alpaka']
   ca_es_producer = 'CAGeometryESProducer@alpaka'
   
   for name in names:
      producers = producers_by_type(process, name)
      has_producer = False
      for producer in producers:
         producer.minHitsPerNtuplet = 3
         has_producer = True

      # assuming we only have either Phase2 or Phase1 producer
      if has_producer and name == 'CAHitNtupletAlpakaPhase1@alpaka':
         for esproducer in esproducers_by_type(process,ca_es_producer):

            esproducer.pairGraph = [  0, 1, 0, 4, 0,
               7, 1, 2, 1, 4,
               1, 7, 4, 5, 7,
               8, 2, 3, 2, 4,
               2, 7, 5, 6, 8,
               9, 0, 2, 1, 3,
               0, 5, 0, 8, 
               4, 6, 7, 9 ]
            esproducer.startingPairs = [i for i in range(8)] + [13, 14, 15, 16, 17, 18, 19]
            esproducer.phiCuts = [522, 730, 730, 522, 626,
               626, 522, 522, 626, 626,
               626, 522, 522, 522, 522,
               522, 522, 522, 522]
            esproducer.minZ = [-20., 0., -30., -22., 10., 
               -30., -70., -70., -22., 15., 
               -30, -70., -70., -20., -22., 
               0, -30., -70., -70.]
            esproducer.maxZ = [20., 30., 0., 22., 30., 
               -10., 70., 70., 22., 30., 
               -15., 70., 70., 20., 22., 
               30., 0., 70., 70.]
            esproducer.maxR = [20., 9., 9., 20., 7., 
               7., 5., 5., 20., 6., 
               6., 5., 5., 20., 20., 
               9., 9., 9., 9.]

         break

      elif has_producer and name == 'CAHitNtupletAlpakaPhase2@alpaka':
         for esproducer in esproducers_by_type(process,ca_es_producer):

            esproducer.pairGraph = [0,  1,  0,  4,  0,  16,  ## BPIX1 (3)
                  1,  2,  1,  4,  1,  16,  ## BPIX2 (6)
                  2,  3,  2,  4,  2,  16,  ## BPIX3 & Forward (9)

                  4,  5,  5,  6,  6,  7,  7,  8,  8,  9,  9,  10, 10, 11,  ## POS (16)
                  16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23,  ## NEG (23)

                  0,  2,  0,  5,  0,  17, 0,  6,  0,  18,  ## BPIX1 Jump (28)
                  1,  3,  1,  5,  1,  17, 1,  6,  1,  18,  ## BPIX2 Jump (33)

                  11, 12, 12, 13, 13, 14, 14, 15,  ## Late POS (37)
                  23, 24, 24, 25, 25, 26, 26, 27,  ## Late NEG (41)

                  4,  6,  5,  7,  6,  8,  7,  9,  8,  10, 9,  11, 10, 12,  ## POS Jump (48)
                  16, 18, 17, 19, 18, 20, 19, 21, 20, 22, 21, 23, 22, 24],  ## NEG Jump (55)]

            esproducer.startingPairs = [f for f in range(32)] # this is actually the same for quads (at the moemnt)
            esproducer.phiCuts = 
            [ 522, 522, 522, 626, 730,
                           730, 626, 730, 730, 522,
                           522, 522, 522, 522, 522,
                           522, 522, 522, 522, 522,
                           522, 522, 522, 522, 522,
                           522, 522, 522, 522, 730,
                           730, 730, 730, 730, 730,
                           730, 730, 730, 730, 730,
                           730, 730, 730, 730, 730,
                           730, 730, 522, 522, 522,
                           522, 522, 522, 522, 522]
            esproducer.minZ = [ -16.0, 4.0,   -22.0, -17.0, 6.0,   -22.0, -18.0, 11.0,  -22.0,  23.0,   30.0,   39.0,   50.0,   65.0,
      82.0,  109.0, -28.0, -35.0, -44.0, -55.0, -70.0, -87.0, -113.0, -16.,   7.0,    -22.0,  11.0,   -22.0,
      -17.0, 9.0,   -22.0, 13.0,  -22.0, 137.0, 173.0, 199.0, 229.0,  -142.0, -177.0, -203.0, -233.0, 23.0,
      30.0,  39.0,  50.0,  65.0,  82.0,  109.0, -28.0, -35.0, -44.0,  -55.0,  -70.0,  -87.0,  -113.0]
            esproducer.maxZ = [17.0, 22.0,  -4.0,  17.0,  22.0,  -6.0,  18.0,  22.0,  -11.0,  28.0,   35.0,   44.0,   55.0,   70.0,
      87.0, 113.0, -23.0, -30.0, -39.0, -50.0, -65.0, -82.0, -109.0, 17.0,   22.0,   -7.0,   22.0,   -10.0,
      17.0, 22.0,  -9.0,  22.0,  -13.0, 142.0, 177.0, 203.0, 233.0,  -137.0, -173.0, -199.0, -229.0, 28.0,
      35.0, 44.0,  55.0,  70.0,  87.0,  113.0, -23.0, -30.0, -39.0,  -50.0,  -65.0,  -82.0,  -109.0]
            esproducer.maxR = [5.0, 5.0, 5.0, 7.0, 8.0, 8.0,  7.0, 7.0, 7.0, 6.0, 6.0, 6.0, 6.0, 5.0,
                                             6.0, 5.0, 6.0, 6.0, 6.0, 6.0,  5.0, 6.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                             5.0, 8.0, 8.0, 8.0, 8.0, 6.0,  5.0, 5.0, 5.0, 6.0, 5.0, 5.0, 5.0, 9.0,
                                             9.0, 9.0, 8.0, 8.0, 8.0, 11.0, 9.0, 9.0, 9.0, 8.0, 8.0, 8.0, 11.0]
         
         break
   
   return process
