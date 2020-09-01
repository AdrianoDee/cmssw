#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoublets_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoublets_h

#include "RecoPixelVertexing/PixelTriplets/plugins/gpuPixelDoubletsAlgos.h"

#define CONSTANT_VAR __constant__
#define PHASE2DEBUG 1

namespace gpuPixelDoublets {

  constexpr int nPairs = 13 + 2 + 4;
  //constexpr int nPairsPhase2 = 45;
  constexpr int nPairsPhase2 = 6 + 14 + 6 + 14 + 11;

  static_assert(nPairs <= CAConstants::maxNumberOfLayerPairs());

  // start constants
  // clang-format off

  CONSTANT_VAR const uint8_t layerPairs[2 * nPairs] = {
      0, 1, 0, 4, 0, 7,              // BPIX1 (3)
      1, 2, 1, 4, 1, 7,              // BPIX2 (5)
      4, 5, 7, 8,                    // FPIX1 (8)
      2, 3, 2, 4, 2, 7, 5, 6, 8, 9,  // BPIX3 & FPIX2 (13)
      0, 2, 1, 3,                    // Jumping Barrel (15)
      0, 5,
      0, 8,                    // Jumping Forward (BPIX1,FPIX2)
      4, 6, 7, 9                     // Jumping Forward (19)
  };


//   CONSTANT_VAR const uint8_t layerPairsPhase2[2 * nPairsPhase2] = {
//
//       0, 1, 0, 4, 0, 16,             // BPIX1 (3)
//
//       4, 5, 16, 17,                  // 5
//       5, 6, 17, 18,                  // 7
//       6, 7, 18, 19,                  // 9
//       7, 8, 19, 20,                  // 11
//       8, 9, 20, 21,                  // 13
//       9, 10, 21, 22,                 // 15
//       10, 11, 22, 23,                // 17
//
//       1, 2, 1, 4, 1, 16,             // 20
//       2, 3, 2, 4, 2, 16,             // 23
//
//       0, 5, 0, 17,                   // 31
//       1, 5, 1, 17,
//       0, 6, 0, 18,                   // 35
//       1, 6, 1, 18,
//       0, 7, 0, 19,                   // 39
//       1, 7, 1, 19,
//
//       11, 12, 23, 24,                // 25
//       12, 13, 24, 25,                // 27
//       13, 14, 25, 26,                // 29
//       14, 15, 26, 27,                // 31
//
//
//       0, 2, 1, 3                     // 45
// };

  CONSTANT_VAR const uint8_t layerPairsPhase2[2 * nPairsPhase2] = {

    0, 1, 0, 4, 0, 16, //BPIX1 (3)
    1, 2, 1, 4, 1, 16, //BPIX2 (6)
    4 ,5 ,5 ,6 ,6 ,7 ,7 ,8 ,8 ,9 ,9 ,10,10,11, //POS (13)
    16,17,17,18,18,19,19,20,20,21,21,22,22,23, //NEG (20)

    0, 2, 0, 5, 0, 17, // BPIX1 Jump (23)
    1, 3, 1, 5, 1, 17, // BPIX2 Jump (26)

    4, 6, 5, 7, 6, 8, 7, 9, 8, 10,9 ,11,10,12, //POS Jump (33)
    16,18,17,19,18,20,19,21,20,22,21,23,22,24, //NEG Jump (40)

    2, 3, 2, 4, 2, 16, //Barrel Jump (43)
    11,12,12,13,13,14,14,15, //Late POS (47)
    23,24,24,25,25,26,26,27 //Late NEG (51)

};


  constexpr int16_t phi0p05 = 521;  // round(521.52189...) = phi2short(0.05);
  constexpr int16_t phi0p06 = 626;  // round(9+625.82270...) = phi2short(0.06);
  constexpr int16_t phi0p07 = 730;  // round(730.12648...) = phi2short(0.07);
  constexpr int16_t phi0p09 = 960;  // round(730.12648...) = phi2short(0.07);

  CONSTANT_VAR const int16_t phicutsPhase2[nPairsPhase2]{
                                           phi0p05, phi0p05, phi0p05, phi0p05, phi0p05, phi0p05,

                                           phi0p05, phi0p05, phi0p05, phi0p05, phi0p05,
                                           phi0p05, phi0p05, phi0p05, phi0p05, phi0p05,
                                           phi0p05, phi0p05, phi0p05, phi0p05, phi0p05,

                                           phi0p05, phi0p05, phi0p05, phi0p05, phi0p05,
                                           phi0p05, phi0p05, phi0p05, phi0p05, phi0p05,
                                           phi0p05, phi0p05, phi0p05, phi0p05, phi0p05,

                                           phi0p05, phi0p05, phi0p05, phi0p05, phi0p05,
                                           phi0p05, phi0p05, phi0p05, phi0p05, phi0p05,
                                           phi0p05, phi0p05, phi0p05, phi0p05, phi0p05,

                                           };

  CONSTANT_VAR const int16_t phicuts[nPairs]{phi0p05,
                                             phi0p07,
                                             phi0p07,
                                             phi0p05,
                                             phi0p06,
                                             phi0p06,
                                             phi0p05,
                                             phi0p05,
                                             phi0p06,
                                             phi0p06,
                                             phi0p06,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05};
  //   phi0p07, phi0p07, phi0p06,phi0p06, phi0p06,phi0p06};  // relaxed cuts

  CONSTANT_VAR float const minz[nPairs] = {
      -20.,  // _-_-_- 0 1
      0.,  // ____ | 0 4
      -30., // | ____ 0 7

      -22., // _-_-_- 1 2
      10., // _-_-_- | 1 4
      -30., // | _-_-_- 1 7


      -70., -70., // || 4 5 & 7 8 minz below min z for pixel tracker

      -22., 15.,  -30,

      -70., -70., -20., -22., 0, -30., -70., -70.};
  CONSTANT_VAR float const maxz[nPairs] = {
      20., 30., 0., 22., 30., -10., 70., 70., 22., 30., -15., 70., 70., 20., 22., 30., 0., 70., 70.};
  CONSTANT_VAR float const maxr[nPairs] = {
      20., 9., 9., 20., 7., 7., 5., 5., 20., 6., 6., 5., 5., 20., 20., 9., 9., 9., 9.};

    //
    // CONSTANT_VAR float const maxzPhase2[nPairsPhase2] = {
    //   13, 23, -3, 15, 23, -5, 28, 5, 18, 23, -8, 35, 5, 10, 12, 23, -23, 28, 5, 23, -19, 999, 999, 999, 999, 999, 999, 44, 56, 70, 89, 113, 143, 178, 204, 234, 5, 5, 5, 5, 5, 5, 5, 5, 5, 35, 56, 89, 143, 204, 5, 5, 5, 5, 5, 28, 44, 70, 113, 178, 5, 5, 5, 5, 5};
    // //   //, 5, 71, 5, 90, 5, 114, 5, 144, 5, 179, 5, 205};
    // //
    // CONSTANT_VAR float const minzPhase2[nPairsPhase2] = {
    //   -13, 3, -23, -15, 5, -23, -5, -32, -18, 8, -23, -5, -39, -10, -12, 15, -23, -5, -39, -2, -23, -999, -999, -999, -999, -999, -999, -5, -5, -5, -5, -5, -5, -5, -5, -5, -47, -59, -74, -92, -116, -147, -182, -208, -237, -5, -5, -5, -5, -5, -39, -59, -92, -147, -208, -5, -5, -5, -5, -5, -32, -47, -74, -116, -182};
    // // //
    // // //   // -71, -5, -90, -5, -114, -5, -144,
    // // //   // -5, -179, -5, -205, -5};
    // // //
    // CONSTANT_VAR float const maxrPhase2[nPairsPhase2] = {
    //   19.5, 22.5, 24.0, 21.0, 25.5, 28.5, 34.5, 34.5, 21.0, 28.5, 31.5, 34.5, 34.5, 25.5, 27.0, 16.5, 16.5, 34.5, 34.5, 163.5, 16.5, 12.0, 12.0, 6.0, 6.0, 0.0, 0.0, 34.5, 34.5, 34.5, 34.5, 34.5, 48.0, 43.5, 43.5, 43.5, 34.5, 34.5, 34.5, 34.5, 34.5, 48.0, 43.5, 43.5, 43.5, 34.5, 34.5, 34.5, 48.0, 43.5, 34.5, 34.5, 34.5, 48.0, 43.5, 34.5, 34.5, 34.5, 48.0, 43.5, 34.5, 34.5, 34.5, 48.0, 43.5};


    CONSTANT_VAR float const maxzPhase2[nPairsPhase2] = {
      13, 23, -3, 15, 23, -5, 28, 35, 44, 56, 70, 89, 113, 5, 5, 5, 5, 5, 5, 5, 10, 25, 25, 12, 25, 25, 28, 35, 44, 56, 70, 89, 113, 5, 5, 5, 5, 5, 5, 5, 18, 23, -8, 143, 178, 204, 234, 5, 5, 5, 5  };
      // 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0,
      // 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0,
      // 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0,
      // 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0,
      // 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0,
      // 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, //1000.0, 1000.0,
      // 1000.0, 1000.0};
      //, 5, 71, 5, 90, 5, 114, 5, 144, 5, 179, 5, 205};

    CONSTANT_VAR float const minzPhase2[nPairsPhase2] = {
      -13, 3, -23, -15, 5, -23, -5, -5, -5, -5, -5, -5, -5, -32, -39, -47, -59, -74, -92, -116, -10, -25, -25, -12, -25, -25, -5, -5, -5, -5, -5, -5, -5, -32, -39, -47, -59, -74, -92, -116, -18, 8, -23, -5, -5, -5, -5, -147, -182, -208, -237};

      // -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0,
      // -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0,
      // -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0,
      // -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0,
      // -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0,
      // -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, //-1000.0, -1000.0
      // -1000.0, -1000.0};
    //
    //   // -71, -5, -90, -5, -114, -5, -144,
    //   // -5, -179, -5, -205, -5};
    //
    CONSTANT_VAR float const maxrPhase2[nPairsPhase2] = { 20.0, 23.0, 24.0, 22.0, 26.0, 29.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 26.0, 17.0, 17.0, 28.0, 12.0, 12.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 48.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 48.0, 22.0, 29.0, 32.0, 48.0, 44.0, 44.0, 44.0, 48.0, 44.0, 44.0, 44.0};

      //
      // 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0,
      // 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0,
      // 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0,
      // 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0,
      // 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0,
      // 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, //1000.0, 1000.0,
      // 1000.0, 1000.0};

    // , 28.5, 28.5, 28.5,
    // 28.5, 42.0, 42.0, 42.0, 42.0, 37.5, 37.5, 37.5,
    // 37.5};
  // end constants
  // clang-format on

  using CellNeighbors = CAConstants::CellNeighbors;
  using CellTracks = CAConstants::CellTracks;
  using CellNeighborsVector = CAConstants::CellNeighborsVector;
  using CellTracksVector = CAConstants::CellTracksVector;

  __global__ void initDoublets(GPUCACell::OuterHitOfCell* isOuterHitOfCell,
                               int nHits,
                               CellNeighborsVector* cellNeighbors,
                               CellNeighbors* cellNeighborsContainer,
                               CellTracksVector* cellTracks,
                               CellTracks* cellTracksContainer) {
    assert(isOuterHitOfCell);
    int first = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = first; i < nHits; i += gridDim.x * blockDim.x)
      isOuterHitOfCell[i].reset();
  }

  constexpr auto getDoubletsFromHistoMaxBlockSize = 64;  // for both x and y
  constexpr auto getDoubletsFromHistoMinBlocksPerMP = 16;

  __global__
#ifdef __CUDACC__
  __launch_bounds__(getDoubletsFromHistoMaxBlockSize, getDoubletsFromHistoMinBlocksPerMP)
#endif
      void getDoubletsFromHisto(GPUCACell* cells,
                                uint32_t* nCells,
                                CellNeighborsVector* cellNeighbors,
                                CellTracksVector* cellTracks,
                                TrackingRecHit2DSOAView const* __restrict__ hhp,
                                GPUCACell::OuterHitOfCell* isOuterHitOfCell,
                                int nActualPairs,
                                bool ideal_cond,
                                bool doClusterCut,
                                bool doZ0Cut,
                                bool doPtCut,
                                uint32_t maxNumOfDoublets, bool upgrade) {
    auto const& __restrict__ hh = *hhp;
    if(!upgrade)
    {
      doubletsFromHisto(layerPairs,
                        nActualPairs,
                        cells,
                        nCells,
                        cellNeighbors,
                        cellTracks,
                        hh,
                        isOuterHitOfCell,
                        phicuts,
                        minz,
                        maxz,
                        maxr,
                        ideal_cond,
                        doClusterCut,
                        doZ0Cut,
                        doPtCut,
                        maxNumOfDoublets,upgrade);
     }else
     {
       doubletsFromHisto(layerPairsPhase2,
                         nActualPairs,
                         cells,
                         nCells,
                         cellNeighbors,
                         cellTracks,
                         hh,
                         isOuterHitOfCell,
                         phicutsPhase2,
                         minzPhase2,
                         maxzPhase2,
                         maxrPhase2,
                         ideal_cond,
                         false,
                         doZ0Cut,
                         doPtCut,
                         maxNumOfDoublets,upgrade);
     }
  }

}  // namespace gpuPixelDoublets

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDouplets_h
