#ifndef Geometry_CommonTopologies_SimplePixelTopology_h
#define Geometry_CommonTopologies_SimplePixelTopology_h

#include <array>
#include <cstdint>
#include <type_traits>
#include "FWCore/Utilities/interface/HostDeviceConstant.h"

namespace pixelTopology {

  // TODO
  // Once CUDA is dropped this could be wrapped in #ifdef CA_TRIPLETS_HOLE
  // see DataFormats/TrackingRecHitSoa/interface/TrackingRecHitSoA.h

  template <typename TrackerTraits>
  struct AverageGeometryT {
    //
    float ladderZ[TrackerTraits::numberOfLaddersInBarrel];
    float ladderX[TrackerTraits::numberOfLaddersInBarrel];
    float ladderY[TrackerTraits::numberOfLaddersInBarrel];
    float ladderR[TrackerTraits::numberOfLaddersInBarrel];
    float ladderMinZ[TrackerTraits::numberOfLaddersInBarrel];
    float ladderMaxZ[TrackerTraits::numberOfLaddersInBarrel];
    float endCapZ[2];  // just for pos and neg Layer1
  };

  constexpr int16_t phi0p05 = 522;  // round(521.52189...) = phi2short(0.05);
  constexpr int16_t phi0p06 = 626;  // round(625.82270...) = phi2short(0.06);
  constexpr int16_t phi0p07 = 730;  // round(730.12648...) = phi2short(0.07);
  constexpr int16_t phi0p09 = 900;

  constexpr uint16_t last_barrel_layer = 3;  // this is common between all the topologies

  template <class Function, std::size_t... Indices>
  constexpr auto map_to_array_helper(Function f, std::index_sequence<Indices...>)
      -> std::array<std::invoke_result_t<Function, std::size_t>, sizeof...(Indices)> {
    return {{f(Indices)...}};
  }

  template <int N, class Function>
  constexpr auto map_to_array(Function f) -> std::array<std::invoke_result_t<Function, std::size_t>, N> {
    return map_to_array_helper(f, std::make_index_sequence<N>{});
  }

  template <typename TrackerTraits>
  constexpr uint16_t findMaxModuleStride() {
    bool go = true;
    int n = 2;
    while (go) {
      for (uint8_t i = 1; i < TrackerTraits::numberOfLayers + 1; ++i) {
        if (TrackerTraits::layerStart[i] % n != 0) {
          go = false;
          break;
        }
      }
      if (!go)
        break;
      n *= 2;
    }
    return n / 2;
  }

  template <typename TrackerTraits>
  constexpr uint16_t maxModuleStride = findMaxModuleStride<TrackerTraits>();

  template <typename TrackerTraits>
  constexpr uint8_t findLayer(uint32_t detId, uint8_t sl = 0) {
    for (uint8_t i = sl; i < TrackerTraits::numberOfLayers + 1; ++i)
      if (detId < TrackerTraits::layerStart[i + 1])
        return i;
    return TrackerTraits::numberOfLayers + 1;
  }

  template <typename TrackerTraits>
  constexpr uint8_t findLayerFromCompact(uint32_t detId) {
    detId *= maxModuleStride<TrackerTraits>;
    for (uint8_t i = 0; i < TrackerTraits::numberOfLayers + 1; ++i)
      if (detId < TrackerTraits::layerStart[i + 1])
        return i;
    return TrackerTraits::numberOfLayers + 1;
  }

  template <typename TrackerTraits>
  constexpr uint32_t layerIndexSize = TrackerTraits::numberOfModules / maxModuleStride<TrackerTraits>;

  template <typename TrackerTraits>
#ifdef __CUDA_ARCH__
  __device__
#endif
      constexpr std::array<uint8_t, layerIndexSize<TrackerTraits>>
          layer = map_to_array<layerIndexSize<TrackerTraits>>(findLayerFromCompact<TrackerTraits>);

  template <typename TrackerTraits>
  constexpr uint8_t getLayer(uint32_t detId) {
    return layer<TrackerTraits>[detId / maxModuleStride<TrackerTraits>];
  }

  template <typename TrackerTraits>
  constexpr bool validateLayerIndex() {
    bool res = true;
    for (auto i = 0U; i < TrackerTraits::numberOfModules; ++i) {
      auto j = i / maxModuleStride<TrackerTraits>;
      res &= (layer<TrackerTraits>[j] < TrackerTraits::numberOfLayers);
      res &= (i >= TrackerTraits::layerStart[layer<TrackerTraits>[j]]);
      res &= (i < TrackerTraits::layerStart[layer<TrackerTraits>[j] + 1]);
    }
    return res;
  }

  template <typename TrackerTraits>
#ifdef __CUDA_ARCH__
  __device__
#endif
      constexpr inline uint32_t layerStart(uint32_t i) {
    return TrackerTraits::layerStart[i];
  }

  constexpr inline uint16_t divu52(uint16_t n) {
    n = n >> 2;
    uint16_t q = (n >> 1) + (n >> 4);
    q = q + (q >> 4) + (q >> 5);
    q = q >> 3;
    uint16_t r = n - q * 13;
    return q + ((r + 3) >> 4);
  }
}  // namespace pixelTopology

namespace phase1PixelTopology {

  using pixelTopology::phi0p05;
  using pixelTopology::phi0p06;
  using pixelTopology::phi0p07;

  constexpr uint32_t numberOfLayers = 10;
  constexpr int nPairs = 13 + 2 + 4;
  constexpr uint16_t numberOfModules = 1856;
  constexpr int nStartingPairs = 3;  // number of layer pairs to start Ntuplet-building from

  constexpr uint32_t maxNumClustersPerModules = 1024;

  constexpr uint32_t max_ladder_bpx0 = 12;
  constexpr uint32_t first_ladder_bpx0 = 0;
  constexpr float module_length_bpx0 = 6.7f;
  constexpr float module_tolerance_bpx0 = 0.4f;  // projection to cylinder is inaccurate on BPIX1
  constexpr uint32_t max_ladder_bpx4 = 64;
  constexpr uint32_t first_ladder_bpx4 = 84;
  constexpr float radius_even_ladder = 15.815f;
  constexpr float radius_odd_ladder = 16.146f;
  constexpr float module_length_bpx4 = 6.7f;
  constexpr float module_tolerance_bpx4 = 0.2f;
  constexpr float barrel_z_length = 26.f;
  constexpr float forward_z_begin = 32.f;

  HOST_DEVICE_CONSTANT uint8_t layerPairs[2 * nPairs] = {
      0, 1, 0, 4, 0, 7,              // BPIX1 (3)
      1, 2, 1, 4, 1, 7,              // BPIX2 (6)
      4, 5, 7, 8,                    // FPIX1 (8)
      2, 3, 2, 4, 2, 7, 5, 6, 8, 9,  // BPIX3 & FPIX2 (13)
      0, 2, 1, 3,                    // Jumping Barrel (15)
      0, 5, 0, 8,                    // Jumping Forward (BPIX1,FPIX2)
      4, 6, 7, 9                     // Jumping Forward (19)
  };

  HOST_DEVICE_CONSTANT uint8_t startingPairs[nStartingPairs] = {0, 1, 2};

  HOST_DEVICE_CONSTANT int16_t phicuts[nPairs]{phi0p05,
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
  HOST_DEVICE_CONSTANT float minInner[nPairs] = {
      -20., 0., -30., -22., 10., -30., -70., -70., -22., 15., -30, -70., -70., -20., -22., 0, -30., -70., -70.};
  HOST_DEVICE_CONSTANT float maxInner[nPairs] = {
      20., 30., 0., 22., 30., -10., 70., 70., 22., 30., -15., 70., 70., 20., 22., 30., 0., 70., 70.};
  HOST_DEVICE_CONSTANT float minOuter[nPairs] = {
      -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100};
  HOST_DEVICE_CONSTANT float maxOuter[nPairs] = {
      100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
  HOST_DEVICE_CONSTANT float minDZ[nPairs] = {
      -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100};
  HOST_DEVICE_CONSTANT float maxDZ[nPairs] = {
      100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
  HOST_DEVICE_CONSTANT float ptCuts[nPairs] = {
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
  HOST_DEVICE_CONSTANT float z0Cuts[nPairs] = {
      12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5};
  HOST_DEVICE_CONSTANT float maxDR[nPairs] = {
      20., 9., 9., 20., 7., 7., 5., 5., 20., 6., 6., 5., 5., 20., 20., 9., 9., 9., 9.};

  HOST_DEVICE_CONSTANT float dcaCuts[numberOfLayers] = {0.15, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25};

  HOST_DEVICE_CONSTANT float thetaCuts[numberOfLayers] = {
      0.002, 0.002, 0.002, 0.002, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003};

  // -------------------------------------------------------------------------------------------------------
  // Deprecated arrays only used in the CUDA version (values have no meaning in alpaka):

  // The layerStart array is only used in the CUDA version (which supports only the non-extended CA).
  // In the alpaka version of the CA the array is built in globalBeginRun from the geometry directly
  // and the values here become irrelevant.
  static constexpr uint32_t layerStart[numberOfLayers + 1] = {0,
                                                              96,
                                                              320,
                                                              672,  // barrel
                                                              1184,
                                                              1296,
                                                              1408,  // positive endcap
                                                              1520,
                                                              1632,
                                                              1744,  // negative endcap
                                                              numberOfModules};
  HOST_DEVICE_CONSTANT float minz[nPairs] = {
      -20., 0., -30., -22., 10., -30., -70., -70., -22., 15., -30, -70., -70., -20., -22., 0, -30., -70., -70.};
  HOST_DEVICE_CONSTANT float maxz[nPairs] = {
      20., 30., 0., 22., 30., -10., 70., 70., 22., 30., -15., 70., 70., 20., 22., 30., 0., 70., 70.};
  HOST_DEVICE_CONSTANT float maxr[nPairs] = {
      20., 9., 9., 20., 7., 7., 5., 5., 20., 6., 6., 5., 5., 20., 20., 9., 9., 9., 9.};
}  // namespace phase1PixelTopology

namespace phase2PixelTopology {

  // The parameters set here include the extension of the CA to the first 3 barrel layers of the OT.
  // This incorporates therefore the general set for Phase-2 and builds the basis for both CA configurations (with or without OT extension).
  // The layer pairs are ordered in such a way that the OT extended pairs are at the end of the arrays. So one can get the non-extended config by
  // chopping off the last elements.
  // The actual implementation of the splitting in these two configs is done below by having two separate structs in the pixelTopology namespace:
  //   - pixelTopology::Phase2    -> no OT extension
  //   - pixelTopology::Phase2OT  -> with OT extension

  constexpr uint32_t nLayersPix = 28;                      // pixel layers
  constexpr uint32_t nLayersOT = 3;                        // considered OT layers
  constexpr uint32_t nLayersTot = nLayersPix + nLayersOT;  // total number of layers for extended CA

  constexpr int nPairsPix = 57;                    // pixel only layer pairs
  constexpr int nPairsOT = 16;                     // layer pairs with OT layers
  constexpr int nPairsTot = nPairsPix + nPairsOT;  // total number of layer pairs for extended CA

  constexpr uint16_t nModulesPix = 4000;                      // pixel modules
  constexpr uint16_t nModulesOT = 2872;                       // considered OT modules (barrel only)
  constexpr uint16_t nModulesTot = nModulesPix + nModulesOT;  // total number of modules for extended CA

  // Module counts for Phase2OTStubs (with barrel + both disk sides)
  // OT barrel: 7288 modules, OT backward disks: 2956, OT forward disks: 2956
  constexpr uint32_t nModulesOTBarrel = 7288;
  constexpr uint32_t nModulesOTBackward = 2956;
  constexpr uint32_t nModulesOTForward = 2956;
  constexpr uint32_t nModulesOTStubs = nModulesOTBarrel + nModulesOTBackward + nModulesOTForward;  // 13200
  constexpr uint32_t nModulesTotStubs = nModulesPix + nModulesOTStubs;                            // 17200

  constexpr int nStartingPairs = 24;  // number of layer pairs to start Ntuplet-building from

  constexpr uint16_t numberOfModules = nModulesPix;
  constexpr uint32_t maxNumClustersPerModules = 1024;

  HOST_DEVICE_CONSTANT uint8_t layerPairs[2 * nPairsTot] = {
      0,  1,  0,  2,  0,  4,  0,  5,  0,  16, 0,  17,          // starting on BPIX1 (6)
      1,  2,  1,  3,  1,  4,  1,  5,  1,  16, 1,  17,          // starting on BPIX2 (12)
      2,  3,  2,  4,  2,  16,                                  // starting on BPIX3 (15)
      4,  5,  4,  6,  5,  6,  5,  7,  6,  7,  6,  8,  7,  8,   // forward endcap (22)
      7,  9,  8,  9,  8,  10, 9,  10, 9,  11, 10, 11, 10, 12,  // forward endcap (29)
      11, 12, 11, 13, 11, 14, 11, 15, 12, 13, 13, 14, 14, 15,  // forward endcap (36)
      16, 17, 16, 18, 17, 18, 17, 19, 18, 19, 18, 20, 19, 20,  // backward endcap (43)
      19, 21, 20, 21, 20, 22, 21, 22, 21, 23, 22, 23, 22, 24,  // backward endcap (50)
      23, 24, 23, 25, 23, 26, 23, 27, 24, 25, 25, 26, 26, 27,  // backward endcap (57)

      2,  28, 2,  28, 2,  28, 3,  28,          // barrel to OT (61)
      4,  28, 5,  28, 6,  28, 7,  28, 8,  28,  // forward endcap to OT (66)
      16, 28, 17, 28, 18, 28, 19, 28, 20, 28,  // backward endcap to OT (71)
      28, 29, 29, 30                           // OT to OT (73)
  };

  HOST_DEVICE_CONSTANT uint8_t startingPairs[nStartingPairs] = {0,  1,  2,  3,  4,  5,  6,  8,  10, 12, 15, 17,
                                                                19, 21, 23, 25, 27, 36, 38, 40, 42, 44, 46, 48};

  HOST_DEVICE_CONSTANT int16_t phicuts[nPairsTot]{
      350,  600,  450,  522,  450,  522,       // BPIX1
      400,  650,  500,  730,  500,  730,       // BPIX2
      350,  400,  400,                         // BPIX3
      300,  522,  300,  522,  250,  522, 250,  // forward endcap
      522,  250,  522,  300,  522,  240, 650,  // forward endcap
      300,  200,  220,  250,  250,  250, 250,  // forward endcap
      300,  522,  300,  522,  250,  522, 250,  // backward endcap
      522,  250,  522,  300,  522,  240, 650,  // backward endcap
      300,  200,  220,  250,  250,  250, 250,  // backward endcap

      1200, 1200, 1200, 1000,        // barrel to OT
      1000, 1000, 1000, 1000, 850,   // forward endcap to OT
      1000, 1000, 1000, 1000, 1000,  // backward endcap to OT
      1100, 1250                     // OT to OT
  };

  HOST_DEVICE_CONSTANT float minInner[nPairsTot] = {
      -17,   -14,  4,      7,   -10000, -10000,      // BPIX1
      -17,   -15,  6,      9,   -10000, -10000,      // BPIX2
      -18,   11,   -10000,                           // BPIX3
      0,     0,    0,      0,   0,      0,      0,   // forward endcap
      0,     0,    0,      0,   0,      0,      12,  // forward endcap
      0,     0,    0,      0,   0,      0,      0,   // forward endcap
      0,     0,    0,      0,   0,      0,      0,   // backward endcap
      0,     0,    0,      0,   0,      0,      12,  // backward endcap
      0,     0,    0,      0,   0,      0,      0,   // backward endcap

      -10,   -20,  10,     -20,     // barrel to O
      11,    11,   11,     11,  0,  // forward end
      11,    11,   11,     11,  0,  // backward en
      -1200, -1200                  // OT to OT
  };

  HOST_DEVICE_CONSTANT float maxInner[nPairsTot] = {
      17,    14,    10000, 10000, -4,    -7,      // BPIX1
      17,    15,    10000, 10000, -6,    -9,      // BPIX2
      18,    10000, -11,                          // BPIX3
      14,    14,    13,    13,    13,    13, 13,  // forward endcap
      13,    13,    13,    13,    13,    13, 16,  // forward endcap
      16,    6,     4,     6,     22,    22, 22,  // forward endcap
      14,    14,    13,    13,    13,    13, 13,  // backward endcap
      13,    13,    13,    13,    13,    13, 16,  // backward endcap
      16,    6,     4,     6,     22,    22, 22,  // backward endcap

      10,    -10,   20,    20,            // barrel to OT
      10000, 10000, 10000, 10000, 10000,  // forward endcap to OT
      10000, 10000, 10000, 10000, 10000,  // backward endcap to OT
      1200,  1200                         // OT to OT
  };

  HOST_DEVICE_CONSTANT float minOuter[nPairsTot] = {
      -10000, -10000, 0,   0,    0,      0,      // BPIX1
      -10000, -10000, 6,   6,    6,      6,      // BPIX2
      -10000, 11,     11,                        // BPIX3
      3,      3,      3,   3,    3,      3, 3,   // forward endcap
      3,      3,      3,   4,    4,      3, 20,  // forward endcap
      6,      0,      0,   0,    7,      7, 7,   // forward endcap
      3,      3,      3,   3,    3,      3, 3,   // backward endcap
      3,      3,      3,   4,    4,      3, 20,  // backward endcap
      6,      0,      0,   0,    7,      7, 7,   // backward endcap

      -30,    -50,    25,  -45,           // barrel to OT
      30,     40,     55,  70,   80,      // forward endcap to OT
      -57,    -70,    -95, -110, -10000,  // backward endcap to OT
      -10000, -10000                      // OT to OT
  };

  HOST_DEVICE_CONSTANT float maxOuter[nPairsTot] = {
      10000, 10000, 10,    10000, 10,    10000,         // BPIX1
      10000, 10000, 10000, 10000, 10000, 10000,         // BPIX2
      10000, 10000, 10000,                              // BPIX3
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // forward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // forward endcap
      21,    7,     7,     10000, 10000, 10000, 10000,  // forward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // backward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // backward endcap
      21,    7,     7,     10000, 10000, 10000, 10000,  // backward endcap

      30,    -25,   50,    45,            // barrel to OT
      57,    80,    95,    110,   10000,  // forward endcap to OT
      -30,   -40,   -55,   -70,   -80,    // backward endcap to OT
      10000, 10000                        // OT to OT
  };

  HOST_DEVICE_CONSTANT float maxDR[nPairsTot] = {
      5.0,     10.0,    8.0,     5.0,     8.0,  5.0,         // BPIX1
      7.0,     10.0,    8.0,     10.0,    8.0,  10.0,        // BPIX2
      7.0,     7.0,     7.0,                                 // BPIX3
      4.5,     9.0,     4.5,     9.0,     4.5,  9.0,  4.5,   // forward endcap
      8.0,     4.0,     8.0,     4.5,     8.0,  4.0,  10.0,  // forward endcap
      5.0,     3.0,     3.0,     4.0,     4.0,  4.0,  3.5,   // forward endcap
      4.5,     9.0,     4.5,     9.0,     4.5,  9.0,  4.5,   // backward endcap
      8.0,     4.0,     8.0,     4.5,     8.0,  4.0,  10.0,  // backward endcap
      5.0,     3.0,     3.0,     4.0,     4.0,  4.0,  3.5,   // backward endcap

      10000.0, 10000.0, 10000.0, 10000.0,        // barrel to OT
      16.0,    16.0,    16.0,    16.0,    14.0,  // forward endcap to OT
      16.0,    16.0,    16.0,    16.0,    14.0,  // backward endcap to OT
      10000.0, 10000.0                           // OT to OT
  };

  HOST_DEVICE_CONSTANT float minDZ[nPairsTot] = {
      -16.0,  -16.0,  0.0,    0.0,    -25.0,  -25.0,           // BPIX1
      -13.0,  -15.0,  0.0,    0.0,    -19.0,  -21.0,           // BPIX2
      -9.0,   0.0,    -13.0,                                   // BPIX3
      -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // forward endcap
      -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // forward endcap
      -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // forward endcap
      -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // backward endcap
      -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // backward endcap
      -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // backward endcap

      -15.0,  -35.0,  10.0,   -22.0,          // barrel to OT
      5.0,    -10.0,  5.0,    15.0,   25.0,   // forward endcap to OT
      -32.5,  -50.0,  -50.0,  -70.0,  -70.0,  // backward endcap to OT
      -50.0,  -40.0                           // OT to OT
  };

  HOST_DEVICE_CONSTANT float maxDZ[nPairsTot] = {
      16.0,  16.0,  25.0,  25.0,  0.0,   0.0,           // BPIX1
      13.0,  15.0,  19.0,  21.0,  0.0,   0.0,           // BPIX2
      9.0,   13.0,  0.0,                                // BPIX3
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // forward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // forward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // forward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // backward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // backward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // backward endcap

      15.0,  -10.0, 35.0,  22.0,          // barrel to OT
      32.5,  50.0,  50.0,  70.0,  70.0,   // forward endcap to OT
      -5.0,  -10.0, -5.0,  -15.0, -25.0,  // backward endcap to OT
      50.0,  40.0                         // OT to OT
  };

  HOST_DEVICE_CONSTANT float ptCuts[nPairsTot] = {
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85,        // BPIX1
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85,        // BPIX2
      0.85, 0.85, 0.85,                          // BPIX3
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // forward endcap
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // forward endcap
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // forward endcap
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // backward endcap
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // backward endcap
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // backward endcap

      2.00, 0.85, 0.85, 0.85,        // barrel to OT
      0.85, 0.85, 0.85, 0.85, 0.85,  // forward endcap to OT
      0.85, 0.85, 0.85, 0.85, 0.85,  // backward endcap to OT
      0.85, 0.85                     // OT to OT
  };

    HOST_DEVICE_CONSTANT float z0Cuts[nPairsTot] = {
      12.5, 12.5, 12.5, 12.5, 12.5, 12.5,        // BPIX1
      12.5, 12.5, 12.5, 12.5, 12.5, 12.5,        // BPIX2
      12.5, 12.5, 12.5,                          // BPIX3
      12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5,  // forward endcap
      12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5,  // forward endcap
      12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5,  // forward endcap
      12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5,  // backward endcap
      12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5,  // backward endcap
      12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5,  // backward endcap

      12.5, 12.5, 12.5, 12.5,        // barrel to OT
      12.5, 12.5, 12.5, 12.5, 12.5,  // forward endcap to OT
      12.5, 12.5, 12.5, 12.5, 12.5,  // backward endcap to OT
      12.5, 12.5                     // OT to OT
  };

  HOST_DEVICE_CONSTANT float dcaCuts[nLayersTot] = {
      0.15,  //BPix1
      0.25, 0.20, 0.20, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
      0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,  // Pixel layers
      0.10, 0.10, 0.10                                                               // OT layers
  };

  HOST_DEVICE_CONSTANT float thetaCuts[nLayersTot] = {
      0.002, 0.002, 0.002, 0.002,  // BPix
      0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,
      0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,  // Pixel layers
      0.003, 0.003, 0.003                                                                  // OT layers
  };

  // Extended arrays for Phase2OTStubs (44 layers: 28 pixel + 6 OT barrel + 10 OT disks)
  constexpr uint32_t nLayersPhase2OTStubs = 44;
  HOST_DEVICE_CONSTANT float dcaCutsPhase2OTStubs[nLayersPhase2OTStubs] = {
      0.15,  //BPix1
      0.25, 0.20, 0.20, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
      0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,  // Pixel layers (28 total)
      0.10, 0.10, 0.10, 0.10, 0.10, 0.10,  // OT barrel layers (6)
      0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10  // OT disk layers (10)
  };

  HOST_DEVICE_CONSTANT float thetaCutsPhase2OTStubs[nLayersPhase2OTStubs] = {
      0.002, 0.002, 0.002, 0.002,  // BPix
      0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,
      0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,  // Pixel layers (28 total)
      0.003, 0.003, 0.003, 0.003, 0.003, 0.003,  // OT barrel layers (6)
      0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003  // OT disk layers (10)
  };

  // =====================================================================================
  // Extended layer pairs for Phase2OTStubs (44 layers)
  // Layer numbering:
  //   0-27:  Pixel layers (same as Phase2)
  //   28-33: OT barrel layers 1-6
  //   34-38: OT backward disk layers 1-5 (z < 0)
  //   39-43: OT forward disk layers 1-5 (z > 0)
  // =====================================================================================

  // Total pairs: 73 (existing) + 21 (OT-OT) + 8 (pixel-to-OT-disk) = 102
  // OT-OT: 3 (barrel chain) + 5 (barrel→fwd disk) + 5 (barrel→bwd disk) + 4 (fwd chain) + 4 (bwd chain)
  // Pixel-to-OT-disk: 4 (fwd pixel D5/D6 → OT fwd D1/D2) + 4 (bwd pixel D5/D6 → OT bwd D1/D2)
  constexpr int nPairsPhase2OTStubs = nPairsTot + 29;
  constexpr int nStartingPairsPhase2OTStubs = nStartingPairs;  // Keep same starting pairs

  HOST_DEVICE_CONSTANT uint8_t layerPairsPhase2OTStubs[2 * nPairsPhase2OTStubs] = {
      // === Existing 73 pairs (indices 0-72) - same as Phase2OT ===
      0,  1,  0,  2,  0,  4,  0,  5,  0,  16, 0,  17,          // starting on BPIX1 (6)
      1,  2,  1,  3,  1,  4,  1,  5,  1,  16, 1,  17,          // starting on BPIX2 (12)
      2,  3,  2,  4,  2,  16,                                  // starting on BPIX3 (15)
      4,  5,  4,  6,  5,  6,  5,  7,  6,  7,  6,  8,  7,  8,   // forward endcap (22)
      7,  9,  8,  9,  8,  10, 9,  10, 9,  11, 10, 11, 10, 12,  // forward endcap (29)
      11, 12, 11, 13, 11, 14, 11, 15, 12, 13, 13, 14, 14, 15,  // forward endcap (36)
      16, 17, 16, 18, 17, 18, 17, 19, 18, 19, 18, 20, 19, 20,  // backward endcap (43)
      19, 21, 20, 21, 20, 22, 21, 22, 21, 23, 22, 23, 22, 24,  // backward endcap (50)
      23, 24, 23, 25, 23, 26, 23, 27, 24, 25, 25, 26, 26, 27,  // backward endcap (57)

      2,  28, 2,  28, 2,  28, 3,  28,          // barrel to OT (61)
      4,  28, 5,  28, 6,  28, 7,  28, 8,  28,  // forward endcap to OT (66)
      16, 28, 17, 28, 18, 28, 19, 28, 20, 28,  // backward endcap to OT (71)
      28, 29, 29, 30,                          // OT barrel to OT barrel (73)

      // === New pairs for extended OT (indices 73-93) ===
      // OT barrel chain continuation (3 pairs, indices 73-75)
      30, 31, 31, 32, 32, 33,

      // OT barrel to forward disks (5 pairs, indices 76-80)
      30, 39, 31, 39, 32, 39, 33, 39,          // barrel layers 3-6 to forward disk 1
      33, 40,                                  // barrel layer 6 to forward disk 2

      // OT barrel to backward disks (5 pairs, indices 81-85)
      30, 34, 31, 34, 32, 34, 33, 34,          // barrel layers 3-6 to backward disk 1
      33, 35,                                  // barrel layer 6 to backward disk 2

      // Forward disk chain (4 pairs, indices 86-89)
      39, 40, 40, 41, 41, 42, 42, 43,          // forward disks 1→2→3→4→5

      // Backward disk chain (4 pairs, indices 90-93)
      34, 35, 35, 36, 36, 37, 37, 38,          // backward disks 1→2→3→4→5

      // Pixel forward endcap disks 5-6 to OT forward disks 1-2 (4 pairs, indices 94-97)
      8, 39, 8, 40, 9, 39, 9, 40,

      // Pixel backward endcap disks 5-6 to OT backward disks 1-2 (4 pairs, indices 98-101)
      20, 34, 20, 35, 21, 34, 21, 35
  };

  // Phi cuts for Phase2OTStubs layer pairs
  HOST_DEVICE_CONSTANT int16_t phicutsPhase2OTStubs[nPairsPhase2OTStubs] = {
      // Existing 73 pairs (same as Phase2OT)
      350,  600,  450,  522,  450,  522,       // BPIX1
      400,  650,  500,  730,  500,  730,       // BPIX2
      350,  400,  400,                         // BPIX3
      300,  522,  300,  522,  250,  522, 250,  // forward endcap
      522,  250,  522,  300,  522,  240, 650,  // forward endcap
      300,  200,  220,  250,  250,  250, 250,  // forward endcap
      300,  522,  300,  522,  250,  522, 250,  // backward endcap
      522,  250,  522,  300,  522,  240, 650,  // backward endcap
      300,  200,  220,  250,  250,  250, 250,  // backward endcap
      1200, 1200, 1200, 1000,        // barrel to OT
      1000, 1000, 1000, 1000, 850,   // forward endcap to OT
      1000, 1000, 1000, 1000, 1000,  // backward endcap to OT
      1100, 1250,                    // OT to OT (existing)
      // New pairs
      1250, 1250, 1250,              // OT barrel chain (30→31, 31→32, 32→33)
      1500, 1500, 1500, 1500, 1500,  // barrel to forward disks
      1500, 1500, 1500, 1500, 1500,  // barrel to backward disks
      1500, 1500, 1500, 1500,        // forward disk chain
      1500, 1500, 1500, 1500,        // backward disk chain
      // Pixel-to-OT-disk pairs
      2000, 2000, 2000, 2000,        // fwd pixel D5/D6 to OT fwd D1/D2
      2000, 2000, 2000, 2000         // bwd pixel D5/D6 to OT bwd D1/D2
  };

  // Min inner coordinate cuts for Phase2OTStubs layer pairs
  HOST_DEVICE_CONSTANT float minInnerPhase2OTStubs[nPairsPhase2OTStubs] = {
      // Existing 73 pairs
      -17,   -14,  4,      7,   -10000, -10000,      // BPIX1
      -17,   -15,  6,      9,   -10000, -10000,      // BPIX2
      -18,   11,   -10000,                           // BPIX3
      0,     0,    0,      0,   0,      0,      0,   // forward endcap
      0,     0,    0,      0,   0,      0,      12,  // forward endcap
      0,     0,    0,      0,   0,      0,      0,   // forward endcap
      0,     0,    0,      0,   0,      0,      0,   // backward endcap
      0,     0,    0,      0,   0,      0,      12,  // backward endcap
      0,     0,    0,      0,   0,      0,      0,   // backward endcap
      -10,   -20,  10,     -20,     // barrel to OT
      11,    11,   11,     11,  0,   // forward endcap to OT
      11,    11,   11,     11,  0,   // backward endcap to OT
      -1200, -1200,                  // OT to OT (existing)
      // New pairs - use relaxed cuts for OT
      -1200, -1200, -1200,           // OT barrel chain
      -1200, -1200, -1200, -1200, -1200,  // barrel to forward disks
      -1200, -1200, -1200, -1200, -1200,  // barrel to backward disks
      20,     20,     20,     20,          // forward disk chain (r-based, OT disk inner ~23cm)
      20,     20,     20,     20,         // backward disk chain (r-based, OT disk inner ~23cm)
      // Pixel-to-OT-disk pairs (r-based, relaxed)
      0,      0,      0,      0,         // fwd pixel D5/D6 to OT fwd D1/D2
      0,      0,      0,      0          // bwd pixel D5/D6 to OT bwd D1/D2
  };

  // Max inner coordinate cuts for Phase2OTStubs layer pairs
  HOST_DEVICE_CONSTANT float maxInnerPhase2OTStubs[nPairsPhase2OTStubs] = {
      // Existing 73 pairs
      17,    14,    10000, 10000, -4,    -7,      // BPIX1
      17,    15,    10000, 10000, -6,    -9,      // BPIX2
      18,    10000, -11,                          // BPIX3
      14,    14,    13,    13,    13,    13, 13,  // forward endcap
      13,    13,    13,    13,    13,    13, 16,  // forward endcap
      16,    6,     4,     6,     22,    22, 22,  // forward endcap
      14,    14,    13,    13,    13,    13, 13,  // backward endcap
      13,    13,    13,    13,    13,    13, 16,  // backward endcap
      16,    6,     4,     6,     22,    22, 22,  // backward endcap
      10,    -10,   20,    20,            // barrel to OT
      10000, 10000, 10000, 10000, 10000,  // forward endcap to OT
      10000, 10000, 10000, 10000, 10000,  // backward endcap to OT
      1200,  1200,                        // OT to OT (existing)
      // New pairs
      1200,  1200,  1200,                 // OT barrel chain
      1200,  1200,  1200,  1200,  1200,   // barrel to forward disks
      1200,  1200,  1200,  1200,  1200,   // barrel to backward disks
      115,   115,   115,   115,            // forward disk chain (r-based, OT disk outer ~110cm)
      115,   115,   115,   115,           // backward disk chain (r-based, OT disk outer ~110cm)
      // Pixel-to-OT-disk pairs (r-based, relaxed)
      10000, 10000, 10000, 10000,        // fwd pixel D5/D6 to OT fwd D1/D2
      10000, 10000, 10000, 10000         // bwd pixel D5/D6 to OT bwd D1/D2
  };

  // Min outer coordinate cuts for Phase2OTStubs layer pairs
  HOST_DEVICE_CONSTANT float minOuterPhase2OTStubs[nPairsPhase2OTStubs] = {
      // Existing 73 pairs
      -10000, -10000, 0,   0,    0,      0,      // BPIX1
      -10000, -10000, 6,   6,    6,      6,      // BPIX2
      -10000, 11,     11,                        // BPIX3
      3,      3,      3,   3,    3,      3, 3,   // forward endcap
      3,      3,      3,   4,    4,      3, 20,  // forward endcap
      6,      0,      0,   0,    7,      7, 7,   // forward endcap
      3,      3,      3,   3,    3,      3, 3,   // backward endcap
      3,      3,      3,   4,    4,      3, 20,  // backward endcap
      6,      0,      0,   0,    7,      7, 7,   // backward endcap
      -30,    -50,    25,  -45,           // barrel to OT
      30,     40,     55,  70,   80,      // forward endcap to OT
      -57,    -70,    -95, -110, -10000,  // backward endcap to OT
      -10000, -10000,                     // OT to OT (existing)
      // New pairs
      -10000, -10000, -10000,             // OT barrel chain (z cut - relaxed)
      20,     20,     20,     20,    20,  // barrel to forward disks (r cut - OT disk inner radius ~23cm)
      20,     20,     20,     20,    20,  // barrel to backward disks (r cut - OT disk inner radius ~23cm)
      20,     20,     20,     20,         // forward disk chain (r cut)
      20,     20,     20,     20,        // backward disk chain (r cut)
      // Pixel-to-OT-disk pairs (r cut - OT disk inner radius ~23cm)
      20,     20,     20,     20,        // fwd pixel D5/D6 to OT fwd D1/D2
      20,     20,     20,     20         // bwd pixel D5/D6 to OT bwd D1/D2
  };

  // Max outer coordinate cuts for Phase2OTStubs layer pairs
  HOST_DEVICE_CONSTANT float maxOuterPhase2OTStubs[nPairsPhase2OTStubs] = {
      // Existing 73 pairs
      10000, 10000, 10,    10000, 10,    10000,         // BPIX1
      10000, 10000, 10000, 10000, 10000, 10000,         // BPIX2
      10000, 10000, 10000,                              // BPIX3
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // forward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // forward endcap
      21,    7,     7,     10000, 10000, 10000, 10000,  // forward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // backward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // backward endcap
      21,    7,     7,     10000, 10000, 10000, 10000,  // backward endcap
      30,    -25,   50,    45,            // barrel to OT
      57,    80,    95,    110,   10000,  // forward endcap to OT
      -30,   -40,   -55,   -70,   -80,    // backward endcap to OT
      10000, 10000,                       // OT to OT (existing)
      // New pairs
      10000, 10000, 10000,                // OT barrel chain (z cut - relaxed)
      120,   120,   120,   120,   120,    // barrel to forward disks (r cut - OT disk outer radius ~110cm)
      120,   120,   120,   120,   120,    // barrel to backward disks (r cut - OT disk outer radius ~110cm)
      120,   120,   120,   120,           // forward disk chain (r cut)
      120,   120,   120,   120,          // backward disk chain (r cut)
      // Pixel-to-OT-disk pairs (r cut - relaxed)
      10000, 10000, 10000, 10000,        // fwd pixel D5/D6 to OT fwd D1/D2
      10000, 10000, 10000, 10000         // bwd pixel D5/D6 to OT bwd D1/D2
  };

  // Max dR cuts for Phase2OTStubs layer pairs
  HOST_DEVICE_CONSTANT float maxDRPhase2OTStubs[nPairsPhase2OTStubs] = {
      // Existing 73 pairs
      5.0,     10.0,    8.0,     5.0,     8.0,  5.0,         // BPIX1
      7.0,     10.0,    8.0,     10.0,    8.0,  10.0,        // BPIX2
      7.0,     7.0,     7.0,                                 // BPIX3
      4.5,     9.0,     4.5,     9.0,     4.5,  9.0,  4.5,   // forward endcap
      8.0,     4.0,     8.0,     4.5,     8.0,  4.0,  10.0,  // forward endcap
      5.0,     3.0,     3.0,     4.0,     4.0,  4.0,  3.5,   // forward endcap
      4.5,     9.0,     4.5,     9.0,     4.5,  9.0,  4.5,   // backward endcap
      8.0,     4.0,     8.0,     4.5,     8.0,  4.0,  10.0,  // backward endcap
      5.0,     3.0,     3.0,     4.0,     4.0,  4.0,  3.5,   // backward endcap
      10000.0, 10000.0, 10000.0, 10000.0,        // barrel to OT
      16.0,    16.0,    16.0,    16.0,    14.0,  // forward endcap to OT
      16.0,    16.0,    16.0,    16.0,    14.0,  // backward endcap to OT
      10000.0, 10000.0,                          // OT to OT (existing)
      // New pairs - OT layers have larger spacing
      10000.0, 10000.0, 10000.0,                 // OT barrel chain
      10000.0, 10000.0, 10000.0, 10000.0, 10000.0,  // barrel to forward disks
      10000.0, 10000.0, 10000.0, 10000.0, 10000.0,  // barrel to backward disks
      60.0,    60.0,    60.0,    60.0,              // forward disk chain (full disk r-extent with margin)
      60.0,    60.0,    60.0,    60.0,             // backward disk chain (full disk r-extent with margin)
      // Pixel-to-OT-disk pairs (relaxed)
      10000.0, 10000.0, 10000.0, 10000.0,          // fwd pixel D5/D6 to OT fwd D1/D2
      10000.0, 10000.0, 10000.0, 10000.0           // bwd pixel D5/D6 to OT bwd D1/D2
  };

  // Min dZ cuts for Phase2OTStubs layer pairs
  HOST_DEVICE_CONSTANT float minDZPhase2OTStubs[nPairsPhase2OTStubs] = {
      // Existing 73 pairs
      -16.0,  -16.0,  0.0,    0.0,    -25.0,  -25.0,           // BPIX1
      -13.0,  -15.0,  0.0,    0.0,    -19.0,  -21.0,           // BPIX2
      -9.0,   0.0,    -13.0,                                   // BPIX3
      -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // forward endcap
      -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // forward endcap
      -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // forward endcap
      -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // backward endcap
      -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // backward endcap
      -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // backward endcap
      -15.0,  -35.0,  10.0,   -22.0,          // barrel to OT
      5.0,    -10.0,  5.0,    15.0,   25.0,   // forward endcap to OT
      -32.5,  -50.0,  -50.0,  -70.0,  -70.0,  // backward endcap to OT
      -50.0,  -40.0,                          // OT to OT (existing)
      // New pairs
      -50.0,  -50.0,  -50.0,                  // OT barrel chain
      0.0,    0.0,    0.0,    0.0,    0.0,    // barrel to forward disks (z > 0)
      -10000, -10000, -10000, -10000, -10000, // barrel to backward disks (z < 0)
      10.0,   20.0,   20.0,   25.0,           // forward disk chain (per-pair dz, D1D2/D2D3/D3D4/D4D5)
      -35.0,  -55.0,  -55.0,  -70.0,        // backward disk chain (per-pair dz, D1D2/D2D3/D3D4/D4D5)
      // Pixel-to-OT-disk pairs (relaxed)
      -10000, -10000, -10000, -10000,        // fwd pixel D5/D6 to OT fwd D1/D2
      -10000, -10000, -10000, -10000         // bwd pixel D5/D6 to OT bwd D1/D2
  };

  // Max dZ cuts for Phase2OTStubs layer pairs
  HOST_DEVICE_CONSTANT float maxDZPhase2OTStubs[nPairsPhase2OTStubs] = {
      // Existing 73 pairs
      16.0,  16.0,  25.0,  25.0,  0.0,   0.0,           // BPIX1
      13.0,  15.0,  19.0,  21.0,  0.0,   0.0,           // BPIX2
      9.0,   13.0,  0.0,                                // BPIX3
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // forward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // forward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // forward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // backward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // backward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // backward endcap
      15.0,  -10.0, 35.0,  22.0,          // barrel to OT
      32.5,  50.0,  50.0,  70.0,  70.0,   // forward endcap to OT
      -5.0,  -10.0, -5.0,  -15.0, -25.0,  // backward endcap to OT
      50.0,  40.0,                        // OT to OT (existing)
      // New pairs
      50.0,  50.0,  50.0,                 // OT barrel chain
      10000, 10000, 10000, 10000, 10000,  // barrel to forward disks
      0.0,   0.0,   0.0,   0.0,   0.0,    // barrel to backward disks (z < 0)
      35.0,  55.0,  55.0,  70.0,           // forward disk chain (per-pair dz, D1D2/D2D3/D3D4/D4D5)
      -10.0, -20.0, -20.0, -25.0,        // backward disk chain (per-pair dz, D1D2/D2D3/D3D4/D4D5)
      // Pixel-to-OT-disk pairs (relaxed)
      10000, 10000, 10000, 10000,        // fwd pixel D5/D6 to OT fwd D1/D2
      10000, 10000, 10000, 10000         // bwd pixel D5/D6 to OT bwd D1/D2
  };

  // pT cuts for Phase2OTStubs layer pairs
  HOST_DEVICE_CONSTANT float ptCutsPhase2OTStubs[nPairsPhase2OTStubs] = {
      // Existing 73 pairs
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85,        // BPIX1
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85,        // BPIX2
      0.85, 0.85, 0.85,                          // BPIX3
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // forward endcap
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // forward endcap
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // forward endcap
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // backward endcap
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // backward endcap
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // backward endcap
      2.00, 0.85, 0.85, 0.85,        // barrel to OT
      0.85, 0.85, 0.85, 0.85, 0.85,  // forward endcap to OT
      0.85, 0.85, 0.85, 0.85, 0.85,  // backward endcap to OT
      0.85, 0.85,                    // OT to OT (existing)
      // New pairs - stubs already have pT filtering from bend
      2.0,  2.0,  2.0,               // OT barrel chain
      2.0,  2.0,  2.0,  2.0,  2.0,   // barrel to forward disks
      2.0,  2.0,  2.0,  2.0,  2.0,   // barrel to backward disks
      2.0,  2.0,  2.0,  2.0,         // forward disk chain
      2.0,  2.0,  2.0,  2.0,        // backward disk chain
      // Pixel-to-OT-disk pairs
      0.85, 0.85, 0.85, 0.85,       // fwd pixel D5/D6 to OT fwd D1/D2
      0.85, 0.85, 0.85, 0.85        // bwd pixel D5/D6 to OT bwd D1/D2
  };

  // Stub-stub pairwise sigma cuts for Phase2OTStubs layer pairs
  // Negative value = cut disabled for this pair
  // Barrel-barrel (flat-flat): kappa-corrected significance cut
  // Forward-forward: raw dPhiDr (= dPhiDz for endcap modules) significance cut
  // Mixed barrel-forward or pixel-involved: disabled (-1.0)
  HOST_DEVICE_CONSTANT float stubSigmaCutsPhase2OTStubs[nPairsPhase2OTStubs] = {
      // Existing 73 pairs
      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,        // BPIX1 (pixel-pixel)
      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,        // BPIX2 (pixel-pixel)
      -1.0, -1.0, -1.0,                           // BPIX3 (pixel-pixel)
      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  // forward pixel endcap
      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  // forward pixel endcap
      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  // forward pixel endcap
      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  // backward pixel endcap
      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  // backward pixel endcap
      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  // backward pixel endcap
      -1.0, -1.0, -1.0, -1.0,        // pixel barrel to OT barrel
      -1.0, -1.0, -1.0, -1.0, -1.0,  // pixel forward endcap to OT barrel
      -1.0, -1.0, -1.0, -1.0, -1.0,  // pixel backward endcap to OT barrel
       5.0,  5.0,                     // OT barrel 28->29, 29->30
      // New pairs (indices 73-93)
       5.0,  5.0,  5.0,               // OT barrel chain (30->31, 31->32, 32->33)
      -1.0, -1.0, -1.0, -1.0, -1.0,  // barrel to forward disks (mixed: disabled)
      -1.0, -1.0, -1.0, -1.0, -1.0,  // barrel to backward disks (mixed: disabled)
      10.0,  5.0,  5.0,  5.0,         // forward disk chain (D1->D2=10sigma, rest=5sigma)
      10.0,  5.0,  5.0,  5.0,        // backward disk chain (D1->D2=10sigma, rest=5sigma)
      // Pixel-to-OT-disk pairs (inner is pixel, not stub -> disabled)
       0.0,  0.0,  0.0,  0.0,        // fwd pixel D5/D6 to OT fwd D1/D2
       0.0,  0.0,  0.0,  0.0         // bwd pixel D5/D6 to OT bwd D1/D2
  };

  // isBarrel flags for Phase2OTStubs layers
  HOST_DEVICE_CONSTANT bool isBarrelPhase2OTStubs[nLayersPhase2OTStubs] = {
      true, true, true, true,     // BPix layers 0-3
      false, false, false, false, false, false, false, false, false, false, false, false,  // Forward pixel (4-15)
      false, false, false, false, false, false, false, false, false, false, false, false,  // Backward pixel (16-27)
      true, true, true, true, true, true,    // OT barrel layers 28-33
      false, false, false, false, false,     // OT backward disks 34-38
      false, false, false, false, false      // OT forward disks 39-43
  };

  // Geometric-vs-stub kappa significance cut for Phase2OTStubs layers
  // Negative = disabled (pixel layers have no stubs). Per-layer, using the middle layer for a triplet.
  HOST_DEVICE_CONSTANT float geomKappaSigmaCutsPhase2OTStubs[nLayersPhase2OTStubs] = {
      // 28 pixel layers: -1.0 (disabled, no stubs)
      -1.0, -1.0, -1.0, -1.0,                                                              // BPix layers 0-3
      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,            // Forward pixel (4-15)
      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,            // Backward pixel (16-27)
      // 6 OT barrel layers: 5.0 (placeholder)
      5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
      // 10 OT endcap disks: 5.0 (placeholder)
      5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0
  };

  // -------------------------------------------------------------------------------------------------------
  // Deprecated arrays only used in the CUDA version (values have no meaning in alpaka or anywhere else):

  // The layerStart array is only used in the CUDA version (which supports only the non-extended CA).
  // In the alpaka version of the CA the array is built in globalBeginRun from the geometry directly
  // and the values here become irrelevant.
  HOST_DEVICE_CONSTANT uint32_t layerStart[nLayersPix + 1] = {
      0,    216,  432,  612,                                                                 // Barrel
      864,  972,  1080, 1188, 1296, 1404, 1512, 1620, 1728, 1904, 2080, 2256,                // Fp
      2432, 2540, 2648, 2756, 2864, 2972, 3080, 3188, 3296, 3472, 3648, 3824, nModulesPix};  // Np
  HOST_DEVICE_CONSTANT float minz[nPairsTot] = {
      -16.0, 4.0,   -22.0, -17.0, 6.0,   -22.0, -18.0, 11.0,  -22.0,  23.0,   30.0,   39.0,   50.0,   65.0,
      82.0,  109.0, -28.0, -35.0, -44.0, -55.0, -70.0, -87.0, -113.0, -16.,   7.0,    -22.0,  11.0,   -22.0,
      -17.0, 9.0,   -22.0, 13.0,  -22.0, 137.0, 173.0, 199.0, 229.0,  -142.0, -177.0, -203.0, -233.0, 23.0,
      30.0,  39.0,  50.0,  65.0,  82.0,  109.0, -28.0, -35.0, -44.0,  -55.0,  -70.0,  -87.0,  -113.0};

  HOST_DEVICE_CONSTANT float maxz[nPairsTot] = {

      17.0, 22.0,  -4.0,  17.0,  22.0,  -6.0,  18.0,  22.0,  -11.0,  28.0,   35.0,   44.0,   55.0,   70.0,
      87.0, 113.0, -23.0, -30.0, -39.0, -50.0, -65.0, -82.0, -109.0, 17.0,   22.0,   -7.0,   22.0,   -10.0,
      17.0, 22.0,  -9.0,  22.0,  -13.0, 142.0, 177.0, 203.0, 233.0,  -137.0, -173.0, -199.0, -229.0, 28.0,
      35.0, 44.0,  55.0,  70.0,  87.0,  113.0, -23.0, -30.0, -39.0,  -50.0,  -65.0,  -82.0,  -109.0};

  HOST_DEVICE_CONSTANT float maxr[nPairsTot] = {5.0, 5.0, 5.0, 7.0, 8.0, 8.0,  7.0, 7.0, 7.0, 6.0, 6.0, 6.0, 6.0, 5.0,
                                                6.0, 5.0, 6.0, 6.0, 6.0, 6.0,  5.0, 6.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                                5.0, 8.0, 8.0, 8.0, 8.0, 6.0,  5.0, 5.0, 5.0, 6.0, 5.0, 5.0, 5.0, 9.0,
                                                9.0, 9.0, 8.0, 8.0, 8.0, 11.0, 9.0, 9.0, 9.0, 8.0, 8.0, 8.0, 11.0};
}  // namespace phase2PixelTopology

namespace phase1HIonPixelTopology {
  // Storing here the needed constants different w.r.t. pp Phase1 topology.
  // All the other defined by inheritance in the HIon topology struct.
  using pixelTopology::phi0p09;

  constexpr uint32_t maxNumClustersPerModules = 2048;

  HOST_DEVICE_CONSTANT int16_t phicuts[phase1PixelTopology::nPairs]{phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09};

  HOST_DEVICE_CONSTANT float dcaCuts[phase1PixelTopology::numberOfLayers] = {
      0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};

  HOST_DEVICE_CONSTANT float thetaCuts[phase1PixelTopology::numberOfLayers] = {
      0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002};

}  // namespace phase1HIonPixelTopology

namespace pixelTopology {

  struct Phase2 {
    // types
    using hindex_type = uint32_t;  // FIXME from siPixelRecHitsHeterogeneousProduct
    using tindex_type = uint32_t;  // for tuples
    using cindex_type = uint32_t;  // for cells

    static constexpr uint32_t maxCellNeighbors = 64;
    static constexpr uint32_t maxCellTracks = 302;
    static constexpr uint32_t maxLayersPerTrack = 13;
    static constexpr uint32_t maxFishboneHitsPerTrack = 8;
    static constexpr uint32_t maxHitsOnTrack = maxLayersPerTrack + maxFishboneHitsPerTrack;
    static constexpr uint32_t maxHitsOnTrackForFullFit = 6;
    static constexpr uint32_t avgHitsPerTrack = 7;
    static constexpr uint32_t maxCellsPerHit = 256;
    static constexpr uint32_t avgTracksPerHit = 10;
    static constexpr uint32_t maxNumberOfTuples = 60 * 1024;
    // this is well above thanks to maxNumberOfTuples
    static constexpr uint32_t maxHitsForContainers = avgHitsPerTrack * maxNumberOfTuples;
    static constexpr uint32_t maxNumberOfDoublets = 6 * 512 * 1024;
    static constexpr uint32_t maxNumOfActiveDoublets = maxNumberOfDoublets / 8;
    static constexpr uint32_t maxNumberOfQuadruplets = maxNumberOfTuples;
    static constexpr uint32_t numberOfLayers = phase2PixelTopology::nLayersPix;
    static constexpr float avgCellsPerHit = 12.;
    static constexpr float avgCellsPerCell = 0.151;
    static constexpr float avgTracksPerCell = 0.04;

    static constexpr uint32_t maxSizeCluster = 2047;

    static constexpr uint32_t getDoubletsFromHistoMaxBlockSize = 128;  // for both x and y
    static constexpr uint32_t getDoubletsFromHistoMinBlocksPerMP = 16;

    static constexpr uint16_t last_bpix1_detIndex = 216;
    static constexpr uint16_t last_bpix2_detIndex = 432;
    static constexpr uint16_t last_barrel_detIndex = 864;

    static constexpr uint32_t maxPixInModule = 6000;
    static constexpr uint32_t maxPixInModuleForMorphing = 0;
    static constexpr uint32_t maxIterClustering = 16;

    static constexpr uint32_t maxNumClustersPerModules = phase2PixelTopology::maxNumClustersPerModules;
    static constexpr uint32_t maxHitsInModule = phase2PixelTopology::maxNumClustersPerModules;

    static constexpr float moduleLength = 4.345f;
    static constexpr float endcapCorrection = 0.0f;

    static constexpr float xerr_barrel_l1_def = 0.00035f;
    static constexpr float yerr_barrel_l1_def = 0.00125f;
    static constexpr float xerr_barrel_ln_def = 0.00035f;
    static constexpr float yerr_barrel_ln_def = 0.00125f;
    static constexpr float xerr_endcap_def = 0.00060f;
    static constexpr float yerr_endcap_def = 0.00180f;

    static constexpr float bigPixXCorrection = 0.0f;
    static constexpr float bigPixYCorrection = 0.0f;

    static constexpr float dzdrFact = 8 * 0.0285 / 0.015;  // from dz/dr to "DY"

    static constexpr int nPairsMinimal = 33;
    static constexpr int nPairsFarForwards = nPairsMinimal + 8;    // include barrel "jumping" layer pairs
    static constexpr int nPairs = phase2PixelTopology::nPairsPix;  // include far forward layer pairs
    static constexpr int nPairsForQuadruplets = nPairs;
    static constexpr int nStartingPairs = phase2PixelTopology::nStartingPairs;

    static constexpr uint16_t numberOfModules = phase2PixelTopology::nModulesPix;

    // 1000 bins < 1024 bins (10 bits) must be:
    // - < 32*32 (warpSize*warpSize for block prefix scan for CUDA)
    // - > number of columns (y) in any module. This is due to the fact
    //     that in pixel clustering we give for granted that in each
    //     bin we only have the pixel belonging to the same column.
    //     See RecoLocalTracker/SiPixelClusterizer/plugins/alpaka/PixelClustering.h#L325-L347
    static constexpr uint16_t clusterBinning = 1000;
    static constexpr uint16_t clusterBits = 10;

    static constexpr uint16_t numberOfModulesInBarrel = 756;
    static constexpr uint16_t numberOfModulesInLadder = 9;
    static constexpr uint16_t numberOfLaddersInBarrel = numberOfModulesInBarrel / numberOfModulesInLadder;

    static constexpr uint16_t firstEndcapPos = 4;
    static constexpr uint16_t firstEndcapNeg = 16;

    static constexpr int16_t xOffset = -1e4;  // not used actually, to suppress static analyzer warnings

    static constexpr char const *nameModifier = "Phase2";

    static constexpr uint32_t const *layerStart = phase2PixelTopology::layerStart;  // only for CUDA

    static constexpr inline bool isBigPixX(uint16_t px) { return false; }
    static constexpr inline bool isBigPixY(uint16_t py) { return false; }

    static constexpr inline uint16_t localX(uint16_t px) { return px; }
    static constexpr inline uint16_t localY(uint16_t py) { return py; }

    // ----------------------------------------
    //  CA cut / geometry parameters
    // ----------------------------------------
    static constexpr uint8_t const *layerPairs = phase2PixelTopology::layerPairs;
    static constexpr uint8_t const *startingPairs = phase2PixelTopology::startingPairs;
    // scalar parameters (doublet building)
    static constexpr int minYsizeB1 = 20;
    static constexpr int minYsizeB2 = 18;
    static constexpr int maxDYsize12 = 12;
    static constexpr int maxDYsize = 10;
    static constexpr int maxDYPred = 24;
    static constexpr float cellZ0Cut = 12.5;
    // vector parameters (doublet building)
    static constexpr float const *minInner = phase2PixelTopology::minInner;
    static constexpr float const *maxInner = phase2PixelTopology::maxInner;
    static constexpr float const *minOuter = phase2PixelTopology::minOuter;
    static constexpr float const *maxOuter = phase2PixelTopology::maxOuter;
    static constexpr float const *maxDR = phase2PixelTopology::maxDR;
    static constexpr float const *minDZ = phase2PixelTopology::minDZ;
    static constexpr float const *maxDZ = phase2PixelTopology::maxDZ;
    static constexpr int16_t const *phicuts = phase2PixelTopology::phicuts;
    static constexpr float const *ptCuts = phase2PixelTopology::ptCuts;
    // scalar parameters (doublet linking)
    // p [GeV/c] = B [T] * R [m] * 0.3 (factor from conversion from J to GeV and q = e = 1.6 * 10e-19 C)
    // 87 cm/GeV = 1/(3.8T * 0.3)
    static constexpr float hardCurvCut = 0.01425;  // corresponds to 800 MeV in 3.8T.
    // vector parameters (doublet linking)
    static constexpr float const *thetaCuts = phase2PixelTopology::thetaCuts;
    static constexpr float const *dcaCuts = phase2PixelTopology::dcaCuts;
    // Deprecated arrays only used in the CUDA version
    static constexpr float const *minz = phase2PixelTopology::minz;
    static constexpr float const *maxz = phase2PixelTopology::maxz;
    static constexpr float const *maxr = phase2PixelTopology::maxr;
    // ----------------------------------------
  };

  struct Phase2OT : public Phase2 {
    static constexpr int nPairs = phase2PixelTopology::nPairsTot;
    static constexpr int nPairsForQuadruplets = nPairs;
    static constexpr uint32_t numberOfLayers = phase2PixelTopology::nLayersTot;    // pixel layers  + OT barrel layers
    static constexpr uint16_t numberOfModules = phase2PixelTopology::nModulesTot;  // pixel modules + OT barrel modules

    static constexpr uint32_t maxNumberOfTuples = 2 * 60 * 1024;
    // this is well above thanks to maxNumberOfTuples
    static constexpr uint32_t maxHitsForContainers = avgHitsPerTrack * maxNumberOfTuples;
    static constexpr uint32_t maxNumberOfDoublets = 12 * 512 * 1024;
    static constexpr uint32_t maxNumOfActiveDoublets = maxNumberOfDoublets / 8;
    static constexpr uint32_t maxNumberOfQuadruplets = maxNumberOfTuples;
    static constexpr float avgCellsPerHit = 17.;
    static constexpr float avgCellsPerCell = 0.5;
    static constexpr float avgTracksPerCell = 0.09;
  };

  struct Phase2OTStubs : public Phase2OT {
    // Extended Phase-2 configuration using OT stubs (barrel + disks) instead of P-side hits
    // CA layers: 28 pixel + 6 OT barrel + 10 OT disks (5 per side) = 44 total
    static constexpr uint32_t numberOfLayers = 44;
    // Total modules: 4000 pixel + 13200 OT (7288 barrel + 2956 backward + 2956 forward)
    static constexpr uint32_t numberOfModules = phase2PixelTopology::nModulesTotStubs;

    // Override number of layer pairs to include extended OT
    static constexpr int nPairs = phase2PixelTopology::nPairsPhase2OTStubs;  // 102 pairs
    static constexpr int nPairsForQuadruplets = nPairs;
    static constexpr int nStartingPairs = phase2PixelTopology::nStartingPairsPhase2OTStubs;

    // Override layer pair arrays to use extended versions
    static constexpr uint8_t const *layerPairs = phase2PixelTopology::layerPairsPhase2OTStubs;

    // Override cut arrays to use extended versions (per-layer cuts)
    static constexpr float const *thetaCuts = phase2PixelTopology::thetaCutsPhase2OTStubs;
    static constexpr float const *dcaCuts = phase2PixelTopology::dcaCutsPhase2OTStubs;

    // Override cut arrays to use extended versions (per-pair cuts)
    static constexpr int16_t const *phicuts = phase2PixelTopology::phicutsPhase2OTStubs;
    static constexpr float const *minInner = phase2PixelTopology::minInnerPhase2OTStubs;
    static constexpr float const *maxInner = phase2PixelTopology::maxInnerPhase2OTStubs;
    static constexpr float const *minOuter = phase2PixelTopology::minOuterPhase2OTStubs;
    static constexpr float const *maxOuter = phase2PixelTopology::maxOuterPhase2OTStubs;
    static constexpr float const *maxDR = phase2PixelTopology::maxDRPhase2OTStubs;
    static constexpr float const *minDZ = phase2PixelTopology::minDZPhase2OTStubs;
    static constexpr float const *maxDZ = phase2PixelTopology::maxDZPhase2OTStubs;
    static constexpr float const *ptCuts = phase2PixelTopology::ptCutsPhase2OTStubs;
    static constexpr float const *geomKappaSigmaCuts = phase2PixelTopology::geomKappaSigmaCutsPhase2OTStubs;

    // Increased capacities for extended tracking (sized from 1k ttbar PU200 peak occupancy)
    static constexpr uint32_t maxNumberOfDoublets = 8 * 1024 * 1024;   // ~13% margin over 7.4M peak
    static constexpr uint32_t maxNumOfActiveDoublets = maxNumberOfDoublets / 8;
    static constexpr uint32_t maxNumberOfTuples = 512 * 1024;
    static constexpr uint32_t avgHitsPerTrack = 10;                    // pixel+OT tracks avg ~7 hits, max ~14
    static constexpr uint32_t maxHitsForContainers = avgHitsPerTrack * maxNumberOfTuples;
    static constexpr uint32_t maxNumberOfQuadruplets = maxNumberOfTuples;
    static constexpr float avgCellsPerHit = 23.;                       // ~12% margin over peak ratio 20.6
    static constexpr float avgCellsPerCell = 0.3;                      // ~32% margin over peak ratio 0.23
    static constexpr float avgTracksPerCell = 0.2;                     // ~46% margin over peak ratio 0.14

    // Override maxHitsOnTrackForFullFit for stubs
    static constexpr uint32_t maxHitsOnTrackForFullFit = 10;
  };

  struct Phase1 {
    // types
    using hindex_type = uint32_t;  // FIXME from siPixelRecHitsHeterogeneousProduct
    using tindex_type = uint16_t;  // for tuples
    using cindex_type = uint32_t;  // for cells

    static constexpr uint32_t maxCellNeighbors = 36;
    static constexpr uint32_t maxCellTracks = 48;
    static constexpr uint32_t maxLayersPerTrack = 7;
    static constexpr uint32_t maxFishboneHitsPerTrack = 2;
    static constexpr uint32_t maxHitsOnTrack = maxLayersPerTrack + maxFishboneHitsPerTrack;
    static constexpr uint32_t maxHitsOnTrackForFullFit = 6;
    static constexpr uint32_t avgHitsPerTrack = 5;
    static constexpr uint32_t maxCellsPerHit = 256;
    static constexpr uint32_t avgTracksPerHit = 6;
    static constexpr uint32_t maxNumberOfTuples = 32 * 1024;
    static constexpr uint32_t maxHitsForContainers = avgHitsPerTrack * maxNumberOfTuples;
    static constexpr uint32_t maxNumberOfDoublets = 512 * 1024;
    static constexpr uint32_t maxNumOfActiveDoublets = maxNumberOfDoublets / 8;
    static constexpr uint32_t maxNumberOfQuadruplets = maxNumberOfTuples;
    static constexpr uint32_t numberOfLayers = 10;
    static constexpr float avgCellsPerHit = 25.;
    static constexpr float avgCellsPerCell = 2.;
    static constexpr float avgTracksPerCell = 1.;

    static constexpr uint32_t maxSizeCluster = 1023;

    static constexpr uint32_t getDoubletsFromHistoMaxBlockSize = 64;  // for both x and y
    static constexpr uint32_t getDoubletsFromHistoMinBlocksPerMP = 16;

    static constexpr uint16_t last_bpix1_detIndex = 96;
    static constexpr uint16_t last_bpix2_detIndex = 320;
    static constexpr uint16_t last_barrel_detIndex = 1184;

    static constexpr uint32_t maxPixInModule = 6000;
    static constexpr uint32_t maxPixInModuleForMorphing = maxPixInModule * 2 / 5;
    static constexpr uint32_t maxIterClustering = 24;

    static constexpr uint32_t maxNumClustersPerModules = phase1PixelTopology::maxNumClustersPerModules;
    static constexpr uint32_t maxHitsInModule = phase1PixelTopology::maxNumClustersPerModules;

    static constexpr float moduleLength = 6.7f;
    static constexpr float endcapCorrection = 1.5f;

    static constexpr float xerr_barrel_l1_def = 0.00200f;
    static constexpr float yerr_barrel_l1_def = 0.00210f;
    static constexpr float xerr_barrel_ln_def = 0.00200f;
    static constexpr float yerr_barrel_ln_def = 0.00210f;
    static constexpr float xerr_endcap_def = 0.0020f;
    static constexpr float yerr_endcap_def = 0.00210f;

    static constexpr float bigPixXCorrection = 1.0f;
    static constexpr float bigPixYCorrection = 8.0f;

    static constexpr float dzdrFact = 8 * 0.0285 / 0.015;  // from dz/dr to "DY"

    static constexpr int nPairsForQuadruplets = 13;                     // quadruplets require hits in all layers
    static constexpr int nPairsForTriplets = nPairsForQuadruplets + 2;  // include barrel "jumping" layer pairs
    static constexpr int nPairs = nPairsForTriplets + 4;                // include forward "jumping" layer pairs
    static constexpr int nStartingPairs = phase1PixelTopology::nStartingPairs;

    static constexpr uint16_t numberOfModules = phase1PixelTopology::numberOfModules;

    static constexpr uint16_t numRowsInRoc = 80;
    static constexpr uint16_t numColsInRoc = 52;
    static constexpr uint16_t lastRowInRoc = numRowsInRoc - 1;
    static constexpr uint16_t lastColInRoc = numColsInRoc - 1;

    static constexpr uint16_t numRowsInModule = 2 * numRowsInRoc;
    static constexpr uint16_t numColsInModule = 8 * numColsInRoc;
    static constexpr uint16_t lastRowInModule = numRowsInModule - 1;
    static constexpr uint16_t lastColInModule = numColsInModule - 1;

    // 418 bins < 512, 9 bits are enough
    static constexpr uint16_t clusterBinning = numColsInModule + 2;
    static constexpr uint16_t clusterBits = 9;

    static constexpr uint16_t numberOfModulesInBarrel = 1184;
    static constexpr uint16_t numberOfModulesInLadder = 8;
    static constexpr uint16_t numberOfLaddersInBarrel = numberOfModulesInBarrel / numberOfModulesInLadder;

    static constexpr uint16_t firstEndcapPos = 4;
    static constexpr uint16_t firstEndcapNeg = 7;

    static constexpr int16_t xOffset = -81;

    static constexpr char const *nameModifier = "";

    static constexpr inline bool isEdgeX(uint16_t px) { return (px == 0) | (px == lastRowInModule); }

    static constexpr inline bool isEdgeY(uint16_t py) { return (py == 0) | (py == lastColInModule); }

    static constexpr inline uint16_t toRocX(uint16_t px) { return (px < numRowsInRoc) ? px : px - numRowsInRoc; }

    static constexpr inline uint16_t toRocY(uint16_t py) {
      auto roc = divu52(py);
      return py - 52 * roc;
    }

    static constexpr inline bool isBigPixX(uint16_t px) { return (px == 79) | (px == 80); }
    static constexpr inline bool isBigPixY(uint16_t py) {
      auto ly = toRocY(py);
      return (ly == 0) | (ly == lastColInRoc);
    }

    static constexpr inline uint16_t localX(uint16_t px) {
      auto shift = 0;
      if (px > lastRowInRoc)
        shift += 1;
      if (px > numRowsInRoc)
        shift += 1;
      return px + shift;
    }

    static constexpr inline uint16_t localY(uint16_t py) {
      auto roc = divu52(py);
      auto shift = 2 * roc;
      auto yInRoc = py - 52 * roc;
      if (yInRoc > 0)
        shift += 1;
      return py + shift;
    }

    static constexpr uint32_t const *layerStart = phase1PixelTopology::layerStart;

    // ----------------------------------------
    //  CA cut / geometry parameters
    // ----------------------------------------
    static constexpr uint8_t const *layerPairs = phase1PixelTopology::layerPairs;
    static constexpr uint8_t const *startingPairs = phase1PixelTopology::startingPairs;
    // scalar parameters (doublet building)
    static constexpr int minYsizeB1 = 1;
    static constexpr int minYsizeB2 = 1;
    static constexpr int maxDYsize12 = 28;
    static constexpr int maxDYsize = 20;
    static constexpr int maxDYPred = 20;
    static constexpr float cellZ0Cut = 12.5;
    // vector parameters (doublet building)
    static constexpr float const *minInner = phase1PixelTopology::minInner;
    static constexpr float const *maxInner = phase1PixelTopology::maxInner;
    static constexpr float const *minOuter = phase1PixelTopology::minOuter;
    static constexpr float const *maxOuter = phase1PixelTopology::maxOuter;
    static constexpr float const *maxDR = phase1PixelTopology::maxDR;
    static constexpr float const *minDZ = phase1PixelTopology::minDZ;
    static constexpr float const *maxDZ = phase1PixelTopology::maxDZ;
    static constexpr int16_t const *phicuts = phase1PixelTopology::phicuts;
    static constexpr float const *ptCuts = phase1PixelTopology::ptCuts;
    // scalar parameters (doublet linking)
    // p [GeV/c] = B [T] * R [m] * 0.3 (factor from conversion from J to GeV and q = e = 1.6 * 10e-19 C)
    // 87 cm/GeV = 1/(3.8T * 0.3)
    static constexpr float hardCurvCut = 1.f / (0.35 * 87.f);  // corresponds to 350 MeV in 3.8T.
    // vector parameters (doublet linking)
    static constexpr float const *thetaCuts = phase1PixelTopology::thetaCuts;
    static constexpr float const *dcaCuts = phase1PixelTopology::dcaCuts;
    // Deprecated arrays only used in the CUDA version
    static constexpr float const *minz = phase1PixelTopology::minz;
    static constexpr float const *maxz = phase1PixelTopology::maxz;
    static constexpr float const *maxr = phase1PixelTopology::maxr;
    // ----------------------------------------
  };

  struct HIonPhase1 : public Phase1 {
    // Storing here the needed constants different w.r.t. pp Phase1 topology.
    // All the other defined by inheritance in the HIon topology struct.

    using tindex_type = uint32_t;  // for tuples

    static constexpr uint32_t maxCellNeighbors = 90;
    static constexpr uint32_t maxCellTracks = 90;
    static constexpr uint32_t maxNumberOfTuples = 256 * 1024;
    static constexpr uint32_t maxNumberOfDoublets = 6 * 512 * 1024;
    static constexpr uint32_t maxHitsForContainers = avgHitsPerTrack * maxNumberOfTuples;
    static constexpr uint32_t maxNumberOfQuadruplets = maxNumberOfTuples;

    static constexpr uint32_t maxPixInModule = 10000;
    static constexpr uint32_t maxPixInModuleForMorphing = maxPixInModule * 1 / 10;
    static constexpr uint32_t maxIterClustering = 32;

    static constexpr uint32_t maxNumOfActiveDoublets =
        maxNumberOfDoublets / 4;  // TODO need to think a better way to avoid this duplication
    static constexpr uint32_t maxCellsPerHit = 256;

    static constexpr uint32_t maxNumClustersPerModules = phase1HIonPixelTopology::maxNumClustersPerModules;
    static constexpr uint32_t maxHitsInModule = phase1HIonPixelTopology::maxNumClustersPerModules;

    static constexpr char const *nameModifier = "HIonPhase1";

    // specified vector cuts for HIon
    static constexpr int16_t const *phicuts = phase1PixelTopology::phicuts;
    static constexpr float const *thetaCuts = phase1PixelTopology::thetaCuts;
    static constexpr float const *dcaCuts = phase1PixelTopology::dcaCuts;
  };

  template <typename T>
  using isPhase1Topology = typename std::enable_if<std::is_base_of<Phase1, T>::value>::type;

  template <typename T>
  using isPhase2Topology = typename std::enable_if<std::is_base_of<Phase2, T>::value>::type;

}  // namespace pixelTopology

#endif  // Geometry_CommonTopologies_SimplePixelTopology_h
