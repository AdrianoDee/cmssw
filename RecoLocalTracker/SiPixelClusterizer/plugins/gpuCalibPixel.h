#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h

#include <cstdint>
#include <cstdio>

#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainForHLTonGPU.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

namespace gpuCalibPixel {

  using gpuClustering::invalidModuleId;

  // valid for run2
  constexpr float VCaltoElectronGain = 47;         // L2-4: 47 +- 4.7
  constexpr float VCaltoElectronGain_L1 = 50;      // L1:   49.6 +- 2.6
  constexpr float VCaltoElectronOffset = -60;      // L2-4: -60 +- 130
  constexpr float VCaltoElectronOffset_L1 = -670;  // L1:   -670 +- 220

  constexpr float    ElectronPerADCGain = 600;
  constexpr int8_t   Phase2ReadoutMode  = 3;
  constexpr uint16_t Phase2DigiBaseline = 1500;
  constexpr uint8_t  Phase2KinkADC      = 8;

  __global__ void calibDigis(bool isRun2,
                             uint16_t* id,
                             uint16_t const* __restrict__ x,
                             uint16_t const* __restrict__ y,
                             uint16_t* adc,
                             SiPixelGainForHLTonGPU const* __restrict__ ped,
                             int numElements,
                             uint32_t* __restrict__ moduleStart,        // just to zero first
                             uint32_t* __restrict__ nClustersInModule,  // just to zero them
                             uint32_t* __restrict__ clusModuleStart     // just to zero first
  ) {
    int first = blockDim.x * blockIdx.x + threadIdx.x;

    // zero for next kernels...
    if (0 == first)
      clusModuleStart[0] = moduleStart[0] = 0;
    for (int i = first; i < gpuClustering::maxNumModules; i += gridDim.x * blockDim.x) {
      nClustersInModule[i] = 0;
    }

    for (int i = first; i < numElements; i += gridDim.x * blockDim.x) {
      if (invalidModuleId == id[i])
        continue;

      float conversionFactor = (isRun2) ? (id[i] < 96 ? VCaltoElectronGain_L1 : VCaltoElectronGain) : 1.f;
      float offset = (isRun2) ? (id[i] < 96 ? VCaltoElectronOffset_L1 : VCaltoElectronOffset) : 0;

      bool isDeadColumn = false, isNoisyColumn = false;

      int row = x[i];
      int col = y[i];
      auto ret = ped->getPedAndGain(id[i], col, row, isDeadColumn, isNoisyColumn);
      float pedestal = ret.first;
      float gain = ret.second;
      // float pedestal = 0; float gain = 1.;
      if (isDeadColumn | isNoisyColumn) {
        id[i] = invalidModuleId;
        adc[i] = 0;
        printf("bad pixel at %d in %d\n", i, id[i]);
      } else {
        float vcal = adc[i] * gain - pedestal * gain;
        adc[i] = std::max(100, int(vcal * conversionFactor + offset));
      }
    }
  }

  __global__ void calibDigisUpgrade(
                             const uint16_t *X, const uint16_t *Y,
                             const uint16_t *A, const uint32_t *P,
                             const uint16_t *M,
                             uint16_t *xx, uint16_t *yy,
                             uint16_t *adc, uint32_t *pdigi,
                             uint16_t *id,
                             int numElements,
                             uint32_t* __restrict__ moduleStart,        // just to zero first
                             uint32_t* __restrict__ nClustersInModule,  // just to zero them
                             uint32_t* __restrict__ clusModuleStart     // just to zero first
  ) {
    int first = blockDim.x * blockIdx.x + threadIdx.x;

    // zero for next kernels...
    // if (0 == first)
    //   clusModuleStart[0] = moduleStart[0] = 0;
    // for (int i = first; i < gpuClustering::MaxNumModules; i += gridDim.x * blockDim.x) {
    //   nClustersInModule[i] = 0;
    // }
    // printf(">calibDigisPhase2 %d\n",__LINE__);
    for (int i = first; i < numElements; i += gridDim.x * blockDim.x) {

      // printf(">calibDigisPhase2 %d\n",__LINE__);
      if (invalidModuleId == M[i])
        continue;

        xx[i]       = X[i];
        yy[i]       = Y[i];
        adc[i]      = A[i];
        pdigi[i]    = P[i];
        id[i]       = M[i];


      int mode = (Phase2ReadoutMode < -1 ? -1 : Phase2ReadoutMode);
      // uint16_t oldadc = adc[i];
      if(mode < 0)
      {
        adc[i] = int(adc[i] * ElectronPerADCGain);
      }
      else
      {
        if (adc[i] < Phase2KinkADC)
        {
          adc[i] = int((adc[i] + 0.5) * ElectronPerADCGain);
        }
        else
        {
          constexpr int8_t dspp = (Phase2ReadoutMode < 10 ? Phase2ReadoutMode : 10);
          constexpr int8_t ds   = int8_t(dspp <= 1 ? 1 : (dspp - 1) * (dspp - 1));

          adc[i] -= Phase2KinkADC;
          adc[i] *= ds;
          adc[i] += Phase2KinkADC;

          adc[i] = uint16_t((adc[i] + 0.5 * ds) * ElectronPerADCGain);
        }

        adc[i] += int(Phase2DigiBaseline);
        }
        
  //  printf("gains %d %d %d %d %d %d\n",oldadc,adc[i],Phase2KinkADC,Phase2ReadoutMode,dspp,ds,ElectronPerADCGain);

  //  printf("digissss %d %d %d %d %d %d\n",id[i],xx[i],yy[i],adc[i],pdigi[i],i);
    }
  }
}  // namespace gpuCalibPixel

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
