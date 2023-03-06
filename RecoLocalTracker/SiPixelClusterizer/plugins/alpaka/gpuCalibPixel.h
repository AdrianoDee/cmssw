#ifndef plugin_SiPixelClusterizer_alpaka_gpuCalibPixel_h
#define plugin_SiPixelClusterizer_alpaka_gpuCalibPixel_h

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/SiPixelClusterSoA/interface/gpuClusteringConstants.h"
#include "DataFormats/SiPixelGainCalibrationForHLTSoA/interface/SiPixelGainCalibrationForHLTLayout.h"
#include "DataFormats/SiPixelGainCalibrationForHLTSoA/interface/alpaka/SiPixelGainCalibrationForHLTDevice.h"
#include "DataFormats/SiPixelGainCalibrationForHLTSoA/interface/alpaka/SiPixelGainCalibrationForHLTUtilities.h"
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace gpuCalibPixel {

    constexpr uint16_t InvId = 9999;  // must be > MaxNumModules

    // valid for run2
    constexpr float VCaltoElectronGain = 47;         // L2-4: 47 +- 4.7
    constexpr float VCaltoElectronGain_L1 = 50;      // L1:   49.6 +- 2.6
    constexpr float VCaltoElectronOffset = -60;      // L2-4: -60 +- 130
    constexpr float VCaltoElectronOffset_L1 = -670;  // L1:   -670 +- 220

    //for phase2
    constexpr float ElectronPerADCGain = 1500;
    constexpr int8_t Phase2ReadoutMode = 3;
    constexpr uint16_t Phase2DigiBaseline = 1000;
    constexpr uint8_t Phase2KinkADC = 8;

    struct calibDigis {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    bool isRun2,
                                    uint16_t* id,
                                    uint16_t const* __restrict__ x,
                                    uint16_t const* __restrict__ y,
                                    uint16_t* adc,
                                    const SiPixelGainCalibrationForHLTSoAConstView& __restrict__ gains,
                                    int numElements,
                                    uint32_t* __restrict__ moduleStart,        // just to zero first
                                    uint32_t* __restrict__ nClustersInModule,  // just to zero them
                                    uint32_t* __restrict__ clusModuleStart     // just to zero first
      ) const {
        const uint32_t threadIdxGlobal(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);

        // zero for next kernels...
        if (threadIdxGlobal == 0) {
          clusModuleStart[0] = moduleStart[0] = 0;
        }

        cms::alpakatools::for_each_element_in_grid_strided(
            acc, gpuClustering::MaxNumModules, [&](uint32_t i) { nClustersInModule[i] = 0; });

        cms::alpakatools::for_each_element_in_grid_strided(acc, numElements, [&](uint32_t i) {
          if (id[i] != InvId) {
            float conversionFactor = (isRun2) ? (id[i] < 96 ? VCaltoElectronGain_L1 : VCaltoElectronGain) : 1.f;
            float offset = (isRun2) ? (id[i] < 96 ? VCaltoElectronOffset_L1 : VCaltoElectronOffset) : 0;

            bool isDeadColumn = false, isNoisyColumn = false;

            int row = x[i];
            int col = y[i];
            
            auto ret = SiPixelClustersUtilities::getPedAndGain(id[i], col, row, isDeadColumn, isNoisyColumn, gains); 

            float pedestal = ret.first;
            float gain = ret.second;
            // float pedestal = 0; float gain = 1.;
            if (isDeadColumn | isNoisyColumn) {
              id[i] = InvId;
              adc[i] = 0;
              printf("bad pixel at %d in %d\n", i, id[i]);
            } else {
              float vcal = adc[i] * gain - pedestal * gain;
              adc[i] = std::max(100, int(vcal * conversionFactor + offset));
            }
          }
        });
      }
    };
    
    struct calibDigisPhase2 {

      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                   uint16_t* id,
                                   uint16_t* adc,
                                   int numElements,
                                   uint32_t* __restrict__ moduleStart,        // just to zero first
                                   uint32_t* __restrict__ nClustersInModule,  // just to zero them
                                   uint32_t* __restrict__ clusModuleStart     // just to zero first
  ) {
    const uint32_t threadIdxGlobal(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
    // zero for next kernels...

    if (0 == threadIdxGlobal)
      clusModuleStart[0] = moduleStart[0] = 0;
    for (int i = threadIdxGlobal; i < phase2PixelTopology::numberOfModules; i += gridDim.x * blockDim.x) {
      nClustersInModule[i] = 0;
    }

    for (int i = threadIdxGlobal; i < numElements; i += gridDim.x * blockDim.x) {
      if (invalidModuleId == id[i])
        continue;

      constexpr int mode = (Phase2ReadoutMode < -1 ? -1 : Phase2ReadoutMode);

      int adc_int = adc[i];

      if constexpr (mode < 0)
        adc_int = int(adc_int * ElectronPerADCGain);
      else {
        if (adc_int < Phase2KinkADC)
          adc_int = int((adc_int + 0.5) * ElectronPerADCGain);
        else {
          constexpr int8_t dspp = (Phase2ReadoutMode < 10 ? Phase2ReadoutMode : 10);
          constexpr int8_t ds = int8_t(dspp <= 1 ? 1 : (dspp - 1) * (dspp - 1));

          adc_int -= Phase2KinkADC;
          adc_int *= ds;
          adc_int += Phase2KinkADC;

          adc_int = ((adc_int + 0.5 * ds) * ElectronPerADCGain);
        }

        adc_int += int(Phase2DigiBaseline);
      }
      adc[i] = std::min(adc_int, int(std::numeric_limits<uint16_t>::max()));
    }
  }


    }
  }  // namespace gpuCalibPixel
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // plugin_SiPixelClusterizer_alpaka_gpuCalibPixel_h