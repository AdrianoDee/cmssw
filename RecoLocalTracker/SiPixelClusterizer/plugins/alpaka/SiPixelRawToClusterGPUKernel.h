#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelRawToClusterGPUKernel_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelRawToClusterGPUKernel_h

#include <algorithm>
#include <alpaka/alpaka.hpp>
#include "DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigiErrorsDevice.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersDevice.h"
#include "FWCore/Utilities/interface/typedefs.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaUtilities/interface/SimpleVector.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelFormatterErrors.h"

// local include(s)
#include "../SiPixelClusterThresholds.h"

struct SiPixelROCsStatusAndMapping;
class SiPixelGainForHLTonGPU;
namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace pixelgpudetails {

    inline namespace phase1geometry {
      const uint32_t layerStartBit = 20;
      const uint32_t ladderStartBit = 12;
      const uint32_t moduleStartBit = 2;

      const uint32_t panelStartBit = 10;
      const uint32_t diskStartBit = 18;
      const uint32_t bladeStartBit = 12;

      const uint32_t layerMask = 0xF;
      const uint32_t ladderMask = 0xFF;
      const uint32_t moduleMask = 0x3FF;
      const uint32_t panelMask = 0x3;
      const uint32_t diskMask = 0xF;
      const uint32_t bladeMask = 0x3F;
    }  // namespace phase1geometry

    const uint32_t maxROCIndex = 8;
    const uint32_t numRowsInRoc = 80;
    const uint32_t numColsInRoc = 52;

    const uint32_t MAX_WORD = 2000;

    struct DetIdGPU {
      uint32_t rawId;
      uint32_t rocInDet;
      uint32_t moduleId;
    };

    struct Pixel {
      uint32_t row;
      uint32_t col;
    };

    inline constexpr pixelchannelidentifierimpl::Packing packing() { return PixelChannelIdentifier::thePacking; }

    inline constexpr uint32_t pack(uint32_t row, uint32_t col, uint32_t adc, uint32_t flag = 0) {
      constexpr pixelchannelidentifierimpl::Packing thePacking = packing();
      adc = std::min(adc, uint32_t(thePacking.max_adc));

      return (row << thePacking.row_shift) | (col << thePacking.column_shift) | (adc << thePacking.adc_shift);
    }

    constexpr uint32_t pixelToChannel(int row, int col) {
      constexpr pixelchannelidentifierimpl::Packing thePacking = packing();
      return (row << thePacking.column_width) | col;
    }

    class SiPixelRawToClusterGPUKernel {
    public:
      class WordFedAppender {
      public:
        WordFedAppender(uint32_t words, Queue& queue)
            : word_{cms::cuda::make_host_unique<unsigned int[]>(words, queue)},
              fedId_{cms::cuda::make_host_unique<unsigned char[]>(words, queue)} {}

        void initializeWordFed(int fedId, unsigned int index, cms_uint32_t const* src, unsigned int length) {
          std::memcpy(word_.data() + index, src, sizeof(cms_uint32_t) * length);
          std::memset(fedId_.data() + index / 2, fedId - FEDNumbering::MINSiPixeluTCAFEDID, length / 2);
        }

        const unsigned int* word() const { return word_.data(); }
        const unsigned char* fedId() const { return fedId_.data(); }

      private:
        cms::alpakatools::host_buffer<unsigned int[]> word_;
        cms::alpakatools::host_buffer<unsigned char[]> fedId_;
      };

      SiPixelRawToClusterGPUKernel() = default;
      ~SiPixelRawToClusterGPUKernel() = default;

      SiPixelRawToClusterGPUKernel(const SiPixelRawToClusterGPUKernel&) = delete;
      SiPixelRawToClusterGPUKernel(SiPixelRawToClusterGPUKernel&&) = delete;
      SiPixelRawToClusterGPUKernel& operator=(const SiPixelRawToClusterGPUKernel&) = delete;
      SiPixelRawToClusterGPUKernel& operator=(SiPixelRawToClusterGPUKernel&&) = delete;

      void makeClustersAsync(bool isRun2,
                             const SiPixelClusterThresholds clusterThresholds,
                             const SiPixelROCsStatusAndMapping* cablingMap,
                             const unsigned char* modToUnp,
                             const SiPixelGainForHLTonGPU* gains,
                             const WordFedAppender& wordFed,
                             SiPixelFormatterErrors&& errors,
                             const uint32_t wordCounter,
                             const uint32_t fedCounter,
                             bool useQualityInfo,
                             bool includeErrors,
                             bool debug,
                             Queue& queue);

      void makePhase2ClustersAsync(const SiPixelClusterThresholds clusterThresholds,
                                   const uint16_t* moduleIds,
                                   const uint16_t* xDigis,
                                   const uint16_t* yDigis,
                                   const uint16_t* adcDigis,
                                   const uint32_t* packedData,
                                   const uint32_t* rawIds,
                                   const uint32_t numDigis,
                                   Queue& queue);

      std::pair<SiPixelDigisDevice, SiPixelClustersDevice> getResults() {
        digis_d.setNModulesDigis(nModules_Clusters_h[0], nDigis);
        assert(nModules_Clusters_h[2] <= nModules_Clusters_h[1]);
        clusters_d.setNClusters(nModules_Clusters_h[1], nModules_Clusters_h[2]);
        // need to explicitly deallocate while the associated CUDA
        // stream is still alive
        //
        // technically the statement above is not true anymore now that
        // the CUDA streams are cached within the cms::cuda::StreamCache, but it is
        // still better to release as early as possible
        nModules_Clusters_h.reset();
        return std::make_pair(std::move(digis_d), std::move(clusters_d));
      }

      SiPixelDigiErrorsCUDA&& getErrors() { return std::move(digiErrors_d); }

    private:
      uint32_t nDigis;

      // Data to be put in the event
      cms::alpakatools::host_buffer<uint32_t[]> nModules_Clusters_h;
      SiPixelDigisDevice digis_d;
      SiPixelClustersDevice clusters_d;
      SiPixelDigiErrorsDevice digiErrors_d;
    };

  }  // namespace pixelgpudetails
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_SiPixelRawToClusterGPUKernel_h
