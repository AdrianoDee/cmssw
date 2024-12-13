// C++ includes
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <type_traits>

// Alpaka includes
#include <alpaka/alpaka.hpp>

// CMSSW includes
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLTLayout.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelMappingLayout.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersSoA.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelDigiConstants.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsSoA.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisSoA.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigiErrorsSoACollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/OneToManyAssoc.h"

#include "SiStripRawToClustersKernels.h"

// #define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  
    // Interface to outside
    void SiStripRawToClustersKernels::makeStripClustersAsync(
        Queue &queue,
        const std::vector<const FEDRawData*>& rawdata,
        const std::vector<std::unique_ptr<sistrip::FEDBuffer>>& buffers,
        const ::reco::StripConditionsDetSoAConstView& detView,
        const ::reco::StripConditionsStripSoAConstView& stripView,
        const ::reco::StripConditionsAPVSoAConstView& apvView)

      {

        int totalSize = 0;
        int offsetSize = 0;
        for (const auto& buff : buffers) {
          if (buff != nullptr) {
            offsetSize++;
            totalSize += buff->bufferSize();
          }
        }

        SequentialContainerView fedMapView;
        auto fedMap = cms::alpakatools::make_host_buffer<SequentialContainer>();
        auto fedMapContainer = cms::alpakatools::make_host_buffer<SequentialContainerStorage[]>(totalSize);
        auto fedMapOffsets = cms::alpakatools::make_host_buffer<SequentialContainerOffsets[]>(offsetSize + 1);
        
        fedMapView.assoc = fedMap.data();
        fedMapView.offStorage = fedMapOffsets.data();
        fedMapView.contentStorage = fedMapContainer.data();
        fedMapView.contentSize = alpaka::getExtentProduct(fedMapContainer);
        fedMapView.offSize = alpaka::getExtentProduct(fedMapOffsets);
        

      //   auto fedRawDataHost = cms::cuda::make_host_unique<uint8_t[]>(totalSize, stream);
      //   auto fedRawDataGPU = cms::cuda::make_device_unique<uint8_t[]>(totalSize, stream);

      //   size_t off = 0;

      //   std::vector<stripgpu::fedId_t> fedIndex_;
      //   std::vector<size_t> fedRawDataOffsets_;

      //   fedRawDataOffsets_.clear();
      //   fedIndex_.clear();
      //   fedIndex_.resize(sistrip::NUMBER_OF_FEDS, stripgpu::invalidFed);

      //   sistrip::FEDReadoutMode mode = sistrip::READOUT_MODE_INVALID;

      //   for (size_t fedi = 0; fedi < buffers.size(); ++fedi) {
      //     auto& buff = buffers[fedi];
      //     if (buff != nullptr) {
      //       const auto raw = rawdata[fedi];
      //       memcpy(fedRawDataHost.get() + off, raw->data(), raw->size());
      //       fedIndex_[stripgpu::fedIndex(fedi)] = fedRawDataOffsets_.size();
      //       fedRawDataOffsets_.push_back(off);
      //       off += raw->size();
      //       if (fedRawDataOffsets_.size() == 1) {
      //         mode = buff->readoutMode();
      //       } else {
      //         if (buff->readoutMode() != mode) {
      //           throw cms::Exception("[SiStripRawToClusterGPUKernel] inconsistent readout mode ")
      //               << buff->readoutMode() << " != " << mode;
      //         }
      //       }
      //     }
      //   }

      // size_t offset = 0;

      // iterate over the detector in DetID/APVPair order
      // mapping out where the data are
      // for (size_t i = 0; i < detView.metadata().size(); ++i) {
      //   auto thisDet = detView[i];
      //   const auto fedId = thisDet.fedID();
      //   const auto fedCh = thisDet.fedCh();
      //   const auto fedi = fedIndex_[stripgpu::fedIndex(fedId)];

      //   // if (fedi != invalidFed) {
      //   //   const auto buffer = buffers[fedId].get();
      //   //   const auto& channel = buffer->channel(thisDet.fedCh());

      //   //   auto len = channel.length();
      //   //   auto off = channel.offset();

      //   //   assert(len >= headerlen || len == 0);

      //   //   if (len >= headerlen) {
      //   //     len -= headerlen;
      //   //     off += headerlen;
      //   //   }

      //   //   // to SoA column setting
      //   //   chanlocs->setChannelLoc(i, channel.data(), off, offset, len, fedId, fedCh, detp.detID()); 
      //   //   inputGPU[i] = fedRawDataGPU.get() + fedRawDataOffsets_[fedi] + (channel.data() - rawdata[fedId]->data());
      //   //   offset += len;

      //   // } else {
      //   //   chanlocs->setChannelLoc(i, nullptr, 0, 0, 0, invalidFed, 0, invalidDet);
      //   //   inputGPU[i] = nullptr;
      //   // }
      // }

      // const auto n_strips = offset;

      // std::cout << "n strips: " << n_strips << std::endl;
        

      }  // end clusterizer scope

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
