#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "RecoLocalTracker/Records/interface/SiStripClusterizerConditionsRcd.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithmFactory.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"

#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditions.h"

#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditionsDevice.h"
#include "CalibFormats/SiStripObjects/interface/alpaka/SiStripClusterizerConditionsCollection.h"

#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Likely.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/SiStripCluster/interface/alpaka/SiStripClusterCollection.h"

// #include "SiStripRawToClusterKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {
  std::unique_ptr<sistrip::FEDBuffer> fillBuffer(int fedId, const FEDRawData& rawData) {
    std::unique_ptr<sistrip::FEDBuffer> buffer;

    // Check on FEDRawData pointer
    const auto st_buffer = sistrip::preconstructCheckFEDBuffer(rawData);
    if UNLIKELY (sistrip::FEDBufferStatusCode::SUCCESS != st_buffer) {
      LogDebug(sistrip::mlRawToCluster_) << "[ClustersFromRawProducer::" << __func__ << "]" << st_buffer
                                         << " for FED ID " << fedId;
      return buffer;
    }
    buffer = std::make_unique<sistrip::FEDBuffer>(rawData);
    const auto st_chan = buffer->findChannels();
    if UNLIKELY (sistrip::FEDBufferStatusCode::SUCCESS != st_chan) {
      LogDebug(sistrip::mlRawToCluster_) << "Exception caught when creating FEDBuffer object for FED " << fedId << ": "
                                         << st_chan;
      buffer.reset();
      return buffer;
    }
    if UNLIKELY (!buffer->doChecks(false)) {
      LogDebug(sistrip::mlRawToCluster_) << "Exception caught when creating FEDBuffer object for FED " << fedId
                                         << ": FED Buffer check fails";
      buffer.reset();
      return buffer;
    }

    return buffer;
  }
}  // namespace

  class SiStripRawToCluster : public stream::SynchronizingEDProducer<> {
  public:
    explicit SiStripRawToCluster(const edm::ParameterSet& iConfig);
    ~SiStripRawToCluster() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    // using Algo = SiStripRawToClusterKernel;

  private:
    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override;
    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;
    
    void getRawData(const FEDRawDataCollection& rawColl, const SiStripClusterizerConditions& conditions);
    void fillBufferRaw(uint32_t idet, const FEDRawDataCollection& rawColl, const SiStripClusterizerConditions& conditions);

    std::vector<std::unique_ptr<sistrip::FEDBuffer>> buffers_;
    std::vector<const FEDRawData*> raw_;

    edm::EDGetTokenT<FEDRawDataCollection> inputToken_;

    const device::ESGetToken<reco::SiStripClusterizerConditionsCollection, SiStripClusterizerConditionsRcd> deviceConditionsToken_;
    edm::ESGetToken<SiStripClusterizerConditions, SiStripClusterizerConditionsRcd> hostConditionsToken_;

    // edm::ESWatcher<SiPixelFedCablingMapRcd> recordWatcher_;
    device::EDPutToken<SiStripClusterSoACollection> outputToken_;

    // Algo Algo_;

  };

  SiStripRawToCluster::SiStripRawToCluster(const edm::ParameterSet& iConfig)
      : buffers_(sistrip::FED_ID_MAX),
        raw_(sistrip::FED_ID_MAX),
        // gpuAlgo_(iConfig.getParameter<edm::ParameterSet>("Clusterizer")),
        inputToken_(consumes(iConfig.getParameter<edm::InputTag>("FEDRawInput"))),
        deviceConditionsToken_(esConsumes(edm::ESInputTag{"", iConfig.getParameter<std::string>("Conditions")})),
        hostConditionsToken_(esConsumes(edm::ESInputTag{"", iConfig.getParameter<std::string>("Conditions")})),
        outputToken_(produces())
  {
  }

  void SiStripRawToCluster::getRawData(const FEDRawDataCollection& rawColl,
                                       const SiStripClusterizerConditions& conditions) {
  // loop over good det in cabling
  for (auto idet : conditions.allDetIds()) {
    fillBufferRaw(idet, rawColl, conditions);
  }  // end loop over dets
}

  void SiStripRawToCluster::fillBufferRaw(uint32_t idet,
                                        const FEDRawDataCollection& rawColl,
                                        const SiStripClusterizerConditions& conditions) {
  auto const& det = conditions.findDetId(idet);
  if (!det.valid())
    return;

  // Loop over apv-pairs of det
  for (auto const conn : conditions.currentConnection(det)) {
    if UNLIKELY (!conn)
      continue;

    const uint16_t fedId = conn->fedId();

    // If fed id is null or connection is invalid continue
    if UNLIKELY (!fedId || !conn->isConnected()) {
      continue;
    }

    // If Fed hasnt already been initialised, extract data and initialise
    sistrip::FEDBuffer* buffer = buffers_[fedId].get();
    if (!buffer) {
      const FEDRawData& rawData = rawColl.FEDData(fedId);
      raw_[fedId] = &rawData;
      buffers_[fedId] = fillBuffer(fedId, rawData);
    }
  }  // end loop over conn
}

  void SiStripRawToCluster::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    // edm::ParameterSetDescription desc;
    // desc.add<bool>("IncludeErrors", true);
    // desc.add<bool>("UseQualityInfo", false);
    // // Note: this parameter is obsolete: it is ignored and will have no effect.
    // // It is kept to avoid breaking older configurations, and will not be printed in the generated cfi.py file.
    // desc.addOptionalNode(edm::ParameterDescription<uint32_t>("MaxFEDWords", 0, true), false)
    //     ->setComment("This parameter is obsolete and will be ignored.");
    // desc.add<int32_t>("clusterThreshold_layer1", pixelClustering::clusterThresholdLayerOne);
    // desc.add<int32_t>("clusterThreshold_otherLayers", pixelClustering::clusterThresholdOtherLayers);
    // desc.add<double>("VCaltoElectronGain", 47.f);
    // desc.add<double>("VCaltoElectronGain_L1", 50.f);
    // desc.add<double>("VCaltoElectronOffset", -60.f);
    // desc.add<double>("VCaltoElectronOffset_L1", -670.f);

    // desc.add<edm::InputTag>("InputLabel", edm::InputTag("rawDataCollector"));
    // {
    //   edm::ParameterSetDescription psd0;
    //   psd0.addOptional<std::vector<edm::InputTag>>("inputs");
    //   psd0.addOptional<std::vector<double>>("deltaPhi");
    //   psd0.addOptional<std::vector<double>>("maxZ");
    //   psd0.addOptional<edm::InputTag>("beamSpot");
    //   desc.add<edm::ParameterSetDescription>("Regions", psd0)
    //       ->setComment("## Empty Regions PSet means complete unpacking");
    // }
    // desc.add<std::string>("CablingMapLabel", "")->setComment("CablingMap label");  //Tav
    // descriptions.addWithDefaultLabel(desc);
  }

  
  void SiStripRawToCluster::acquire(device::Event const& iEvent, device::EventSetup const& iSetup) {
    const auto& dConditions = iSetup.getData(deviceConditionsToken_); 
    const auto& hConditions = iSetup.getData(hostConditionsToken_); 
    
    // get raw data
    auto const& rawData = iEvent.get(inputToken_);
    getRawData(rawData, hConditions);

    int totalSize = 0;
    for (const auto& buff : buffers_) {
      if (buff != nullptr) {
        totalSize += buff->bufferSize();
      }
    }
    
  //   auto const& hMap = iSetup.getData(mapToken_);
  //   auto const& dGains = iSetup.getData(gainsToken_);

  //   // initialize cabling map or update if necessary
  //   if (recordWatcher_.check(iSetup)) {
  //     // cabling map, which maps online address (fed->link->ROC->local pixel) to offline (DetId->global pixel)
  //     cablingMap_ = &iSetup.getData(cablingMapToken_);
  //     fedIds_ = cablingMap_->fedIds();
  //     cabling_ = cablingMap_->cablingTree();
  //     LogDebug("map version:") << cablingMap_->version();
  //   }

  //   // if used, the buffer is guaranteed to stay alive until the after the execution of makePhase1ClustersAsync completes
  //   std::optional<cms::alpakatools::device_buffer<Device, unsigned char[]>> modulesToUnpackRegional;
  //   const unsigned char* modulesToUnpack;
  //   if (regions_) {
  //     regions_->run(iEvent, iSetup);
  //     LogDebug("SiStripRawToCluster") << "region2unpack #feds: " << regions_->nFEDs();
  //     LogDebug("SiStripRawToCluster") << "region2unpack #modules (BPIX,EPIX,total): " << regions_->nBarrelModules()
  //                                     << " " << regions_->nForwardModules() << " " << regions_->nModules();

  //     modulesToUnpackRegional = SiPixelMappingUtilities::getModToUnpRegionalAsync(
  //         *(regions_->modulesToUnpack()), cabling_.get(), fedIds_, iEvent.queue());
  //     modulesToUnpack = modulesToUnpackRegional->data();
  //   } else {
  //     modulesToUnpack = hMap->modToUnpDefault();
  //   }

  //   const auto& buffers = iEvent.get(rawGetToken_);

  //   errors_.clear();

  //   // GPU specific: Data extraction for RawToDigi GPU
  //   unsigned int wordCounter = 0;
  //   unsigned int fedCounter = 0;
  //   bool errorsInEvent = false;
  //   std::vector<unsigned int> index(fedIds_.size(), 0);
  //   std::vector<cms_uint32_t const*> start(fedIds_.size(), nullptr);
  //   std::vector<ptrdiff_t> words(fedIds_.size(), 0);
  //   // In CPU algorithm this loop is part of PixelDataFormatter::interpretRawData()
  //   ErrorChecker errorcheck;
  //   for (uint32_t i = 0; i < fedIds_.size(); ++i) {
  //     const int fedId = fedIds_[i];
  //     if (regions_ && !regions_->mayUnpackFED(fedId))
  //       continue;

  //     // for GPU
  //     // first 150 index stores the fedId and next 150 will store the
  //     // start index of word in that fed
  //     assert(fedId >= FEDNumbering::MINSiPixeluTCAFEDID);
  //     fedCounter++;

  //     // get event data for this fed
  //     const FEDRawData& rawData = buffers.FEDData(fedId);

  //     // GPU specific
  //     int nWords = rawData.size() / sizeof(cms_uint64_t);
  //     if (nWords == 0) {
  //       continue;
  //     }
  //     // check CRC bit
  //     const cms_uint64_t* trailer = reinterpret_cast<const cms_uint64_t*>(rawData.data()) + (nWords - 1);
  //     if (not errorcheck.checkCRC(errorsInEvent, fedId, trailer, errors_)) {
  //       continue;
  //     }
  //     // check headers
  //     const cms_uint64_t* header = reinterpret_cast<const cms_uint64_t*>(rawData.data());
  //     header--;
  //     bool moreHeaders = true;
  //     while (moreHeaders) {
  //       header++;
  //       bool headerStatus = errorcheck.checkHeader(errorsInEvent, fedId, header, errors_);
  //       moreHeaders = headerStatus;
  //     }

  //     // check trailers
  //     bool moreTrailers = true;
  //     trailer++;
  //     while (moreTrailers) {
  //       trailer--;
  //       bool trailerStatus = errorcheck.checkTrailer(errorsInEvent, fedId, nWords, trailer, errors_);
  //       moreTrailers = trailerStatus;
  //     }

  //     const cms_uint32_t* bw = (const cms_uint32_t*)(header + 1);
  //     const cms_uint32_t* ew = (const cms_uint32_t*)(trailer);

  //     assert(0 == (ew - bw) % 2);
  //     index[i] = wordCounter;
  //     start[i] = bw;
  //     words[i] = (ew - bw);
  //     wordCounter += (ew - bw);

  //   }  // end of for loop
  //   nDigis_ = wordCounter;
  //   if (nDigis_ == 0)
  //     return;

  //   // copy the FED data to a single cpu buffer
  //   pixelDetails::WordFedAppender wordFedAppender(iEvent.queue(), nDigis_);
  //   for (uint32_t i = 0; i < fedIds_.size(); ++i) {
  //     wordFedAppender.initializeWordFed(fedIds_[i], index[i], start[i], words[i]);
  //   }
  //   Algo_.makePhase1ClustersAsync(iEvent.queue(),
  //                                 clusterThresholds_,
  //                                 hMap.const_view(),
  //                                 modulesToUnpack,
  //                                 dGains.const_view(),
  //                                 wordFedAppender,
  //                                 wordCounter,
  //                                 fedCounter,
  //                                 useQuality_,
  //                                 includeErrors_,
  //                                 edm::MessageDrop::instance()->debugEnabled);
  // }
  }

  
  void SiStripRawToCluster::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
  //   if (nDigis_ == 0) {
  //     // Cannot use the default constructor here, as it would not allocate memory.
  //     // In the case of no digis, clusters_d are not being instantiated, but are
  //     // still used downstream to initialize TrackingRecHitSoADevice. If there
  //     // are no valid pointers to clusters' Collection columns, instantiation
  //     // of TrackingRecHits fail. Example: workflow 11604.0

  //     iEvent.emplace(digiPutToken_, nDigis_, iEvent.queue());
  //     iEvent.emplace(clusterPutToken_, pixelTopology::Phase1::numberOfModules, iEvent.queue());
  //     if (includeErrors_) {
  //       iEvent.emplace(digiErrorPutToken_);
  //       iEvent.emplace(fmtErrorToken_);
  //     }
  //     return;
  //   }

  //   iEvent.emplace(digiPutToken_, Algo_.getDigis());
  //   iEvent.emplace(clusterPutToken_, Algo_.getClusters());
  //   if (includeErrors_) {
  //     iEvent.emplace(digiErrorPutToken_, Algo_.getErrors());
  //     iEvent.emplace(fmtErrorToken_, std::move(errors_));
  //   }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// define as framework plugin
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(SiStripRawToCluster);
