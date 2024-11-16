#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLTHost.h"
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationForHLTRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "RecoLocalTracker/Records/interface/SiStripClusterizerConditionsRcd.h"

#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditionsSoA.h"
#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditionsHost.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"

#include <memory>
#include <numeric>
#include <algorithm> 

#define GPU_STRIP_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace stripSoAUtilities
  {

    static constexpr uint16_t badBit = 1 << 15;

    using ConstDetView   = ::reco::StripConditionsDetSoAConstView;  
    using ConstStripView = ::reco::StripConditionsStripSoAConstView;
    using ConstAPVView   = ::reco::StripConditionsAPVSoAConstView;

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint16_t fedIndex(uint16_t fed) { return fed - sistrip::FED_ID_MIN; }
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t stripIndex(uint16_t fed, uint8_t channel, uint16_t strip) {
    return fedIndex(fed) * sistrip::FEDCH_PER_FED * sistrip::STRIPS_PER_FEDCH + channel * sistrip::STRIPS_PER_FEDCH +
           (strip % sistrip::STRIPS_PER_FEDCH);
  }
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t apvIndex(uint16_t fed, uint8_t channel, uint16_t strip) {
    return fedIndex(fed) * sistrip::APVS_PER_FEDCH * sistrip::FEDCH_PER_FED + sistrip::APVS_PER_CHAN * channel +
           (strip % sistrip::STRIPS_PER_FEDCH) / sistrip::STRIPS_PER_APV;
  }
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t channelIndex(uint16_t fed, uint8_t channel) {
    return fedIndex(fed) * sistrip::FEDCH_PER_FED + channel;
  }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t detId(ConstDetView const& soa, uint16_t fed, uint8_t channel) {
      return soa.detId()[channelIndex(fed, channel)];
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE uint16_t iPair(ConstDetView const& soa, uint16_t fed, uint8_t channel) {
      return soa.iPair()[channelIndex(fed, channel)];
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE float invthick(ConstDetView const& soa, uint16_t fed, uint8_t channel) {
      return soa.invthick()[channelIndex(fed, channel)];
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE float noise(ConstStripView const& soa, uint16_t fed, uint8_t channel, uint16_t strip) {
      // noise is stored as 9 bits with a fixed point scale factor of 0.1
      return 0.1f * (soa.noise()[stripIndex(fed, channel, strip)] & ~badBit);
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE float gain(ConstAPVView const& soa, uint16_t fed, uint8_t channel, uint16_t strip) {
      return soa.gain()[apvIndex(fed, channel, strip)];
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool bad(ConstStripView const& soa, uint16_t fed, uint8_t channel, uint16_t strip) {
      return badBit == (soa.noise()[stripIndex(fed, channel, strip)] & badBit);
    }
  
  }
  class SiStripClustersConditionsESProducer : public ESProducer {
  public:
    explicit SiStripClustersConditionsESProducer(const edm::ParameterSet& iConfig);
    std::optional<::reco::SiStripClusterizerConditionsHost> produce(const SiStripClusterizerConditionsRcd& iRecord);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
    edm::ESGetToken<SiStripGain, SiStripGainRcd> gainToken_;
    edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> noisesToken_;
    edm::ESGetToken<SiStripQuality, SiStripQualityRcd> qualityToken_;
  };

  SiStripClustersConditionsESProducer::SiStripClustersConditionsESProducer(const edm::ParameterSet& iConfig)
      : ESProducer(iConfig) {

    auto cc = setWhatProduced(this);

    gainToken_ = cc.consumesFrom<SiStripGain, SiStripGainRcd>();
    noisesToken_ = cc.consumesFrom<SiStripNoises, SiStripNoisesRcd>();
    qualityToken_ = cc.consumesFrom<SiStripQuality, SiStripQualityRcd>(
      edm::ESInputTag{"", iConfig.getParameter<std::string>("QualityLabel")});

  }

  void SiStripClustersConditionsESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("QualityLabel", "");
    desc.add<std::string>("Label", "");
    descriptions.addWithDefaultLabel(desc);
  }

  std::optional<::reco::SiStripClusterizerConditionsHost> SiStripClustersConditionsESProducer::produce(
      const SiStripClusterizerConditionsRcd& iRecord) {
    
    alpaka::QueueCpuBlocking queue{cms::alpakatools::host()};

    std::cout << "SiStripClustersConditionsESProducer::produce" << std::endl;
    using namespace stripSoAUtilities;
    auto gains = iRecord.getTransientHandle(gainToken_);
    const auto& noises = iRecord.get(noisesToken_);
    const auto& quality = iRecord.get(qualityToken_);

    constexpr auto nStrips = sistrip::NUMBER_OF_FEDS * sistrip::FEDCH_PER_FED * sistrip::STRIPS_PER_FEDCH;
    constexpr auto nChannels = sistrip::NUMBER_OF_FEDS * sistrip::FEDCH_PER_FED;
    constexpr auto nApvs   = sistrip::NUMBER_OF_FEDS * sistrip::FEDCH_PER_FED * sistrip::STRIPS_PER_FEDCH;
    
    const auto& connected = quality.cabling()->connected();
    const auto& detCabling = quality.cabling()->getDetCabling();

    // // Count first
    // auto detToFedsSize = 0;
    // for (const auto& conn : connected) {
    //   const auto det = conn.first;
    //   if (!quality.IsModuleBad(det)) {
    //     const auto detConn_it = detCabling.find(det);

    //     if (detCabling.end() != detConn_it) {
    //       for (const auto& chan : (*detConn_it).second) {
    //         if (chan && chan->fedId() && chan->isConnected()) {
    //         ++detToFedsSize;
    //         }
    //       }
    //     }
    //   }
    // }

    ::reco::SiStripClusterizerConditionsHost product({{nChannels, nStrips, nApvs}}, cms::alpakatools::host());
    auto soaDet = product.view<::reco::StripConditionsDetSoA>();
    auto soaStr = product.view<::reco::StripConditionsStripSoA>();
    auto soaApv = product.view<::reco::StripConditionsAPVSoA>();
    
    auto detIdView = cms::alpakatools::make_host_view<uint32_t>(soaDet.detId(), nChannels);
    std::vector<uint32_t> dummyDet(nChannels,std::numeric_limits<uint32_t>::max());
    auto dummyDet_view = cms::alpakatools::make_host_view<uint32_t>(dummyDet.data(),nChannels);
    alpaka::memcpy(queue, detIdView, dummyDet_view);

    for (const auto& conn : connected) {
      const auto det = conn.first;
      if (!quality.IsModuleBad(det)) {
        const auto detConn_it = detCabling.find(det);

        if (detCabling.end() != detConn_it) {
          for (const auto& chan : (*detConn_it).second) {
            if (chan && chan->fedId() && chan->isConnected()) {
              const auto detId = chan->detId();
              const auto fedId = chan->fedId();
              const auto fedCh = chan->fedCh();
              const auto iPair = chan->apvPairNumber();
              const auto chId = channelIndex(fedId, fedCh);


              // detToFeds_.emplace_back(detId, iPair, fedId, fedCh);

              soaDet.detId()[chId] = detId;
              soaDet.iPair()[chId] = iPair;
              soaDet.fedCh()[chId] = fedCh;
              soaDet.fedId()[chId] = fedId;
              std::cout << "chId: " << chId << " - " << detId << std::endl;
              soaDet.invthick()[chId] = siStripClusterTools::sensorThicknessInverse(detId);
              auto offset = 256 * iPair;

              for (auto strip = 0; strip < 256; ++strip) {
                const auto gainRange = gains->getRange(det);

                const auto detstrip = strip + offset;
                const uint16_t noise = SiStripNoises::getRawNoise(detstrip, noises.getRange(det));
                const auto gain = SiStripGain::getStripGain(detstrip, gainRange);
                const auto bad = quality.IsStripBad(quality.getRange(det), detstrip);

                soaApv.gain()[apvIndex(fedId, fedCh, detstrip)] = gain;
                soaStr.noise()[stripIndex(fedId, fedCh, detstrip)] = noise;
                if (bad) {
                  soaStr.noise()[stripIndex(fedId, fedCh, detstrip)] |= badBit;
                }

              }
            }
          }
        }
      }
    }

    std::vector<uint32_t> idx(nChannels);
    iota(idx.begin(), idx.end(), 0);

    std::stable_sort(idx.begin(), idx.end(), [&soaDet](size_t i1, size_t i2) {
      
      return (soaDet.detId()[i2] == 0 && soaDet.detId()[i1] !=0) || (soaDet.detId()[i1] < soaDet.detId()[i2] || (soaDet.detId()[i1] == soaDet.detId()[i2] && soaDet.iPair()[i1] < soaDet.iPair()[i2]));
      });
    
    std::copy(idx.begin(), idx.end(), &soaDet.order()[0]);
    
    #ifdef GPU_STRIP_DEBUG
    auto cc = 0;
    for (int i = 0; i < nChannels; i++)
    {
      auto I = soaDet.order()[i];
      cc++;
      if(soaDet.detId()[I] == std::numeric_limits<uint32_t>::max())
        break;
      std::cout << "detToFeds: " << I << " " << soaDet.detId()[I] << " - " << soaDet.fedCh()[I] << std::endl;
    }
    
    
    std::cout << "THIS IS DONE: nClusters/nChannels: " << cc <<"/"<< nChannels << std::endl;
    #endif
       
    return product;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(SiStripClustersConditionsESProducer);
