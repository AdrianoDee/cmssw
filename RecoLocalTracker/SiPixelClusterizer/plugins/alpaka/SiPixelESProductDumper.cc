#include <fstream>
#include <filesystem>
#include <memory>
#include <stdexcept>

#include "CondFormats/SiPixelObjects/interface/SiPixelMappingHost.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLTHost.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"

#include "CalibTracker/Records/interface/SiPixelGainCalibrationForHLTSoARcd.h"
#include "CalibTracker/Records/interface/SiPixelMappingSoARecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelESProductDumper : public stream::SynchronizingEDProducer<> {
  public:
    explicit SiPixelESProductDumper(const edm::ParameterSet& iConfig);
    ~SiPixelESProductDumper() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override;
    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;

    // ES tokens
    const edm::ESGetToken<SiPixelMappingHost, SiPixelMappingSoARecord> mappingToken_;
    const edm::ESGetToken<SiPixelGainCalibrationForHLTHost, SiPixelGainCalibrationForHLTSoARcd> gainToken_;

    // Output token (dummy)
    device::EDPutToken<SiPixelClustersSoACollection> clusterPutToken_;

    // Output directory for dumps
    const std::filesystem::path outputDir_;
  };

  // ------------------- Constructor -------------------
  SiPixelESProductDumper::SiPixelESProductDumper(const edm::ParameterSet& iConfig)
      : SynchronizingEDProducer(iConfig),
        mappingToken_(esConsumes()),
        gainToken_(esConsumes()),
        clusterPutToken_(produces()),
        outputDir_(iConfig.getParameter<std::string>("outputDir")) {}

  // ------------------- Fill Descriptions -------------------
  void SiPixelESProductDumper::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("outputDir", ".")->setComment("Directory where mapping and gains will be dumped.");
    descriptions.addWithDefaultLabel(desc);
  }

  // ------------------- Acquire -------------------
  void SiPixelESProductDumper::acquire(device::Event const& iEvent, device::EventSetup const& iSetup) {
    auto const& mapping = iSetup.getData(mappingToken_);
    auto const& gains = iSetup.getData(gainToken_);

    std::filesystem::create_directories(outputDir_);

    // ----------- Dump SiPixelMappingHost -----------
    {
      std::ofstream out(outputDir_ / "SiPixelMappingHost.bin", std::ios::binary);
      out.exceptions(std::ofstream::badbit | std::ofstream::failbit);

      const auto view = mapping.const_view();
      const unsigned int size = view.metadata().size();
      const bool hasQuality = view.hasQuality();

      out.write(reinterpret_cast<const char*>(&size), sizeof(unsigned int));
      out.write(reinterpret_cast<const char*>(&hasQuality), sizeof(bool));

      out.write(reinterpret_cast<const char*>(view.fed()), sizeof(unsigned int) * size);
      out.write(reinterpret_cast<const char*>(view.link()), sizeof(unsigned int) * size);
      out.write(reinterpret_cast<const char*>(view.roc()), sizeof(unsigned int) * size);
      out.write(reinterpret_cast<const char*>(view.rawId()), sizeof(unsigned int) * size);
      out.write(reinterpret_cast<const char*>(view.rocInDet()), sizeof(unsigned int) * size);
      out.write(reinterpret_cast<const char*>(view.moduleId()), sizeof(unsigned int) * size);
      out.write(reinterpret_cast<const char*>(view.badRocs()), sizeof(bool) * size);
      out.write(reinterpret_cast<const char*>(view.modToUnpDefault()), sizeof(unsigned char) * size);
      out.close();
    }

    // ----------- Dump SiPixelGainCalibrationForHLTHost -----------
    {
      std::ofstream out(outputDir_ / "SiPixelGainCalibrationForHLTHost.bin", std::ios::binary);
      out.exceptions(std::ofstream::badbit | std::ofstream::failbit);

      const auto view = gains.const_view();
      const unsigned int size = view.metadata().size();
      out.write(reinterpret_cast<const char*>(&size), sizeof(unsigned int));

      // column of DecodingStructure
      out.write(reinterpret_cast<const char*>(view.v_pedestals()),
                sizeof(siPixelGainsSoA::DecodingStructure) * size);

      // scalars (arrays and floats)
      out.write(reinterpret_cast<const char*>(&view.modStarts()), sizeof(view.modStarts()));
      out.write(reinterpret_cast<const char*>(&view.modEnds()), sizeof(view.modEnds()));
      out.write(reinterpret_cast<const char*>(&view.modCols()), sizeof(view.modCols()));

      out.write(reinterpret_cast<const char*>(&view.minPed()), sizeof(float));
      out.write(reinterpret_cast<const char*>(&view.maxPed()), sizeof(float));
      out.write(reinterpret_cast<const char*>(&view.minGain()), sizeof(float));
      out.write(reinterpret_cast<const char*>(&view.maxGain()), sizeof(float));
      out.write(reinterpret_cast<const char*>(&view.pedPrecision()), sizeof(float));
      out.write(reinterpret_cast<const char*>(&view.gainPrecision()), sizeof(float));

      out.write(reinterpret_cast<const char*>(&view.numberOfRowsAveragedOver()), sizeof(unsigned int));
      out.write(reinterpret_cast<const char*>(&view.nBinsToUseForEncoding()), sizeof(unsigned int));
      out.write(reinterpret_cast<const char*>(&view.deadFlag()), sizeof(unsigned int));
      out.write(reinterpret_cast<const char*>(&view.noisyFlag()), sizeof(unsigned int));
      out.write(reinterpret_cast<const char*>(&view.link()), sizeof(float));
      out.close();
    }

    edm::LogInfo("SiPixelESProductDumper") << "Dumped SiPixelMappingHost and SiPixelGainCalibrationForHLTHost to "
                                           << outputDir_.string();
  }

  // ------------------- Produce -------------------
  void SiPixelESProductDumper::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    // Produce an empty SiPixelClustersSoACollection to stay a valid producer
    iEvent.emplace(clusterPutToken_, 0, iEvent.queue());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// Define plugin
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(SiPixelESProductDumper);