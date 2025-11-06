#include <fstream>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <cassert>
#include <vector>
#include <iostream>  // <-- added for debug output

#include "CondFormats/SiPixelObjects/interface/SiPixelMappingHost.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLTHost.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"

#include "CalibTracker/Records/interface/SiPixelGainCalibrationForHLTSoARcd.h"
#include "CalibTracker/Records/interface/SiPixelMappingSoARecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelESProductReader : public stream::SynchronizingEDProducer<> {
  public:
    explicit SiPixelESProductReader(const edm::ParameterSet& iConfig);
    ~SiPixelESProductReader() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override;
    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;

    const edm::ESGetToken<SiPixelMappingHost, SiPixelMappingSoARecord> mappingToken_;
    const edm::ESGetToken<SiPixelGainCalibrationForHLTHost, SiPixelGainCalibrationForHLTSoARcd> gainToken_;

    device::EDPutToken<SiPixelClustersSoACollection> clusterPutToken_;

    const std::filesystem::path inputDir_;
  };

  SiPixelESProductReader::SiPixelESProductReader(const edm::ParameterSet& iConfig)
      : SynchronizingEDProducer(iConfig),
        mappingToken_(esConsumes()),
        gainToken_(esConsumes()),
        clusterPutToken_(produces()),
        inputDir_(iConfig.getParameter<std::string>("inputDir")) {}

  void SiPixelESProductReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("inputDir", ".")->setComment("Directory containing the dumped .bin files");
    descriptions.addWithDefaultLabel(desc);
  }

  template <typename T>
  void readBinary(std::ifstream& in, T& val) {
    in.read(reinterpret_cast<char*>(&val), sizeof(T));
  }

  void SiPixelESProductReader::acquire(device::Event const& iEvent, device::EventSetup const& iSetup) {
    std::cout << "[Reader] Starting acquire()..." << std::endl;
    auto const& mapping = iSetup.getData(mappingToken_);
    auto const& gains = iSetup.getData(gainToken_);
    std::cout << "[Reader] Loaded ESProducts from EventSetup." << std::endl;

    {
      std::cout << "[Reader] Reading SiPixelMappingHost from " << (inputDir_ / "SiPixelMappingHost.bin") << std::endl;
      std::ifstream in(inputDir_ / "SiPixelMappingHost.bin", std::ios::binary);
      in.exceptions(std::ifstream::badbit | std::ifstream::failbit);

      const auto view = mapping.const_view();
      unsigned int refSize = view.metadata().size();
      bool refHasQuality = view.hasQuality();
      std::cout << "  ES mapping size: " << refSize << ", hasQuality: " << refHasQuality << std::endl;

      unsigned int size;
      bool hasQuality;
      readBinary(in, size);
      readBinary(in, hasQuality);
      std::cout << "  BIN mapping size: " << size << ", hasQuality: " << hasQuality << std::endl;

      assert(size == refSize && "Mapping size mismatch");
      assert(hasQuality == refHasQuality && "Mapping hasQuality mismatch");

      std::vector<unsigned int> fed(size), link(size), roc(size), rawId(size), rocInDet(size), moduleId(size);
      std::vector<uint8_t> badRocs(size);
      std::vector<unsigned char> modToUnpDefault(size);

      in.read(reinterpret_cast<char*>(fed.data()), sizeof(unsigned int) * size);
      in.read(reinterpret_cast<char*>(link.data()), sizeof(unsigned int) * size);
      in.read(reinterpret_cast<char*>(roc.data()), sizeof(unsigned int) * size);
      in.read(reinterpret_cast<char*>(rawId.data()), sizeof(unsigned int) * size);
      in.read(reinterpret_cast<char*>(rocInDet.data()), sizeof(unsigned int) * size);
      in.read(reinterpret_cast<char*>(moduleId.data()), sizeof(unsigned int) * size);
      in.read(reinterpret_cast<char*>(badRocs.data()), sizeof(uint8_t) * size);
      in.read(reinterpret_cast<char*>(modToUnpDefault.data()), sizeof(unsigned char) * size);
      in.close();
      std::cout << "  Finished reading mapping arrays." << std::endl;

      for (unsigned int i = 0; i < size; ++i) {
        if (i < 5)
          std::cout << "   [Entry " << i << "] fed=" << fed[i] << " link=" << link[i] << " roc=" << roc[i] << std::endl;
        assert(fed[i] == view.fed(i));
        assert(link[i] == view.link(i));
        assert(roc[i] == view.roc(i));
        assert(rawId[i] == view.rawId(i));
        assert(rocInDet[i] == view.rocInDet(i));
        assert(moduleId[i] == view.moduleId(i));
        assert(badRocs[i] == view.badRocs(i));
        assert(modToUnpDefault[i] == view.modToUnpDefault(i));
      }
      std::cout << "  ✅ SiPixelMappingHost check passed." << std::endl;
    }

    {
      std::cout << "[Reader] Reading SiPixelGainCalibrationForHLTHost from "
                << (inputDir_ / "SiPixelGainCalibrationForHLTHost.bin") << std::endl;
      std::ifstream in(inputDir_ / "SiPixelGainCalibrationForHLTHost.bin", std::ios::binary);
      in.exceptions(std::ifstream::badbit | std::ifstream::failbit);

      const auto view = gains.const_view();
      unsigned int refSize = view.metadata().size();
      std::cout << "  ES gain size: " << refSize << std::endl;

      unsigned int size;
      readBinary(in, size);
      std::cout << "  BIN gain size: " << size << std::endl;
      assert(size == refSize && "Gain size mismatch");

      std::vector<siPixelGainsSoA::DecodingStructure> v_ped(size);
      in.read(reinterpret_cast<char*>(v_ped.data()), sizeof(siPixelGainsSoA::DecodingStructure) * size);
      std::cout << "  Read v_pedestals array (" << size << " entries)" << std::endl;

      siPixelGainsSoA::Ranges modStarts, modEnds;
      siPixelGainsSoA::Cols modCols;
      readBinary(in, modStarts);
      readBinary(in, modEnds);
      readBinary(in, modCols);

      float minPed, maxPed, minGain, maxGain, pedPrecision, gainPrecision;
      readBinary(in, minPed);
      readBinary(in, maxPed);
      readBinary(in, minGain);
      readBinary(in, maxGain);
      readBinary(in, pedPrecision);
      readBinary(in, gainPrecision);

      unsigned int numberOfRowsAveragedOver, nBinsToUseForEncoding, deadFlag, noisyFlag;
      readBinary(in, numberOfRowsAveragedOver);
      readBinary(in, nBinsToUseForEncoding);
      readBinary(in, deadFlag);
      readBinary(in, noisyFlag);

      float link;
      readBinary(in, link);
      in.close();

      for (unsigned int i = 0; i < size && i < 5; ++i)
        std::cout << "   [Ped entry " << i << "] gain=" << (int)v_ped[i].gain
                  << " ped=" << (int)v_ped[i].ped << std::endl;

      for (unsigned int i = 0; i < size; ++i) {
        assert(v_ped[i].gain == view.v_pedestals(i).gain);
        assert(v_ped[i].ped == view.v_pedestals(i).ped);
      }

      assert(modStarts == view.modStarts());
      assert(modEnds == view.modEnds());
      assert(modCols == view.modCols());
      assert(minPed == view.minPed());
      assert(maxPed == view.maxPed());
      assert(minGain == view.minGain());
      assert(maxGain == view.maxGain());
      assert(pedPrecision == view.pedPrecision());
      assert(gainPrecision == view.gainPrecision());
      assert(numberOfRowsAveragedOver == view.numberOfRowsAveragedOver());
      assert(nBinsToUseForEncoding == view.nBinsToUseForEncoding());
      assert(deadFlag == view.deadFlag());
      assert(noisyFlag == view.noisyFlag());
      assert(link == view.link());
      std::cout << "  ✅ SiPixelGainCalibrationForHLTHost check passed." << std::endl;
    }

    edm::LogInfo("SiPixelESProductReader") << "✅ All comparisons passed successfully for files in "
                                           << inputDir_.string();
    std::cout << "[Reader] Completed all checks successfully." << std::endl;
  }

  void SiPixelESProductReader::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    iEvent.emplace(clusterPutToken_, 0, iEvent.queue());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(SiPixelESProductReader);
