#include <fstream>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFastParamsHost.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

#include "RecoLocalTracker/Records/interface/PixelCPEFastParamsRecord.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/alpaka/PixelCPEFastParamsCollection.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  class SiPixelCPEESProductDumper : public stream::SynchronizingEDProducer<> {
  public:
    explicit SiPixelCPEESProductDumper(const edm::ParameterSet& iConfig);
    ~SiPixelCPEESProductDumper() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override;
    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;

    edm::ESGetToken<PixelCPEFastParamsHost<TrackerTraits>, PixelCPEFastParamsRecord> cpeToken_;
    device::EDPutToken<SiPixelClustersSoACollection> dummyPutToken_;

    const std::filesystem::path outputDir_;
  };

  // ---------- Constructor ----------
  template <typename TrackerTraits>
  SiPixelCPEESProductDumper<TrackerTraits>::SiPixelCPEESProductDumper(const edm::ParameterSet& iConfig)
      : SynchronizingEDProducer(iConfig),
        cpeToken_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("CPE")))),
        dummyPutToken_(produces()),
        outputDir_(iConfig.getParameter<std::string>("outputDir")) {}

  // ---------- FillDescriptions ----------
  template <typename TrackerTraits>
  void SiPixelCPEESProductDumper<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    std::string cpe = "PixelCPEFastParams";
    cpe += TrackerTraits::nameModifier;
    desc.add<std::string>("CPE", cpe);
    desc.add<std::string>("outputDir", ".")->setComment("Directory where CPE parameters will be dumped.");
    descriptions.addWithDefaultLabel(desc);
  }

  // ---------- Acquire ----------
  template <typename TrackerTraits>
  void SiPixelCPEESProductDumper<TrackerTraits>::acquire(device::Event const& iEvent,
                                                        device::EventSetup const& iSetup) {
    namespace fs = std::filesystem;

    auto const& cpe = iSetup.getData(cpeToken_);
    auto const buffer = cpe.const_buffer();
    auto const* params = buffer.data();  // pointer to ParamsOnDeviceT<TrackerTraits>

    fs::create_directories(outputDir_);

    fs::path outpath = outputDir_ / (std::string("PixelCPEFastRun2_") + TrackerTraits::nameModifier + ".bin");
    std::ofstream out(outpath, std::ios::binary);
    out.exceptions(std::ofstream::badbit | std::ofstream::failbit);

    // --- Write CommonParams
    out.write(reinterpret_cast<const char*>(&params->m_commonParams),
              sizeof(pixelCPEforDevice::CommonParams));

    // --- Write number of modules
    unsigned int nModules = TrackerTraits::numberOfModules;
    out.write(reinterpret_cast<const char*>(&nModules), sizeof(unsigned int));

    // --- Write DetParams array
    out.write(reinterpret_cast<const char*>(&params->m_detParams),
              nModules * sizeof(pixelCPEforDevice::DetParams));

    out.close();

    edm::LogInfo("SiPixelCPEESProductDumper")
        << "Dumped PixelCPEFastParamsHost<" << TrackerTraits::nameModifier << "> to " << outpath.string();
  }

  // ---------- Produce ----------
  template <typename TrackerTraits>
  void SiPixelCPEESProductDumper<TrackerTraits>::produce(device::Event& iEvent,
                                                        device::EventSetup const& iSetup) {
    // Dummy product to stay valid as producer
    iEvent.emplace(dummyPutToken_, 0, iEvent.queue());
  }

  // ---------- Type aliases for concrete instantiations ----------
  using SiPixelCPEESProductDumperPhase1 = SiPixelCPEESProductDumper<pixelTopology::Phase1>;
  using SiPixelCPEESProductDumperPhase2 = SiPixelCPEESProductDumper<pixelTopology::Phase2>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// ---------- Register both as plugins ----------
DEFINE_FWK_ALPAKA_MODULE(SiPixelCPEESProductDumperPhase1);
DEFINE_FWK_ALPAKA_MODULE(SiPixelCPEESProductDumperPhase2);
