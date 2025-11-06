#include <fstream>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"


#define GPU_DEBUG

class TrackingRecHitHostDumper : public edm::global::EDProducer<> {
public:
  explicit TrackingRecHitHostDumper(const edm::ParameterSet& iConfig);
  ~TrackingRecHitHostDumper() override = default;

  using HitsOnHost = ::reco::TrackingRecHitHost;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::EDGetTokenT<HitsOnHost> tokenHit_;

  const std::filesystem::path outputDir_;
  const std::string fileName_;

  static constexpr uint32_t kVersion = 1;
  static constexpr uint32_t kEndianness = 0x01020304;
  static constexpr char kMagic[4] = {'T', 'R', 'H', '1'};

  static inline bool headerWritten_ = false;
  static inline uint32_t nEventsWritten_ = 0;

  edm::EDPutTokenT<SiPixelClustersHost> dummyToken_;
};

TrackingRecHitHostDumper::TrackingRecHitHostDumper(const edm::ParameterSet& iConfig)
    : tokenHit_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
      outputDir_(iConfig.getParameter<std::string>("outputDir")),
      fileName_((outputDir_ / "TrackingRecHitHost.bin").string()),
      dummyToken_(produces()) {
#ifdef GPU_DEBUG
  std::cout << "[TrackingRecHitHostDumper] Output file: " << fileName_ << std::endl;
#endif
  std::filesystem::create_directories(outputDir_);
}

void TrackingRecHitHostDumper::produce(edm::StreamID streamID,
                                         edm::Event& iEvent,
                                         const edm::EventSetup& iSetup) const  {
  auto const& hits = iEvent.get(tokenHit_);

  auto hitView = hits.view<::reco::TrackingRecHitSoA>();
  auto modView = hits.view<::reco::HitModuleSoA>();

  uint32_t nHits = hits.nHits();
  uint32_t nModules = hits.nModules();

  std::ofstream out(fileName_, std::ios::binary | std::ios::app);
  if (!out.is_open()) {
    throw cms::Exception("FileOpenError") << "Cannot open " << fileName_ << " for writing.";
  }

  // Write header once
  if (!headerWritten_) {
    out.write(kMagic, 4);
    out.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
    out.write(reinterpret_cast<const char*>(&kEndianness), sizeof(kEndianness));

    uint32_t dummyCount = 0;
    out.write(reinterpret_cast<const char*>(&dummyCount), sizeof(dummyCount));

    headerWritten_ = true;
#ifdef GPU_DEBUG
    std::cout << "[TrackingRecHitHostDumper] Wrote file header." << std::endl;
#endif
  }

#ifdef GPU_DEBUG
  std::cout << "[TrackingRecHitHostDumper] Dumping event "
            << nEventsWritten_ << " with " << nHits
            << " hits and " << nModules << " modules.\n";
#endif

  // Event header
  out.write(reinterpret_cast<const char*>(&nHits), sizeof(nHits));
  out.write(reinterpret_cast<const char*>(&nModules), sizeof(nModules));

  // Module start
  out.write(reinterpret_cast<const char*>(modView.moduleStart()),
            (nModules + 1) * sizeof(uint32_t));

  auto write_column = [&](auto ptr) {
      using ElemT = std::remove_cv_t<std::remove_pointer_t<decltype(ptr)>>;
      out.write(reinterpret_cast<const char*>(ptr), nHits * sizeof(ElemT));
  };

  out.write(reinterpret_cast<const char*>(modView.moduleStart()),
        (nModules + 1) * sizeof(uint32_t));

  write_column(hitView.xLocal());
  write_column(hitView.yLocal());
  write_column(hitView.xerrLocal());
  write_column(hitView.yerrLocal());
  write_column(hitView.xGlobal());
  write_column(hitView.yGlobal());
  write_column(hitView.zGlobal());
  write_column(hitView.rGlobal());
  write_column(hitView.iphi());
  write_column(hitView.chargeAndStatus());
  write_column(hitView.clusterSizeX());
  write_column(hitView.clusterSizeY());
  write_column(hitView.detectorIndex());

#ifdef GPU_DEBUG
  if (nHits > 0) {
    std::cout << "  First hit (x,y,z,r)="
              << hitView.xGlobal()[0] << ", "
              << hitView.yGlobal()[0] << ", "
              << hitView.zGlobal()[0] << ", "
              << hitView.rGlobal()[0]
              << " detIdx=" << hitView.detectorIndex()[0] << std::endl;
  }
#endif

  ++nEventsWritten_;
  out.close();

  // produce dummy output (empty SoA)
  iEvent.emplace(dummyToken_, 0, cms::alpakatools::host());
}

void TrackingRecHitHostDumper::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("outputDir", ".")
      ->setComment("Directory where TrackingRecHitHost SoA will be dumped in binary format.");
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("ssiPixelRecHitsPreSplittingSoA"))
      ->setComment("Input hits.");
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackingRecHitHostDumper);
