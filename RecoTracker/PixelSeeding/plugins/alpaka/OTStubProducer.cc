// OT Stub Producer - Bend-based stub formation for Phase-2 Outer Tracker
//
// Debug mode: Define OTSTUB_DEBUG_VERBOSE to enable verbose GPU->CPU debug transfers
// In BuildFile.xml: <flags CXXFLAGS="-DOTSTUB_DEBUG_VERBOSE"/>

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

#include "DataFormats/TrackingRecHitSoA/interface/alpaka/OTRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/StubsSoACollection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/StackedModuleGeometrySoACollection.h"
#include "RecoTracker/PixelSeeding/plugins/alpaka/OTStubFormationKernelsWrapper.h"
#include "RecoTracker/Record/interface/StackedModuleGeometryRecord.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class OTStubProducer : public stream::SynchronizingEDProducer<> {
  public:
    explicit OTStubProducer(edm::ParameterSet const& iConfig);
    ~OTStubProducer() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override;
    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;

    // Input tokens
    device::EDGetToken<reco::OTRecHitsSoACollection> hitToken_;

    // EventSetup tokens - using device::ESGetToken for automatic host-to-device transfer
    device::ESGetToken<reco::StackedModuleGeometrySoACollection, StackedModuleGeometryRecord> geomToken_;

    // Output token
    device::EDPutToken<reco::StubsSoACollection> stubToken_;

    // Configuration - per-layer cluster size cuts
    std::vector<int32_t> barrelFlatMaxCSDiff_;
    std::vector<int32_t> barrelTiltedMaxCSDiff_;
    std::vector<int32_t> endcapMaxCSDiff_;
    std::vector<int32_t> barrelFlatMaxCS_;
    std::vector<int32_t> barrelTiltedMaxCS_;
    std::vector<int32_t> endcapMaxCS_;
    std::vector<int32_t> barrelFlatMaxCSSum_;
    std::vector<int32_t> barrelTiltedMaxCSSum_;
    std::vector<int32_t> endcapMaxCSSum_;

    // Member variables for acquire/produce communication
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> stubOffsets_d_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> barrelFlatMaxCSDiff_d_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> barrelTiltedMaxCSDiff_d_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> endcapMaxCSDiff_d_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> barrelFlatMaxCS_d_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> barrelTiltedMaxCS_d_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> endcapMaxCS_d_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> barrelFlatMaxCSSum_d_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> barrelTiltedMaxCSSum_d_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> endcapMaxCSSum_d_;
    std::optional<cms::alpakatools::host_buffer<uint32_t[]>> nStubs_h_;
    uint32_t nModules_ = 0;

    // CA module offsets (computed when geometry is loaded)
    otStubFormation::CAModuleOffsets caOffsets_;
  };

  OTStubProducer::OTStubProducer(edm::ParameterSet const& iConfig)
      : SynchronizingEDProducer<>(iConfig),
        hitToken_(consumes(iConfig.getParameter<edm::InputTag>("otRecHitsSoA"))),
        geomToken_(esConsumes()),
        stubToken_(produces()),
        barrelFlatMaxCSDiff_(iConfig.getParameter<std::vector<int32_t>>("maxClusterSizeDiffBarrelFlat")),
        barrelTiltedMaxCSDiff_(iConfig.getParameter<std::vector<int32_t>>("maxClusterSizeDiffBarrelTilted")),
        endcapMaxCSDiff_(iConfig.getParameter<std::vector<int32_t>>("maxClusterSizeDiffEndcap")),
        barrelFlatMaxCS_(iConfig.getParameter<std::vector<int32_t>>("maxClusterSizeBarrelFlat")),
        barrelTiltedMaxCS_(iConfig.getParameter<std::vector<int32_t>>("maxClusterSizeBarrelTilted")),
        endcapMaxCS_(iConfig.getParameter<std::vector<int32_t>>("maxClusterSizeEndcap")),
        barrelFlatMaxCSSum_(iConfig.getParameter<std::vector<int32_t>>("maxClusterSizeBarrelFlatSum")),
        barrelTiltedMaxCSSum_(iConfig.getParameter<std::vector<int32_t>>("maxClusterSizeSumBarrelTilted")),
        endcapMaxCSSum_(iConfig.getParameter<std::vector<int32_t>>("maxClusterSizeSumEndcap")) {}

  void OTStubProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("otRecHitsSoA", edm::InputTag("otRecHitsSoAConverter"))
        ->setComment("Input OT RecHits SoA collection");
    desc.add<std::vector<int32_t>>("maxClusterSizeDiffBarrelFlat", {999, 999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max |clusterSize_lower - clusterSize_upper| for flat barrel (layers 0-6). 999 = disabled.");
    desc.add<std::vector<int32_t>>("maxClusterSizeDiffBarrelTilted", {999, 999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max |clusterSize_lower - clusterSize_upper| for tilted barrel (layers 0-6). 999 = disabled.");
    desc.add<std::vector<int32_t>>("maxClusterSizeDiffEndcap", {999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max |clusterSize_lower - clusterSize_upper| for endcap (layers 0-5). 999 = disabled.");
    desc.add<std::vector<int32_t>>("maxClusterSizeBarrelFlat", {999, 999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max cluster size for flat barrel (layers 0-6). 999 = disabled.");
    desc.add<std::vector<int32_t>>("maxClusterSizeBarrelTilted", {999, 999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max cluster size for tilted barrel (layers 0-6). 999 = disabled.");
    desc.add<std::vector<int32_t>>("maxClusterSizeEndcap", {999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max cluster size for endcap (layers 0-5). 999 = disabled.");
    desc.add<std::vector<int32_t>>("maxClusterSizeBarrelFlatSum", {999, 999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max (clusterSize_lower + clusterSize_upper) for flat barrel (layers 0-6). 999 = disabled.");
    desc.add<std::vector<int32_t>>("maxClusterSizeSumBarrelTilted", {999, 999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max (clusterSize_lower + clusterSize_upper) for tilted barrel (layers 0-6). 999 = disabled.");
    desc.add<std::vector<int32_t>>("maxClusterSizeSumEndcap", {999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max (clusterSize_lower + clusterSize_upper) for endcap (layers 0-5). 999 = disabled.");
    descriptions.addWithDefaultLabel(desc);
  }

  void OTStubProducer::acquire(device::Event const& iEvent, device::EventSetup const& iSetup) {
    // Get queue for kernel launches
    auto queue = iEvent.queue();

    // Get input hits SoA (on device)
    auto const& otRecHits = iEvent.get(hitToken_);

    // Get geometry from EventSetup (automatically transferred to device by the framework)
    auto const& geomDevice = iSetup.getData(geomToken_);
    auto const& geomView = geomDevice.const_view();
    nModules_ = geomView.metadata().size();

    // CA module offsets: Since StackedModuleGeometry is now sorted in CA order
    // (barrel by layer, backward, forward), the CA module index is simply:
    // caModuleIndex = nPixelModules + geometryIndex
    // The CAModuleOffsets struct is kept for interface compatibility but only nPixelModules is used
    caOffsets_.barrelStartGeom = 0;
    caOffsets_.backwardStartGeom = 0;
    caOffsets_.forwardStartGeom = 0;
    caOffsets_.nBarrelModules = 0;
    caOffsets_.nBackwardModules = 0;
    caOffsets_.nForwardModules = 0;

    // Get views of input hits (on device)
    auto const& hitsView = otRecHits.const_view().otRecHits();
    auto const& moduleView = otRecHits.const_view().otHitModules();

    // Copy per-layer cluster size cut arrays to device
    barrelFlatMaxCSDiff_d_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, barrelFlatMaxCSDiff_.size());
    barrelTiltedMaxCSDiff_d_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, barrelTiltedMaxCSDiff_.size());
    endcapMaxCSDiff_d_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, endcapMaxCSDiff_.size());
    barrelFlatMaxCS_d_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, barrelFlatMaxCS_.size());
    barrelTiltedMaxCS_d_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, barrelTiltedMaxCS_.size());
    endcapMaxCS_d_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, endcapMaxCS_.size());

    alpaka::memcpy(queue, *barrelFlatMaxCSDiff_d_,
                   cms::alpakatools::make_host_view(barrelFlatMaxCSDiff_.data(), barrelFlatMaxCSDiff_.size()));
    alpaka::memcpy(queue, *barrelTiltedMaxCSDiff_d_,
                   cms::alpakatools::make_host_view(barrelTiltedMaxCSDiff_.data(), barrelTiltedMaxCSDiff_.size()));
    alpaka::memcpy(queue, *endcapMaxCSDiff_d_,
                   cms::alpakatools::make_host_view(endcapMaxCSDiff_.data(), endcapMaxCSDiff_.size()));
    alpaka::memcpy(queue, *barrelFlatMaxCS_d_,
                   cms::alpakatools::make_host_view(barrelFlatMaxCS_.data(), barrelFlatMaxCS_.size()));
    alpaka::memcpy(queue, *barrelTiltedMaxCS_d_,
                   cms::alpakatools::make_host_view(barrelTiltedMaxCS_.data(), barrelTiltedMaxCS_.size()));
    alpaka::memcpy(queue, *endcapMaxCS_d_,
                   cms::alpakatools::make_host_view(endcapMaxCS_.data(), endcapMaxCS_.size()));

    barrelFlatMaxCSSum_d_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, barrelFlatMaxCSSum_.size());
    barrelTiltedMaxCSSum_d_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, barrelTiltedMaxCSSum_.size());
    endcapMaxCSSum_d_ = cms::alpakatools::make_device_buffer<int32_t[]>(queue, endcapMaxCSSum_.size());

    alpaka::memcpy(queue, *barrelFlatMaxCSSum_d_,
                   cms::alpakatools::make_host_view(barrelFlatMaxCSSum_.data(), barrelFlatMaxCSSum_.size()));
    alpaka::memcpy(queue, *barrelTiltedMaxCSSum_d_,
                   cms::alpakatools::make_host_view(barrelTiltedMaxCSSum_.data(), barrelTiltedMaxCSSum_.size()));
    alpaka::memcpy(queue, *endcapMaxCSSum_d_,
                   cms::alpakatools::make_host_view(endcapMaxCSSum_.data(), endcapMaxCSSum_.size()));

    // Create kernel wrapper
    OTStubFormationKernelsWrapper kernels(queue);

    // Allocate device buffer for stub offsets (nModules + 1 to hold total at end)
    stubOffsets_d_ = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nModules_ + 1);
    alpaka::memset(queue, *stubOffsets_d_, 0);

    // Launch counting kernel - fills stubOffsets with counts
    kernels.countStubs(queue,
                       hitsView,
                       moduleView,
                       geomView,
                       barrelFlatMaxCSDiff_d_->data(),
                       barrelTiltedMaxCSDiff_d_->data(),
                       endcapMaxCSDiff_d_->data(),
                       barrelFlatMaxCS_d_->data(),
                       barrelTiltedMaxCS_d_->data(),
                       endcapMaxCS_d_->data(),
                       barrelFlatMaxCSSum_d_->data(),
                       barrelTiltedMaxCSSum_d_->data(),
                       endcapMaxCSSum_d_->data(),
                       stubOffsets_d_->data(),
                       nModules_);

#ifdef OTSTUB_DEBUG_VERBOSE
    // DEBUG: Check counts before prefix scan (GPU->CPU transfer for debugging only)
    alpaka::wait(queue);
    auto stubCounts_h = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, nModules_ + 1);
    alpaka::memcpy(queue, stubCounts_h, *stubOffsets_d_);
    alpaka::wait(queue);
    edm::LogPrint("OTStubProducer") << "=== Stub COUNTS (before prefix scan) ===";
    edm::LogPrint("OTStubProducer") << "First 20 count values:";
    for (uint32_t i = 0; i < std::min(20u, nModules_ + 1); ++i) {
      edm::LogPrint("OTStubProducer") << "  stubCounts[" << i << "] = " << stubCounts_h.data()[i];
    }
    edm::LogPrint("OTStubProducer") << "Counts for known modules with hits:";
    std::vector<uint32_t> checkModules = {6158, 6224, 6851, 6998, 7951, 8227, 9082, 9662, 10305, 11054, 11840, 12783};
    for (uint32_t idx : checkModules) {
      if (idx < nModules_) {
        edm::LogPrint("OTStubProducer") << "  stubCounts[" << idx << "] = " << stubCounts_h.data()[idx];
      }
    }
    edm::LogPrint("OTStubProducer") << "";
#endif

    // Finalize offsets using device-side prefix scan
    kernels.finalizeOffsets(queue, stubOffsets_d_->data(), nModules_);

    // Copy ONLY the total number of stubs (1 uint32_t) - needed for allocation in produce()
    // This is an async copy; framework will synchronize queue before produce()
    nStubs_h_ = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, 1u);
    auto nStubs_view_d =
        cms::alpakatools::make_device_view(alpaka::getDev(queue), stubOffsets_d_->data() + nModules_, 1u);
    alpaka::memcpy(queue, *nStubs_h_, nStubs_view_d);

#ifdef OTSTUB_DEBUG_VERBOSE
    // Debug mode: Wait and print debug info
    alpaka::wait(queue);
    uint32_t totalStubs = nStubs_h_->data()[0];
    edm::LogPrint("OTStubProducer") << "=== OTStubProducer Debug ===";
    edm::LogPrint("OTStubProducer") << "Total stubs counted: " << totalStubs;

    // Additional GPU->CPU transfer for debugging stubOffsets array
    auto stubOffsets_h = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, nModules_ + 1);
    alpaka::memcpy(queue, stubOffsets_h, *stubOffsets_d_);
    alpaka::wait(queue);

    edm::LogPrint("OTStubProducer") << "First 20 stubOffsets values:";
    for (uint32_t i = 0; i < std::min(20u, nModules_ + 1); ++i) {
      edm::LogPrint("OTStubProducer") << "  stubOffsets[" << i << "] = " << stubOffsets_h.data()[i];
    }

    edm::LogPrint("OTStubProducer") << "\nStub offsets for modules with stubs:";
    for (uint32_t i = 0; i < std::min(50u, nModules_ + 1); ++i) {
      if (i > 0 && stubOffsets_h.data()[i] != stubOffsets_h.data()[i-1]) {
        edm::LogPrint("OTStubProducer") << "  Module " << i << ": stubOffsets[" << i << "] = " << stubOffsets_h.data()[i]
                                          << " (count: " << (stubOffsets_h.data()[i] - stubOffsets_h.data()[i-1]) << ")";
      }
    }
    edm::LogPrint("OTStubProducer") << "";
#endif

    // Production mode: No wait - framework synchronizes before produce()
  }

  void OTStubProducer::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    // Get queue for kernel launches
    auto queue = iEvent.queue();

    // Framework has synchronized - read total stub count
    uint32_t nStubs = nStubs_h_->data()[0];

    // Allocate output stubs collection
    reco::StubsSoACollection outputStubs(queue, nStubs, nModules_);

    if (nStubs > 0) {
      // Get input hits and geometry again for views
      auto const& otRecHits = iEvent.get(hitToken_);
      auto const& geomDevice = iSetup.getData(geomToken_);
      auto const& geomView = geomDevice.const_view();

      // Get views
      auto const& hitsView = otRecHits.const_view().otRecHits();
      auto const& moduleView = otRecHits.const_view().otHitModules();

      // Get output views
      auto stubsView = outputStubs.view().stubs();
      auto stubModuleView = outputStubs.view().stubModules();

      // Create kernel wrapper and launch formation kernel
      OTStubFormationKernelsWrapper kernels(queue);
      kernels.formStubs(queue,
                        hitsView,
                        moduleView,
                        geomView,
                        barrelFlatMaxCSDiff_d_->data(),
                        barrelTiltedMaxCSDiff_d_->data(),
                        endcapMaxCSDiff_d_->data(),
                        barrelFlatMaxCS_d_->data(),
                        barrelTiltedMaxCS_d_->data(),
                        endcapMaxCS_d_->data(),
                        barrelFlatMaxCSSum_d_->data(),
                        barrelTiltedMaxCSSum_d_->data(),
                        endcapMaxCSSum_d_->data(),
                        stubOffsets_d_->data(),
                        stubsView,
                        nModules_,
                        caOffsets_);

      // Set offsetStubs scalar (must use memcpy for device memory)
      int32_t zero = 0;
      auto offset_h = cms::alpakatools::make_host_view(zero);
      auto offset_d = cms::alpakatools::make_device_view(queue, stubsView.offsetStubs());
      alpaka::memcpy(queue, offset_d, offset_h);

      // Fill module start array from stubOffsets (on device)
      kernels.fillModuleStart(queue, stubOffsets_d_->data(), stubModuleView, nModules_);
    }

    // Put output
    iEvent.emplace(stubToken_, std::move(outputStubs));

    // Clean up member variables
    stubOffsets_d_.reset();
    barrelFlatMaxCSDiff_d_.reset();
    barrelTiltedMaxCSDiff_d_.reset();
    endcapMaxCSDiff_d_.reset();
    barrelFlatMaxCS_d_.reset();
    barrelTiltedMaxCS_d_.reset();
    endcapMaxCS_d_.reset();
    barrelFlatMaxCSSum_d_.reset();
    barrelTiltedMaxCSSum_d_.reset();
    endcapMaxCSSum_d_.reset();
    nStubs_h_.reset();
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(OTStubProducer);
