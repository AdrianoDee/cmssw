// OT Stub Producer VectorHitStyle - VectorHits-style stub formation for Phase-2 Outer Tracker
//
// Debug mode: Define OTSTUB_DEBUG_VERBOSE to enable verbose GPU->CPU debug transfers
// In BuildFile.xml: <flags CXXFLAGS="-DOTSTUB_DEBUG_VERBOSE"/>

#include <cmath>
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
#include "RecoTracker/PixelSeeding/plugins/alpaka/OTStubFormationVectorHitStyleKernelsWrapper.h"
#include "RecoTracker/Record/interface/StackedModuleGeometryRecord.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class OTStubProducerVectorHitStyle : public stream::SynchronizingEDProducer<> {
  public:
    explicit OTStubProducerVectorHitStyle(edm::ParameterSet const& iConfig);
    ~OTStubProducerVectorHitStyle() override = default;

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

    // Configuration
    std::vector<double> barrelFlatCut_;
    std::vector<double> barrelTiltedCut_;
    std::vector<double> endcapCut_;
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
    std::optional<cms::alpakatools::device_buffer<Device, float[]>> barrelFlatCut_d_;
    std::optional<cms::alpakatools::device_buffer<Device, float[]>> barrelTiltedCut_d_;
    std::optional<cms::alpakatools::device_buffer<Device, float[]>> endcapCut_d_;
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

  OTStubProducerVectorHitStyle::OTStubProducerVectorHitStyle(edm::ParameterSet const& iConfig)
      : SynchronizingEDProducer<>(iConfig),
        hitToken_(consumes(iConfig.getParameter<edm::InputTag>("otRecHitsSoA"))),
        geomToken_(esConsumes()),
        stubToken_(produces()),
        barrelFlatCut_(iConfig.getParameter<std::vector<double>>("barrelFlatCut")),
        barrelTiltedCut_(iConfig.getParameter<std::vector<double>>("barrelTiltedCut")),
        endcapCut_(iConfig.getParameter<std::vector<double>>("endcapCut")),
        barrelFlatMaxCSDiff_(iConfig.getParameter<std::vector<int32_t>>("barrelFlatMaxClusterSizeDiff")),
        barrelTiltedMaxCSDiff_(iConfig.getParameter<std::vector<int32_t>>("barrelTiltedMaxClusterSizeDiff")),
        endcapMaxCSDiff_(iConfig.getParameter<std::vector<int32_t>>("endcapMaxClusterSizeDiff")),
        barrelFlatMaxCS_(iConfig.getParameter<std::vector<int32_t>>("barrelFlatMaxClusterSize")),
        barrelTiltedMaxCS_(iConfig.getParameter<std::vector<int32_t>>("barrelTiltedMaxClusterSize")),
        endcapMaxCS_(iConfig.getParameter<std::vector<int32_t>>("endcapMaxClusterSize")),
        barrelFlatMaxCSSum_(iConfig.getParameter<std::vector<int32_t>>("barrelFlatMaxClusterSizeSum")),
        barrelTiltedMaxCSSum_(iConfig.getParameter<std::vector<int32_t>>("barrelTiltedMaxClusterSizeSum")),
        endcapMaxCSSum_(iConfig.getParameter<std::vector<int32_t>>("endcapMaxClusterSizeSum")) {}

  void OTStubProducerVectorHitStyle::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("otRecHitsSoA", edm::InputTag("otRecHitsSoAConverter"))
        ->setComment("Input OT RecHits SoA collection");
    desc.add<std::vector<double>>("barrelFlatCut", {0.0, 0.05, 0.06, 0.08, 0.09, 0.12, 0.2})
        ->setComment("Layer-dependent width cuts for flat barrel modules (cm), indexed by layer 0-6");
    desc.add<std::vector<double>>("barrelTiltedCut", {0.0, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15})
        ->setComment("Layer-dependent width cuts for tilted barrel modules (cm), indexed by layer 0-6. Only layers 1-3 have tilted modules.");
    desc.add<std::vector<double>>("endcapCut", {0.0, 0.1, 0.1, 0.1, 0.1, 0.1})
        ->setComment("Layer-dependent width cuts for endcap (cm), indexed by layer 0-5");
    desc.add<std::vector<int32_t>>("barrelFlatMaxClusterSizeDiff", {999, 999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max |clusterSize_lower - clusterSize_upper| for flat barrel (layers 0-6). 999 = disabled.");
    desc.add<std::vector<int32_t>>("barrelTiltedMaxClusterSizeDiff", {999, 999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max |clusterSize_lower - clusterSize_upper| for tilted barrel (layers 0-6). 999 = disabled.");
    desc.add<std::vector<int32_t>>("endcapMaxClusterSizeDiff", {999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max |clusterSize_lower - clusterSize_upper| for endcap (layers 0-5). 999 = disabled.");
    desc.add<std::vector<int32_t>>("barrelFlatMaxClusterSize", {999, 999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max cluster size for flat barrel (layers 0-6). 999 = disabled.");
    desc.add<std::vector<int32_t>>("barrelTiltedMaxClusterSize", {999, 999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max cluster size for tilted barrel (layers 0-6). 999 = disabled.");
    desc.add<std::vector<int32_t>>("endcapMaxClusterSize", {999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max cluster size for endcap (layers 0-5). 999 = disabled.");
    desc.add<std::vector<int32_t>>("barrelFlatMaxClusterSizeSum", {999, 999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max (clusterSize_lower + clusterSize_upper) for flat barrel (layers 0-6). 999 = disabled.");
    desc.add<std::vector<int32_t>>("barrelTiltedMaxClusterSizeSum", {999, 999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max (clusterSize_lower + clusterSize_upper) for tilted barrel (layers 0-6). 999 = disabled.");
    desc.add<std::vector<int32_t>>("endcapMaxClusterSizeSum", {999, 999, 999, 999, 999, 999})
        ->setComment("Per-layer max (clusterSize_lower + clusterSize_upper) for endcap (layers 0-5). 999 = disabled.");
    descriptions.addWithDefaultLabel(desc);
  }

  void OTStubProducerVectorHitStyle::acquire(device::Event const& iEvent, device::EventSetup const& iSetup) {
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
    // Note: Do NOT iterate over geomView here - on GPU backends, it points to device memory
    // which cannot be accessed from host code.
    caOffsets_.barrelStartGeom = 0;
    caOffsets_.backwardStartGeom = 0;
    caOffsets_.forwardStartGeom = 0;
    caOffsets_.nBarrelModules = 0;
    caOffsets_.nBackwardModules = 0;
    caOffsets_.nForwardModules = 0;

    // Get views of input hits (on device)
    auto const& hitsView = otRecHits.const_view().otRecHits();
    auto const& moduleView = otRecHits.const_view().otHitModules();

    // Copy cut arrays to device
    std::vector<float> barrelFlatCutFloat(barrelFlatCut_.begin(), barrelFlatCut_.end());
    std::vector<float> barrelTiltedCutFloat(barrelTiltedCut_.begin(), barrelTiltedCut_.end());
    std::vector<float> endcapCutFloat(endcapCut_.begin(), endcapCut_.end());

#ifdef OTSTUB_DEBUG_VERBOSE
    // Debug: print cut arrays
    edm::LogPrint("OTStubProducerVectorHitStyle") << "Barrel flat cuts:";
    for (size_t i = 0; i < barrelFlatCutFloat.size(); ++i) {
      edm::LogPrint("OTStubProducerVectorHitStyle") << "  barrelFlatCut[" << i << "] = " << barrelFlatCutFloat[i];
    }
    edm::LogPrint("OTStubProducerVectorHitStyle") << "Barrel tilted cuts:";
    for (size_t i = 0; i < barrelTiltedCutFloat.size(); ++i) {
      edm::LogPrint("OTStubProducerVectorHitStyle") << "  barrelTiltedCut[" << i << "] = " << barrelTiltedCutFloat[i];
    }
#endif

    barrelFlatCut_d_ = cms::alpakatools::make_device_buffer<float[]>(queue, barrelFlatCutFloat.size());
    barrelTiltedCut_d_ = cms::alpakatools::make_device_buffer<float[]>(queue, barrelTiltedCutFloat.size());
    endcapCut_d_ = cms::alpakatools::make_device_buffer<float[]>(queue, endcapCutFloat.size());

    alpaka::memcpy(
        queue, *barrelFlatCut_d_, cms::alpakatools::make_host_view(barrelFlatCutFloat.data(), barrelFlatCutFloat.size()));
    alpaka::memcpy(
        queue, *barrelTiltedCut_d_, cms::alpakatools::make_host_view(barrelTiltedCutFloat.data(), barrelTiltedCutFloat.size()));
    alpaka::memcpy(
        queue, *endcapCut_d_, cms::alpakatools::make_host_view(endcapCutFloat.data(), endcapCutFloat.size()));

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
    OTStubFormationVectorHitStyleKernelsWrapper kernels(queue);

    // Allocate device buffer for stub offsets (nModules + 1 to hold total at end)
    stubOffsets_d_ = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nModules_ + 1);
    alpaka::memset(queue, *stubOffsets_d_, 0);

    // Launch counting kernel - fills stubOffsets with counts
    kernels.countStubs(queue,
                       hitsView,
                       moduleView,
                       geomView,
                       barrelFlatCut_d_->data(),
                       barrelTiltedCut_d_->data(),
                       endcapCut_d_->data(),
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

    // Finalize offsets using device-side prefix scan (converts to exclusive scan)
    kernels.finalizeOffsets(queue, stubOffsets_d_->data(), nModules_);

    // Copy ONLY the total number of stubs (1 uint32_t) - needed for allocation in produce()
    // This is an async copy; framework will synchronize queue before produce()
    nStubs_h_ = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, 1u);
    auto nStubs_view_d =
        cms::alpakatools::make_device_view(alpaka::getDev(queue), stubOffsets_d_->data() + nModules_, 1u);
    alpaka::memcpy(queue, *nStubs_h_, nStubs_view_d);

    // Always print total stub count for CPU vs GPU comparison diagnostic
    alpaka::wait(queue);
    uint32_t totalStubs = nStubs_h_->data()[0];
    edm::LogPrint("OTStubDebug") << "EVENT STUBS: total=" << totalStubs;

    // Per-module stub count diagnostic: copy full offsets array to host and print per-module counts
    {
      auto offsets_h = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, nModules_ + 1);
      alpaka::memcpy(queue, offsets_h, *stubOffsets_d_);
      alpaka::wait(queue);
      auto const* offsets = alpaka::getPtrNative(offsets_h);

      // Per-layer summary
      // We need geometry to know which layer each module belongs to, but geometry is on device.
      // Instead, print a hash of all per-module counts for quick mismatch detection.
      uint64_t moduleHash = 0;
      uint32_t nModulesWithStubs = 0;
      for (uint32_t i = 0; i < nModules_; ++i) {
        uint32_t count = offsets[i + 1] - offsets[i];
        if (count > 0) {
          nModulesWithStubs++;
          // Simple hash: combine module index and count
          moduleHash ^= (uint64_t(i) * 2654435761u) ^ (uint64_t(count) * 40503u);
        }
      }
      // edm::LogPrint("OTStubDebug") << "EVENT STUBS: modulesWithStubs=" << nModulesWithStubs
      //                               << " moduleHash=0x" << std::hex << moduleHash << std::dec;

      // Print first 20 modules with stubs for detailed comparison
      // uint32_t printed = 0;
      // for (uint32_t i = 0; i < nModules_ && printed < 20; ++i) {
      //   uint32_t count = offsets[i + 1] - offsets[i];
      //   if (count > 0) {
      //     edm::LogPrint("OTStubDebug") << "  module=" << i << " stubs=" << count;
      //     printed++;
      //   }
      // }
    }

#ifdef OTSTUB_DEBUG_VERBOSE
    // Debug mode: print additional info
    edm::LogPrint("OTStubProducerVectorHitStyle") << "VectorHitStyle: Total stubs counted: " << totalStubs;
#endif
  }

  void OTStubProducerVectorHitStyle::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
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
      OTStubFormationVectorHitStyleKernelsWrapper kernels(queue);
      kernels.formStubs(queue,
                        hitsView,
                        moduleView,
                        geomView,
                        barrelFlatCut_d_->data(),
                        barrelTiltedCut_d_->data(),
                        endcapCut_d_->data(),
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
    barrelFlatCut_d_.reset();
    barrelTiltedCut_d_.reset();
    endcapCut_d_.reset();
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
DEFINE_FWK_ALPAKA_MODULE(OTStubProducerVectorHitStyle);
