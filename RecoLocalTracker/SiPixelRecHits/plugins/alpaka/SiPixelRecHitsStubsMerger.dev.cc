#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/StubsSoACollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

//#define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  /**
   * @brief Merges pixel RecHits with OT stubs into a unified TrackingRecHitsSoACollection
   *
   * This module combines:
   * - Pixel hits (from pixel detector)
   * - OT stub-derived hits (from outer tracker stubs with direction information)
   *
   * The merged collection is used by the Cellular Automaton tracking for hybrid pixel+stub tracking.
   * Stub-derived hits have isStub=true and contain direction information (dPhiDr) for improved tracking.
   *
   * CRITICAL: The merged collection uses a dense moduleStart array indexed by CA module indices:
   * - Modules 0-3999: Pixel modules
   * - Modules 4000-17199: OT modules (barrel, backward disks, forward disks)
   *
   * Stubs are copied preserving their StubsSoA order (already grouped by CA module from
   * the formation kernel's prefix scan). This ensures stubIndex = hitIndex - offsetStubs
   * correctly maps back to the original StubsSoA for the fitter's SS stub expansion.
   *
   * The offsetStubs scalar is set to nPixHits, allowing:
   * - stubIndex = hitIndex - offsetStubs (for hits from stubs)
   * - Access to original stub data via StubsSoACollection[stubIndex]
   */
  class SiPixelRecHitsStubsMerger : public global::EDProducer<> {
  public:
    explicit SiPixelRecHitsStubsMerger(const edm::ParameterSet& iConfig);
    ~SiPixelRecHitsStubsMerger() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::StreamID streamID, device::Event& iEvent, const device::EventSetup& iSetup) const override;

    const device::EDGetToken<reco::TrackingRecHitsSoACollection> pixelRecHitToken_;
    const device::EDGetToken<reco::StubsSoACollection> stubsToken_;

    const device::EDPutToken<reco::TrackingRecHitsSoACollection> outputRecHitsSoAToken_;
    const device::EDPutToken<reco::TrackingRecHitsMaskingCollection> outputRecHitsMaskToken_;
  };

  SiPixelRecHitsStubsMerger::SiPixelRecHitsStubsMerger(const edm::ParameterSet& iConfig)
      : EDProducer(iConfig),
        pixelRecHitToken_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitsSoA"))),
        stubsToken_(consumes(iConfig.getParameter<edm::InputTag>("stubsSoA"))),
        outputRecHitsSoAToken_(produces()),
        outputRecHitsMaskToken_(produces()) {}

  void SiPixelRecHitsStubsMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("pixelRecHitsSoA", edm::InputTag("siPixelRecHitsPreSplittingAlpaka"));
    desc.add<edm::InputTag>("stubsSoA", edm::InputTag("otStubProducer"));

    descriptions.addWithDefaultLabel(desc);
  }

  namespace {
    // Module count for Phase2OTStubs (from SimplePixelTopology)
    constexpr uint32_t nModulesCA = phase2PixelTopology::nModulesTotStubs;  // 17200
    constexpr uint32_t nPixelModules = phase2PixelTopology::nModulesPix;   // 4000

    struct LaunchZerosPixelMask {
    template <typename TAcc, typename MaskView>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    MaskView mask) const {
        for(uint32_t ic : cms::alpakatools::independent_group_elements(acc, mask.metadata().size())){
          assert(ic < (uint32_t)mask.metadata().size());
          mask[ic].recHitMask() = 0;
        }
      }
    };

    // Kernel to initialize stub fields for pixel hits (set to zero/false)
    struct InitializePixelStubFieldsKernel {
      template <typename TAcc, typename HitsView>
      ALPAKA_FN_ACC void operator()(const TAcc& acc, HitsView hitsView, uint32_t nPixHits) const {
        for (uint32_t hitIdx : cms::alpakatools::uniform_elements(acc, nPixHits)) {
          hitsView.isStub()[hitIdx] = false;
          hitsView.stubType()[hitIdx] = 0;
          hitsView.dPhiDr()[hitIdx] = 0.0f;
          hitsView.dPhiDrError()[hitIdx] = -1.0f;
          hitsView.ptEst()[hitIdx] = 0.0f;
          // Set pHitGroupId to max value (invalid) for pixel hits
          hitsView.pHitGroupId()[hitIdx] = UINT32_MAX;
          hitsView.stubFlags()[hitIdx] = 0;
        }
      }
    };

    // Kernel to initialize moduleStart array for OT modules
    // Sets all OT entries to point to the end of stub hits (indicating empty)
    struct InitializeOTModuleStartKernel {
      template <typename TAcc, typename ModuleView>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    ModuleView moduleView,
                                    uint32_t nPixHits,
                                    uint32_t nStubs,
                                    uint32_t nOTModulesActual) const {
        uint32_t endOffset = nPixHits + nStubs;
        // Initialize OT module entries (4000 to 17200 inclusive)
        for (uint32_t i : cms::alpakatools::uniform_elements(acc, nOTModulesActual + 1)) {
          moduleView.moduleStart()[nPixelModules + i] = endOffset;
        }
      }
    };

    // Kernel to count stubs per CA module
    // Input: stubs in geometry order with detectorIndex set to CA module
    // Output: stubCounts[caModule] = number of stubs in that CA module
    struct CountStubsPerCAModuleKernel {
      template <typename TAcc, typename StubsView>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    StubsView stubsView,
                                    uint32_t* stubCounts,
                                    uint32_t nStubs) const {
        for (uint32_t stubIdx : cms::alpakatools::uniform_elements(acc, nStubs)) {
          uint32_t caModule = stubsView.detectorIndex()[stubIdx];
          // Atomic increment for this CA module
          alpaka::atomicAdd(acc, &stubCounts[caModule], 1u, alpaka::hierarchy::Blocks{});
        }
      }
    };

    // Kernel to copy stubs to the merged hit collection preserving StubsSoA order.
    // Stubs are placed at hitIdx = nPixHits + stubIdx, so that the fitter's
    // mapping stubIdx = hitIdx - offsetStubs correctly indexes the StubsSoA.
    // The StubsSoA already has stubs grouped by module (from the formation kernel's
    // prefix scan), matching the moduleStart layout in the merged collection.
    struct CopyStubsToHitsKernel {
      template <typename TAcc, typename StubsView, typename HitsView>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    StubsView stubsView,
                                    HitsView hitsView,
                                    uint32_t nPixHits,
                                    uint32_t nStubs) const {
        for (uint32_t stubIdx : cms::alpakatools::uniform_elements(acc, nStubs)) {
          uint32_t hitIdx = nPixHits + stubIdx;

          // Copy position information
          hitsView.xLocal()[hitIdx] = stubsView.xLocal()[stubIdx];
          hitsView.yLocal()[hitIdx] = stubsView.yLocal()[stubIdx];
          hitsView.xerrLocal()[hitIdx] = stubsView.xerrLocal()[stubIdx];
          hitsView.yerrLocal()[hitIdx] = stubsView.yerrLocal()[stubIdx];

          hitsView.xGlobal()[hitIdx] = stubsView.xGlobal()[stubIdx];
          hitsView.yGlobal()[hitIdx] = stubsView.yGlobal()[stubIdx];
          hitsView.zGlobal()[hitIdx] = stubsView.zGlobal()[stubIdx];
          hitsView.rGlobal()[hitIdx] = stubsView.rGlobal()[stubIdx];
          hitsView.iphi()[hitIdx] = stubsView.iphi()[stubIdx];

          // Set cluster info (stubs don't have charge)
          hitsView.chargeAndStatus()[hitIdx].charge = 0;
          hitsView.chargeAndStatus()[hitIdx].status.isBigX = false;
          hitsView.chargeAndStatus()[hitIdx].status.isOneX = false;
          hitsView.chargeAndStatus()[hitIdx].status.isBigY = false;
          hitsView.chargeAndStatus()[hitIdx].status.isOneY = false;
          hitsView.chargeAndStatus()[hitIdx].status.qBin = 0;

          hitsView.clusterSizeX()[hitIdx] = 0;
          hitsView.clusterSizeY()[hitIdx] = 0;

          // Set detector index (CA module)
          hitsView.detectorIndex()[hitIdx] = stubsView.detectorIndex()[stubIdx];

          // Set stub-specific fields
          hitsView.isStub()[hitIdx] = true;
          hitsView.stubType()[hitIdx] = stubsView.stubType()[stubIdx];
          hitsView.dPhiDr()[hitIdx] = stubsView.dPhiDr()[stubIdx];
          hitsView.dPhiDrError()[hitIdx] = stubsView.dPhiDrError()[stubIdx];
          hitsView.ptEst()[hitIdx] = stubsView.ptEst()[stubIdx];
          // Copy pHitGroupId for duplicate stub handling in CAFishbone
          hitsView.pHitGroupId()[hitIdx] = stubsView.pHitGroupId()[stubIdx];
          // Copy stub flags (isBarrel, isFlat, isValid, layer) for pairwise compatibility cuts
          hitsView.stubFlags()[hitIdx] = stubsView.flags()[stubIdx];
        }
      }
    };

    // Kernel to build moduleStart from stub counts after sorting
    // This runs after stubs are scattered, so we rebuild moduleStart correctly
    struct BuildModuleStartKernel {
      template <typename TAcc, typename ModuleView>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    ModuleView moduleView,
                                    uint32_t const* stubCounts,
                                    uint32_t nPixHits,
                                    uint32_t nOTModulesActual) const {
        // Single-threaded prefix sum for OT modules
        // moduleStart[4000] = nPixHits (first OT hit)
        // moduleStart[4001] = nPixHits + stubCounts[4000]
        // etc.
        if (cms::alpakatools::once_per_grid(acc)) {
          uint32_t cumSum = nPixHits;
          for (uint32_t i = 0; i <= nOTModulesActual; ++i) {
            moduleView.moduleStart()[nPixelModules + i] = cumSum;
            if (i < nOTModulesActual) {
              cumSum += stubCounts[nPixelModules + i];
            }
          }
        }
      }
    };

  }  // namespace

  void SiPixelRecHitsStubsMerger::produce(edm::StreamID streamID,
                                          device::Event& iEvent,
                                          const device::EventSetup& es) const {
    auto queue = iEvent.queue();
    const auto& pixColl = iEvent.get(pixelRecHitToken_);
    const auto& stubsColl = iEvent.get(stubsToken_);

    const uint32_t nPixHits = pixColl.nHits();
    const uint32_t nStubs = stubsColl.nStubs();
    // Get the actual OT module count from the stubs collection (set by the geometry)
    // This may differ from the hardcoded nModulesCA constant
    const uint32_t nOTModulesActual = stubsColl.nModules();
    const uint32_t nModulesTotal = nPixelModules + nOTModulesActual;

#ifdef GPU_DEBUG
    std::cout << "----------------- Merging Pixel RecHits and OT Stubs -----------------\n"
              << "Number of Pixel recHits: " << nPixHits << '\n'
              << "Number of OT Stubs: " << nStubs << '\n'
              << "Total number of hits: " << (nPixHits + nStubs) << '\n'
              << "Number of CA modules (hardcoded): " << nModulesCA << '\n'
              << "Number of CA modules (actual): " << nModulesTotal << '\n'
              << "  Pixel modules: " << nPixelModules << '\n'
              << "  OT modules (actual): " << nOTModulesActual << '\n'
              << "offsetStubs will be set to: " << nPixHits << '\n'
              << "----------------------------------------------------------------------\n";
#endif

    // Allocate output collection with actual module count from geometry
    auto output = reco::TrackingRecHitsSoACollection(queue, nPixHits + nStubs, nModulesTotal);

    // Get sub-views for individual blocks (SoABlocks pattern)
    auto outHitsView = output.view().trackingHits();
    auto pixHitsView = pixColl.view().trackingHits();
    auto outModuleView = output.view().hitModules();
    auto stubsSubView = stubsColl.view().stubs();

    // Step 1: Copy pixel hits (columns only, scalars come from pixel collection)
    using HitsViewType = decltype(outHitsView);
    using HitsLayoutType = typename HitsViewType::Metadata::TypeOf_Layout;

    auto outDesc = HitsLayoutType::Descriptor(outHitsView);
    auto pixDesc = HitsLayoutType::ConstDescriptor(pixHitsView);

    constexpr std::size_t N = std::tuple_size_v<decltype(outDesc.buff)>;

    auto copyColumns = [&](auto columnIndex) {
      auto& outCol = std::get<columnIndex>(outDesc.buff);
      const auto& pixCol = std::get<columnIndex>(pixDesc.buff);

      if constexpr (std::get<columnIndex>(outDesc.columnTypes) == cms::soa::SoAColumnType::scalar) {
        alpaka::memcpy(queue,
                       cms::alpakatools::make_device_view(queue, outCol.data(), 1),
                       cms::alpakatools::make_device_view(queue, pixCol.data(), 1));
      } else {
        if (nPixHits > 0) {
          alpaka::memcpy(queue,
                         cms::alpakatools::make_device_view(queue, outCol.data(), nPixHits),
                         cms::alpakatools::make_device_view(queue, pixCol.data(), nPixHits));
        }
      }
    };

    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      (copyColumns(std::integral_constant<std::size_t, Is>{}), ...);
    }(std::make_index_sequence<N>{});

    // Step 2: Set offsetStubs scalar via memcpy (cannot directly assign to device memory from host)
    {
      uint32_t offsetStubsValue = nPixHits;
      auto offsetStubsHostView = cms::alpakatools::make_host_view(offsetStubsValue);
      auto offsetStubsDeviceView = cms::alpakatools::make_device_view(queue, outHitsView.offsetStubs());
      alpaka::memcpy(queue, offsetStubsDeviceView, offsetStubsHostView);
    }

    // Step 3: Initialize stub fields for pixel hits
    constexpr uint32_t threadsPerBlock = 256;
    if (nPixHits > 0) {
      uint32_t blocksInit = (nPixHits + threadsPerBlock - 1) / threadsPerBlock;
      auto workDivInit = cms::alpakatools::make_workdiv<Acc1D>(blocksInit, threadsPerBlock);
      alpaka::exec<Acc1D>(queue, workDivInit, InitializePixelStubFieldsKernel{}, outHitsView, nPixHits);
    }

    // Step 4: Copy pixel moduleStart (entries 0 to nPixelModules)
    // The pixel collection has dense moduleStart indexed by module ID 0-3999
    alpaka::memcpy(queue,
                   cms::alpakatools::make_device_view(
                       queue, outModuleView.moduleStart().data(), nPixelModules + 1),
                   cms::alpakatools::make_device_view(
                       queue, pixColl.view().hitModules().moduleStart().data(), nPixelModules + 1));

    // Step 5: Handle OT stubs
    if (nStubs > 0) {
      auto outModuleView = output.view().hitModules();

      // Allocate temporary buffer for stub counts per CA module
      auto stubCounts = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nModulesTotal + 1);

      // Initialize stub counts to zero
      alpaka::memset(queue, stubCounts, 0);

      // Count stubs per CA module
      {
        uint32_t blocksCount = (nStubs + threadsPerBlock - 1) / threadsPerBlock;
        auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocksCount, threadsPerBlock);
        alpaka::exec<Acc1D>(queue, workDiv, CountStubsPerCAModuleKernel{}, stubsSubView, stubCounts.data(), nStubs);
      }

      // Build moduleStart from counts using single-threaded prefix sum
      {
        auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1);
        alpaka::exec<Acc1D>(
            queue, workDiv, BuildModuleStartKernel{}, outModuleView, stubCounts.data(), nPixHits, nOTModulesActual);
      }

      // Copy stubs to hit collection preserving StubsSoA order.
      // Stubs are placed at hitIdx = nPixHits + stubIdx so that the fitter's
      // mapping stubIdx = hitIdx - offsetStubs correctly indexes the StubsSoA.
      // The StubsSoA already has stubs grouped by module from the formation
      // kernel's prefix scan, matching the moduleStart layout computed above.
      {
        uint32_t blocksCopy = (nStubs + threadsPerBlock - 1) / threadsPerBlock;
        auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocksCopy, threadsPerBlock);
        alpaka::exec<Acc1D>(queue, workDiv, CopyStubsToHitsKernel{}, stubsSubView, outHitsView, nPixHits, nStubs);
      }
    } else {
      // No stubs: initialize OT moduleStart to empty (all point to nPixHits)
      auto outModuleView = output.view().hitModules();
      uint32_t blocksOT = (nOTModulesActual + 1 + threadsPerBlock - 1) / threadsPerBlock;
      auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocksOT, threadsPerBlock);
      alpaka::exec<Acc1D>(queue, workDiv, InitializeOTModuleStartKernel{}, outModuleView, nPixHits, 0, nOTModulesActual);
    }

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    // Read offsetStubs from device via memcpy (cannot directly read device memory from host)
    uint32_t offsetStubsDebug = 0;
    auto offsetStubsDebugHostView = cms::alpakatools::make_host_view(offsetStubsDebug);
    auto offsetStubsDebugDeviceView = cms::alpakatools::make_device_view(queue, outHitsView.offsetStubs());
    alpaka::memcpy(queue, offsetStubsDebugHostView, offsetStubsDebugDeviceView);
    alpaka::wait(queue);
    std::cout << "Merge complete. offsetStubs = " << offsetStubsDebug << '\n';
#endif

    // Update cached information and emit
    output.updateFromDevice(queue);
    iEvent.emplace(outputRecHitsSoAToken_, std::move(output));

    auto mask_d = reco::TrackingRecHitsMaskingCollection(queue, static_cast<uint32_t>(output.nHits()));

    uint32_t threadsZeroPixelMask = 128;
    uint32_t blocksZeroPixelMask = static_cast<uint32_t>(output.nHits());
    auto workDivZeroPixelMask = cms::alpakatools::make_workdiv<Acc1D>(blocksZeroPixelMask, threadsZeroPixelMask);
    alpaka::exec<Acc1D>(queue, workDivZeroPixelMask, LaunchZerosPixelMask{}, mask_d.view());

    // create masking vector with zeros and emplace in the event
    iEvent.emplace(outputRecHitsMaskToken_, std::move(mask_d));

  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(SiPixelRecHitsStubsMerger);
