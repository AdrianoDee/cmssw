#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <iomanip>
#include <map>

#include <TFormula.h>
#include "CommonTools/Utils/interface/FormulaEvaluator.h"

#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/StackedModuleGeometryRecord.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/CAGeometrySoACollection.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometryHost.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometryHost.h"
#include "CAHitNtupletGenerator.h"

#include "HeterogeneousCore/AlpakaCore/interface/MoveToDeviceCache.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#define GPU_DEBUG

namespace reco {
  struct CAGeometryParams {
    //Constructor from ParameterSet
    CAGeometryParams(edm::ParameterSet const& iConfig)
        : caThetaCuts_(iConfig.getParameter<std::vector<double>>("caThetaCuts")),
          caDCACuts_(iConfig.getParameter<std::vector<double>>("caDCACuts")),
          pairGraph_(iConfig.getParameter<std::vector<unsigned int>>("pairGraph")),
          startingPairs_(iConfig.getParameter<std::vector<unsigned int>>("startingPairs")),
          phiCuts_(iConfig.getParameter<std::vector<int>>("phiCuts")),
          ptCuts_(iConfig.getParameter<std::vector<double>>("ptCuts")),
          minInner_(iConfig.getParameter<std::vector<double>>("minInner")),
          maxInner_(iConfig.getParameter<std::vector<double>>("maxInner")),
          minOuter_(iConfig.getParameter<std::vector<double>>("minOuter")),
          maxOuter_(iConfig.getParameter<std::vector<double>>("maxOuter")),
          maxDZ_(iConfig.getParameter<std::vector<double>>("maxDZ")),
          minDZ_(iConfig.getParameter<std::vector<double>>("minDZ")),
          maxDR_(iConfig.getParameter<std::vector<double>>("maxDR")),
          stubSigmaCuts_(iConfig.existsAs<std::vector<double>>("stubSigmaCuts")
                             ? iConfig.getParameter<std::vector<double>>("stubSigmaCuts")
                             : std::vector<double>{}),
          geomKappaSigmaCuts_(iConfig.existsAs<std::vector<double>>("geomKappaSigmaCuts")
                                  ? iConfig.getParameter<std::vector<double>>("geomKappaSigmaCuts")
                                  : std::vector<double>{}),
          caPhiMiddleCuts_(iConfig.existsAs<std::vector<double>>("caPhiMiddleCuts")
                               ? iConfig.getParameter<std::vector<double>>("caPhiMiddleCuts")
                               : std::vector<double>{}),
          caThetaCut1SSCuts_(iConfig.existsAs<std::vector<double>>("caThetaCut1SSCuts")
                                 ? iConfig.getParameter<std::vector<double>>("caThetaCut1SSCuts")
                                 : std::vector<double>{}),
          caThetaCut2SSCuts_(iConfig.existsAs<std::vector<double>>("caThetaCut2SSCuts")
                                 ? iConfig.getParameter<std::vector<double>>("caThetaCut2SSCuts")
                                 : std::vector<double>{}) {
      startNoBPix1_ = false;
      for (const unsigned int& i : startingPairs_) {
        if (pairGraph_[2 * i] > 0) {
          startNoBPix1_ = true;
          break;
        }
      }
#ifdef GPU_DEBUG
      std::cout << "\n========== CAGeometryParams CONSTRUCTOR ==========" << std::endl;
      std::cout << "Reading geometry from Python ParameterSet..." << std::endl;
      std::cout << "  caThetaCuts size: " << caThetaCuts_.size() << std::endl;
      std::cout << "  caDCACuts size: " << caDCACuts_.size() << std::endl;
      std::cout << "  pairGraph size: " << pairGraph_.size() << " (= " << pairGraph_.size() / 2 << " pairs)" << std::endl;
      std::cout << "  startingPairs size: " << startingPairs_.size() << std::endl;
      std::cout << "  phiCuts size: " << phiCuts_.size() << std::endl;
      std::cout << "  First 5 phiCuts values: ";
      for (size_t i = 0; i < std::min(size_t(5), phiCuts_.size()); ++i) {
        std::cout << phiCuts_[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "==================================================\n" << std::endl;
#endif
    }

    // Layers params
    const std::vector<double> caThetaCuts_;
    const std::vector<double> caDCACuts_;
    const std::vector<int> isBarrel_;

    // Cells params
    const std::vector<unsigned int> pairGraph_;
    const std::vector<unsigned int> startingPairs_;
    const std::vector<int> phiCuts_;
    const std::vector<double> ptCuts_;
    const std::vector<double> minInner_;
    const std::vector<double> maxInner_;
    const std::vector<double> minOuter_;
    const std::vector<double> maxOuter_;
    const std::vector<double> maxDZ_;
    const std::vector<double> minDZ_;
    const std::vector<double> maxDR_;
    const std::vector<double> stubSigmaCuts_;  // Stub-stub pairwise sigma cut (empty = disabled)
    const std::vector<double> geomKappaSigmaCuts_;  // Geometric-vs-stub kappa significance cut (empty = disabled)
    const std::vector<double> caPhiMiddleCuts_;  // Phi residual at middle hit cut [rad] (empty = disabled)
    const std::vector<double> caThetaCut1SSCuts_;  // Theta cut with 1 SS stub (empty = fallback)
    const std::vector<double> caThetaCut2SSCuts_;  // Theta cut with 2+ SS stubs (empty = fallback)

    bool startNoBPix1_;

    mutable edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tokenGeometry_;
    mutable edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tokenTopology_;
    mutable edm::ESGetToken<::reco::StackedModuleGeometryHost, StackedModuleGeometryRecord> tokenStackedGeometry_;
  };

}  // namespace reco

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  class CAHitNtupletAlpaka
      : public stream::EDProducer<edm::GlobalCache<::reco::CAGeometryParams>,
                                  edm::RunCache<cms::alpakatools::MoveToDeviceCache<Device, ::reco::CAGeometryHost>>> {
    using HitsConstView = ::reco::TrackingRecHitConstView;
    using HitsOnDevice = reco::TrackingRecHitsSoACollection;
    using HitsOnHost = ::reco::TrackingRecHitHost;

    using TkSoAHost = ::reco::TracksHost;
    using TkSoADevice = reco::TracksSoACollection;

    using Algo = CAHitNtupletGenerator<TrackerTraits>;

    using CAGeometryCache = cms::alpakatools::MoveToDeviceCache<Device, ::reco::CAGeometryHost>;
    using Rotation = SOARotation<float>;
    using Frame = SOAFrame<float>;

  public:
    explicit CAHitNtupletAlpaka(const edm::ParameterSet& iConfig, const ::reco::CAGeometryParams* iCache);
    ~CAHitNtupletAlpaka() override = default;

    void produce(device::Event& iEvent, const device::EventSetup& es) override;

    static void globalEndJob(::reco::CAGeometryParams const*) { /* Do nothing */ };
    static void globalEndRun(edm::Run const& iRun,
                             edm::EventSetup const&,
                             RunContext const* iContext) { /* Do nothing */ };

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    static std::shared_ptr<CAGeometryCache> globalBeginRun(edm::Run const& iRun,
                                                           edm::EventSetup const& iSetup,
                                                           GlobalCache const* iCache) {
      assert(iCache->maxDR_.size() == iCache->minInner_.size());
      assert(iCache->maxDR_.size() == iCache->maxInner_.size());
      assert(iCache->maxDR_.size() == iCache->minOuter_.size());
      assert(iCache->maxDR_.size() == iCache->maxOuter_.size());
      assert(iCache->maxDR_.size() == iCache->maxDZ_.size());
      assert(iCache->maxDR_.size() == iCache->minDZ_.size());
      assert(iCache->maxDR_.size() == iCache->phiCuts_.size());
      assert(iCache->maxDR_.size() == iCache->ptCuts_.size());

      assert(iCache->caThetaCuts_.size() == iCache->caDCACuts_.size());

      int n_layers = iCache->caThetaCuts_.size();
      int n_pairs = iCache->pairGraph_.size() / 2;
      int n_modules = 0;

#ifdef GPU_DEBUG
      std::cout << "No. Layers to be used = " << n_layers << std::endl;
      std::cout << "No. Pairs to be used = " << n_pairs << std::endl;
#endif

      assert(int(n_pairs) == int(iCache->maxDR_.size()));
      assert(int(*std::max_element(iCache->startingPairs_.begin(), iCache->startingPairs_.end())) < n_pairs);
      assert(int(*std::max_element(iCache->pairGraph_.begin(), iCache->pairGraph_.end())) < n_layers);

      auto const& trackerGeometry = iSetup.getData(iCache->tokenGeometry_);
      auto const& trackerTopology = iSetup.getData(iCache->tokenTopology_);
      auto const& dets = trackerGeometry.dets();

      // Get stacked module geometry for Phase-2 OT with stubs
      ::reco::StackedModuleGeometryHost const* stackedGeometry = nullptr;
      if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
        stackedGeometry = &iSetup.getData(iCache->tokenStackedGeometry_);
      }

#ifdef GPU_DEBUG
      auto subSystem = 0;
      auto subSystemName = GeomDetEnumerators::tkDetEnum[subSystem];
      std::cout
          << "========================================================================================================="
          << std::endl;
#endif

      auto oldLayer = 0u;
      auto layerCount = 0u;

      std::vector<bool> layerIsBarrel(n_layers);
      std::vector<int> layerStarts(n_layers + 1);
      //^ why n_layers + 1? This is a cumulative sum of the number
      // of modules each layer has. And we need the  extra spot
      // at the end to hold the total number of modules.

      std::vector<int> moduleToindexInDets;

      auto isPinPSinOTBarrel = [&](DetId detId) {
        // Select only P-hits from the OT barrel
        return (trackerGeometry.getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP &&
                detId.subdetId() == StripSubdetector::TOB);
      };
      auto isPixel = [&](DetId detId) {
        auto subId = detId.subdetId();
        return (subId == PixelSubdetector::PixelBarrel || subId == PixelSubdetector::PixelEndcap);
      };
      auto isBarrel = [&](DetId detId) {
        auto subId = detId.subdetId();
        auto subDetector = trackerGeometry.geomDetSubDetector(subId);
        return GeomDetEnumerators::isBarrel(subDetector);
      };

      // loop over all detector modules and build the CA layers
      int counter = 0;
      for (auto& det : dets) {
        DetId detid = det->geographicalId();
        auto layer = trackerTopology.layer(detid);
        // Logic:
        // - if we are not inside pixels, we need to ignore anything **but** the OT.
        // - for the time being, this is assuming that the CA extension will
        //   only cover the OT barrel part, and will ignore the OT forward.

#ifdef GPU_DEBUG
        auto subId = detid.subdetId();
        if (subSystemName != trackerGeometry.geomDetSubDetector(subId)) {
          subSystemName = trackerGeometry.geomDetSubDetector(subId);
          std::cout << " ===================== Subsystem: " << subSystemName << std::endl;
        }
#endif

        // Modules of the pixel layers
        if (isPixel(detid)) {
          if (layer != oldLayer) {
#ifdef GPU_DEBUG
            std::cout << "Pixel LayerStart: CA layer " << layerCount << " at subdetector layer " << layer
                      << " starts at module " << n_modules << " and is " << (isBarrel(detid) ? "barrel" : "not barrel")
                      << std::endl;
#endif
            layerIsBarrel[layerCount] = isBarrel(detid);
            layerStarts[layerCount++] = n_modules;
            if (layerCount >= layerStarts.size())
              break;
            oldLayer = layer;
          }
          moduleToindexInDets.push_back(counter);
          n_modules++;
        }

        // if we are using the CA extension for Phase-2,
        // we also have to collect the modules from the considered OT layers
        if constexpr (std::is_same_v<pixelTopology::Phase2OT, TrackerTraits>) {
          auto const& detUnits = det->components();
          for (auto& detUnit : detUnits) {
            DetId unitDetId(detUnit->geographicalId());
            // Modules of the considered OT layers
            if (isPinPSinOTBarrel(unitDetId)) {
              if (layer != oldLayer) {
#ifdef GPU_DEBUG
                std::cout << "OT LayerStart: CA layer " << layerCount << " at subdetector layer " << layer
                          << " starts at module " << n_modules << " and is "
                          << (isBarrel(detid) ? "barrel" : "not barrel") << std::endl;
#endif
                layerIsBarrel[layerCount] = isBarrel(detid);
                layerStarts[layerCount++] = n_modules;
                if (layerCount >= layerStarts.size())
                  break;
                oldLayer = layer;
              }
              moduleToindexInDets.push_back(counter);
              n_modules++;
            }
          }
        }
        counter++;
      }

      // Process OT stacked modules for Phase-2 with stubs
      // CA layers follow inside-out ordering:
      // - CA layers 28-33: Barrel (layers 1-6)
      // - CA layers 34-38: Backward disks (layers 1-5)
      // - CA layers 39-43: Forward disks (layers 1-5)
      //
      // IMPORTANT: StackedModuleGeometry is ALREADY sorted in CA order by StackedModuleGeometryESProducer
      // (barrel by layer -> backward by layer -> forward by layer) using stable_sort.
      // We iterate through it in index order WITHOUT re-sorting to ensure the frame array
      // index matches the detectorIndex assigned to hits/stubs (detectorIndex = nPixelModules + geomIndex).
      if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
        if (stackedGeometry != nullptr) {
          auto stackedView = stackedGeometry->view();
          const uint32_t nStackedModules = static_cast<uint32_t>(stackedView.metadata().size());

          bool firstModule = true;
          uint8_t prevLayer = 0;
          bool prevBarrel = false;
          int prevCategory = -1;  // 0=barrel, 1=backward, 2=forward

          // Iterate through StackedModuleGeometry in index order (already sorted in CA order)
          for (uint32_t i = 0; i < nStackedModules; ++i) {
            bool isBarrel = stackedView.isBarrel()[i];
            bool isFwdEndcap = stackedView.isFwdEndcap()[i];
            uint8_t otLayer = stackedView.layer()[i];
            DetId stackedDetId(stackedView.stackedDetId()[i]);

            // Determine category: 0=barrel, 1=backward, 2=forward
            int category = isBarrel ? 0 : (isFwdEndcap ? 2 : 1);

            // Check if we've transitioned to a new CA layer
            // A new layer starts when category changes OR layer number changes within same category
            if (firstModule || category != prevCategory || otLayer != prevLayer) {
              // Start new CA layer
              if (layerCount < layerStarts.size()) {
                layerIsBarrel[layerCount] = isBarrel;
                layerStarts[layerCount++] = n_modules;

#ifdef GPU_DEBUG
                const char* categoryName = isBarrel ? "barrel" : (isFwdEndcap ? "forward" : "backward");
                std::cout << "OT LayerStart: CA layer " << (layerCount - 1) << " starts at module " << n_modules
                          << " (" << categoryName << " layer " << int(otLayer) << ")" << std::endl;
#endif
              }
              prevCategory = category;
              prevLayer = otLayer;
              prevBarrel = isBarrel;
              firstModule = false;
            }

            // Find this module in TrackerGeometry dets list
            bool found = false;
            for (int detIdx = 0; detIdx < static_cast<int>(dets.size()); ++detIdx) {
              if (dets[detIdx]->geographicalId() == stackedDetId) {
                moduleToindexInDets.push_back(detIdx);
                n_modules++;
                found = true;
                break;
              }
            }
            if (!found) {
              edm::LogWarning("CAHitNtuplet")
                  << "Could not find stacked module " << stackedDetId.rawId() << " in TrackerGeometry";
            }
          }
        }
      }

#ifdef GPU_DEBUG
      std::cout << "Full CA LayerStart: " << n_layers << " layers with " << n_modules << " modules in total."
                << std::endl;
#endif
      layerStarts[n_layers] = n_modules;

      reco::CAGeometryHost product{cms::alpakatools::host(), n_layers + 1, n_pairs, n_modules};

      auto layerSoA = product.view().layers();
      auto cellSoA = product.view().graph();
      auto modulesSoA = product.view().modules();

      for (int i = 0; i < n_modules; ++i) {
        auto idx = moduleToindexInDets[i];
        auto det = dets[idx];
        auto vv = det->surface().position();
        auto rr = Rotation(det->surface().rotation());
        modulesSoA[i].detFrame() = Frame(vv.x(), vv.y(), vv.z(), rr);
#ifdef GPU_DEBUG
        auto const& detUnits = det->components();
        for (auto& detUnit : detUnits) {
          DetId unitDetId(detUnit->geographicalId());
          if (isPinPSinOTBarrel(unitDetId)) {
            std::cout << "Filling frame at index " << idx << " in SoA position " << i << " for det "
                      << det->geographicalId() << " and detUnit->index: " << detUnit->index() << std::endl;
          }
        }
        std::cout << "Filling frame at index " << idx << " in SoA position " << i << " for det "
                  << det->geographicalId() << std::endl;
        std::cout << "Position: " << vv << " with Rotation: " << det->surface().rotation() << std::endl;
        std::cout << "Rotation in z-r plane: "
                  << atan2(det->surface().normalVector().perp(), det->surface().normalVector().z()) * 180. / M_PI
                  << std::endl;
#endif
      }

      for (int i = 0; i < n_layers; ++i) {
        layerSoA.layerStarts()[i] = layerStarts[i];
        layerSoA.caThetaCut()[i] = iCache->caThetaCuts_[i];
        layerSoA.caDCACut()[i] = iCache->caDCACuts_[i];
        layerSoA.isBarrel()[i] = layerIsBarrel[i];
        layerSoA.geomKappaSigmaCut()[i] =
            (!iCache->geomKappaSigmaCuts_.empty()) ? static_cast<float>(iCache->geomKappaSigmaCuts_[i]) : -1.0f;
        layerSoA.caPhiMiddleCut()[i] =
            (!iCache->caPhiMiddleCuts_.empty()) ? static_cast<float>(iCache->caPhiMiddleCuts_[i]) : -1.0f;
        layerSoA.caThetaCut1SS()[i] =
            (!iCache->caThetaCut1SSCuts_.empty()) ? static_cast<float>(iCache->caThetaCut1SSCuts_[i]) : -1.0f;
        layerSoA.caThetaCut2SS()[i] =
            (!iCache->caThetaCut2SSCuts_.empty()) ? static_cast<float>(iCache->caThetaCut2SSCuts_[i]) : -1.0f;
      }

      layerSoA.layerStarts()[n_layers] = layerStarts[n_layers];

      for (int i = 0; i < n_pairs; ++i) {
        cellSoA.graph()[i] = {{uint32_t(iCache->pairGraph_[2 * i]), uint32_t(iCache->pairGraph_[2 * i + 1])}};
        cellSoA.phiCuts()[i] = iCache->phiCuts_[i];
        // convert ptCut in curvature radius in cm
        // 1 GeV track has 1 GeV/c / (e * 3.8T) ~ 87 cm radius in a 3.8T field
        const float minRadius = iCache->ptCuts_[i] * 87.78f;
        // Use minRadius^2/4 in the CA to avoid sqrt
        const float minRadius2T4 = 4.f * minRadius * minRadius;
        cellSoA.ptCuts()[i] = minRadius2T4;
        cellSoA.minInner()[i] = iCache->minInner_[i];
        cellSoA.maxInner()[i] = iCache->maxInner_[i];
        cellSoA.minOuter()[i] = iCache->minOuter_[i];
        cellSoA.maxOuter()[i] = iCache->maxOuter_[i];
        cellSoA.maxDZ()[i] = iCache->maxDZ_[i];
        cellSoA.minDZ()[i] = iCache->minDZ_[i];
        cellSoA.maxDR()[i] = iCache->maxDR_[i];
        // Stub-stub pairwise sigma cut: use from config if available, otherwise -1.0 (disabled)
        cellSoA.stubSigmaCut()[i] =
            (!iCache->stubSigmaCuts_.empty()) ? static_cast<float>(iCache->stubSigmaCuts_[i]) : -1.0f;
        cellSoA.startingPair()[i] = false;
      }

      for (const unsigned int& i : iCache->startingPairs_)
        cellSoA.startingPair()[i] = true;

#ifdef GPU_DEBUG
      // Debug output: Print geometry values from Python config
      std::cout << "\n========== CA GEOMETRY FROM PYTHON CONFIG ==========" << std::endl;
      std::cout << "Number of layers: " << n_layers << std::endl;
      std::cout << "Number of layer pairs: " << n_pairs << std::endl;
      std::cout << "Number of starting pairs: " << iCache->startingPairs_.size() << std::endl;

      std::cout << "\n--- Layer Pair Geometry (first 20 pairs) ---" << std::endl;
      std::cout << "Pair | Inner | Outer | phiCut | minIn | maxIn | minOut | maxOut | maxDR | minDZ | maxDZ | start" << std::endl;
      std::cout << "-----|-------|-------|--------|-------|-------|--------|--------|-------|-------|-------|------" << std::endl;
      for (int i = 0; i < std::min(20, n_pairs); ++i) {
        std::cout << std::setw(4) << i << " | "
                  << std::setw(5) << iCache->pairGraph_[2 * i] << " | "
                  << std::setw(5) << iCache->pairGraph_[2 * i + 1] << " | "
                  << std::setw(6) << iCache->phiCuts_[i] << " | "
                  << std::setw(5) << iCache->minInner_[i] << " | "
                  << std::setw(5) << iCache->maxInner_[i] << " | "
                  << std::setw(6) << iCache->minOuter_[i] << " | "
                  << std::setw(6) << iCache->maxOuter_[i] << " | "
                  << std::setw(5) << iCache->maxDR_[i] << " | "
                  << std::setw(5) << iCache->minDZ_[i] << " | "
                  << std::setw(5) << iCache->maxDZ_[i] << " | "
                  << (cellSoA.startingPair()[i] ? "Y" : "N") << std::endl;
      }
      if (n_pairs > 20) {
        std::cout << "... (" << (n_pairs - 20) << " more pairs not shown)" << std::endl;
      }

      std::cout << "\n--- Layer Cuts (all layers) ---" << std::endl;
      std::cout << "Layer | isBarrel | caThetaCut | caDCACut" << std::endl;
      std::cout << "------|----------|------------|----------" << std::endl;
      for (int i = 0; i < n_layers; ++i) {
        std::cout << std::setw(5) << i << " | "
                  << std::setw(8) << (layerIsBarrel[i] ? "Y" : "N") << " | "
                  << std::setw(10) << iCache->caThetaCuts_[i] << " | "
                  << std::setw(10) << iCache->caDCACuts_[i] << std::endl;
      }
      std::cout << "====================================================\n" << std::endl;
#endif

      return std::make_shared<CAGeometryCache>(std::move(product));
    }

    static std::unique_ptr<::reco::CAGeometryParams> initializeGlobalCache(edm::ParameterSet const& iConfig) {
      return std::make_unique<::reco::CAGeometryParams>(iConfig.getParameterSet("geometry"));
    }

  private:
    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tokenField_;
    const device::EDGetToken<HitsOnDevice> tokenHit_;
    const device::EDPutToken<TkSoADevice> tokenTrack_;

    // Conditional tokens for OT stubs (only for Phase2OTStubs)
    [[no_unique_address]] std::conditional_t<std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>,
                                             device::EDGetToken<reco::OTRecHitsSoACollection>,
                                             std::monostate> tokenOTRecHits_;
    [[no_unique_address]] std::conditional_t<std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>,
                                             device::EDGetToken<reco::StubsSoACollection>,
                                             std::monostate> tokenStubs_;

    const ::reco::FormulaEvaluator maxNumberOfDoublets_;
    const ::reco::FormulaEvaluator maxNumberOfTuples_;

    Algo deviceAlgo_;
  };

  template <typename TrackerTraits>
  CAHitNtupletAlpaka<TrackerTraits>::CAHitNtupletAlpaka(const edm::ParameterSet& iConfig,
                                                        const ::reco::CAGeometryParams* iCache)
      : EDProducer(iConfig),
        tokenField_(esConsumes()),
        tokenHit_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
        tokenTrack_(produces()),
        maxNumberOfDoublets_(iConfig.getParameter<std::string>("maxNumberOfDoublets")),
        maxNumberOfTuples_(iConfig.getParameter<std::string>("maxNumberOfTuples")),
        deviceAlgo_(iConfig) {
    iCache->tokenGeometry_ = esConsumes<edm::Transition::BeginRun>();
    iCache->tokenTopology_ = esConsumes<edm::Transition::BeginRun>();
    if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
      iCache->tokenStackedGeometry_ = esConsumes<edm::Transition::BeginRun>();
      // Initialize OT tokens for stub-based tracking
      tokenOTRecHits_ = consumes(iConfig.getParameter<edm::InputTag>("otRecHitsSrc"));
      tokenStubs_ = consumes(iConfig.getParameter<edm::InputTag>("stubsSrc"));
    }
  }

  template <typename TrackerTraits>
  void CAHitNtupletAlpaka<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsPreSplittingAlpaka"));

    // Add OT input tags for Phase2OTStubs configuration
    if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
      desc.add<edm::InputTag>("otRecHitsSrc", edm::InputTag("phase2OTRecHits"));
      desc.add<edm::InputTag>("stubsSrc", edm::InputTag("otStubProducer"));
    }

    Algo::fillPSetDescription(desc);
    descriptions.addWithDefaultLabel(desc);
  }

  template <typename TrackerTraits>
  void CAHitNtupletAlpaka<TrackerTraits>::produce(device::Event& iEvent, const device::EventSetup& es) {
    auto bf = 1. / es.getData(tokenField_).inverseBzAtOriginInGeV();

    auto const& geometry = runCache()->get(iEvent.queue());
    auto const& hits = iEvent.get(tokenHit_);

    /// Don't bother if no hits on BPix1 and no good graph for that
    /// (so no staring pair without BPix1 as first layer).
    /// TODO: this could be extended to a more general check for
    /// no hits on any of the starting layers.

    if (globalCache()->startNoBPix1_ or hits.offsetBPIX2() > 0) {
      std::array<double, 1> nHitsV = {{double(hits.nHits())}};
      std::array<double, 1> emptyV;

      uint32_t const maxTuples = maxNumberOfTuples_.evaluate(nHitsV, emptyV);
      uint32_t const maxDoublets = maxNumberOfDoublets_.evaluate(nHitsV, emptyV);

#ifdef CA_PIPELINE_COUNTERS
      printf("[CA Pipeline] Event: run=%u lumi=%u event=%llu nHits=%u\n",
             iEvent.id().run(),
             iEvent.id().luminosityBlock(),
             (unsigned long long)iEvent.id().event(),
             hits.nHits());
#endif

      // Conditionally pass OT collections for stub-based tracking
      if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
        auto const& otRecHits = iEvent.get(tokenOTRecHits_);
        auto const& stubs = iEvent.get(tokenStubs_);
        iEvent.emplace(tokenTrack_,
                       deviceAlgo_.makeTuplesAsync(
                           hits, geometry, bf, maxDoublets, maxTuples, iEvent.queue(), otRecHits, stubs));
      } else {
        iEvent.emplace(tokenTrack_,
                       deviceAlgo_.makeTuplesAsync(hits, geometry, bf, maxDoublets, maxTuples, iEvent.queue()));
      }

    } else {
      edm::LogWarning("CAHitNtupletAlpaka") << "No hit on BPix1 (" << hits.offsetBPIX2()
                                            << ") and all the starting pairs has BPix1 as inner layer.\nIt's useless "
                                            << "to run the CA. Returning with 0 tracks!";
      auto& queue = iEvent.queue();
      reco::TracksSoACollection tracks(queue, 0, 0);
      auto ntracks_d = cms::alpakatools::make_device_view(queue, tracks.view().tracks().nTracks());
      alpaka::memset(queue, ntracks_d, 0);
      iEvent.emplace(tokenTrack_, std::move(tracks));
    }
  }

  using CAHitNtupletAlpakaPhase1 = CAHitNtupletAlpaka<pixelTopology::Phase1>;
  using CAHitNtupletAlpakaHIonPhase1 = CAHitNtupletAlpaka<pixelTopology::HIonPhase1>;
  using CAHitNtupletAlpakaPhase2 = CAHitNtupletAlpaka<pixelTopology::Phase2>;
  using CAHitNtupletAlpakaPhase2OT = CAHitNtupletAlpaka<pixelTopology::Phase2OT>;
  using CAHitNtupletAlpakaPhase2OTStubs = CAHitNtupletAlpaka<pixelTopology::Phase2OTStubs>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"

DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase1);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaHIonPhase1);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase2);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase2OT);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase2OTStubs);
