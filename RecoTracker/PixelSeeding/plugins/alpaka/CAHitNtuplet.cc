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
    CAGeometryParams(edm::ParameterSet const& graphConfig,
                     edm::ParameterSet const& doubletCutConfig,
                     edm::ParameterSet const& tripletCutConfig,
                     edm::ParameterSet const& ntupletCutConfig,
                     std::vector<double> const& fishboneCuts)
        : layerPairs_(graphConfig.getParameter<std::vector<unsigned int>>("layerPairs")),
          startingPair_(graphConfig.getParameter<std::vector<unsigned int>>("startingPair")),
          skipsLayers_(graphConfig.getParameter<std::vector<unsigned int>>("skipsLayers")),
          maxDPhi_(doubletCutConfig.getParameter<std::vector<int>>("maxDPhi")),
          minInner_(doubletCutConfig.getParameter<std::vector<double>>("minInner")),
          maxInner_(doubletCutConfig.getParameter<std::vector<double>>("maxInner")),
          minOuter_(doubletCutConfig.getParameter<std::vector<double>>("minOuter")),
          maxOuter_(doubletCutConfig.getParameter<std::vector<double>>("maxOuter")),
          maxDR_(doubletCutConfig.getParameter<std::vector<double>>("maxDR")),
          minDZ_(doubletCutConfig.getParameter<std::vector<double>>("minDZ")),
          maxDZ_(doubletCutConfig.getParameter<std::vector<double>>("maxDZ")),
          minPt_(doubletCutConfig.getParameter<std::vector<double>>("minPt")),
          maxZ0_(doubletCutConfig.getParameter<std::vector<double>>("maxZ0")),
          maxStubCurvSigma_(doubletCutConfig.existsAs<std::vector<double>>("maxStubCurvSigma")
                                ? doubletCutConfig.getParameter<std::vector<double>>("maxStubCurvSigma")
                                : std::vector<double>{}),
          dzdrFact_(doubletCutConfig.getParameter<double>("dzdrFact")),
          minInnerSizeB1_(doubletCutConfig.getParameter<int>("minInnerSizeB1")),
          minInnerSizeB2_(doubletCutConfig.getParameter<int>("minInnerSizeB2")),
          maxDSizeB1_(doubletCutConfig.getParameter<int>("maxDSizeB1")),
          maxDSize_(doubletCutConfig.getParameter<int>("maxDSize")),
          maxDSizePred_(doubletCutConfig.getParameter<int>("maxDSizePred")),
          maxRZTolerance_(tripletCutConfig.getParameter<std::vector<double>>("maxRZTolerance")),
          maxDCA_(tripletCutConfig.getParameter<std::vector<double>>("maxDCA")),
          floorDCA_(tripletCutConfig.existsAs<std::vector<double>>("floorDCA")
                        ? tripletCutConfig.getParameter<std::vector<double>>("floorDCA")
                        : std::vector<double>{}),
          maxStubGeomCurvSigma_(tripletCutConfig.existsAs<std::vector<double>>("maxStubGeomCurvSigma")
                                    ? tripletCutConfig.getParameter<std::vector<double>>("maxStubGeomCurvSigma")
                                    : std::vector<double>{}),
          maxStubInnerDoubletDCurv_(tripletCutConfig.existsAs<std::vector<double>>("maxStubInnerDoubletDCurv")
                                        ? tripletCutConfig.getParameter<std::vector<double>>("maxStubInnerDoubletDCurv")
                                        : std::vector<double>{}),
          ptmin_(tripletCutConfig.getParameter<double>("ptmin")),
          maxCurv_(tripletCutConfig.getParameter<double>("maxCurv")),
          maxPhiResid_(tripletCutConfig.getParameter<double>("maxPhiResid")),
          sameDPhiSign_(tripletCutConfig.getParameter<bool>("sameDPhiSign")),
          startMaxInnerR_(ntupletCutConfig.getParameter<std::vector<double>>("startMaxInnerR")),
          maxDCurv_(ntupletCutConfig.getParameter<std::vector<double>>("maxDCurv")),
          floorDCurv_(ntupletCutConfig.getParameter<std::vector<double>>("floorDCurv")),
          fishboneCuts_(fishboneCuts) {
      startNoBPix1_ = false;
      for (size_t i{0}; i < layerPairs_.size() / 2; ++i) {
        if (startingPair_[i] && layerPairs_[2 * i] > 0) {
          startNoBPix1_ = true;
          break;
        }
      }
#ifdef GPU_DEBUG
      std::cout << "\n========== CAGeometryParams CONSTRUCTOR ==========" << std::endl;
      std::cout << "Reading geometry from Python ParameterSet..." << std::endl;
      std::cout << "  maxRZTolerance size: " << maxRZTolerance_.size() << std::endl;
      std::cout << "  maxDCA size: " << maxDCA_.size() << std::endl;
      std::cout << "  pairGraph size: " << layerPairs_.size() << " (= " << layerPairs_.size() / 2 << " pairs)"
                << std::endl;
      auto nStartingPairs = std::count(startingPair_.begin(), startingPair_.end(), 1);
      std::cout << "  startingPairs number: " << nStartingPairs << std::endl;
      std::cout << "  maxDPhi size: " << maxDPhi_.size() << std::endl;
      std::cout << "  First 5 maxDPhi values: ";
      for (size_t i = 0; i < std::min(size_t(5), maxDPhi_.size()); ++i) {
        std::cout << maxDPhi_[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "==================================================\n" << std::endl;
#endif
    }

    // graph
    const std::vector<unsigned int> layerPairs_;
    const std::vector<unsigned int> startingPair_;
    const std::vector<unsigned int> skipsLayers_;

    // doublet cuts
    const std::vector<int> maxDPhi_;
    const std::vector<double> minInner_;
    const std::vector<double> maxInner_;
    const std::vector<double> minOuter_;
    const std::vector<double> maxOuter_;
    const std::vector<double> maxDR_;
    const std::vector<double> minDZ_;
    const std::vector<double> maxDZ_;
    const std::vector<double> minPt_;
    const std::vector<double> maxZ0_;
    const std::vector<double> maxStubCurvSigma_;
    const double dzdrFact_;
    const int minInnerSizeB1_;
    const int minInnerSizeB2_;
    const int maxDSizeB1_;
    const int maxDSize_;
    const int maxDSizePred_;

    // triplet cuts
    const std::vector<double> maxRZTolerance_;
    const std::vector<double> maxDCA_;
    const std::vector<double> floorDCA_;
    const std::vector<double> maxStubGeomCurvSigma_;
    const std::vector<double> maxStubInnerDoubletDCurv_;
    const double ptmin_;
    const double maxCurv_;
    const double maxPhiResid_;
    const bool sameDPhiSign_;

    // ntuplet cuts
    const std::vector<double> startMaxInnerR_;
    const std::vector<double> maxDCurv_;
    const std::vector<double> floorDCurv_;

    const std::vector<double> fishboneCuts_;

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
      assert(iCache->maxDR_.size() == iCache->skipsLayers_.size());
      assert(iCache->maxDR_.size() == iCache->startingPair_.size());
      assert(iCache->maxDR_.size() == iCache->maxDPhi_.size());
      assert(iCache->maxDR_.size() == iCache->minInner_.size());
      assert(iCache->maxDR_.size() == iCache->maxInner_.size());
      assert(iCache->maxDR_.size() == iCache->minOuter_.size());
      assert(iCache->maxDR_.size() == iCache->maxOuter_.size());
      assert(iCache->maxDR_.size() == iCache->maxDZ_.size());
      assert(iCache->maxDR_.size() == iCache->minDZ_.size());
      assert(iCache->maxDR_.size() == iCache->minPt_.size());
      assert(iCache->maxDR_.size() == iCache->maxZ0_.size());
      assert(iCache->maxDR_.size() == iCache->maxRZTolerance_.size());
      assert(iCache->maxDR_.size() == iCache->maxDCA_.size());

      // stubs-specific cuts are optional, but if they are present they need to have the same size as the number of layer pairs
      if (iCache->maxStubCurvSigma_.size() > 0) {
        assert(iCache->maxDR_.size() == iCache->maxStubCurvSigma_.size());
        assert(iCache->maxDR_.size() == iCache->floorDCA_.size());
        assert(iCache->maxDR_.size() == iCache->maxStubGeomCurvSigma_.size());
        assert(iCache->maxDR_.size() == iCache->maxStubInnerDoubletDCurv_.size());
      }
      assert(iCache->fishboneCuts_.size() == iCache->maxDCurv_.size());
      assert(iCache->fishboneCuts_.size() == iCache->floorDCurv_.size());
      assert(iCache->fishboneCuts_.size() == iCache->startMaxInnerR_.size());

      int n_layers = iCache->fishboneCuts_.size();
      int n_pairs = iCache->layerPairs_.size() / 2;
      int n_modules = 0;

#ifdef GPU_DEBUG
      std::cout << "No. Layers to be used = " << n_layers << std::endl;
      std::cout << "No. Pairs to be used = " << n_pairs << std::endl;
#endif

      assert(int(n_pairs) == int(iCache->maxDR_.size()));
      assert(int(*std::max_element(iCache->layerPairs_.begin(), iCache->layerPairs_.end())) < n_layers);

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
      std::vector<bool> layerIsOT(n_layers);
      std::vector<bool> layerIsSS(n_layers);
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
            layerIsOT[layerCount] = false;  // we are in the loop over pixel dets, so these are not OT layers
            layerIsSS[layerCount] = false;  // we are in the loop over pixel dets, so these are not SS layers
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
                layerIsOT[layerCount] = true;   // we are in the loop over PS modules, so these are all OT layers
                layerIsSS[layerCount] = false;  // we are in the loop over PS modules, so these are not SS layers
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
      // - CA layers 34-43: Backward disks (layers 1-5) split in two groups:
      //                    34-38 with PS modules first, then SS; 39-43 with no PS/SS split
      // - CA layers 44-53: Forward disks (layers 1-5) split in two groups:
      //                    44-48 with PS modules first, then SS; 49-53 with no PS/SS split
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
          bool prevIsPS = false;
          int prevCategory = -1;  // 0=barrel, 1=backward, 2=forward

          // Iterate through StackedModuleGeometry in index order (already sorted in CA order)
          for (uint32_t i = 0; i < nStackedModules; ++i) {
            bool isBarrel = stackedView.isBarrel()[i];
            bool isFwdEndcap = stackedView.isFwdEndcap()[i];
            bool isPS = stackedView.isPS()[i];
            uint8_t otLayer = stackedView.layer()[i];
            DetId stackedDetId(stackedView.stackedDetId()[i]);

            // Determine category: 0=barrel, 1=backward, 2=forward
            int category = isBarrel ? 0 : (isFwdEndcap ? 2 : 1);

            // Check if we've transitioned to a new CA layer
            // A new layer starts when category changes OR layer number changes within same category
            if (firstModule || category != prevCategory || otLayer != prevLayer || isPS != prevIsPS) {
              // Start new CA layer
              if (layerCount < layerStarts.size()) {
                layerIsBarrel[layerCount] = isBarrel;
                layerIsOT[layerCount] =
                    true;  // we are in the loop over StackedModuleGeometry, so these are all OT layers
                layerIsSS[layerCount] = !isPS;
                layerStarts[layerCount++] = n_modules;

#ifdef GPU_DEBUG
                const char* categoryName = isBarrel ? "barrel" : (isFwdEndcap ? "forward" : "backward");
                std::cout << "OT LayerStart: CA layer " << (layerCount - 1) << " starts at module " << n_modules << " ("
                          << categoryName << " layer " << int(otLayer) << ")" << std::endl;
#endif
              }
              prevCategory = category;
              prevLayer = otLayer;
              prevBarrel = isBarrel;
              prevIsPS = isPS;
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

      reco::CAGeometryHost product{cms::alpakatools::host(), n_layers + 1, n_pairs, n_pairs, n_pairs, n_layers, n_modules};

      auto layerSoA = product.view().layers();
      auto graphSoA = product.view().graph();
      auto doubletCutsSoA = product.view().doubletCuts();
      auto tripletCutsSoA = product.view().tripletCuts();
      auto ntupletCutsSoA = product.view().ntupletCuts();
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
        layerSoA.fishboneCut()[i] = iCache->fishboneCuts_[i];
        layerSoA.layerStarts()[i] = layerStarts[i];
        layerSoA.isBarrel()[i] = layerIsBarrel[i];
        layerSoA.isOT()[i] = layerIsOT[i];
        layerSoA.isSS()[i] = layerIsSS[i];
        ntupletCutsSoA.startMaxInnerR()[i] = iCache->startMaxInnerR_[i];
        ntupletCutsSoA.maxDCurv()[i] = iCache->maxDCurv_[i];
        ntupletCutsSoA.floorDCurv()[i] = iCache->floorDCurv_[i];
      }

      layerSoA.layerStarts()[n_layers] = layerStarts[n_layers];

      for (int i = 0; i < n_pairs; ++i) {
        graphSoA.layerPair()[i] = {{uint32_t(iCache->layerPairs_[2 * i]), uint32_t(iCache->layerPairs_[2 * i + 1])}};
        graphSoA.skipsLayers()[i] = uint16_t(bool(iCache->skipsLayers_[i]));
        graphSoA.startingPair()[i] = iCache->startingPair_[i];
        doubletCutsSoA.maxDPhi()[i] = iCache->maxDPhi_[i];
        doubletCutsSoA.minInner()[i] = iCache->minInner_[i];
        doubletCutsSoA.maxInner()[i] = iCache->maxInner_[i];
        doubletCutsSoA.minOuter()[i] = iCache->minOuter_[i];
        doubletCutsSoA.maxOuter()[i] = iCache->maxOuter_[i];
        doubletCutsSoA.maxDZ()[i] = iCache->maxDZ_[i];
        doubletCutsSoA.minDZ()[i] = iCache->minDZ_[i];
        doubletCutsSoA.maxDR()[i] = iCache->maxDR_[i];
        // convert ptCut in curvature radius in cm
        // 1 GeV track has 1 GeV/c / (e * 3.8T) ~ 87 cm radius in a 3.8T field
        const float minRadius = iCache->minPt_[i] * 87.78f;
        // Use minRadius^2/4 in the CA to avoid sqrt
        const float minRadius2T4 = 4.f * minRadius * minRadius;
        doubletCutsSoA.minPt()[i] = minRadius2T4;
        doubletCutsSoA.maxZ0()[i] = iCache->maxZ0_[i];
        // Stub-stub pairwise sigma cut: use from config if available, otherwise -1.0 (disabled)
        doubletCutsSoA.maxStubCurvSigma()[i] =
            (!iCache->maxStubCurvSigma_.empty()) ? static_cast<float>(iCache->maxStubCurvSigma_[i]) : -1.0f;
        tripletCutsSoA.maxRZTolerance()[i] = iCache->maxRZTolerance_[i];
        tripletCutsSoA.maxDCA()[i] = iCache->maxDCA_[i];
        tripletCutsSoA.floorDCA()[i] = (!iCache->floorDCA_.empty()) ? static_cast<float>(iCache->floorDCA_[i]) : -1.0f;
        tripletCutsSoA.maxStubGeomCurvSigma()[i] =
            (!iCache->maxStubGeomCurvSigma_.empty()) ? static_cast<float>(iCache->maxStubGeomCurvSigma_[i]) : -1.0f;
        tripletCutsSoA.maxStubInnerDoubletDCurv()[i] = (!iCache->maxStubInnerDoubletDCurv_.empty())
                                                           ? static_cast<float>(iCache->maxStubInnerDoubletDCurv_[i])
                                                           : -1.0f;
      }

      doubletCutsSoA.dzdrFact() = iCache->dzdrFact_;
      doubletCutsSoA.minInnerSizeB1() = iCache->minInnerSizeB1_;
      doubletCutsSoA.minInnerSizeB2() = iCache->minInnerSizeB2_;
      doubletCutsSoA.maxDSizeB1() = iCache->maxDSizeB1_;
      doubletCutsSoA.maxDSize() = iCache->maxDSize_;
      doubletCutsSoA.maxDSizePred() = iCache->maxDSizePred_;

      tripletCutsSoA.ptmin() = iCache->ptmin_;
      tripletCutsSoA.maxCurv() = iCache->maxCurv_;
      tripletCutsSoA.maxPhiResid() = iCache->maxPhiResid_;
      tripletCutsSoA.sameDPhiSign() = iCache->sameDPhiSign_;

#ifdef GPU_DEBUG
      // Debug output: Print geometry values from Python config
      std::cout << "\n========== CA GEOMETRY FROM PYTHON CONFIG ==========" << std::endl;
      std::cout << "Number of layers: " << n_layers << std::endl;
      std::cout << "Number of layer pairs: " << n_pairs << std::endl;

      std::cout << "\n--- Layer Pair Geometry (all pairs) ---" << std::endl;
      std::cout << "Pair | Inner | Outer | phiCut | minIn | maxIn | minOut | maxOut | maxDR | minDZ | maxDZ | minPt | "
                   "stubSigma | start"
                << std::endl;
      std::cout << "-----|-------|-------|--------|-------|-------|--------|--------|-------|-------|-------|-------|--"
                   "---------|------"
                << std::endl;
      for (int i = 0; i < n_pairs; ++i) {
        float stubSig = (!iCache->maxStubCurvSigma_.empty()) ? iCache->maxStubCurvSigma_[i] : -1.0;
        std::cout << std::setw(4) << i << " | " << std::setw(5) << iCache->layerPairs_[2 * i] << " | " << std::setw(5)
                  << iCache->layerPairs_[2 * i + 1] << " | " << std::setw(6) << iCache->maxDPhi_[i] << " | "
                  << std::setw(5) << iCache->minInner_[i] << " | " << std::setw(5) << iCache->maxInner_[i] << " | "
                  << std::setw(6) << iCache->minOuter_[i] << " | " << std::setw(6) << iCache->maxOuter_[i] << " | "
                  << std::setw(5) << iCache->maxDR_[i] << " | " << std::setw(5) << iCache->minDZ_[i] << " | "
                  << std::setw(5) << iCache->maxDZ_[i] << " | " << std::setw(5) << iCache->minPt_[i] << " | "
                  << std::setw(9) << stubSig << " | " << (graphSoA.startingPair()[i] ? "Y" : "N") << std::endl;
      }

      std::cout << "\n--- Layer Cuts (all layers) ---" << std::endl;
      std::cout << "Layer | isBarrel | caThetaCut | caDCACut | geomKappa | caPhiMiddle" << std::endl;
      std::cout << "------|----------|------------|----------|-----------|-------------|------------|------------"
                << std::endl;
      for (int i = 0; i < n_layers; ++i) {
        float geomK = (!iCache->maxStubGeomCurvSigma_.empty()) ? iCache->maxStubGeomCurvSigma_[i] : -1.0;
        float phiMid = (!iCache->maxStubInnerDoubletDCurv_.empty()) ? iCache->maxStubInnerDoubletDCurv_[i] : -1.0;
        std::cout << std::setw(5) << i << " | " << std::setw(8) << (layerIsBarrel[i] ? "Y" : "N") << " | "
                  << std::setw(10) << iCache->maxRZTolerance_[i] << " | " << std::setw(8) << iCache->maxDCA_[i] << " | "
                  << std::setw(9) << geomK << " | " << std::setw(11) << phiMid << " | " << std::endl;
      }
      std::cout << "====================================================\n" << std::endl;
#endif

      return std::make_shared<CAGeometryCache>(std::move(product));
    }

    static std::unique_ptr<::reco::CAGeometryParams> initializeGlobalCache(edm::ParameterSet const& iConfig) {
      return std::make_unique<::reco::CAGeometryParams>(iConfig.getParameterSet("graph"),
                                                        iConfig.getParameterSet("doubletCuts"),
                                                        iConfig.getParameterSet("tripletCuts"),
                                                        iConfig.getParameterSet("ntupletCuts"),
                                                        iConfig.getParameter<std::vector<double>>("fishboneCuts"));
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
        iEvent.emplace(
            tokenTrack_,
            deviceAlgo_.makeTuplesAsync(hits, geometry, bf, maxDoublets, maxTuples, iEvent.queue(), otRecHits, stubs));
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
