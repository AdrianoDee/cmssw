#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAGeometryBuild_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAGeometryBuild_h

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometryHost.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometryHost.h"

namespace reco {

  // Build the geometry-only blocks of the CA geometry SoA, a pure function of (TrackerGeometry,
  // TrackerTopology, StackedModuleGeometry, nLayers):
  //   layers  : layerStarts, isBarrel, isOT, isSS   (nLayers + 1 rows; the extra layerStarts slot
  //             holds the total module count)
  //   modules : detFrame, innerSensorFrame          (one row per CA module)
  // Nothing per-iteration is written, so one copy serves every CA iteration over the same geometry.
  //   stackedGeometry        : the CA-ordered OT stacked-module geometry, non-null only for the
  //                            Phase2OTStubs topology.
  //   nLayers                : the CA layer count (sizes the layers block).
  //   nCutPairs / nCutLayers : row counts of the cut blocks, allocated here but never written; both
  //                            default to 0, and a CA producer that keeps its cuts in the same product
  //                            passes (n_pairs, n_layers) and fills them in place.
  template <typename TrackerTraits>
  reco::CAGeometryHost buildCAGeometryHost(TrackerGeometry const& trackerGeometry,
                                           TrackerTopology const& trackerTopology,
                                           reco::StackedModuleGeometryHost const* stackedGeometry,
                                           int nLayers,
                                           int nCutPairs = 0,
                                           int nCutLayers = 0) {
    using Rotation = SOARotation<float>;
    using Frame = SOAFrame<float>;

    int n_layers = nLayers;
    int n_modules = 0;

    assert(n_layers > 0);
    assert(nCutPairs >= 0);
    assert(nCutLayers == 0 || nCutLayers == n_layers);

    auto const& dets = trackerGeometry.dets();

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

      // Modules of the pixel layers
      if (isPixel(detid)) {
        if (layer != oldLayer) {
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

    // CA layers follow inside-out ordering:
    // - CA layers 28-33: Barrel (layers 1-6)
    // - CA layers 34-43: disks 1-5 at z > 0, each disk as a PS layer (even id) then a 2S layer (odd id)
    // - CA layers 44-53: disks 1-5 at z < 0, same PS/2S alternation
    //
    // StackedModuleGeometry is already sorted in CA order by StackedModuleGeometryESProducer (barrel by
    // layer, then z > 0 disks by layer, then z < 0 disks by layer, PS before 2S). It is walked in index
    // order so that the frame array index matches the detectorIndex assigned to hits and stubs
    // (detectorIndex = nPixelModules + geomIndex).
    // Number of pixel modules already processed; OT modules start at this CA module index.
    const int nPixelModulesInCA = n_modules;

    if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
      if (stackedGeometry != nullptr) {
        auto stackedView = stackedGeometry->view();
        const uint32_t nStackedModules = static_cast<uint32_t>(stackedView.metadata().size());

        bool firstModule = true;
        uint8_t prevLayer = 0;
        bool prevIsPS = false;
        int prevCategory = -1;  // 0=barrel, 1=z < 0 endcap, 2=z > 0 endcap

        for (uint32_t i = 0; i < nStackedModules; ++i) {
          bool isBarrelMod = stackedView.isBarrel()[i];
          bool isFwdEndcap = stackedView.isFwdEndcap()[i];
          bool isPS = stackedView.isPS()[i];
          uint8_t otLayer = stackedView.layer()[i];
          DetId stackedDetId(stackedView.stackedDetId()[i]);

          // Determine category: 0=barrel, 1=z < 0 endcap, 2=z > 0 endcap
          int category = isBarrelMod ? 0 : (isFwdEndcap ? 2 : 1);

          // A new CA layer starts when the category changes, or the layer number changes within one.
          if (firstModule || category != prevCategory || otLayer != prevLayer || isPS != prevIsPS) {
            if (layerCount < layerStarts.size()) {
              layerIsBarrel[layerCount] = isBarrelMod;
              layerIsOT[layerCount] =
                  true;  // we are in the loop over StackedModuleGeometry, so these are all OT layers
              layerIsSS[layerCount] = !isPS;
              layerStarts[layerCount++] = n_modules;
            }
            prevCategory = category;
            prevLayer = otLayer;
            prevIsPS = isPS;
            firstModule = false;
          }

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
            edm::LogWarning("CAGeometryBuild")
                << "Could not find stacked module " << stackedDetId.rawId() << " in TrackerGeometry";
          }
        }
      }
    }

    layerStarts[n_layers] = n_modules;

    // Block sizes, in CALayoutTemplate order: layers, graph, doubletCuts, tripletCuts, ntupletCuts,
    // modules. With the default nCutPairs/nCutLayers = 0 the four configuration blocks get zero rows.
    reco::CAGeometryHost product{
        cms::alpakatools::host(), n_layers + 1, nCutPairs, nCutPairs, nCutPairs, nCutLayers, n_modules};

    auto layerSoA = product.view().layers();
    auto modulesSoA = product.view().modules();

    // For Phase2OTStubs, stackedView picks the per-sensor frame for each OT module's `innerSensorFrame`:
    //   - PSP (moduleType==0): inner = lower sensor (P-side).
    //   - PSS (moduleType==1): inner = upper sensor (P-side).
    //   - SS:                   inner = physically-inner = lower if !isFlipped, else upper.
    auto pickInnerFrame = [&](int iCA) -> Frame {
      // Default fallback: identity-handling via detFrame for non-OT or missing stacked info.
      if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
        if (stackedGeometry != nullptr && iCA >= nPixelModulesInCA) {
          auto stackedView = stackedGeometry->view();
          const uint32_t nStackedModules = static_cast<uint32_t>(stackedView.metadata().size());
          const uint32_t iGeom = static_cast<uint32_t>(iCA - nPixelModulesInCA);
          if (iGeom < nStackedModules) {
            uint8_t mt = stackedView.moduleType()[iGeom];
            bool isFlipped = stackedView.isFlipped()[iGeom];
            bool useUpper;
            if (mt == 0)
              useUpper = false;  // PSP: pixel is lower
            else if (mt == 1)
              useUpper = true;  // PSS: pixel is upper
            else
              useUpper = isFlipped;  // SS: inner = lower if !flipped, else upper
            return useUpper ? Frame(stackedView[iGeom].upperSensorFrame())
                            : Frame(stackedView[iGeom].lowerSensorFrame());
          }
        }
      }
      // Pixel modules (or non-OTStubs topology): innerSensorFrame == detFrame.
      return modulesSoA[iCA].detFrame();
    };

    for (int i = 0; i < n_modules; ++i) {
      auto idx = moduleToindexInDets[i];
      auto det = dets[idx];
      auto vv = det->surface().position();
      auto rr = Rotation(det->surface().rotation());
      modulesSoA[i].detFrame() = Frame(vv.x(), vv.y(), vv.z(), rr);
      modulesSoA[i].innerSensorFrame() = pickInnerFrame(i);
    }

    for (int i = 0; i < n_layers; ++i) {
      layerSoA.layerStarts()[i] = layerStarts[i];
      layerSoA.isBarrel()[i] = layerIsBarrel[i];
      layerSoA.isOT()[i] = layerIsOT[i];
      layerSoA.isSS()[i] = layerIsSS[i];
    }

    layerSoA.layerStarts()[n_layers] = layerStarts[n_layers];

    return product;
  }

}  // namespace reco

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAGeometryBuild_h
