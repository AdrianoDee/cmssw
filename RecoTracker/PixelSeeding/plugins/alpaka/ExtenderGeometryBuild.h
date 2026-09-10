#ifndef RecoTracker_PixelSeeding_plugins_alpaka_ExtenderGeometryBuild_h
#define RecoTracker_PixelSeeding_plugins_alpaka_ExtenderGeometryBuild_h

// Once-per-run host build of the extender layer geometry (mean layer surfaces, module-envelope
// half-extents, module-to-layer starts), walking TrackerGeometry and StackedModuleGeometry in CA
// module order.

#include <sstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometryHost.h"

#include "ExtenderGeometry.h"

namespace reco_extender {

  inline ExtenderGeometryHost buildExtenderLayerGeometry(const TrackerGeometry& trackerGeometry,
                                                         const TrackerTopology& trackerTopology,
                                                         const ::reco::StackedModuleGeometryHost& stackedGeometry) {
    auto const& dets = trackerGeometry.dets();

    auto isPixel = [&](DetId detId) {
      auto subId = detId.subdetId();
      return (subId == PixelSubdetector::PixelBarrel || subId == PixelSubdetector::PixelEndcap);
    };
    auto isBarrel = [&](DetId detId) {
      auto subId = detId.subdetId();
      auto subDetector = trackerGeometry.geomDetSubDetector(subId);
      return GeomDetEnumerators::isBarrel(subDetector);
    };

    constexpr int kMaxLayers = ::reco::kMaxCALayers;  // reserve() hint; the bound is enforced below
    std::vector<uint32_t> layerStarts;
    layerStarts.reserve(kMaxLayers + 1);
    std::vector<bool> layerIsBarrel;
    layerIsBarrel.reserve(kMaxLayers);
    std::vector<double> sumR, sumZ;
    std::vector<int> nDets;
    std::vector<double> minR, maxR, minZ, maxZ;
    sumR.reserve(kMaxLayers);
    sumZ.reserve(kMaxLayers);
    nDets.reserve(kMaxLayers);
    minR.reserve(kMaxLayers);
    maxR.reserve(kMaxLayers);
    minZ.reserve(kMaxLayers);
    maxZ.reserve(kMaxLayers);

    auto pushNewLayer = [&](bool barrel) {
      layerIsBarrel.push_back(barrel);
      sumR.push_back(0.);
      sumZ.push_back(0.);
      nDets.push_back(0);
      minR.push_back(+1e30);
      maxR.push_back(-1e30);
      minZ.push_back(+1e30);
      maxZ.push_back(-1e30);
    };
    auto accumulateDet = [&](GeomDet const* det) {
      auto const& p = det->surface().position();
      const double r = std::sqrt(p.x() * p.x() + p.y() * p.y());
      sumR.back() += r;
      // Signed z: endcap sides must stay distinguishable downstream; barrel means are ~0.
      sumZ.back() += p.z();
      nDets.back() += 1;
      if (r < minR.back())
        minR.back() = r;
      if (r > maxR.back())
        maxR.back() = r;
      if (p.z() < minZ.back())
        minZ.back() = p.z();
      if (p.z() > maxZ.back())
        maxZ.back() = p.z();
    };

    int n_modules = 0;
    auto oldLayer = 0u;

    // Pass 1: pixel modules (barrel + endcap).
    for (auto& det : dets) {
      DetId detid = det->geographicalId();
      if (!isPixel(detid))
        continue;
      auto layer = trackerTopology.layer(detid);
      if (layer != oldLayer) {
        layerStarts.push_back(n_modules);
        pushNewLayer(isBarrel(detid));
        oldLayer = layer;
      }
      accumulateDet(det);
      ++n_modules;
    }

    // Pass 2: OT stacked modules, already CA-sorted; following the CAHitNtuplet convention, a new
    // layer starts on any category, OT-layer-number or PS-vs-SS transition.
    {
      auto stackedView = stackedGeometry.view();
      const uint32_t nStackedModules = static_cast<uint32_t>(stackedView.metadata().size());

      bool firstModule = true;
      uint8_t prevLayer = 0;
      bool prevIsPS = false;
      int prevCategory = -1;  // 0=barrel, 1=backward, 2=forward

      for (uint32_t i = 0; i < nStackedModules; ++i) {
        const bool barrel = stackedView.isBarrel()[i];
        const bool isFwd = stackedView.isFwdEndcap()[i];
        const bool isPS = stackedView.isPS()[i];
        const uint8_t otLayer = stackedView.layer()[i];
        const int category = barrel ? 0 : (isFwd ? 2 : 1);

        const bool newCAlayer = firstModule || category != prevCategory || otLayer != prevLayer || isPS != prevIsPS;
        if (newCAlayer) {
          layerStarts.push_back(n_modules);
          pushNewLayer(barrel);
          prevCategory = category;
          prevLayer = otLayer;
          prevIsPS = isPS;
          firstModule = false;
        }

        // Once per run, so a linear search is fine.
        const DetId stackedDetId(stackedView.stackedDetId()[i]);
        for (auto const& det : dets) {
          if (det->geographicalId() == stackedDetId) {
            accumulateDet(det);
            ++n_modules;
            break;
          }
        }
      }
    }

    layerStarts.push_back(n_modules);
    const int n_layers = static_cast<int>(layerIsBarrel.size());
    // The OT-extension walk keeps its per-layer state in arrays sized at reco::kMaxCALayers and in a
    // uint64_t layer bitmask; more layers than that would silently overrun both.
    if (n_layers > ::reco::kMaxCALayers)
      throw cms::Exception("Configuration")
          << "buildExtenderLayerGeometry: the tracker geometry yields " << n_layers
          << " CA layers, above the OT-extension walk's hard bound of " << ::reco::kMaxCALayers << ".";

    ::reco_extender::ExtenderGeometryHost host(cms::alpakatools::host(), n_layers + 1);
    auto view = host.view();
    for (int i = 0; i < n_layers; ++i) {
      view.layerStarts()[i] = layerStarts[i];
      view.isBarrel()[i] = layerIsBarrel[i];
      const float invN = nDets[i] > 0 ? 1.f / float(nDets[i]) : 0.f;
      view.layerR()[i] = static_cast<float>(sumR[i]) * invN;
      view.layerZ()[i] = static_cast<float>(sumZ[i]) * invN;
      view.halfExtentR()[i] = nDets[i] > 0 ? static_cast<float>(0.5 * (maxR[i] - minR[i])) : 0.f;
      view.halfExtentZ()[i] = nDets[i] > 0 ? static_cast<float>(0.5 * (maxZ[i] - minZ[i])) : 0.f;
    }
    view.layerStarts()[n_layers] = layerStarts[n_layers];
    view.isBarrel()[n_layers] = false;
    view.layerR()[n_layers] = 0.f;
    view.layerZ()[n_layers] = 0.f;
    view.halfExtentR()[n_layers] = 0.f;
    view.halfExtentZ()[n_layers] = 0.f;

#ifdef EXTENDER_DEBUG_GEOMDUMP
    {
      std::stringstream ss;
      ss << "[ExtenderGeometryBuild] Built extender geometry: " << n_layers << " CA layers over " << n_modules
         << " modules\n";
      ss << "  Layer | start | nMod | type   | <r>   | <z>   | dR/2  | dZ/2\n";
      ss << "  ------+-------+------+--------+-------+-------+-------+--------\n";
      for (int i = 0; i < n_layers; ++i) {
        const uint32_t start = layerStarts[i];
        const uint32_t end = layerStarts[i + 1];
        const float r = nDets[i] > 0 ? float(sumR[i]) / float(nDets[i]) : 0.f;
        const float z = nDets[i] > 0 ? float(sumZ[i]) / float(nDets[i]) : 0.f;
        const float dr = nDets[i] > 0 ? float(0.5 * (maxR[i] - minR[i])) : 0.f;
        const float dz = nDets[i] > 0 ? float(0.5 * (maxZ[i] - minZ[i])) : 0.f;
        ss << "  " << std::setw(5) << i << " | " << std::setw(5) << start << " | " << std::setw(4) << (end - start)
           << " | " << (layerIsBarrel[i] ? "barrel" : "endcap") << " | " << std::fixed << std::setprecision(2)
           << std::setw(5) << r << " | " << std::setw(6) << z << " | " << std::setw(5) << dr << " | " << std::setw(6)
           << dz << "\n";
      }
      std::cout << ss.str();
    }
#endif  // EXTENDER_DEBUG_GEOMDUMP

    return host;
  }

}  // namespace reco_extender

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_ExtenderGeometryBuild_h
