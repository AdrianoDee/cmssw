#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/approx_atan2.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/OTRecHitsSoACollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometryHost.h"
#include "RecoTracker/Record/interface/StackedModuleGeometryRecord.h"

// #define OTREC_HITS_SOA_CONVERTER_DEBUG
// #define OT_HIT_POSITION_DEBUG  // Compare OT hit positions with P-hits and PS stubs

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelSeedingOTRecHitsSoAConverter : public stream::EDProducer<> {
  public:
    explicit PixelSeedingOTRecHitsSoAConverter(const edm::ParameterSet& iConfig);
    ~PixelSeedingOTRecHitsSoAConverter() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    void beginRun(edm::Run const& run, edm::EventSetup const& setup) override;

  private:
    void produce(device::Event& iEvent, const device::EventSetup& es) override;

    // Input tokens
    edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> recHitToken_;
    edm::EDGetTokenT<::reco::BeamSpot> beamSpotToken_;

    // EventSetup tokens for BeginRun transition
    edm::ESGetToken<::reco::StackedModuleGeometryHost, StackedModuleGeometryRecord> geomToken_;

    // EventSetup tokens for Event transition
    edm::ESGetToken<::reco::StackedModuleGeometryHost, StackedModuleGeometryRecord> geomTokenEvent_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;

    // Output tokens - always produce Host collection, framework handles device transfer
    edm::EDPutTokenT<::reco::OTRecHitsHost> otRecHitsToken_;
    // Also produce module start indices for legacy track converter (HMSstorage = std::vector<uint32_t>)
    edm::EDPutTokenT<std::vector<uint32_t>> hmsToken_;

    // Maps built in beginRun
    std::map<uint32_t, uint32_t> stackDetIdToGeomIdx_;  // stack DetId -> StackedModuleGeometry index (0-13199)
    std::vector<uint32_t> orderedStacks_;               // ordered list of stack DetIds
    uint32_t nModules_;                                 // total number of stacked modules in geometry

    // CA module offsets (computed in beginRun for logging)
    static constexpr uint32_t nPixelModules_ = 4000;  // Phase-2 has fixed 4000 pixel modules
    uint32_t nBarrelModules_;                         // OT barrel modules (for logging)
    uint32_t nBackwardModules_;                       // OT backward disk modules (for logging)
    uint32_t nForwardModules_;                        // OT forward disk modules (for logging)
  };

  PixelSeedingOTRecHitsSoAConverter::PixelSeedingOTRecHitsSoAConverter(const edm::ParameterSet& iConfig)
      : stream::EDProducer<>(iConfig),
        recHitToken_(consumes(iConfig.getParameter<edm::InputTag>("otRecHitSource"))),
        beamSpotToken_(consumes(iConfig.getParameter<edm::InputTag>("beamSpot"))),
        geomToken_(esConsumes<edm::Transition::BeginRun>()),
        geomTokenEvent_(esConsumes()),
        trackerGeomToken_(esConsumes()),
        topoToken_(esConsumes()),
        otRecHitsToken_(produces()),
        hmsToken_(produces()) {}

  void PixelSeedingOTRecHitsSoAConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("otRecHitSource", edm::InputTag("siPhase2RecHits"))
        ->setComment("Input Phase2TrackerRecHit1D collection");
    desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"))
        ->setComment("Beam spot for position correction");
    descriptions.addWithDefaultLabel(desc);
  }

  void PixelSeedingOTRecHitsSoAConverter::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
    stackDetIdToGeomIdx_.clear();
    orderedStacks_.clear();

    const auto& geomHost = iSetup.getData(geomToken_);
    auto const& geomView = geomHost.const_view();
    nModules_ = geomView.metadata().size();

    // Build map: stack DetId -> StackedModuleGeometry index (0-nModules)
    // IMPORTANT: orderedStacks_ must stay in geometry index order (not sorted by DetId)
    // because moduleStart is indexed by geometry index
    for (uint32_t iGeom = 0; iGeom < nModules_; ++iGeom) {
      uint32_t stackDetId = geomView[iGeom].stackedDetId();
      stackDetIdToGeomIdx_[stackDetId] = iGeom;
      orderedStacks_.push_back(stackDetId);
    }
    // Note: Do NOT sort orderedStacks_ - it must remain in geometry index order
    // to match how moduleStart is indexed

    // Count modules by category for verification logging
    // Note: StackedModuleGeometry is now sorted in CA order (barrel, backward, forward by layer)
    // so caModuleIndex = nPixelModules_ + geometryIndex
    nForwardModules_ = 0;
    nBackwardModules_ = 0;
    nBarrelModules_ = 0;

    for (uint32_t iGeom = 0; iGeom < nModules_; ++iGeom) {
      bool isBarrel = geomView[iGeom].isBarrel();
      bool isFwdEndcap = geomView[iGeom].isFwdEndcap();

      if (isBarrel) {
        nBarrelModules_++;
      } else if (isFwdEndcap) {
        nForwardModules_++;
      } else {
        nBackwardModules_++;
      }
    }

    // Compute CA offsets (for logging)
    uint32_t caBarrelOffset = nPixelModules_;
    uint32_t caBackwardOffset = caBarrelOffset + nBarrelModules_;
    uint32_t caForwardOffset = caBackwardOffset + nBackwardModules_;

    edm::LogInfo("PixelSeedingOTRecHitsSoAConverter")
        << "OT Module Configuration (geometry sorted in CA order):\n"
        << "  Pixel modules: " << nPixelModules_ << " (CA index 0-" << (nPixelModules_ - 1) << ")\n"
        << "  OT Barrel: " << nBarrelModules_ << " modules (CA index " << caBarrelOffset << "-"
        << (caBackwardOffset - 1) << ")\n"
        << "  OT Backward: " << nBackwardModules_ << " modules (CA index " << caBackwardOffset << "-"
        << (caForwardOffset - 1) << ")\n"
        << "  OT Forward: " << nForwardModules_ << " modules (CA index " << caForwardOffset << "-"
        << (caForwardOffset + nForwardModules_ - 1) << ")";
  }

  void PixelSeedingOTRecHitsSoAConverter::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    auto queue = iEvent.queue();
    const auto& bs = iEvent.get(beamSpotToken_);
    const auto& trackerGeom = iSetup.getData(trackerGeomToken_);
    const auto& topo = iSetup.getData(topoToken_);
    const auto& recHits = iEvent.get(recHitToken_);

    // Get StackedModuleGeometry to map to CA module indices
    const auto& geomHost = iSetup.getData(geomTokenEvent_);
    [[maybe_unused]] auto const& geomView = geomHost.const_view();

    // Structure to temporarily hold hits on host
    struct HitData {
      float xLocal, yLocal, xerrLocal, yerrLocal;
      float xGlobal, yGlobal, zGlobal, rGlobal;
      float xGlobalErr, xyGlobalErr, yGlobalErr;
      float xzGlobalErr, yzGlobalErr, zGlobalErr;
      int16_t iphi;
      uint16_t detectorIndex;
      uint32_t stackDetId;
      uint32_t sensorDetId;
      bool isLower;
      uint32_t origRecHitIdx;
      uint16_t clusterSize;
    };

    // Organize hits by stack DetId, then by sensor (lower first, upper second)
    std::map<uint32_t, std::vector<HitData>> hitsByStack;

    // Process all hits and group by stack
    // Maintain a flat running index across all RecHits for truth matching
    uint32_t flatRecHitIdx = 0;
    for (const auto& detSet : recHits) {
      DetId detId(detSet.detId());

      // Skip if not a stacked module sensor
      uint32_t stackDetId = topo.stack(detId);
      if (stackDetId == 0) {
        flatRecHitIdx += detSet.size();
        continue;
      }

      // Check if we have geometry for this stack
      auto geomIt = stackDetIdToGeomIdx_.find(stackDetId);
      if (geomIt == stackDetIdToGeomIdx_.end()) {
        flatRecHitIdx += detSet.size();
        continue;
      }

      uint32_t iGeom = geomIt->second;
      bool isLower = topo.isLower(detId);
      auto det = trackerGeom.idToDet(detId);

      // Process all hits in this DetSet
      for (const auto& recHit : detSet) {
        HitData hit;

        // Local position and errors
        auto localPos = recHit.localPosition();
        auto localErr = recHit.localPositionError();
        hit.xLocal = localPos.x();
        hit.yLocal = localPos.y();
        hit.xerrLocal = localErr.xx();  // Store variance, not sigma (consistent with pixel hits)
        hit.yerrLocal = localErr.yy();  // Store variance, not sigma (consistent with pixel hits)

        // Global error covariance matrix (6 elements)
        // Transform local errors to global using rotation matrix
        // Formula from SOAFrame::toGlobal(cxx, cxy, cyy, gl)
        float cxx = localErr.xx();
        float cxy = 0.f;  // Local correlation is zero for strip sensors
        float cyy = localErr.yy();

        auto const& rot = det->surface().rotation();
        // gl[0] = r.xx() * (r.xx() * cxx + r.yx() * cxy) + r.yx() * (r.xx() * cxy + r.yx() * cyy);
        hit.xGlobalErr = rot.xx() * (rot.xx() * cxx + rot.yx() * cxy) + rot.yx() * (rot.xx() * cxy + rot.yx() * cyy);
        // gl[1] = r.xx() * (r.xy() * cxx + r.yy() * cxy) + r.yx() * (r.xy() * cxy + r.yy() * cyy);
        hit.xyGlobalErr = rot.xx() * (rot.xy() * cxx + rot.yy() * cxy) + rot.yx() * (rot.xy() * cxy + rot.yy() * cyy);
        // gl[2] = r.xy() * (r.xy() * cxx + r.yy() * cxy) + r.yy() * (r.xy() * cxy + r.yy() * cyy);
        hit.yGlobalErr = rot.xy() * (rot.xy() * cxx + rot.yy() * cxy) + rot.yy() * (rot.xy() * cxy + rot.yy() * cyy);
        // gl[3] = r.xx() * (r.xz() * cxx + r.yz() * cxy) + r.yx() * (r.xz() * cxy + r.yz() * cyy);
        hit.xzGlobalErr = rot.xx() * (rot.xz() * cxx + rot.yz() * cxy) + rot.yx() * (rot.xz() * cxy + rot.yz() * cyy);
        // gl[4] = r.xy() * (r.xz() * cxx + r.yz() * cxy) + r.yy() * (r.xz() * cxy + r.yz() * cyy);
        hit.yzGlobalErr = rot.xy() * (rot.xz() * cxx + rot.yz() * cxy) + rot.yy() * (rot.xz() * cxy + rot.yz() * cyy);
        // gl[5] = r.xz() * (r.xz() * cxx + r.yz() * cxy) + r.yz() * (r.xz() * cxy + r.yz() * cyy);
        hit.zGlobalErr = rot.xz() * (rot.xz() * cxx + rot.yz() * cxy) + rot.yz() * (rot.xz() * cxy + rot.yz() * cyy);

        // Global position (beam-spot corrected, consistent with pixel hits)
        auto globalPos = det->toGlobal(localPos);
        float gx = globalPos.x() - bs.x0();
        float gy = globalPos.y() - bs.y0();
        float gz = globalPos.z() - bs.z0();
        hit.xGlobal = gx;
        hit.yGlobal = gy;
        hit.zGlobal = gz;
        hit.rGlobal = std::sqrt(gx * gx + gy * gy);

        // Discretized phi (1024 bins over 2π)
        hit.iphi = unsafe_atan2s<7>(gy, gx);

        // Detector index - map to CA module index
        // StackedModuleGeometry is sorted in CA order (barrel by layer, backward, forward)
        // so CA module index is simply nPixelModules + geometry index
        uint32_t caModuleIndex = nPixelModules_ + iGeom;
        hit.detectorIndex = static_cast<uint16_t>(caModuleIndex);

        // Stack and sensor IDs
        hit.stackDetId = stackDetId;
        hit.sensorDetId = detId.rawId();
        hit.isLower = isLower;
        hit.origRecHitIdx = flatRecHitIdx;
        hit.clusterSize = recHit.cluster()->size();

        // Add to the appropriate stack's hit list
        hitsByStack[stackDetId].push_back(hit);
        flatRecHitIdx++;
      }
    }

    // Count total hits and organize by stack
    // Each stack has lower hits followed by upper hits
    uint32_t totalHits = 0;
    std::vector<uint32_t> hitsPerStack(orderedStacks_.size(), 0);
    std::vector<uint32_t> lowerHitsPerStack(orderedStacks_.size(), 0);
    std::vector<uint32_t> upperHitsPerStack(orderedStacks_.size(), 0);

    for (size_t iStack = 0; iStack < orderedStacks_.size(); ++iStack) {
      uint32_t stackDetId = orderedStacks_[iStack];
      auto& stackHits = hitsByStack[stackDetId];

      // Separate lower and upper hits
      std::vector<HitData> lowerHits, upperHits;
      for (const auto& hit : stackHits) {
        if (hit.isLower) {
          lowerHits.push_back(hit);
        } else {
          upperHits.push_back(hit);
        }
      }

      lowerHitsPerStack[iStack] = lowerHits.size();
      upperHitsPerStack[iStack] = upperHits.size();
      hitsPerStack[iStack] = lowerHits.size() + upperHits.size();

      // Re-order: lower hits first, then upper hits
      stackHits.clear();
      stackHits.insert(stackHits.end(), lowerHits.begin(), lowerHits.end());
      stackHits.insert(stackHits.end(), upperHits.begin(), upperHits.end());

      totalHits += hitsPerStack[iStack];
    }

    // Create HOST collection first - fill everything on host
    // moduleView sized for StackedModuleGeometry indices (0 to nModules)
    // Note: OTRecHitsHost constructor already adds +1 for the sentinel element
    const uint32_t nStacks = orderedStacks_.size();
    ::reco::OTRecHitsHost hostHits(cms::alpakatools::host(), totalHits, nModules_);

    // Get host views
    auto hitsView = hostHits.view().otRecHits();
    auto moduleView = hostHits.view().otHitModules();

    // Initialize all module entries to sentinel values (end of hit array)
    for (uint32_t i = 0; i <= nModules_; ++i) {
      moduleView[i].moduleStart() = totalHits;
      moduleView[i].upperSensorStart() = totalHits;
    }

#ifdef OTREC_HITS_SOA_CONVERTER_DEBUG
    // Debug: verify initialization
    edm::LogPrint("PixelSeedingOTRecHitsSoAConverter")
        << "After initialization (totalHits=" << totalHits << ", nModules=" << nModules_ << "):";
    for (uint32_t i = 0; i < std::min(10u, nModules_ + 1); ++i) {
      edm::LogPrint("PixelSeedingOTRecHitsSoAConverter")
          << "  moduleView[" << i << "].moduleStart() = " << moduleView[i].moduleStart()
          << ", upperSensorStart() = " << moduleView[i].upperSensorStart();
    }
#endif

    // Fill moduleView directly with hit ranges for each stack
    // moduleView is indexed by StackedModuleGeometry index (iGeom)
    uint32_t hitOffset = 0;
    for (size_t iStack = 0; iStack < nStacks; ++iStack) {
      uint32_t stackDetId = orderedStacks_[iStack];
      uint32_t iGeom = stackDetIdToGeomIdx_[stackDetId];

      // Fill module entry (indexed by iGeom) with hit ranges
      moduleView[iGeom].moduleStart() = hitOffset;
      moduleView[iGeom].upperSensorStart() = hitOffset + lowerHitsPerStack[iStack];

      hitOffset += hitsPerStack[iStack];
    }

    // Set final sentinel value at end of array
    moduleView[nModules_].moduleStart() = totalHits;
    moduleView[nModules_].upperSensorStart() = totalHits;

    // Fix sentinel values for modules without hits
    // Modules without hits have moduleStart = totalHits (from initialization)
    // We need to set them to point to the next valid module's start
    // This ensures hitEnd = moduleView[iModule+1].moduleStart() works correctly
    uint32_t nextStart = totalHits;
    uint32_t nextUpperStart = totalHits;
    for (int32_t i = static_cast<int32_t>(nModules_); i >= 0; --i) {
      if (moduleView[i].moduleStart() == totalHits) {
        // This module has no hits, point to next valid offset
        moduleView[i].moduleStart() = nextStart;
        moduleView[i].upperSensorStart() = nextUpperStart;
      } else {
        // This module has hits, update next valid offsets
        nextStart = moduleView[i].moduleStart();
        nextUpperStart = moduleView[i].upperSensorStart();
      }
    }

#ifdef OTREC_HITS_SOA_CONVERTER_DEBUG
    // Debug: verify module filling for first few stacks
    edm::LogPrint("PixelSeedingOTRecHitsSoAConverter") << "After filling modules with hits:";
    for (size_t iStack = 0; iStack < std::min(size_t(3), size_t(nStacks)); ++iStack) {
      uint32_t stackDetId = orderedStacks_[iStack];
      uint32_t iGeom = stackDetIdToGeomIdx_[stackDetId];
      edm::LogPrint("PixelSeedingOTRecHitsSoAConverter")
          << "  Stack " << iStack << " (DetId=" << stackDetId << ", iGeom=" << iGeom
          << "): moduleStart=" << moduleView[iGeom].moduleStart()
          << ", upperSensorStart=" << moduleView[iGeom].upperSensorStart();
    }
    edm::LogPrint("PixelSeedingOTRecHitsSoAConverter") << "First 10 modules after filling:";
    for (uint32_t i = 0; i < std::min(10u, nModules_ + 1); ++i) {
      edm::LogPrint("PixelSeedingOTRecHitsSoAConverter")
          << "  moduleView[" << i << "].moduleStart() = " << moduleView[i].moduleStart();
    }

    // Debug: check modules around those with hits to verify forward-fill
    edm::LogPrint("PixelSeedingOTRecHitsSoAConverter") << "Modules around those with hits:";
    std::vector<uint32_t> checkModules = {6157, 6158, 6159, 6160, 6223, 6224, 6225};
    for (uint32_t i : checkModules) {
      if (i <= nModules_) {
        edm::LogPrint("PixelSeedingOTRecHitsSoAConverter")
            << "  moduleView[" << i << "]: moduleStart=" << moduleView[i].moduleStart()
            << ", upperSensorStart=" << moduleView[i].upperSensorStart();
      }
    }
#endif
    // Fill hit data
    hitOffset = 0;
    for (size_t iStack = 0; iStack < nStacks; ++iStack) {
      uint32_t stackDetId = orderedStacks_[iStack];
      const auto& stackHits = hitsByStack[stackDetId];

      for (const auto& hit : stackHits) {
        hitsView[hitOffset].xLocal() = hit.xLocal;
        hitsView[hitOffset].yLocal() = hit.yLocal;
        hitsView[hitOffset].xerrLocal() = hit.xerrLocal;
        hitsView[hitOffset].yerrLocal() = hit.yerrLocal;
        hitsView[hitOffset].xGlobal() = hit.xGlobal;
        hitsView[hitOffset].yGlobal() = hit.yGlobal;
        hitsView[hitOffset].zGlobal() = hit.zGlobal;
        hitsView[hitOffset].rGlobal() = hit.rGlobal;
        hitsView[hitOffset].xGlobalErr() = hit.xGlobalErr;
        hitsView[hitOffset].xyGlobalErr() = hit.xyGlobalErr;
        hitsView[hitOffset].yGlobalErr() = hit.yGlobalErr;
        hitsView[hitOffset].xzGlobalErr() = hit.xzGlobalErr;
        hitsView[hitOffset].yzGlobalErr() = hit.yzGlobalErr;
        hitsView[hitOffset].zGlobalErr() = hit.zGlobalErr;
        hitsView[hitOffset].iphi() = hit.iphi;
        hitsView[hitOffset].detectorIndex() = hit.detectorIndex;
        hitsView[hitOffset].stackDetId() = hit.stackDetId;
        hitsView[hitOffset].sensorDetId() = hit.sensorDetId;
        hitsView[hitOffset].isLower() = hit.isLower;
        hitsView[hitOffset].origRecHitIdx() = hit.origRecHitIdx;
        hitsView[hitOffset].clusterSize() = hit.clusterSize;
#ifdef OT_HIT_POSITION_DEBUG
        // Debug: print OT hit position for comparison with P-hits
        // Only print lower sensor hits (which correspond to P-hits in PS modules)
        if (hit.isLower) {
          edm::LogPrint("PixelSeedingOTRecHitsSoAConverter")
              << "OT_HIT sensorDetId=" << hit.sensorDetId << " hitOffset=" << hitOffset
              << " x=" << std::fixed << std::setprecision(6) << hit.xGlobal
              << " y=" << hit.yGlobal << " z=" << hit.zGlobal
              << " r=" << std::setprecision(4) << hit.rGlobal
              << " iphi=" << hit.iphi
              << " xerrLocal=" << std::scientific << std::setprecision(4) << hit.xerrLocal
              << " yerrLocal=" << hit.yerrLocal
              << " xGlobalErr=" << hit.xGlobalErr
              << " yGlobalErr=" << hit.yGlobalErr
              << " zGlobalErr=" << hit.zGlobalErr
              << " stackDetId=" << hit.stackDetId
              << " detIdx=" << hit.detectorIndex;
        }
#endif
        hitOffset++;
      }
    }

    LogDebug("PixelSeedingOTRecHitsSoAConverter")
        << "Converted " << totalHits << " OT hits from " << nStacks << " stacked modules (geometry has " << nModules_
        << " total modules)";

    // Create HMSstorage (module start indices) for legacy track converter
    // This is indexed by geometry index (0 to nModules_), same as moduleView
    std::vector<uint32_t> hmsStorage(nModules_ + 1);
    for (uint32_t i = 0; i <= nModules_; ++i) {
      hmsStorage[i] = moduleView[i].moduleStart();
    }

    // Put outputs - framework will handle device transfer if needed
    iEvent.emplace(hmsToken_, std::move(hmsStorage));
    iEvent.emplace(otRecHitsToken_, std::move(hostHits));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PixelSeedingOTRecHitsSoAConverter);
