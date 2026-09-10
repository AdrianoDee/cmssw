#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsHost.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/PixelSeeding/interface/OTHitTag.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/FitUtils.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"

#include "storeTracks.h"

/**
 * This class creates "legacy" reco::Track
 * objects from the output of SoA CA.
 */

// #define GPU_DEBUG
// struct that holds two maps for detIds of the OT modules
struct DetIdMaps {
  DetIdMaps() : detIdToOTModuleId_(), detIdIsUsedOTModule_() {}

  // map from the detId of OT modules to the moduleId among the used OT modules
  // (starting from 0 for first module of first OT layer)
  std::map<uint32_t, uint32_t> detIdToOTModuleId_;
  // map from detId to bool if used as OT extension
  std::map<uint32_t, bool> detIdIsUsedOTModule_;
};

class PixelTrackProducerFromSoAAlpaka : public edm::global::EDProducer<edm::RunCache<DetIdMaps>> {
  using TrackSoAHost = reco::TracksHost;
  using HMSstorage = std::vector<uint32_t>;
  using IndToEdm = std::vector<uint32_t>;
  using TrackHitSoA = reco::TrackHitSoA;

public:
  explicit PixelTrackProducerFromSoAAlpaka(const edm::ParameterSet &iConfig);
  ~PixelTrackProducerFromSoAAlpaka() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  std::shared_ptr<DetIdMaps> globalBeginRun(edm::Run const &, edm::EventSetup const &) const override;
  void globalEndRun(edm::Run const &, edm::EventSetup const &) const override {};

private:
  void produce(edm::StreamID streamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const override;

  // Event Data tokens
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  const edm::EDGetTokenT<TrackSoAHost> trackSoAToken_;
  const edm::EDGetTokenT<SiPixelRecHitCollectionNew> pixelRecHitsToken_;
  edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> otRecHitsToken_;
  const edm::EDGetTokenT<HMSstorage> pixelHMSToken_;
  edm::EDGetTokenT<HMSstorage> otHMSToken_;
  edm::EDGetTokenT<reco::OTRecHitsHost> otRecHitsSoAToken_;
  edm::EDGetTokenT<reco::StubsHost> stubsSoAToken_;
  // Event Setup tokens
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> idealMagneticFieldToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryTokenRun_;

  int32_t const minNumberOfHits_;
  pixelTrack::Quality const minQuality_;
  const bool useOTExtension_;
  const bool throwOnMissing_;
  const bool expandStubs_;
  const bool requireQuadsFromConsecutiveLayers_;
  const bool verbose_;
};

PixelTrackProducerFromSoAAlpaka::PixelTrackProducerFromSoAAlpaka(const edm::ParameterSet &iConfig)
    : beamSpotToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      trackSoAToken_(consumes(iConfig.getParameter<edm::InputTag>("trackSrc"))),
      pixelRecHitsToken_(
          consumes<SiPixelRecHitCollectionNew>(iConfig.getParameter<edm::InputTag>("pixelRecHitLegacySrc"))),
      pixelHMSToken_(consumes<HMSstorage>(iConfig.getParameter<edm::InputTag>("pixelRecHitLegacySrc"))),
      idealMagneticFieldToken_(esConsumes()),
      trackerTopologyToken_(esConsumes()),
      trackerGeometryTokenRun_(esConsumes<edm::Transition::BeginRun>()),
      minNumberOfHits_(iConfig.getParameter<int>("minNumberOfHits")),
      minQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"))),
      useOTExtension_(iConfig.getParameter<bool>("useOTExtension")),
      throwOnMissing_(iConfig.getParameter<bool>("throwOnMissing")),
      expandStubs_(iConfig.getParameter<bool>("expandStubs")),
      requireQuadsFromConsecutiveLayers_(iConfig.getParameter<bool>("requireQuadsFromConsecutiveLayers")),
      verbose_(iConfig.getUntrackedParameter<bool>("verbose")) {
  if (minQuality_ == pixelTrack::Quality::notQuality) {
    throw cms::Exception("PixelTrackConfiguration")
        << iConfig.getParameter<std::string>("minQuality") + " is not a pixelTrack::Quality";
  }
  if (minQuality_ < pixelTrack::Quality::dup) {
    throw cms::Exception("PixelTrackConfiguration")
        << iConfig.getParameter<std::string>("minQuality") + " not supported";
  }
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
  // TrackCollection refers to TrackingRechit and TrackExtra
  // collections, need to declare its production after them to work
  // around a rare race condition in framework scheduling
  produces<reco::TrackCollection>();
  produces<IndToEdm>();

  // if useOTExtension consume the OT RecHits
  if (useOTExtension_) {
    otRecHitsToken_ =
        consumes<Phase2TrackerRecHit1DCollectionNew>(iConfig.getParameter<edm::InputTag>("outerTrackerRecHitSrc"));
    otHMSToken_ = consumes<HMSstorage>(iConfig.getParameter<edm::InputTag>("outerTrackerRecHitSoAConverterSrc"));
  }

  // if expandStubs consume the OTRecHitsSoA and StubsSoA collections
  if (expandStubs_) {
    otRecHitsSoAToken_ = consumes<reco::OTRecHitsHost>(iConfig.getParameter<edm::InputTag>("otRecHitsSoASrc"));
    stubsSoAToken_ = consumes<reco::StubsHost>(iConfig.getParameter<edm::InputTag>("stubsSoASrc"));
  }
}

std::shared_ptr<DetIdMaps> PixelTrackProducerFromSoAAlpaka::globalBeginRun(const edm::Run &iRun,
                                                                           const edm::EventSetup &iSetup) const {
  // make the maps object
  auto detIdMaps = std::make_shared<DetIdMaps>();

  // if OT RecHits are used in PixelTracks, fill the detIdToOTModuleId_ map
  if (useOTExtension_) {
    // get track geometry
    const auto &trackerGeometry = &iSetup.getData(trackerGeometryTokenRun_);

    // function to check if given module is used as OT for CA
    auto isPinPSinOTBarrel = [&](DetId detId) {
      // Select only P-hits from the OT barrel
      return (trackerGeometry->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP &&
              detId.subdetId() == StripSubdetector::TOB);
    };

    // loop over all modules and fill the map detIdToOTModuleId_
    auto const &detUnits = trackerGeometry->detUnits();
    for (uint32_t otModuleId{0}; auto &detUnit : detUnits) {
      DetId detId(detUnit->geographicalId());
      // check if the module is used for OT extension
      bool isUsedOTModule = isPinPSinOTBarrel(detId);
      detIdMaps->detIdIsUsedOTModule_[detUnit->geographicalId()] = isUsedOTModule;
      if (isUsedOTModule) {
        // save the module index among the extension modules
        detIdMaps->detIdToOTModuleId_[detUnit->geographicalId()] = otModuleId;
        otModuleId++;
      }
    }
  }

  return detIdMaps;
}

void PixelTrackProducerFromSoAAlpaka::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("trackSrc", edm::InputTag("pixelTracksAlpaka"));
  desc.add<edm::InputTag>("pixelRecHitLegacySrc", edm::InputTag("siPixelRecHitsPreSplittingLegacy"));
  desc.add<edm::InputTag>("outerTrackerRecHitSrc", edm::InputTag("hltSiPhase2RecHits"));
  desc.add<edm::InputTag>("outerTrackerRecHitSoAConverterSrc", edm::InputTag("phase2OTRecHitsSoAConverter"));
  desc.add<edm::InputTag>("otRecHitsSoASrc", edm::InputTag("pixelSeedingOTRecHitsSoA"));
  desc.add<edm::InputTag>("stubsSoASrc", edm::InputTag("otStubProducer"));
  desc.add<int>("minNumberOfHits", 0);
  desc.add<std::string>("minQuality", "loose");
  desc.add<bool>("useOTExtension", false);
  desc.add<bool>("throwOnMissing", true)
      ->setComment(
          "Throw when the track SoA is absent; false makes the validation clones write empty collections for the "
          "events "
          "whose HLT paths did not run the pixel tracking");
  desc.add<bool>("expandStubs", false);

  // this option for removing tracks with exactly 4 hits is a temporary solution to reduce the fake rate in Phase-2
  // and is to be replaced by a smarter inclusive track selection in the CA directly
  desc.add<bool>("requireQuadsFromConsecutiveLayers", false);

  // Per-event hit-conservation diagnostic (OT-extra -> legacy rechit resolution). OFF by default.
  desc.addUntracked<bool>("verbose", false);

  descriptions.addWithDefaultLabel(desc);
}

void PixelTrackProducerFromSoAAlpaka::produce(edm::StreamID streamID,
                                              edm::Event &iEvent,
                                              const edm::EventSetup &iSetup) const {
  // enum class Quality : uint8_t { bad = 0, edup, dup, loose, strict, tight, highPurity };
  reco::TrackBase::TrackQuality recoQuality[] = {reco::TrackBase::undefQuality,
                                                 reco::TrackBase::undefQuality,
                                                 reco::TrackBase::discarded,
                                                 reco::TrackBase::loose,
                                                 reco::TrackBase::tight,
                                                 reco::TrackBase::tight,
                                                 reco::TrackBase::highPurity};
  assert(reco::TrackBase::highPurity == recoQuality[int(pixelTrack::Quality::highPurity)]);

#ifdef GPU_DEBUG
  std::cout << "Converting soa helix in reco tracks" << std::endl;
#endif

  // index map: trackId(in SoA) -> trackId(in legacy edm)
  auto indToEdmP = std::make_unique<IndToEdm>();
  auto &indToEdm = *indToEdmP;

  auto const &idealField = iSetup.getData(idealMagneticFieldToken_);

  // prepare container for legacy tracks
  pixeltrackfitting::TracksWithRecHits tracks;

  // get trackerTopology
  auto const &trackerTopology = iSetup.getData(trackerTopologyToken_);

  // get the maps for the detId of the OT modules
  auto const &detIdIsUsedOTModule = runCache(iEvent.getRun().index())->detIdIsUsedOTModule_;
  auto const &detIdToOTModuleId = runCache(iEvent.getRun().index())->detIdToOTModuleId_;

  // Validation clones run for every event, including those whose HLT paths did not run the
  // pixel tracking: without the track SoA they write empty collections.
  if (not throwOnMissing_ and not iEvent.getHandle(trackSoAToken_).isValid()) {
    iEvent.put(std::make_unique<TrackingRecHitCollection>());
    iEvent.put(std::make_unique<reco::TrackExtraCollection>());
    iEvent.put(std::make_unique<reco::TrackCollection>());
    iEvent.put(std::move(indToEdmP));
    return;
  }

  // get beamspot
  const auto &bsh = iEvent.get(beamSpotToken_);
  GlobalPoint bs(bsh.x0(), bsh.y0(), bsh.z0());

  // get the module's starting indices in the hit collection
  auto const &pixelHitsModuleStart = iEvent.get(pixelHMSToken_);

  // get Pixel RecHits
  auto const &pixelRecHitsDSV = iEvent.get(pixelRecHitsToken_);
  auto const &pixelRecHits = pixelRecHitsDSV.data();
  auto const nPixelHits = pixelRecHits.size();

  // get OT RecHits if needed
  size_t nOTHits = 0;
  const Phase2TrackerRecHit1DCollectionNew *otRecHitsDSV = nullptr;
  if (useOTExtension_) {
    otRecHitsDSV = &iEvent.get(otRecHitsToken_);
    nOTHits = otRecHitsDSV->dataSize();
  }

  size_t nTotalHits = nPixelHits + nOTHits;

  // get OTRecHitsSoA and StubsSoA if stub expansion is enabled
  const reco::OTRecHitsHost *otRecHitsSoAHost = nullptr;
  const reco::StubsHost *stubsSoAHost = nullptr;
  reco::OTRecHitsConstView otRecHitsSoAView;
  reco::StubsConstView stubsSoAView;
  int32_t offsetStubs = -1;

  if (expandStubs_) {
    otRecHitsSoAHost = &iEvent.get(otRecHitsSoAToken_);
    otRecHitsSoAView = otRecHitsSoAHost->const_view().otRecHits();

    stubsSoAHost = &iEvent.get(stubsSoAToken_);
    stubsSoAView = stubsSoAHost->const_view().stubs();

    // Stubs start after pixel hits in the merged collection
    offsetStubs = static_cast<int32_t>(nPixelHits);
  }

  // hitmap to go from a unique RecHit identifier to the RecHit in the legacy collection
  // (unique hit identifier is equivalent to the position of the hit in the RecHit SoA)
  std::vector<TrackingRecHit const *> hitmap;
  hitmap.resize(nTotalHits, nullptr);

  // loop over pixel RecHits to fill the hitmap
  for (auto const &pixelHit : pixelRecHits) {
    auto const &thit = static_cast<BaseTrackerRecHit const &>(pixelHit);
    auto const detI = thit.det()->index();
    auto const &clus = thit.firstClusterRef();
    assert(clus.isPixel());

    // get hit identifier as (hit offset of the module) + (hit index in this module)
    auto const idx = pixelHitsModuleStart[detI] + clus.pixelCluster().originalId();

    assert(nullptr == hitmap[idx]);
    hitmap[idx] = &pixelHit;
  }

  // if OT RecHits are used in PixelTracks, fill the hitmap also with those
  if (useOTExtension_) {
    if (expandStubs_ && otRecHitsSoAHost != nullptr) {
      // The OT hits in the SoA are organized by StackedModuleGeometry index, not by
      // detUnit->index(). Each SoA hit stores origRecHitIdx, the flat index into the legacy
      // Phase2TrackerRecHit1DCollectionNew assigned while iterating the DetSets in legacy order,
      // so each SoA hit maps straight to its legacy RecHit by that index.
      auto const &otData = otRecHitsDSV->data();
      auto otHitsView = otRecHitsSoAHost->const_view().otRecHits();
      uint32_t nOTHitsSoA = otHitsView.metadata().size();
      for (uint32_t i = 0; i < nOTHitsSoA; ++i) {
        uint32_t flatIdx = otHitsView[i].origRecHitIdx();
        assert(flatIdx < otData.size());
        hitmap[nPixelHits + i] = &otData[flatIdx];
      }
    } else {
      // Without stub expansion: OT hits organized by detUnit->index() for Ph2PSP TOB modules.
      // The RecHits in the SoA are ordered according to the detUnit->index()
      // of the respective OT module. For this reason, we need the map from the
      // detId to the moduleId among all used OT modules. This otModuleId corresponds
      // to the module's position in the otHitsModuleStart that we get from the event.

      // get the module's starting indices in the hit collection
      auto const &otHitsModuleStart = iEvent.get(otHMSToken_);

      // perform the exact same loop of how the SoA is initially filled with OT hits
      // and get the index by counting the hits (starting from the correpondign HitStartModule)
      for (auto const &detSet : *otRecHitsDSV) {
        auto detId = detSet.detId();

        // check if module is used in extension
        if (detIdIsUsedOTModule.find(detId)->second) {
          // get the corresponding otModuleId
          auto otModuleId = detIdToOTModuleId.find(detId)->second;

          // loop over the RecHits of the module and fill the hitmap
          for (int idx = otHitsModuleStart[otModuleId]; auto const &recHit : detSet) {
            assert(nullptr == hitmap[idx]);
            hitmap[idx] = &recHit;
            idx++;
          }
        }
      }
    }
  }

  // function that returns the number of skipped layers for a given pair of RecHits
  // for the case where the inner RecHit is in the pixel barrel.
  auto getNSkippedLayersInnerInBarrel = [&](const DetId &innerDetId,
                                            const DetId &outerDetId,
                                            const TrackingRecHit *innerRecHit) {
    int nSkippedLayers = 0;
    switch (outerDetId.subdetId()) {
      case PixelSubdetector::PixelBarrel:
        nSkippedLayers = trackerTopology.pxbLayer(outerDetId) - trackerTopology.pxbLayer(innerDetId) - 1;
        break;
      case PixelSubdetector::PixelEndcap:
        nSkippedLayers = trackerTopology.pxfDisk(outerDetId) - 1;  // -1 because first disk has Id 1
        break;
      case StripSubdetector::TOB:
        // if the inner RecHit is at the edge of the barrel layer, consider the jump to the first OT layer as no skip
        if (std::abs(innerRecHit->globalPosition().z()) > 17)
          nSkippedLayers = trackerTopology.getOTLayerNumber(outerDetId) - 1;  // -1 because first barrel has Id 1
        else
          nSkippedLayers = trackerTopology.getOTLayerNumber(outerDetId) + 4 - trackerTopology.pxbLayer(innerDetId) - 1;
        break;
      case StripSubdetector::TID:
        // Pixel barrel to OT endcap disk: transition region, no skipped layers
        nSkippedLayers = 0;
        break;
    }
    return nSkippedLayers;
  };

  // function that returns the number of skipped layers for a given pair of RecHits
  // for the case where the inner RecHit is in the pixel endcap.
  auto getNSkippedLayersInnerInEndcap = [&](const DetId &innerDetId, const DetId &outerDetId) {
    int nSkippedLayers = 0;
    switch (outerDetId.subdetId()) {
      case PixelSubdetector::PixelEndcap:
        nSkippedLayers = trackerTopology.pxfDisk(outerDetId) - trackerTopology.pxfDisk(innerDetId) - 1;
        break;
      case StripSubdetector::TOB:
        nSkippedLayers = trackerTopology.getOTLayerNumber(outerDetId) - 1;  // -1 because first disk has Id 1
        break;
      case StripSubdetector::TID:
        // Pixel endcap to OT endcap disk: transition region, no skipped layers
        nSkippedLayers = 0;
        break;
    }
    return nSkippedLayers;
  };

  // function that returns the number of skipped layers for a given pair of RecHits
  // for the case where the inner RecHit is in the OT (barrel or endcap).
  auto getNSkippedLayersInnerInOT = [&](const DetId &innerDetId, const DetId &outerDetId) {
    int nSkippedLayers = 0;
    if (innerDetId.subdetId() == StripSubdetector::TOB && outerDetId.subdetId() == StripSubdetector::TOB) {
      // Both in OT barrel: compute layer difference
      nSkippedLayers = trackerTopology.getOTLayerNumber(outerDetId) - trackerTopology.getOTLayerNumber(innerDetId) - 1;
    } else if (innerDetId.subdetId() == StripSubdetector::TID && outerDetId.subdetId() == StripSubdetector::TID) {
      // Both in OT endcap: compute disk difference (same side)
      int innerDisk = trackerTopology.tidWheel(innerDetId);
      int outerDisk = trackerTopology.tidWheel(outerDetId);
      nSkippedLayers = outerDisk - innerDisk - 1;
    }
    // Barrel-to-endcap or endcap-to-barrel transitions: 0 skipped layers
    return nSkippedLayers;
  };

  // function that returns the number of skipped layers for a given pair of RecHits
  // It works only for Phase-2, as this feature does not make sense for Phase-1 due to the smaller number of layers.
  // (needed for layer-skipping quadruplet rejection)
  auto getNSkippedLayers = [&](const TrackingRecHit *innerRecHit, const TrackingRecHit *outerRecHit) {
    // get detIds and subdetectors of the hits to determine their layers
    auto innerDetId = innerRecHit->geographicalId();
    auto outerDetId = outerRecHit->geographicalId();

    int nSkippedLayers = 0;

    switch (innerDetId.subdetId()) {
      case PixelSubdetector::PixelBarrel:
        nSkippedLayers = getNSkippedLayersInnerInBarrel(innerDetId, outerDetId, innerRecHit);
        break;
      case PixelSubdetector::PixelEndcap:
        nSkippedLayers = getNSkippedLayersInnerInEndcap(innerDetId, outerDetId);
        break;
      case StripSubdetector::TOB:
      case StripSubdetector::TID:
        nSkippedLayers = getNSkippedLayersInnerInOT(innerDetId, outerDetId);
        break;
    }
    return nSkippedLayers;
  };

  std::vector<const TrackingRecHit *> hits;
  hits.reserve(5);  //TODO move to a configurable parameter?

  auto const &tsoa = iEvent.get(trackSoAToken_);
  auto const quality = tsoa.view().tracks().quality();
  auto const hitOffs = tsoa.view().tracks().hitOffsets();
  // Plain column accessor for pt, used by the sort comparator below: tsoa.view().tracks()[i].pt()
  // would build a full element proxy per comparison, and that proxy's constructor builds the
  // Eigen::Map members of the layout's two Eigen columns.
  auto const trackPt = tsoa.view().tracks().pt();
  auto const hitIdxs = tsoa.view().trackHits().id();
  auto nTracks = tsoa.view().tracks().nTracks();

  tracks.reserve(nTracks);

  int32_t nt = 0;

  // sort index by pt
  std::vector<int32_t> sortIdxs(nTracks);
  std::iota(sortIdxs.begin(), sortIdxs.end(), 0);
  // sort good-quality tracks by pt, keep bad-quality tracks at the bottom
  std::sort(sortIdxs.begin(), sortIdxs.end(), [&](int32_t const i1, int32_t const i2) {
    if (quality[i1] >= minQuality_ && quality[i2] >= minQuality_)
      return trackPt[i1] > trackPt[i2];
    else
      return quality[i1] > quality[i2];
  });

  indToEdm.resize(nTracks, -1);

  // A track-hit id with caOTHitTag::kOTHitTag set is a raw OT rechit attached by the extension
  // stage; its low bits are the OT SoA row. It resolves to a legacy Phase2TrackerRecHit1D via
  // the OT portion of the hitmap (hitmap[nPixelHits + row]), populated only on the expandStubs OT
  // path. Where that map is unavailable the tagged extra is dropped, never crashing.
  const bool otTagResolvable = useOTExtension_ && expandStubs_ && otRecHitsSoAHost != nullptr;
  // Per-event diagnostic tallies of the tagged-OT-extra branch (one-shot print below).
  uint32_t nOTExtrasResolved = 0, nOTExtrasDropped = 0;

  // loop over (sorted) tracks
  for (const auto &it : sortIdxs) {
    auto nHits = reco::nHits(tsoa.view().tracks(), it);
    assert(nHits >= 3);
    auto q = quality[it];

    // apply cuts on quality and number of hits
    if (q < minQuality_)
      // since the tracks are sorted according to quality,
      // we can break after the first track with low quality
      break;
    if (nHits < minNumberOfHits_)  //move to nLayers?
      continue;

    auto start = (it == 0) ? 0 : hitOffs[it - 1];
    auto end = hitOffs[it];
    int nRemovedHits{0};
    int nExpandedHits{0};

    // First pass: count how many hits we'll have after stub expansion
    if (expandStubs_ && stubsSoAHost != nullptr && offsetStubs >= 0) {
      for (auto iHit = start; iHit < end; ++iHit) {
        auto hitIdx = hitIdxs[iHit];
        if (caOTHitTag::isOTId(hitIdx)) {
          // Tagged raw-OT extra: one legacy rechit, no stub expansion. Neither expanded nor removed
          // when resolvable, dropped otherwise, matching the fill pass.
          const uint32_t o = caOTHitTag::otIdx(hitIdx);
          if (!(otTagResolvable && (nPixelHits + o) < nTotalHits))
            nRemovedHits++;
          continue;
        }
        if (hitIdx < nTotalHits) {
          if (hitIdx >= static_cast<uint32_t>(offsetStubs)) {
            uint32_t stubIdx = hitIdx - offsetStubs;
            if (isStub(stubsSoAView, stubIdx)) {
              nExpandedHits++;  // Regular stub expands to 2 hits, so we add 1 more
            }
            // PHitOnly stubs have only 1 hit (inner), so no expansion needed
          }
        } else {
          nRemovedHits++;
        }
      }
    } else {
      for (auto iHit = start; iHit < end; ++iHit) {
        auto hitIdx = hitIdxs[iHit];
        if (caOTHitTag::isOTId(hitIdx)) {
          // Tagged OT extra: resolvable only on the expandStubs OT path (false here) -> dropped.
          const uint32_t o = caOTHitTag::otIdx(hitIdx);
          if (!(otTagResolvable && (nPixelHits + o) < nTotalHits))
            nRemovedHits++;
          continue;
        }
        if (hitIdx >= nTotalHits) {
          nRemovedHits++;
        }
      }
    }

    hits.resize(nHits - nRemovedHits + nExpandedHits);

    int hitOutputIdx = 0;
    for (auto iHit = start; iHit < end; ++iHit) {
      auto hitIdx = hitIdxs[iHit];
      if (caOTHitTag::isOTId(hitIdx)) {
        // Tagged raw-OT extra -> its legacy Phase2TrackerRecHit1D via the OT hitmap (same lookup as
        // a stub's lower/upper sensor hit: hitmap[nPixelHits + otSoARow]). Unresolvable tagged ids
        // are dropped (counted as removed above), keeping the hits vector correctly sized.
        const uint32_t o = caOTHitTag::otIdx(hitIdx);
        if (otTagResolvable && (nPixelHits + o) < nTotalHits) {
          hits[hitOutputIdx++] = hitmap[nPixelHits + o];
          ++nOTExtrasResolved;
        } else {
          ++nOTExtrasDropped;
        }
        continue;
      }
      if (hitIdx < nTotalHits) {
        if (expandStubs_ && stubsSoAHost != nullptr && offsetStubs >= 0 &&
            hitIdx >= static_cast<uint32_t>(offsetStubs)) {
          uint32_t stubIdx = hitIdx - offsetStubs;
          uint32_t lowerHitIdx = stubsSoAView[stubIdx].lowerHitIdx();

          hits[hitOutputIdx++] = hitmap[nPixelHits + lowerHitIdx];

          // Add outer sensor hit only if not PHitOnly (PHitOnly stubs have invalid upperHitIdx)
          if (isStub(stubsSoAView, stubIdx)) {
            uint32_t upperHitIdx = stubsSoAView[stubIdx].upperHitIdx();
            hits[hitOutputIdx++] = hitmap[nPixelHits + upperHitIdx];
          }
        } else {
          hits[hitOutputIdx++] = hitmap[hitIdx];
        }
      }
      // else: removed hits are skipped
    }

    end = end - nRemovedHits;

    // implement custome requirement for quadruplets coming from consecutive layers
    if (requireQuadsFromConsecutiveLayers_ && (nHits == 4)) {
      bool skipThisTrack{false};
      // loop over layer pairs and check if they skip
      for (auto iHit = start; iHit < end - 1; ++iHit) {
        // if the inner (iHit-start) to outer (iHit-start+1) hit layer-change skips 1 or more
        // layers skipt the track
        if (getNSkippedLayers(hits[iHit - start], hits[iHit - start + 1]) > 0) {
          skipThisTrack = true;
          break;
        }
      }
      if (skipThisTrack) {
        indToEdm[it] = pixelTrack::skippedTrack;  // mark as skipped
        continue;
      }
    }

#ifdef CA_DEBUG
    std::cout << "track soa : " << it << " with hits: ";
    for (auto iHit = start; iHit < end; ++iHit)
      std::cout << hitIdxs[iHit] << " - ";
    std::cout << std::endl;
#endif

    // store the index of the SoA:
    // indToEdm[index_SoAtrack] -> index_edmTrack (if it exists)
    indToEdm[it] = nt;
    ++nt;

    // mind: this values are respect the beamspot!
    float chi2 = tsoa.view().tracks()[it].chi2();
    float phi = reco::phi(tsoa.view().tracks(), it);

    riemannFit::Vector5d ipar, opar;
    riemannFit::Matrix5d icov, ocov;
    reco::copyToDense<riemannFit::Vector5d, riemannFit::Matrix5d>(tsoa.view().tracks(), ipar, icov, it);
    riemannFit::transformToPerigeePlane(ipar, icov, opar, ocov);

    LocalTrajectoryParameters lpar(opar(0), opar(1), opar(2), opar(3), opar(4), 1.);
    AlgebraicSymMatrix55 m;
    for (int i = 0; i < 5; ++i)
      for (int j = i; j < 5; ++j)
        m(i, j) = ocov(i, j);

    float sp = std::sin(phi);
    float cp = std::cos(phi);
    Surface::RotationType rot(sp, -cp, 0, 0, 0, -1.f, cp, sp, 0);

    Plane impPointPlane(bs, rot);
    GlobalTrajectoryParameters gp(
        impPointPlane.toGlobal(lpar.position()), impPointPlane.toGlobal(lpar.momentum()), lpar.charge(), &idealField);
    JacobianLocalToCurvilinear jl2c(impPointPlane, lpar, idealField);

    AlgebraicSymMatrix55 mo = ROOT::Math::Similarity(jl2c.jacobian(), m);

    // ndof follows the hit-count convention 2 * nhits - 5 on every path that does not expand stubs.
    // A stub-expanded track carries both outer-tracker rechits of every stub in `hits` while the fit
    // used one position per stub, so there ndof counts the positions the fit used.
    int ndof = 2 * int(hits.size()) - 5;
    if (expandStubs_) {
      constexpr int maxHitsOnTrackForFullFit = 6;  // fallback only, see below
      ndof = 2 * std::min(nHits, maxHitsOnTrackForFullFit) - 5;
      // The fit kernel stamps the degrees of freedom of the positions it fitted (2N-5 for the
      // N <= maxHitsOnTrackForFullFit positions used) into the SoA; prefer it.
      const int ndofSoA = tsoa.view().tracks()[it].ndof();
      if (ndofSoA > 0)
        ndof = ndofSoA;
    }
    chi2 = chi2 * ndof;
    GlobalPoint vv = gp.position();
    math::XYZPoint pos(vv.x(), vv.y(), vv.z());
    GlobalVector pp = gp.momentum();
    math::XYZVector mom(pp.x(), pp.y(), pp.z());

    // A device fit that failed numerically can leave finite SoA parameters that still map to a
    // non-finite reco trajectory (through transformToPerigeePlane / the local->global transform).
    // Never emit such a track: drop it and free the edm index reserved above.
    if (not(std::isfinite(chi2) and std::isfinite(mom.x()) and std::isfinite(mom.y()) and std::isfinite(mom.z()))) {
      indToEdm[it] = pixelTrack::skippedTrack;
      --nt;
      continue;
    }

    auto track = std::make_unique<reco::Track>(chi2, ndof, pos, mom, gp.charge(), CurvilinearTrajectoryError(mo));

    // bad and edup not supported as fit not present or not reliable
    auto tkq = recoQuality[int(q)];
    track->setQuality(tkq);
    // loose,tight and HP are inclusive
    if (reco::TrackBase::highPurity == tkq) {
      track->setQuality(reco::TrackBase::tight);
      track->setQuality(reco::TrackBase::loose);
    } else if (reco::TrackBase::tight == tkq) {
      track->setQuality(reco::TrackBase::loose);
    }
    track->setQuality(tkq);
    // filter???
    tracks.emplace_back(track.release(), hits);
  }

#ifdef GPU_DEBUG
  std::cout << "processed " << nt << " good tuples " << tracks.size() << " out of " << indToEdm.size() << std::endl;
#endif

  // Diagnostic, printed only for events with tagged OT extras: "dropped" counts unresolvable tags
  // (no OT hitmap or out-of-range row). MessageLogger rate-limits per category, and produce() is
  // const (edm::global::EDProducer), so no local counter is possible anyway.
  if (verbose_ && nOTExtrasResolved + nOTExtrasDropped > 0)
    edm::LogInfo("PixelTrackProducerFromSoAAlpaka")
        << "tagged OT extras -> legacy hits: resolved=" << nOTExtrasResolved << " dropped=" << nOTExtrasDropped;

  // store tracks
  storeTracks(iEvent, tracks, trackerTopology);
  iEvent.put(std::move(indToEdmP));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PixelTrackProducerFromSoAAlpaka);
