
// #include <cuda_runtime.h>

// #include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoAHost.h"
#include "CUDADataFormats/Common/interface/PortableHostCollection.h"
#include "CUDADataFormats/Common/interface/HostProduct.h"
// #include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/CommonTopologies/interface/GluedGeomDet.h"

template <typename TrackerTraits>
class SiStripRecHitSoAHost : public edm::global::EDProducer<> {
public:
  explicit SiStripRecHitSoAHost(const edm::ParameterSet& iConfig);
  ~SiStripRecHitSoAHost() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  using HitModuleStart = std::array<uint32_t, TrackerTraits::numberOfModules + 1>;
  using HMSstorage = HostProduct<uint32_t[]>;
  using HitsOnHost = TrackingRecHitSoAHost<TrackerTraits>;

private:
  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
//   const edm::EDGetTokenT<reco::BeamSpot> bsGetToken_;
  const edm::EDGetTokenT<SiStripMatchedRecHit2DCollection> recHitToken_;
  const edm::EDGetTokenT<HitsOnHost> pixelRecHitSoAToken_;
  const edm::EDGetTokenT<HMSstorage> pixelModuleStartToken_;
  const edm::EDPutTokenT<HitsOnHost> tokenHit_;
  const edm::EDPutTokenT<HMSstorage> tokenModuleStart_;
};

template <typename TrackerTraits>
SiStripRecHitSoAHost<TrackerTraits>::SiStripRecHitSoAHost(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes()),
    //   bsGetToken_{consumes(iConfig.getParameter<edm::InputTag>("beamSpot"))},
      recHitToken_{consumes(iConfig.getParameter<edm::InputTag>("stripRecHitSource"))},
      pixelRecHitSoAToken_{consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSoASource"))},
      pixelModuleStartToken_{consumes(iConfig.getParameter<edm::InputTag>("pixelModuleStartSource"))},
      tokenHit_{produces()},
      tokenModuleStart_{produces()} {
}

template <typename TrackerTraits>
void SiStripRecHitSoAHost<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

//   desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("src", edm::InputTag("matchedRecHit"));

  descriptions.addWithDefaultLabel(desc);
}

template <typename TrackerTraits>
void SiStripRecHitSoAHost<TrackerTraits>::produce(edm::StreamID streamID,
                                                         edm::Event& iEvent,
                                                         const edm::EventSetup& es) const {
  
  // Get the objects that we need
  const TrackerGeometry* trackerGeometry = &es.getData(geomToken_);

  edm::Handle<SiStripMatchedRecHit2DCollection> stripRecHitHandle;
  iEvent.getByToken(recHitToken_, stripRecHitHandle);

  edm::Handle<HitsOnHost> pixelRecHitSoAHandle;
  iEvent.getByToken(pixelRecHitSoAToken_, pixelRecHitSoAHandle);

  // Count strip hits
  size_t nStripHits = 0;
  for (const auto& detSet : *stripRecHitHandle) {
    const GluedGeomDet* det = static_cast<const GluedGeomDet*>(trackerGeometry->idToDet(detSet.detId()));
    if (det->stereoDet()->index() < TrackerTraits::numberOfModules)
        nStripHits += detSet.size();
  }

  size_t nPixelHits = pixelRecHitSoAHandle->view().nHits();

  // Create output collection with the right size
  TrackingRecHitSoAHost<TrackerTraits> result(
    nPixelHits + nStripHits, 
    pixelRecHitSoAHandle->view().offsetBPIX2(),
    &pixelRecHitSoAHandle->view().cpeParams(),
    pixelRecHitSoAHandle->view().hitsModuleStart().begin()
  );

  // Copy pixel data
  std::copy(pixelRecHitSoAHandle->view().xLocal(), pixelRecHitSoAHandle->view().xLocal() + nPixelHits, result.view().xLocal());
  std::copy(pixelRecHitSoAHandle->view().yLocal(), pixelRecHitSoAHandle->view().yLocal() + nPixelHits, result.view().yLocal());
  std::copy(pixelRecHitSoAHandle->view().xerrLocal(), pixelRecHitSoAHandle->view().xerrLocal() + nPixelHits, result.view().xerrLocal());
  std::copy(pixelRecHitSoAHandle->view().yerrLocal(), pixelRecHitSoAHandle->view().yerrLocal() + nPixelHits, result.view().yerrLocal());
  std::copy(pixelRecHitSoAHandle->view().xGlobal(), pixelRecHitSoAHandle->view().xGlobal() + nPixelHits, result.view().xGlobal());
  std::copy(pixelRecHitSoAHandle->view().yGlobal(), pixelRecHitSoAHandle->view().yGlobal() + nPixelHits, result.view().yGlobal());
  std::copy(pixelRecHitSoAHandle->view().zGlobal(), pixelRecHitSoAHandle->view().zGlobal() + nPixelHits, result.view().zGlobal());
  std::copy(pixelRecHitSoAHandle->view().rGlobal(), pixelRecHitSoAHandle->view().rGlobal() + nPixelHits, result.view().rGlobal());
  std::copy(pixelRecHitSoAHandle->view().iphi(), pixelRecHitSoAHandle->view().iphi() + nPixelHits, result.view().iphi());
  std::copy(pixelRecHitSoAHandle->view().chargeAndStatus(), pixelRecHitSoAHandle->view().chargeAndStatus() + nPixelHits, result.view().chargeAndStatus());
  std::copy(pixelRecHitSoAHandle->view().clusterSizeX(), pixelRecHitSoAHandle->view().clusterSizeX() + nPixelHits, result.view().clusterSizeX());
  std::copy(pixelRecHitSoAHandle->view().clusterSizeY(), pixelRecHitSoAHandle->view().clusterSizeY() + nPixelHits, result.view().clusterSizeY());
  std::copy(pixelRecHitSoAHandle->view().detectorIndex(), pixelRecHitSoAHandle->view().detectorIndex() + nPixelHits, result.view().detectorIndex());
  
  // result.view().phiBinnerStorage() = pixelRecHitSoAHandle->view().phiBinnerStorage();
  result.view().hitsModuleStart() = pixelRecHitSoAHandle->view().hitsModuleStart();
  result.view().hitsLayerStart() = pixelRecHitSoAHandle->view().hitsLayerStart();
  result.view().cpeParams() = pixelRecHitSoAHandle->view().cpeParams();
  result.view().averageGeometry() = pixelRecHitSoAHandle->view().averageGeometry();
  result.view().phiBinner() = pixelRecHitSoAHandle->view().phiBinner();

  // Loop over strip RecHits
  size_t i = 0;
  for (const auto& detSet : *stripRecHitHandle) {
    const GluedGeomDet* det = static_cast<const GluedGeomDet*>(trackerGeometry->idToDet(detSet.detId()));
    if (det->stereoDet()->index() < TrackerTraits::numberOfModules)
      for (const auto& recHit : detSet) {
        result.view()[nPixelHits + i].xLocal() = recHit.localPosition().x();
        result.view()[nPixelHits + i].yLocal() = recHit.localPosition().y();
        result.view()[nPixelHits + i].xerrLocal() = recHit.localPositionError().xx();
        result.view()[nPixelHits + i].yerrLocal() = recHit.localPositionError().yy();
        auto globalPosition = det->toGlobal(recHit.localPosition());
        result.view()[nPixelHits + i].xGlobal() = globalPosition.x();
        result.view()[nPixelHits + i].yGlobal() = globalPosition.y();
        result.view()[nPixelHits + i].zGlobal() = globalPosition.z();
        result.view()[nPixelHits + i].rGlobal() = globalPosition.transverse();
        // result.view()[nPixelHits + i].clusterSizeX() = recHit.monoClusterRef().cluster_strip();
        // result.view()[nPixelHits + i].clusterSizeY() = recHit.monoClusterRef().cluster_strip();
        result.view()[nPixelHits + i].detectorIndex() = det->stereoDet()->index();
        // ???
        ++i;
      }
  }

  iEvent.emplace(tokenHit_, std::move(result));

}

using SiStripRecHitSoAHostPhase1 = SiStripRecHitSoAHost<pixelTopology::Phase1>;
DEFINE_FWK_MODULE(SiStripRecHitSoAHostPhase1);

// using SiPixelRecHitSoAFromLegacyPhase2 = SiStripRecHitSoAHost<pixelTopology::Phase2>;
// DEFINE_FWK_MODULE(SiPixelRecHitSoAFromLegacyPhase2);
