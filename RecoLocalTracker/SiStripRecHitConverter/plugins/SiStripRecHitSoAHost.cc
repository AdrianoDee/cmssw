
#include <cuda_runtime.h>

#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"
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

#include "gpuPixelRecHits.h"

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
      pixeModuleStartToken_{consumes(iConfig.getParameter<edm::InputTag>("pixelModuleStartSource"))}
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
  const TrackerGeometry* trackerGeometry = &es.getData(geomToken_);

  edm::Handle<SiStripMatchedRecHit2DCollection> stripRecHitHandle;
  iEvent.getByToken(recHitToken_, stripRecHitHandle);


  edm::Handle<HitsOnHost> pixelRecHitSoAHandle;
  iEvent.getByToken(pixelRecHitSoAToken_, pixelRecHitSoAHandle);

  size_t nStripHits = 0
  for (const auto& detSet : *stripRecHitHandle) 
    if (trackerGeometry->idToDet(detSet.subdetId())->stereoDet()->index() < TrackerTraits::numberOfModules)
        nStripHits += detSet.size();

  TrackingRecHitSoAHost<TrackerTraits> result(pixelRecHitSoAHandle->)


}

// using SiPixelRecHitSoAFromLegacyPhase1 = SiStripRecHitSoAHost<pixelTopology::Phase1>;
// DEFINE_FWK_MODULE(SiPixelRecHitSoAFromLegacyPhase1);

// using SiPixelRecHitSoAFromLegacyPhase2 = SiStripRecHitSoAHost<pixelTopology::Phase2>;
// DEFINE_FWK_MODULE(SiPixelRecHitSoAFromLegacyPhase2);
