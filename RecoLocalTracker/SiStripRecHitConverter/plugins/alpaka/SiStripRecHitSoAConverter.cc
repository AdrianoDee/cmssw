
// #include <cstdint>
// #include <memory>
// #include <vector>
// #include "DataFormats/BeamSpot/interface/BeamSpot.h"
// #include "DataFormats/GeometryVector/interface/GlobalPoint.h"
// #include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
// #include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
// #include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
// #include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// #include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
// #include "FWCore/ParameterSet/interface/ParameterSet.h"
// #include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
// #include "FWCore/Utilities/interface/InputTag.h"

// // #include "DataFormats/BeamSpot/interface/BeamSpot.h"
// #include "DataFormats/Common/interface/DetSetVectorNew.h"
// #include "DataFormats/Common/interface/Handle.h"
// // #include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
// // #include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
// #include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
// #include "DataFormats/Math/interface/approx_atan2.h"

// #include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
// #include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
// // #include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
// // #include "Geometry/CommonTopologies/interface/GluedGeomDet.h"

// #include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"

// #include "DataFormats/TrackerCommon/interface/TrackerTopology.h"


// namespace ALPAKA_ACCELERATOR_NAMESPACE {

// class SiStripRecHitSoA : public global::EDProducer<> {

//   using Hits = ::reco::TrackingRecHitsSoACollection;
//   using HitsHost = ::reco::TrackingRecHitHost;
// public:
//   explicit SiStripRecHitSoA(const edm::ParameterSet& iConfig);
//   ~SiStripRecHitSoA() override = default;

//   static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

// private:
//   void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override {};
//   void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;

//   const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
//   const edm::EDGetTokenT<SiStripMatchedRecHit2DCollection> recHitToken_;
//   const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;

//   const device::EDPutToken<Hits> stripSoA_;
// };


// SiStripRecHitSoA::SiStripRecHitSoA(const edm::ParameterSet& iConfig)
//     : geomToken_(esConsumes()),
//       // topologyToken_(esConsumes()),
//       recHitToken_{consumes(iConfig.getParameter<edm::InputTag>("stripRecHitSource"))},
//       //beamSpotToken(consumes(edm::InputTag("offlineBeamSpot"))),
//       beamSpotToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
//       stripSoA_{produces()}
// {
  
// }


// void SiStripRecHitSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
//   edm::ParameterSetDescription desc;

//   desc.add<edm::InputTag>("stripRecHitSource", edm::InputTag("siStripMatchedRecHits", "matchedRecHit"));
//   desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
//   descriptions.addWithDefaultLabel(desc);

// }

// //https://github.com/cms-sw/cmssw/blob/3f06ef32d66bd2a7fa04e411fa4db4845193bd3c/RecoTracker/MkFit/plugins/convertHits.h

// void SiStripRecHitSoA::produce(device::Event& iEvent, device::EventSetup const& iSetup) {

//   // const auto &trackerTopology = &iSetup.getData(topologyToken_);
//   const auto &trackerGeometry = &iSetup.getData(geomToken_);

//   auto const& stripHits = iEvent.get(recHitToken_);
//   auto& bs = iEvent.get(beamSpotToken_);

//   // Count strip hits
//   size_t nStripHits = 0;
//   //std::cout << "number of modules: " << TrackerTraits::numberOfModules << std::endl;
//   //std::cout << "stripHits size: " << stripHits.size() << std::endl;
//   for (const auto& detSet : stripHits) 
//     nStripHits += detSet.size();

//   std::cout << "nStripHits = " << nStripHits << std::endl;

//   HitsHost stripHitsHost(iEvent.queue(), nStripHits, detSet.size() + 1);

//   auto& hitModuleView = stripHitsHost.view<HitModuleSoA>()
//   for(int i = 0; i < hitModuleView.metadata().size(); i++)
//     hitModuleView[i].moduleStart() = 0;
//   // Loop over strip RecHits
//   auto moduleCount = 0u;
  
//   for (const auto& detSet : stripHits) {
    
//     auto index = trackerGeometry->idToDet(detSet.detId());
//     moduleCount++;
//     hitModuleView[moduleCount].moduleStart() = detSet.size() + hitModuleView[moduleCount-1].moduleStart();

//     for (const auto& recHit : detSet) {
//       stripHitsHost.view()[nPixelHits + i].xLocal() = recHit.localPosition().x();
//       stripHitsHost.view()[nPixelHits + i].yLocal() = recHit.localPosition().y();
//       stripHitsHost.view()[nPixelHits + i].xerrLocal() = recHit.localPositionError().xx();
//       stripHitsHost.view()[nPixelHits + i].yerrLocal() = recHit.localPositionError().yy();
//       auto globalPosition = det->toGlobal(recHit.localPosition());
//       double gx = globalPosition.x() - bs.x0();
//       double gy = globalPosition.y() - bs.y0();
//       double gz = globalPosition.z() - bs.z0();
//       stripHitsHost.view()[nPixelHits + i].xGlobal() = gx;
//       stripHitsHost.view()[nPixelHits + i].yGlobal() = gy;
//       stripHitsHost.view()[nPixelHits + i].zGlobal() = gz;
//       stripHitsHost.view()[nPixelHits + i].rGlobal() = sqrt(gx * gx + gy * gy);
//       stripHitsHost.view()[nPixelHits + i].iphi() = unsafe_atan2s<7>(gy, gx);
//       // stripHitsHost.view()[nPixelHits + i].chargeAndStatus().charge = ?
//       // stripHitsHost.view()[nPixelHits + i].chargeAndStatus().status = ?
//       // stripHitsHost.view()[nPixelHits + i].clusterSizeX() = ?
//       // stripHitsHost.view()[nPixelHits + i].clusterSizeY() = ?
//       stripHitsHost.view()[nPixelHits + i].detectorIndex() = index;
//       // ???
//     }
//   }

//   assert(hitModuleView[hitModuleView.metadata().size()] == nStripHits);

//   iEvent.emplace(stripSoA_, std::move(stripHitsHost));
  
  
// }

// }

// #include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
// DEFINE_FWK_ALPAKA_MODULE(SiStripRecHitSoAConverter);
