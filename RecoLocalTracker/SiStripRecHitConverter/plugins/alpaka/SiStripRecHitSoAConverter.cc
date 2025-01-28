
#include <cstdint>
#include <memory>
#include <vector>
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

// #include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
// #include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
// #include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/Math/interface/approx_atan2.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
// #include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
// #include "Geometry/CommonTopologies/interface/GluedGeomDet.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/CAGeometrySoA.h"
#include "RecoTracker/Record/interface/CAGeometryHost.h"

#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

class SiStripRecHitSoAConverter : public stream::EDProducer<> {

  using Hits = reco::TrackingRecHitsSoACollection;
  using HitsHost = ::reco::TrackingRecHitHost;
  using HMSstorage = typename std::vector<uint32_t>;
  
public:
  explicit SiStripRecHitSoAConverter(const edm::ParameterSet& iConfig);
  ~SiStripRecHitSoAConverter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(device::Event& iEvent, const device::EventSetup& es) override;

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<::reco::CAGeometryHost, TrackerRecoGeometryRecord> geometrySoA_;  
  const edm::EDGetTokenT<SiStripMatchedRecHit2DCollection> recHitToken_;
  const edm::EDGetTokenT<::reco::BeamSpot> beamSpotToken_;
  const edm::EDGetTokenT<HitsHost> pixelHitsSoA_;

  // const edm::EDPutTokenT<HitsHost> stripSoAHost_;
  const device::EDPutToken<Hits> stripSoADevice_;
  const edm::EDPutTokenT<HMSstorage> hitModuleStart_;
};


SiStripRecHitSoAConverter::SiStripRecHitSoAConverter(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes()),
      geometrySoA_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("caGeometry")))),
      recHitToken_{consumes(iConfig.getParameter<edm::InputTag>("stripRecHitSource"))},
      beamSpotToken_(consumes<::reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      pixelHitsSoA_{consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSoASource"))},
      // stripSoAHost_{produces()},
      stripSoADevice_{produces()},
      hitModuleStart_{produces()}
{
  
}


void SiStripRecHitSoAConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("pixelRecHitSoASource", edm::InputTag("siPixelRecHitsPreSplittingAlpaka"));
  desc.add<edm::InputTag>("stripRecHitSource", edm::InputTag("siStripMatchedRecHits", "matchedRecHit"));
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<std::string>("caGeometry", std::string("caGeometry"));

  descriptions.addWithDefaultLabel(desc);

}

//https://github.com/cms-sw/cmssw/blob/3f06ef32d66bd2a7fa04e411fa4db4845193bd3c/RecoTracker/MkFit/plugins/convertHits.h

void SiStripRecHitSoAConverter::produce(device::Event& iEvent, device::EventSetup const& iSetup) {

  auto& bs = iEvent.get(beamSpotToken_);

  auto queue = iEvent.queue();

  const auto &trackerGeometry = &iSetup.getData(geomToken_);
  auto const& geometrySoA = iSetup.getData(geometrySoA_);
  int nModules = geometrySoA.view<::reco::CAModulesSoA>().metadata().size();

  auto const& pixelHitsHost = iEvent.get(pixelHitsSoA_);
  int nPixelHits = pixelHitsHost.view().metadata().size();
  auto& pixelHitsModuleView = pixelHitsHost.view<::reco::HitModuleSoA>();
  int nPixelModules = pixelHitsModuleView.metadata().size() - 1;
  auto pixelBufferExtent = alpaka::getExtentProduct(pixelHitsHost.buffer());
  
  auto const& stripHits = iEvent.get(recHitToken_);

  // Count strip hits
  const int nStripHits = stripHits.data().size();
  const int activeStripModules = stripHits.size();

  std::cout << "nStripHits = " << nStripHits << std::endl;
  assert(nModules >= nPixelModules);
  const int nStripModules = nModules - nPixelModules;
  HitsHost stripHitsHost(queue, nStripHits + nPixelHits, nModules + 1);
  alpaka::memcpy(queue, stripHitsHost.buffer(), pixelHitsHost.buffer(), pixelBufferExtent);
  alpaka::wait(queue);
  std::cout << "Tot number of modules           : " << nModules << std::endl;
  std::cout << "Number of pixel modules         : " << nPixelModules << std::endl;
  std::cout << "Number of strip (active) modules: " << activeStripModules << std::endl;
  std::cout << "Number of strip modules: " << nStripModules << std::endl;
  std::cout << "Number of pixel hits: " << nPixelHits << std::endl;
  std::cout << "Number of strip hits: " << nStripHits << std::endl;

  auto& stripHitsModuleView = stripHitsHost.view<::reco::HitModuleSoA>();
  

  // ugly but let's do like this for the moment
  for(int i = 0; i <= nPixelModules; i++)
    stripHitsModuleView[i].moduleStart() = pixelHitsModuleView[i].moduleStart();

  // hitModuleStart[nPixelModules] == number of pixel hits
  std::cout << stripHitsModuleView[0].moduleStart() << " - " << nPixelHits << std::endl;
  std::cout << stripHitsModuleView[nPixelModules-1].moduleStart() << " - " << nPixelHits << std::endl;
  std::cout << stripHitsModuleView[nPixelModules].moduleStart() << " - " << nPixelHits << std::endl;
  assert(int(stripHitsModuleView[nPixelModules].moduleStart()) == nPixelHits);
 
  // this most probably is redundant because by default they should be zero-ed
  for(int i = nPixelModules + 1; i < stripHitsModuleView.metadata().size(); i++)
    stripHitsModuleView[i].moduleStart() = 0;

  uint32_t n_modules = nPixelModules;
  uint32_t n_hits = nPixelHits;
  
  auto const& detUnits = trackerGeometry->detUnits();
  std::map<uint32_t,uint16_t> detIdToIndex;
  for (auto& detUnit : detUnits)
  {
    detIdToIndex[detUnit->geographicalId()] = detUnit->index();
    std::cout << detUnit->geographicalId() << " - " << detUnit->index() << std::endl;
  }

  std::vector<int> stripDetSizes(nStripModules,0);
  for (const auto& detSet : stripHits) {
    
    // if (detSet.size() < 1)
    //   continue;

    // const StripGeomDetUnit* det = dynamic_cast<const StripGeomDetUnit*> (trackerGeometry->idToDet(detSet.begin()->geographicalId()));
    auto firstHit = detSet.begin();
    auto geo = firstHit->rawId();
    auto det = trackerGeometry->idToDet(geo);
    auto hitDetUnit = firstHit->detUnit();
    auto hitDet = firstHit->det()->geographicalId();
    auto index = detIdToIndex[geo];
    n_modules++;
    // stripDetSizes[index-nPixelModules] 
    std::cout << geo << " - " << detSet.detId() << " - " << hitDet << " - " << index << " - " << nPixelModules << " - " << detSet.size() << std::endl;
    std::cout << hitDetUnit->index() << " - " << hitDetUnit->geographicalId() << std::endl;
    stripHitsModuleView[index-nPixelModules].moduleStart() = n_hits; //detSet.size();
    // stripHitsModuleView[index].moduleStart() =  + stripHitsModuleView[n_modules-1].moduleStart();
    
    for (const auto& recHit : detSet) {

      stripHitsHost.view()[n_hits].xLocal() = recHit.localPosition().x();
      stripHitsHost.view()[n_hits].yLocal() = recHit.localPosition().y();
      stripHitsHost.view()[n_hits].xerrLocal() = recHit.localPositionError().xx();
      stripHitsHost.view()[n_hits].yerrLocal() = recHit.localPositionError().yy();
      auto globalPosition = det->toGlobal(recHit.localPosition());
      double gx = globalPosition.x() - bs.x0();
      double gy = globalPosition.y() - bs.y0();
      double gz = globalPosition.z() - bs.z0();
      std::cout << gx << std::endl;
      stripHitsHost.view()[n_hits].xGlobal() = gx;
      stripHitsHost.view()[n_hits].yGlobal() = gy;
      stripHitsHost.view()[n_hits].zGlobal() = gz;
      stripHitsHost.view()[n_hits].rGlobal() = sqrt(gx * gx + gy * gy);
      stripHitsHost.view()[n_hits].iphi() = unsafe_atan2s<7>(gy, gx);
      // stripHitsHost.view()[n_hits].chargeAndStatus().charge = ?
      // stripHitsHost.view()[n_hits].chargeAndStatus().status = ?
      // stripHitsHost.view()[n_hits].clusterSizeX() = ?
      // stripHitsHost.view()[n_hits].clusterSizeY() = ?
      stripHitsHost.view()[n_hits].detectorIndex() = index;
      // ???
      n_hits++;
    }

    std::cout << index-nPixelModules << " - >" << n_hits;
  }

  // // this most probably is redundant because by default they should be zero-ed
  // for(int i = nPixelModules + 1; i < stripHitsModuleView.metadata().size(); i++)
  //   stripHitsModuleView[i].moduleStart() = stripHitsModuleView[i-1].moduleStart() + stripDetSizes[i-nPixelModules];
  
  std::cout << stripHitsModuleView[nPixelModules].moduleStart() << " - >" << nPixelHits << std::endl;
  std::cout << stripHitsModuleView[stripHitsModuleView.metadata().size()-1].moduleStart() << " - >" << nStripHits << std::endl;
  assert(stripHitsModuleView[stripHitsModuleView.metadata().size()-1].moduleStart() == n_hits);
  
  auto moduleStartView = cms::alpakatools::make_host_view<uint32_t>(stripHitsModuleView.moduleStart(), stripHitsModuleView.metadata().size());
  HMSstorage moduleStartVec(stripHitsModuleView.metadata().size());

  // Put in the event the hit module start vector.
  // Now, this could  be avoided having the Host Hit SoA 
  // consumed by the downstream module (converters to legacy formats).
  // But this is the common practice at the moment
  // also for legacy data formats.
  alpaka::memcpy(queue, moduleStartVec, moduleStartView);
  iEvent.emplace(hitModuleStart_, std::move(moduleStartVec));

  Hits stripHitsDevice(queue, stripHitsHost.view().metadata().size(), stripHitsModuleView.metadata().size()); 
  alpaka::memcpy(queue, stripHitsDevice.buffer(), stripHitsHost.buffer());
  stripHitsDevice.updateFromDevice(queue);

  // Would be useful to have a way to prompt a special CopyToDevice for EDProducers
  // iEvent.emplace(stripSoAHost_, std::move(stripHitsHost));
  iEvent.emplace(stripSoADevice_, std::move(stripHitsDevice));
  
}

}

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(SiStripRecHitSoAConverter);
