#include <cuda_runtime.h>

#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"
#include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "CUDADataFormats/Common/interface/HostProduct.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
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
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFast.h"

#include "gpuPixelRecHits.h"

class SiPixelRecHitSoAProducer : public edm::global::EDProducer<> {
public:
  explicit SiPixelRecHitSoAProducer(const edm::ParameterSet& iConfig);
  ~SiPixelRecHitSoAProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  using HitModuleStart = std::array<uint32_t, gpuClustering::maxNumModules + 1>;
  using HMSstorage = HostProduct<uint32_t[]>;

private:
  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> cpeToken_;
  const edm::EDGetTokenT<reco::BeamSpot> bsGetToken_;
  const edm::EDGetTokenT<SiPixelClusterCollectionNew> clusterToken_;  // Legacy Clusters
  const edm::EDGetTokenT<SiPixelRecHitCollectionNew> hitToken_;
// const edm::EDPutTokenT<TrackingRecHit2DGPU> tokenHitGPU_;
  const edm::EDPutTokenT<HMSstorage> tokenModuleStart_;
  const edm::EDPutTokenT<TrackingRecHit2DCPU> tokenHitCPU_;
  const bool onGPU_;
  const bool isPhase2_;

  static constexpr uint32_t n16 = 4;                 // number of elements in m_store16
  static constexpr uint32_t n32 = 10;
};

SiPixelRecHitSoAProducer::SiPixelRecHitSoAProducer(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes()),
      cpeToken_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("CPE")))),
      bsGetToken_{consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))},
	    clusterToken_{consumes<SiPixelClusterCollectionNew>(iConfig.getParameter<edm::InputTag>("clusters"))},
	    hitToken_{consumes<SiPixelRecHitCollectionNew>(iConfig.getParameter<edm::InputTag>("hits"))},
      tokenModuleStart_{produces<HMSstorage>()},
      tokenHitCPU_{produces<TrackingRecHit2DCPU>()},
      onGPU_(iConfig.getParameter<bool>("onGPU")),
      isPhase2_(iConfig.getParameter<bool>("isPhase2"))
       {
}

void SiPixelRecHitSoAProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("clusters", edm::InputTag("siPixelClustersPreSplitting"));
  desc.add<edm::InputTag>("hits", edm::InputTag("siPixelRecHitsPreSplitting"));
  desc.add<std::string>("CPE", "PixelCPEFast");
  desc.add<bool>("convertToLegacy", false);
  desc.add<bool>("onGPU", false);
  desc.add<bool>("isPhase2", false);
  descriptions.addWithDefaultLabel(desc);
}

void SiPixelRecHitSoAProducer::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& es) const {
  const TrackerGeometry* geom_ = &es.getData(geomToken_);

  PixelCPEFast const* fcpe = dynamic_cast<const PixelCPEFast*>(&es.getData(cpeToken_));
  if (not fcpe) {
    throw cms::Exception("Configuration") << "SiPixelRecHitSoAFromLegacy can only use a CPE of type PixelCPEFast";
  }

  std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
  auto const& cpeView = fcpe->getCPUProduct();

  const reco::BeamSpot& bs = iEvent.get(bsGetToken_);

  edm::Handle<SiPixelClusterCollectionNew> hclusters;
  iEvent.getByToken(clusterToken_, hclusters);
  auto const& clusters = *hclusters;

  edm::Handle<SiPixelRecHitCollectionNew> hhits;
  iEvent.getByToken(hitToken_, hhits);
  auto const& hits = *hhits;

  const int nMaxModules = isPhase2_ ? phase2PixelTopology::numberOfModules : phase1PixelTopology::numberOfModules;
  const int startBPIX2 = isPhase2_ ? phase2PixelTopology::layerStart[1] : phase1PixelTopology::layerStart[1];

  assert(nMaxModules < gpuClustering::maxNumModules);
  assert(startBPIX2 < nMaxModules);

  // allocate a buffer for the indices of the clusters
  auto hmsp = std::make_unique<uint32_t[]>(nMaxModules + 1);
  // hitsModuleStart is a non-owning pointer to the buffer
  auto hitsModuleStart = hmsp.get();
  // wrap the buffer in a HostProduct
  auto hms = std::make_unique<HMSstorage>(std::move(hmsp));
  // move the HostProduct to the Event, without reallocating the buffer or affecting hitsModuleStart
  iEvent.put(tokenModuleStart_, std::move(hms));

  // legacy output
  auto legacyOutput = std::make_unique<SiPixelRecHitCollectionNew>();

  std::vector<edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster>> clusterRef;

  constexpr uint32_t maxHitsInModule = gpuClustering::maxHitsInModule();

  HitModuleStart moduleStart_;  // index of the first pixel of each module
  HitModuleStart clusInModule_;
  memset(&clusInModule_, 0, sizeof(HitModuleStart));  // needed??
  memset(&moduleStart_, 0, sizeof(HitModuleStart));
  assert(gpuClustering::maxNumModules + 1 == clusInModule_.size());
  assert(0 == clusInModule_[gpuClustering::maxNumModules]);
  uint32_t moduleId_;
  moduleStart_[1] = 0;  // we run sequentially....

  SiPixelClustersCUDA::SiPixelClustersCUDASOAView clusterView{
      moduleStart_.data(), clusInModule_.data(), &moduleId_, hitsModuleStart};
  std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
  // fill cluster arrays
  int numberOfClusters = 0;
  for (auto const& dsv : clusters) {
    unsigned int detid = dsv.detId();
    DetId detIdObject(detid);
    const GeomDetUnit* genericDet = geom_->idToDetUnit(detIdObject);
    auto gind = genericDet->index();
    assert(gind < nMaxModules);
    auto const nclus = dsv.size();
    clusInModule_[gind] = nclus;
    numberOfClusters += nclus;
  }
  hitsModuleStart[0] = 0;

  for (int i = 1, n = nMaxModules + 1; i < n; ++i)
    hitsModuleStart[i] = hitsModuleStart[i - 1] + clusInModule_[i - 1];

  // assert(numberOfClusters == int(hits.size()));

  std::vector<float> store32(numberOfClusters*n32); //move to *static_cast<int>(TrackingRecHit2DCPU::n32)
  std::vector<uint16_t> store16(numberOfClusters*n16);

  /*  enum class Storage32 {
    kXLocal = 0,
    kYLocal = 1,
    kXerror = 2,
    kYerror = 3,
    kCharge = 4,
    kXGlobal = 5,
    kYGlobal = 6,
    kZGlobal = 7,
    kRGlobal = 8,
    kPhiStorage = 9,
    kLayers = 10
  };

  enum class Storage16 {
    kDetId = 0,
    kPhi = 1,
    kXSize = 2,
    kYSize = 3,
  };
  */

  std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
  //todo beamspot correct //done
  int numberOfHits = 0;
  int numberOfDetUnitsHits = 0;

  for (auto const& dsv : hits) {

    numberOfDetUnitsHits++;
    unsigned int detid = dsv.detId();
    DetId detIdObject(detid);
    const GeomDetUnit* genericDet = geom_->idToDetUnit(detIdObject);
    auto const gind = genericDet->index();
    assert(gind < nMaxModules);

    auto const nclus = dsv.size();
    assert(clusInModule_[gind] == nclus);

    for (auto const& h : dsv) {
  	auto local = h.localPosition();
  	auto localErr = h.localPositionError();
  	auto global = h.globalPosition();
  	auto clust = h.cluster();

  	store32[numberOfHits+numberOfClusters*static_cast<int>(TrackingRecHit2DCPU::Storage32::kXLocal)] = local.x();
  	store32[numberOfHits+numberOfClusters*1] = local.y();
  	store32[numberOfHits+numberOfClusters*2] = localErr.xx();
  	store32[numberOfHits+numberOfClusters*3] = localErr.yy();

  	//FIXME
  	store32[numberOfHits+numberOfClusters*4] = 0.0; //localErr.y();

  	auto x = global.x() - bs.x0();
  	auto y = global.y() - bs.y0();
  	store32[numberOfHits+numberOfClusters*5] = global.x() - bs.x0();
  	store32[numberOfHits+numberOfClusters*6] = global.y() - bs.y0();
  	store32[numberOfHits+numberOfClusters*7] = global.z() - bs.z0();
  	//std::sqrt(xg * xg + yg * yg);
  	store32[numberOfHits+numberOfClusters*8] = std::sqrt(x * x + y * y);//global.r();
  	//store32[numberOfHits*n32+9] = global.phi();
  	//store32[numberOfHits*n32+10] = global.phi();
      //store32[numberOfHits*n32+10] = global.phi();

  	store16[numberOfHits+numberOfClusters*0] = uint16_t(h.det()->index());
  	store16[numberOfHits+numberOfClusters*1] = unsafe_atan2s<7>(y, x);
  	store16[numberOfHits+numberOfClusters*2] = uint16_t(clust->sizeX());
  	store16[numberOfHits+numberOfClusters*3] = uint16_t(clust->sizeY());
  	numberOfHits++;
    // std::cout << numberOfHits << ";"<<uint16_t(h.det()->index()) << ";";
    }
}
  std::cout << std::endl;
  std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
  std::cout << store16[0] << std::endl;
  std::cout << store16[1] << std::endl;
  std::cout << store16[2] << std::endl;
  std::cout << store16[3] << std::endl;
  assert(numberOfHits==numberOfClusters);

  // output SoA
  // element 96 is the start of BPIX2 (i.e. the number of clusters in BPIX1)

  auto output = std::make_unique<TrackingRecHit2DCPU>(store32.data(),store16.data(), hmsp.get(), numberOfHits, isPhase2_, hitsModuleStart[startBPIX2], nullptr);
  std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
  if (0 == numberOfClusters) {
    iEvent.put(std::move(output));
    if (onGPU_)
      iEvent.put(std::move(legacyOutput));
    return;
  }


  int numberOfDetUnits = 0;

  for (auto const& dsv : clusters) {
    numberOfDetUnits++;
    unsigned int detid = dsv.detId();
    DetId detIdObject(detid);
    const GeomDetUnit* genericDet = geom_->idToDetUnit(detIdObject);
    auto const gind = genericDet->index();
    assert(gind < nMaxModules);
    const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>(genericDet);
    assert(pixDet);
    auto const nclus = dsv.size();
    assert(clusInModule_[gind] == nclus);
    if (0 == nclus)
      continue;  // is this really possible?

    auto const fc = hitsModuleStart[gind];
    auto const lc = hitsModuleStart[gind + 1];
    assert(lc > fc);
    //LogDebug("SiPixelRecHitSoAProducer")
    std::cout << "SiPixelRecHitSoAProducer" << "in det " << gind << ": conv " << nclus << " hits from " << dsv.size()
                                           << " legacy clusters" << ' ' << fc << ',' << lc;
    assert((lc - fc) == nclus);
    if (nclus > maxHitsInModule)
      printf(
          "WARNING: too many clusters %d in Module %d. Only first %d Hits converted\n", nclus, gind, maxHitsInModule);


	for (auto h = fc; h < lc; ++h)
      if (h - fc < maxHitsInModule)
        {
          if(gind!=output->view()->detectorIndex(h))
            std::cout << "AIA " << gind << " - " <<  output->view()->detectorIndex(h) << " - " << output->view()->iphi(h)<<std::endl;
          assert(gind == output->view()->detectorIndex(h));}
      else
        assert(gpuClustering::invalidModuleId == output->view()->detectorIndex(h));
  }


  // fill data structure to support CA
  const auto nLayers = isPhase2_ ? phase2PixelTopology::numberOfLayers : phase1PixelTopology::numberOfLayers;
  for (auto i = 0U; i < nLayers + 1; ++i) {
    output->hitsLayerStart()[i] = hitsModuleStart[cpeView.layerGeometry().layerStart[i]];
    LogDebug("SiPixelRecHitSoAProducer")
        << "Layer n." << i << " - starting at module: " << cpeView.layerGeometry().layerStart[i]
        << " - starts ad cluster: " << output->hitsLayerStart()[i] << "\n";
  }
  std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
  cms::cuda::fillManyFromVector(output->phiBinner(),
                                nLayers,
                                output->iphi(),
                                output->hitsLayerStart(),
                                numberOfHits,
                                256,
                                output->phiBinnerStorage());

  LogDebug("SiPixelRecHitSoAProducer") << "created HitSoa for " << numberOfClusters << " clusters in "
                                         << numberOfDetUnits << " Dets";
  iEvent.put(std::move(output));
  //if (onGPU_)
  //  iEvent.put(std::move(legacyOutput));
}

DEFINE_FWK_MODULE(SiPixelRecHitSoAProducer);
