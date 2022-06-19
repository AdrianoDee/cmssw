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
#include "DataFormats/Math/interface/approx_atan2.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFast.h"

#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"

// #include "gpuPixelRecHits.h"

class SiPixelRecHitSoAProducer : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit SiPixelRecHitSoAProducer(const edm::ParameterSet& iConfig);
  ~SiPixelRecHitSoAProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  using HitModuleStart = std::array<uint32_t, gpuClustering::maxNumModules + 1>;
  using HMSstorage = HostProduct<uint32_t[]>;

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  cms::cuda::ContextState ctxState_;

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> cpeToken_;
  const edm::EDGetTokenT<reco::BeamSpot> bsGetToken_;
  const edm::EDGetTokenT<SiPixelRecHitCollectionNew> hitToken_;
// const edm::EDPutTokenT<TrackingRecHit2DGPU> tokenHitGPU_;
  const edm::EDPutTokenT<HMSstorage> tokenModuleStart_;

  edm::EDPutTokenT<TrackingRecHit2DCPU> tokenHitCPU_;
  edm::EDPutTokenT<cms::cuda::Product<TrackingRecHit2DGPU>> tokenHitGPU_;

  const bool onGPU_;
  const bool isPhase2_;

  static constexpr uint32_t n16 = 4;                 // number of elements in m_store16
  static constexpr uint32_t n32 = 10;

  static constexpr uint32_t chargeMask() { return (1 << 24) - 1; }
};

SiPixelRecHitSoAProducer::SiPixelRecHitSoAProducer(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes()),
      cpeToken_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("CPE")))),
      bsGetToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
	    hitToken_(consumes<SiPixelRecHitCollectionNew>(iConfig.getParameter<edm::InputTag>("hits"))),
      tokenModuleStart_(produces<HMSstorage>()),
      tokenHitCPU_(produces<TrackingRecHit2DCPU>()),
      onGPU_(iConfig.getParameter<bool>("onGPU")),
      isPhase2_(iConfig.getParameter<bool>("isPhase2"))
       {
}

void SiPixelRecHitSoAProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("hits", edm::InputTag("siPixelRecHitsPreSplitting"));
  desc.add<std::string>("CPE", "PixelCPEFast");
  desc.add<bool>("convertToLegacy", false);
  desc.add<bool>("onGPU", false);
  desc.add<bool>("isPhase2", false);
  descriptions.addWithDefaultLabel(desc);
}


void SiPixelRecHitSoAProducer::acquire(const edm::Event& iEvent,
                                      const edm::EventSetup& iSetup,
                                      edm::WaitingTaskWithArenaHolder waitingTaskHolder)
                                      {
cms::cuda::ScopedContextAcquire ctx{iEvent.streamID(), std::move(waitingTaskHolder), ctxState_};

                                      }
void SiPixelRecHitSoAProducer::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {

  cms::cuda::ScopedContextProduce ctx{ctxState_};

  const TrackerGeometry* geom_ = &iSetup.getData(geomToken_);

  const PixelCPEBase* cpe  = dynamic_cast<const PixelCPEBase*>(&iSetup.getData(cpeToken_));

  std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;

  const reco::BeamSpot& bs = iEvent.get(bsGetToken_);

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
  HitModuleStart hitsInModule_;
  memset(&hitsInModule_, 0, sizeof(HitModuleStart));  // needed??
  memset(&moduleStart_, 0, sizeof(HitModuleStart));
  assert(gpuClustering::maxNumModules + 1 == hitsInModule_.size());
  assert(0 == hitsInModule_[gpuClustering::maxNumModules]);
  moduleStart_[1] = 0;  // we run sequentially....


  std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
  // fill hit arrays
  int numberOfHits = 0;

  int numberOfDetUnits = 0;
  for (auto const& dsv : hits) {
    numberOfDetUnits++;
    unsigned int detid = dsv.detId();
    DetId detIdObject(detid);
    const GeomDetUnit* genericDet = geom_->idToDetUnit(detIdObject);
    auto gind = genericDet->index();
    assert(gind < nMaxModules);
    auto const nhits = dsv.size();
    hitsInModule_[gind] = nhits;
    numberOfHits += nhits;

    std::cout << "SiPixelRecHitSoAProducer - Clusters " << "in det " << gind << ": conv " << nhits << " hits from " << dsv.size()
                                           << " legacy hits" << std::endl;
  }
  hitsModuleStart[0] = 0;

  for (int i = 1, n = nMaxModules + 1; i < n; ++i)
    hitsModuleStart[i] = hitsModuleStart[i - 1] + hitsInModule_[i - 1];

  std::vector<float> store32(numberOfHits * n32);
  std::vector<uint16_t> store16(numberOfHits * n16);

  std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
  //todo beamspot correct //done

  int c = 0;
  for (auto const& dsv : hits) {

    unsigned int detid = dsv.detId();
    DetId detIdObject(detid);
    const GeomDetUnit* genericDet = geom_->idToDetUnit(detIdObject);
    auto const gind = genericDet->index();
    assert(gind < nMaxModules);
    PixelCPEBase::DetParam const& theDetParam = cpe->detParam(*genericDet);

    auto const nhits = dsv.size();
    assert(hitsInModule_[gind] == nhits);

    for (auto const& h : dsv) {
  	auto local = h.localPosition();
  	auto localErr = h.localPositionError();
  	auto global = h.globalPosition();
  	auto clust = h.cluster();

    std::cout << "SiPixelRecHitSoAProducer - Hits " << "in det " << gind << ": conv " << nhits << " hits from " << dsv.size() << " legacy hits" << std::endl;

  	store32[c + numberOfHits*static_cast<int>(TrackingRecHit2DCPU::Storage32::kXLocal)] = local.x();
  	store32[c + numberOfHits*static_cast<int>(TrackingRecHit2DCPU::Storage32::kYLocal)] = local.y();
  	store32[c + numberOfHits*static_cast<int>(TrackingRecHit2DCPU::Storage32::kXerror)] = localErr.xx();
  	store32[c + numberOfHits*static_cast<int>(TrackingRecHit2DCPU::Storage32::kYerror)] = localErr.yy();
    std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
  	//FIXME

    SiPixelHitStatus status;

    auto minInX = clust->minPixelRow();
    auto minInY = clust->minPixelCol();
    auto maxInX = clust->maxPixelRow();
    auto maxInY = clust->maxPixelCol();
    std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
    auto sx = maxInX - minInX;
    auto sy = maxInY - minInY;

    int min_row(0), min_col(0);
    int max_row = theDetParam.theRecTopol->nrows() - 1;
    int max_col = theDetParam.theRecTopol->ncolumns() - 1;
    std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
    bool isEdgeX = (minInX == min_row) | (maxInX == max_row);
    bool isEdgeY = (minInY == min_col) | (maxInY == max_col);
    bool isOneX = (0 == sx);
    bool isOneY = (0 == sy);
    bool isBigX = isPhase2_ ? false : theDetParam.theRecTopol->containsBigPixelInX(minInX, maxInX);
    bool isBigY = isPhase2_ ? false : theDetParam.theRecTopol->containsBigPixelInY(minInY, maxInY);

    status.qBin = h.qBin();
    status.isOneX = isOneX;
    status.isBigX = (isOneX & isBigX) | isEdgeX;
    status.isOneY = isOneY;
    status.isBigY = (isOneY & isBigY) | isEdgeY;
    std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
    uint32_t ich = clust->charge();
    ich = std::min(ich, chargeMask());
    uint32_t w = *reinterpret_cast<uint8_t*>(&status);
    ich |= (w << 24);

  	store32[c + numberOfHits*static_cast<int>(TrackingRecHit2DCPU::Storage32::kCharge)] = ich; //FIXME

  	auto x = global.x() - bs.x0();
  	auto y = global.y() - bs.y0();
  	store32[c + numberOfHits*static_cast<int>(TrackingRecHit2DCPU::Storage32::kXGlobal)] = x; //correcting for the bs
  	store32[c + numberOfHits*static_cast<int>(TrackingRecHit2DCPU::Storage32::kYGlobal)] = y;
  	store32[c + numberOfHits*static_cast<int>(TrackingRecHit2DCPU::Storage32::kZGlobal)] = global.z() - bs.z0();
  	store32[c + numberOfHits*static_cast<int>(TrackingRecHit2DCPU::Storage32::kRGlobal)] = std::sqrt(x * x + y * y);

  	store16[c + numberOfHits*static_cast<int>(TrackingRecHit2DCPU::Storage16::kDetId)] = uint16_t(h.det()->index());
  	store16[c + numberOfHits*static_cast<int>(TrackingRecHit2DCPU::Storage16::kPhi)] = unsafe_atan2s<7>(y, x);
  	store16[c + numberOfHits*static_cast<int>(TrackingRecHit2DCPU::Storage16::kXSize)] = uint16_t(clust->sizeX()); //FIXME AND CALCULATE ME or CHANGE CUTS
  	store16[c + numberOfHits*static_cast<int>(TrackingRecHit2DCPU::Storage16::kYSize)] = uint16_t(clust->sizeY());
  	c++;

    }
}
  std::cout << std::endl;
  std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
  std::cout << store16[0] << std::endl;
  std::cout << store16[1] << std::endl;
  std::cout << store16[2] << std::endl;
  std::cout << store16[3] << std::endl;
  assert(numberOfHits==c);

  std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
  // output SoA
  // element 96 is the start of BPIX2 (i.e. the number of clusters in BPIX1)

  auto theStream = ctx.stream();
  // auto output = std::make_unique<TrackingRecHit2DCPU>(store32.data(),store16.data(), hmsp.get(), numberOfHits, isPhase2_, nullptr);
  TrackingRecHit2DCPU output(store32.data(),store16.data(), hmsp.get(), numberOfHits, hitsModuleStart[startBPIX2], isPhase2_, theStream);
  std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;


  // int numberOfDetUnits = 0;
  // TOD ADD cluster filler

  for (auto const& dsv : hits) {

    unsigned int detid = dsv.detId();
    DetId detIdObject(detid);
    const GeomDetUnit* genericDet = geom_->idToDetUnit(detIdObject);
    auto const gind = genericDet->index();
    assert(gind < nMaxModules);
    const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>(genericDet);
    assert(pixDet);
    auto const nhits = dsv.size();
    assert(hitsInModule_[gind] == nhits);
    if (0 == nhits)
      continue;  // is this really possible?

    auto const fc = hitsModuleStart[gind];
    auto const lc = hitsModuleStart[gind + 1];
    assert(lc > fc);
    //LogDebug("SiPixelRecHitSoAProducer")
    std::cout << "SiPixelRecHitSoAProducer " << "in det " << gind << ": conv " << nhits << " hits from " << dsv.size()
                                           << " legacy clusters" << ' ' << fc << ',' << lc << std::endl;
    assert((lc - fc) == nhits);
    if (nhits > maxHitsInModule)
      printf(
          "WARNING: too many clusters %d in Module %d. Only first %d Hits converted\n", nhits, gind, maxHitsInModule);


	// for (auto h = fc; h < lc; ++h)
  //     if (h - fc < maxHitsInModule)
  //       {
  //         if(gind!=output.view()->detectorIndex(h))
  //           std::cout << "AIA " << gind << " - " <<  output.view()->detectorIndex(h) << " - " << output.view()->iphi(h)<<std::endl;
  //         else
  //           std::cout << "GOO " << gind << " - " <<  output.view()->detectorIndex(h) << " - " << output.view()->iphi(h)<<std::endl;
  //         assert(gind == output.view()->detectorIndex(h));}
  //     else
  //       assert(gpuClustering::invalidModuleId == output.view()->detectorIndex(h));
  }

  std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
  // fill data structure to support CA
  const auto nLayers = 10; //isPhase2_ ? phase2PixelTopology::numberOfLayers : phase1PixelTopology::numberOfLayers;

  constexpr int L[11] = {0, 96, 320, 672, 1184, 1296, 1408,  1520, 1632, 1744, 1856};

  std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
  for (size_t i = 0; i < 11; i++) {
    std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << " " << L[i] << std::endl;
    auto start = L[i];
    // auto start = 96;//layerStart[i];//isPhase2_ ? phase2PixelTopology::layerStart[i] : phase1PixelTopology::layerStart[i];
    std::cout << "SiPixelRecHitSoAProducer" << "- starting at module: " << start;
    output.hitsLayerStart()[i] = hitsModuleStart[start];

        std::cout << " - starts ad cluster: " << output.hitsLayerStart()[i] << " -" << output.iphi()[i] << "\n";
  }
  std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;


    typename TrackingRecHit2DSOAView::PhiBinner::View view = {output.phiBinner(), nullptr, output.phiBinnerStorage(), -1, numberOfHits};
    cms::cuda::launchZero(view, theStream);
    cms::cuda::countFromVector(output.phiBinner(), nLayers, output.iphi(), output.hitsLayerStart());
    std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
    output.phiBinner()->finalize();
    std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
    cms::cuda::fillFromVector(output.phiBinner(), nLayers, output.iphi(), output.hitsLayerStart());
    std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;

    LogDebug("SiPixelRecHitSoAProducer") << "created HitSoa for " << numberOfHits << " clusters in "
                                           << numberOfDetUnits << " Dets";
    std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
    iEvent.emplace(tokenHitCPU_, std::move(output));
    std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
    if (onGPU_)
    {
      std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
      // auto outputGPU = std::make_unique<TrackingRecHit2DGPU>
      auto store32ForGPU = output.store32ToHostAsync(theStream).get();
      std::cout << "SiPixelRecHitSoAProducer" << __LINE__ << std::endl;
      TrackingRecHit2DGPU outputGPU(store32ForGPU, store16.data(), hmsp.get(), numberOfHits, hitsModuleStart[startBPIX2], isPhase2_, theStream);
      ctx.emplace(iEvent, tokenHitGPU_, std::move(outputGPU));

    }

}

DEFINE_FWK_MODULE(SiPixelRecHitSoAProducer);
