
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
  // const edm::EDPutTokenT<HMSstorage> tokenModuleStart_;
};

template <typename TrackerTraits>
SiStripRecHitSoAHost<TrackerTraits>::SiStripRecHitSoAHost(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes()),
    //   bsGetToken_{consumes(iConfig.getParameter<edm::InputTag>("beamSpot"))},
      recHitToken_{consumes<SiStripMatchedRecHit2DCollection>(iConfig.getParameter<edm::InputTag>("stripRecHitSource"))},
      pixelRecHitSoAToken_{consumes<HitsOnHost>(iConfig.getParameter<edm::InputTag>("pixelRecHitSoASource"))},
      // pixelModuleStartToken_{consumes(iConfig.getParameter<edm::InputTag>("pixelModuleStartSource"))},
      tokenHit_{produces()}
      // tokenModuleStart_{produces()} 
      {
}

template <typename TrackerTraits>
void SiStripRecHitSoAHost<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

//   desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  // desc.add<edm::InputTag>("stripRecHitSource", edm::InputTag("matchedRecHit"));
  desc.setUnknown();
  descriptions.addDefault(desc);

  // descriptions.addWithDefaultLabel(desc);
}

template <typename TrackerTraits>
void SiStripRecHitSoAHost<TrackerTraits>::produce(edm::StreamID streamID,
                                                         edm::Event& iEvent,
                                                         const edm::EventSetup& es) const {
  
  std::cout << "produce" << std::endl;
  
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

  std::cout << "nStripHits = " << nStripHits << std::endl;

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
  std::copy(
    pixelRecHitSoAHandle->view().hitsModuleStart().begin(),
    pixelRecHitSoAHandle->view().hitsModuleStart().end(),
    result.view().hitsModuleStart().begin()
  );

  std::copy(
    pixelRecHitSoAHandle->view().hitsModuleStart().begin(),
    pixelRecHitSoAHandle->view().hitsModuleStart().end(),
    result.view().hitsModuleStart().begin()
  );

  result.view().cpeParams() = pixelRecHitSoAHandle->view().cpeParams();
  result.view().averageGeometry() = pixelRecHitSoAHandle->view().averageGeometry();
  result.view().phiBinner() = pixelRecHitSoAHandle->view().phiBinner();

  // Loop over strip RecHits
  size_t i = 0;
  std::vector<size_t> nHitsPerModule;
  nHitsPerModule.reserve(TrackerTraits::numberOfStripModules + 1);
  nHitsPerModule.push_back(pixelRecHitSoAHandle->view().hitsModuleStart().back());
  
  for (const auto& detSet : *stripRecHitHandle) {
    const GluedGeomDet* det = static_cast<const GluedGeomDet*>(trackerGeometry->idToDet(detSet.detId()));
    if (det->stereoDet()->index() >= TrackerTraits::numberOfModules)
      break;

    nHitsPerModule.push_back(detSet.size());

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
  
  std::partial_sum(
    nHitsPerModule.begin(), 
    nHitsPerModule.end(), 
    result.view().hitsModuleStart().begin() + TrackerTraits::numberOfPixelModules - 1
  );

  for (auto layer = 0U; layer < TrackerTraits::numberOfStripLayers + 1; ++layer) {
    result.view().hitsLayerStart()[TrackerTraits::numberOfPixelLayers + i] = 
      result.view().hitsModuleStart()[TrackerTraits::layerStart[layer]];
  }

  using PhiBinner = typename TrackingRecHitSoA<TrackerTraits>::PhiBinner;

  auto& hh = result.view();

  auto const& __restrict__ phiBinner = hh.phiBinner();
  uint32_t const* __restrict__ offsets = hh.hitsLayerStart().data();
  assert(offsets);

  auto layerSize = [=](uint8_t li) { return offsets[li + 1] - offsets[li]; };

  // nPairsMax to be optimized later (originally was 64).
  // If it should be much bigger, consider using a block-wide parallel prefix scan,
  // e.g. see  https://nvlabs.github.io/cub/classcub_1_1_warp_scan.html

  __shared__ uint32_t innerLayerCumulativeSize[TrackerTraits::nPairs];
  __shared__ uint32_t ntot;
  if (threadIdx.y == 0 && threadIdx.x == 0) {
    innerLayerCumulativeSize[0] = layerSize(TrackerTraits::layerPairs[0]);
    for (uint32_t i = 1; i < TrackerTraits::nPairs; ++i) {
      innerLayerCumulativeSize[i] = innerLayerCumulativeSize[i - 1] + layerSize(TrackerTraits::layerPairs[2 * i]);
    }
    ntot = innerLayerCumulativeSize[TrackerTraits::nPairs - 1];
  }
  // __syncthreads();

  // x runs faster
  auto idy = blockIdx.y * blockDim.y + threadIdx.y;
  auto first = threadIdx.x;
  auto stride = blockDim.x;

  uint32_t pairLayerId = 0;  // cannot go backward

  for (auto j = idy; j < ntot; j += blockDim.y * gridDim.y) {
    while (j >= innerLayerCumulativeSize[pairLayerId++])
      ;
    --pairLayerId;  // move to lower_bound ??

    assert(pairLayerId < TrackerTraits::nPairs);
    assert(j < innerLayerCumulativeSize[pairLayerId]);
    assert(0 == pairLayerId || j >= innerLayerCumulativeSize[pairLayerId - 1]);

    uint8_t inner = TrackerTraits::layerPairs[2 * pairLayerId];
    uint8_t outer = TrackerTraits::layerPairs[2 * pairLayerId + 1];
    assert(outer > inner);

    auto hoff = PhiBinner::histOff(outer);
    auto i = (0 == pairLayerId) ? j : j - innerLayerCumulativeSize[pairLayerId - 1];
    i += offsets[inner];

    printf("i = %d", i);
    printf("inner = %d", inner);
    printf("offsets[inner] = %d", offsets[inner]);
    printf("offsets[inner + 1] = %d", offsets[inner + 1]);

    // assert(i >= offsets[inner]);
    std::cout << (i >= offsets[inner]) << std::endl;
    // assert(i < offsets[inner + 1]);
    std::cout << (i < offsets[inner + 1]) << std::endl;
  }

  iEvent.emplace(tokenHit_, std::move(result));
}

using SiStripRecHitSoAHostPhase1 = SiStripRecHitSoAHost<pixelTopology::Phase1>;
DEFINE_FWK_MODULE(SiStripRecHitSoAHostPhase1);

// using SiPixelRecHitSoAFromLegacyPhase2 = SiStripRecHitSoAHost<pixelTopology::Phase2>;
// DEFINE_FWK_MODULE(SiPixelRecHitSoAFromLegacyPhase2);
