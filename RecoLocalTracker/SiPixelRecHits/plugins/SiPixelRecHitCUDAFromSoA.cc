#include <cuda_runtime.h>

#include <fmt/printf.h>

#include "CUDADataFormats/Common/interface/HostProduct.h"
#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoAHost.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoADevice.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFast.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"

#include "PixelRecHitGPUKernel.h"

template <typename TrackerTraits>
class SiPixelRecHitCUDAFromSoAT : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit SiPixelRecHitCUDAFromSoAT(const edm::ParameterSet& iConfig);
  ~SiPixelRecHitCUDAFromSoAT() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  using HMSstorage = HostProduct<uint32_t[]>;
  using HitsOnHost = TrackingRecHitSoAHost<TrackerTraits>;
  using HitsOnDevice = TrackingRecHitSoADevice<TrackerTraits>;

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;
  
  const edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> cpeToken_;
  const edm::EDGetTokenT<HitsOnHost> hitsTokenCPU_;  // CPU hits
  edm::EDPutTokenT<cms::cuda::Product<HitsOnDevice>> hitsPutTokenGPU_;

  cms::cuda::ContextState ctxState_;

  const pixelgpudetails::PixelRecHitGPUKernel<TrackerTraits> gpuAlgo_;
};

template <typename TrackerTraits>
SiPixelRecHitCUDAFromSoAT<TrackerTraits>::SiPixelRecHitCUDAFromSoAT(const edm::ParameterSet& iConfig)
    : cpeToken_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("CPE")))),
      hitsTokenCPU_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
      hitsPutTokenGPU_(produces<cms::cuda::Product<HitsOnDevice>>())
      {} 

template <typename TrackerTraits>
void SiPixelRecHitCUDAFromSoAT<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsPreSplitting"));

  std::string cpe = "PixelCPEFast";
  cpe += TrackerTraits::nameModifier;
  desc.add<std::string>("CPE", cpe);
  
  descriptions.addWithDefaultLabel(desc);
}

template <typename TrackerTraits>
void SiPixelRecHitCUDAFromSoAT<TrackerTraits>::acquire(edm::Event const& iEvent,
                                                       edm::EventSetup const& iSetup,
                                                       edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::ScopedContextAcquire ctx{iEvent.streamID(), std::move(waitingTaskHolder), ctxState_};

  // nHits_ = inputData.nHits();
  // hits_d_ = HitsOnDevice(nHits_, ctx.stream());
  // cudaCheck(cudaMemcpyAsync(hits_d_.buffer().get(),
  //                           inputData.const_buffer().get(),
  //                           inputData.bufferSize(),
  //                           cudaMemcpyHostToDevice,
  //                           ctx.stream()));  // Copy data from Device to Host
  
  // constexpr auto nLayers = TrackerTraits::numberOfLayers;

  // cms::cuda::fillManyFromVector(&(hits_d_.view().phiBinner()),
  //                                     nLayers,
  //                                     hits_d_.view().iphi(),
  //                                     hits_d_.view().hitsLayerStart().data(),
  //                                     nHits_,
  //                                     256,
  //                                     hits_d_.view().phiBinnerStorage(),
  //                                     ctx.stream()); 
  // cudaCheck(cudaGetLastError());
}

template <typename TrackerTraits>
void SiPixelRecHitCUDAFromSoAT<TrackerTraits>::produce(edm::Event& iEvent, edm::EventSetup const& es) {

  cms::cuda::ScopedContextProduce ctx{ctxState_};

  PixelCPEFast<TrackerTraits> const* fcpe = dynamic_cast<const PixelCPEFast<TrackerTraits>*>(&es.getData(cpeToken_));
  if (not fcpe) {
    throw cms::Exception("Configuration") << "SiPixelRecHitCUDA can only use a CPE of type PixelCPEFast";
  }

  auto& hits_h = iEvent.get(hitsTokenCPU_);
  // auto const& inputData = iEvent.get(hitsTokenCPU_);
  LogDebug("SiPixelRecHitCUDAFromSoA") << "copying to GPU the CPU SoA with" << hits_h.nHits() << " Hits";

  ctx.emplace(iEvent,
              hitsPutTokenGPU_,
              gpuAlgo_.fillHitsAsync(hits_h, fcpe->getGPUProductAsync(ctx.stream()), ctx.stream()));

  // ctx.emplace(iEvent,hitsPutTokenGPU_, std::move(hits_d_));
}

using SiPixelRecHitCUDAFromSoAPhase1 = SiPixelRecHitCUDAFromSoAT<pixelTopology::Phase1>;
DEFINE_FWK_MODULE(SiPixelRecHitCUDAFromSoAPhase1);

using SiPixelRecHitCUDAFromSoAPhase2 = SiPixelRecHitCUDAFromSoAT<pixelTopology::Phase2>;
DEFINE_FWK_MODULE(SiPixelRecHitCUDAFromSoAPhase2);

using SiPixelRecHitCUDAFromSoAHIonPhase1 = SiPixelRecHitCUDAFromSoAT<pixelTopology::HIonPhase1>;
DEFINE_FWK_MODULE(SiPixelRecHitCUDAFromSoAHIonPhase1);
