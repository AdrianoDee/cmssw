#include <cuda_runtime.h>

#include <fmt/printf.h>

#include "CUDADataFormats/Common/interface/HostProduct.h"
#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
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

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoAHost.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoADevice.h"

class SiPixelRecHitSoAFromCUDA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit SiPixelRecHitSoAFromCUDA(const edm::ParameterSet& iConfig);
  ~SiPixelRecHitSoAFromCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  using HMSstorage = HostProduct<uint32_t[]>;
  using TrackingRecHitSoADevice = trackingRecHit::TrackingRecHitSoADevice;
  using TrackingRecHitSoAHost = trackingRecHit::TrackingRecHitSoAHost;

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  const edm::EDGetTokenT<cms::cuda::Product<TrackingRecHitSoADevice>> hitsTokenGPU_;  // CUDA hits
  const edm::EDPutTokenT<TrackingRecHitSoAHost> hitsPutTokenCPU_;
  const edm::EDPutTokenT<HMSstorage> hostPutToken_;

  uint32_t nHits_;
  TrackingRecHitSoAHost hits_h_;

  uint32_t nMaxModules_;

  // cms::cuda::host::unique_ptr<float[]> store32_;
  // cms::cuda::host::unique_ptr<uint16_t[]> store16_;
  // cms::cuda::host::unique_ptr<uint32_t[]> hitsModuleStart_;
};

SiPixelRecHitSoAFromCUDA::SiPixelRecHitSoAFromCUDA(const edm::ParameterSet& iConfig)
    : hitsTokenGPU_(
          consumes<cms::cuda::Product<TrackingRecHitSoADevice>>(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
      hitsPutTokenCPU_(produces<TrackingRecHitSoAHost>()),
      hostPutToken_(produces<HMSstorage>()) {}

void SiPixelRecHitSoAFromCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsPreSplittingCUDA"));
  descriptions.addWithDefaultLabel(desc);
}

void SiPixelRecHitSoAFromCUDA::acquire(edm::Event const& iEvent,
                                       edm::EventSetup const& iSetup,
                                       edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::Product<TrackingRecHitSoADevice> const& inputDataWrapped = iEvent.get(hitsTokenGPU_);
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  nHits_ = inputData.nHits();

  if (0 == nHits_)
    return;

  nMaxModules_ = inputData.nModules();

  hits_h_ = TrackingRecHitSoAHost(nHits_, ctx.stream());
  cudaCheck(cudaMemcpyAsync(hits_h_.buffer().get(),
                            inputData.const_buffer().get(),
                            inputData.bufferSize(),
                            cudaMemcpyDeviceToHost,
                            ctx.stream()));  // Copy data from Device to Host
  cudaCheck(cudaGetLastError());

  LogDebug("SiPixelRecHitSoAFromCUDA") << "copying to cpu SoA" << inputData.nHits() << " Hits";

  // store32_ = inputData.store32ToHostAsync(ctx.stream());
  // store16_ = inputData.store16ToHostAsync(ctx.stream());
  // hitsModuleStart_ = inputData.hitsModuleStartToHostAsync(ctx.stream());
}

void SiPixelRecHitSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& es) {
  auto hmsp = std::make_unique<uint32_t[]>(nMaxModules_ + 1);

  if (nHits_ > 0)
    std::copy(hits_h_.view().hitsModuleStart().begin(), hits_h_.view().hitsModuleStart().end(), hmsp.get());
  // std::copy(hitsModuleStart_.get(), hitsModuleStart_.get() + nMaxModules_ + 1, hmsp.get());

  iEvent.emplace(hostPutToken_, std::move(hmsp));
  iEvent.emplace(hitsPutTokenCPU_,
                 std::move(hits_h_));  //store32_.get(), store16_.get(), hitsModuleStart_.get(), nHits_);
}

DEFINE_FWK_MODULE(SiPixelRecHitSoAFromCUDA);
