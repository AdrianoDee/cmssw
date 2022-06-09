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

class SiPixelRecHitCUDAFromSoA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit SiPixelRecHitCUDAFromSoA(const edm::ParameterSet& iConfig);
  ~SiPixelRecHitCUDAFromSoA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  const edm::EDGetTokenT<TrackingRecHit2DCPU> hitsTokenCPU_;
  const edm::EDPutTokenT<cms::cuda::Product<TrackingRecHit2DGPU>> hitsTokenGPU_;  // CUDA hits

  cms::cuda::ContextState ctxState_;

};

SiPixelRecHitCUDAFromSoA::SiPixelRecHitCUDAFromSoA(const edm::ParameterSet& iConfig)
    : hitsTokenCPU_(consumes<TrackingRecHit2DCPU>(iConfig.getParameter<edm::InputTag>("pixelRecHitSoASrc"))),
      hitsTokenGPU_(produces<cms::cuda::Product<TrackingRecHit2DGPU>>())
      {      }

void SiPixelRecHitCUDAFromSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelRecHitSoASrc", edm::InputTag("siPixelRecHitsPreSplittingSoA"));
  descriptions.addWithDefaultLabel(desc);
}

void SiPixelRecHitCUDAFromSoA::acquire(edm::Event const& iEvent,
                                       edm::EventSetup const& iSetup,
                                       edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::ScopedContextAcquire ctx{iEvent.streamID(), std::move(waitingTaskHolder), ctxState_};
}

void SiPixelRecHitCUDAFromSoA::produce(edm::Event& iEvent, edm::EventSetup const& es) {

  cms::cuda::ScopedContextProduce ctx{ctxState_};

  auto const& hits = iEvent.get(hitsTokenCPU_);
  ctx.emplace(iEvent, hitsTokenGPU_, hits.store32().get(), hits.store16().get(), hits.modules().get(), hits.nHits());

}

DEFINE_FWK_MODULE(SiPixelRecHitCUDAFromSoA);
