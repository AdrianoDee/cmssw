#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/Product.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "CAHitNtupletGeneratorOnGPU.h"
#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"

#include "RecoPixelVertexing/PixelTriplets/plugins/CAHitNtupletGeneratorKernels.h"

class CAHitNtupletCUDA : public edm::global::EDProducer<> {
public:

  // using VertexRegion = cAHitNtupletGenerator::VertexRegion;

  explicit CAHitNtupletCUDA(const edm::ParameterSet& iConfig);
  ~CAHitNtupletCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  bool m_OnGPU;
  bool m_Upgrade;

  edm::EDGetTokenT<cms::cuda::Product<TrackingRecHit2DGPU>> tokenHitGPU_;
  edm::EDPutTokenT<cms::cuda::Product<PixelTrackHeterogeneous>> tokenTrackGPU_;
  edm::EDGetTokenT<TrackingRecHit2DCPU> tokenHitCPU_;
  edm::EDPutTokenT<PixelTrackHeterogeneous> tokenTrackCPU_;

  edm::EDGetTokenT<VertexRegion> vertexTokenCPU_;

  CAHitNtupletGeneratorOnGPU gpuAlgo_;
};

CAHitNtupletCUDA::CAHitNtupletCUDA(const edm::ParameterSet& iConfig)
    : m_OnGPU(iConfig.getParameter<bool>("onGPU")),gpuAlgo_(iConfig, consumesCollector()) {
  if (m_OnGPU) {
    tokenHitGPU_ =
        consumes<cms::cuda::Product<TrackingRecHit2DGPU>>(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"));
    tokenTrackGPU_ = produces<cms::cuda::Product<PixelTrackHeterogeneous>>();
  } else {
    tokenHitCPU_ = consumes<TrackingRecHit2DCPU>(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"));
    tokenTrackCPU_ = produces<PixelTrackHeterogeneous>();
  }

  vertexTokenCPU_ = consumes<VertexRegion>(iConfig.getParameter<edm::InputTag>("vertexRegion"));
}

void CAHitNtupletCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<bool>("onGPU", true);
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsCUDAPreSplitting"));
  desc.add<edm::InputTag>("vertexRegion", edm::InputTag("pixelVertexCoordinates"));

  CAHitNtupletGeneratorOnGPU::fillDescriptions(desc);
  auto label = "caHitNtupletCUDA";
  descriptions.add(label, desc);
}

void CAHitNtupletCUDA::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& es) const {
  auto bf = 1. / PixelRecoUtilities::fieldInInvGev(es);

  if (m_OnGPU) {
    edm::Handle<cms::cuda::Product<TrackingRecHit2DCUDA>> hHits;
    iEvent.getByToken(tokenHitGPU_, hHits);

    cms::cuda::ScopedContextProduce ctx{*hHits};
    auto const& hits = ctx.get(*hHits);

    ctx.emplace(iEvent, tokenTrackGPU_, gpuAlgo_.makeTuplesAsync(hits, bf, ctx.stream()));
  } else {
    auto const& hits = iEvent.get(tokenHitCPU_);
    auto const& vtxs = iEvent.get(vertexTokenCPU_);
    iEvent.emplace(tokenTrackCPU_, gpuAlgo_.makeTuples(hits, vtxs, bf));
  }
}

DEFINE_FWK_MODULE(CAHitNtupletCUDA);
