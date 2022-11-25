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
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "CAHitNtupletGeneratorOnGPU.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousHost.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousDevice.h"
// #include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoADevice.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoAHost.h"

class CAHitNtupletCUDA : public edm::global::EDProducer<> {
public:
  using TrackingRecHitSoAHost = trackingRecHit::TrackingRecHitSoAHost;
  using TrackingRecHitSoADevice = trackingRecHit::TrackingRecHitSoADevice;
  explicit CAHitNtupletCUDA(const edm::ParameterSet& iConfig);
  ~CAHitNtupletCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void endJob() override;

  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  bool onGPU_;

  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tokenField_;
  // GPU
  // Produces a view on GPU, which is used by PixelTrackSoAFromCUDA
  edm::EDGetTokenT<cms::cuda::Product<TrackingRecHitSoADevice>> tokenHitGPU_;
  edm::EDPutTokenT<cms::cuda::Product<pixelTrack::TrackSoADevice>> tokenTrackGPU_;

  // CPU
  // Produces a view on CPU, which is used by PixelTrackProducerFromSoA
  edm::EDGetTokenT<TrackingRecHitSoAHost> tokenHitCPU_;
  edm::EDPutTokenT<pixelTrack::TrackSoAHost> tokenTrackCPU_;

  CAHitNtupletGeneratorOnGPU gpuAlgo_;
};

CAHitNtupletCUDA::CAHitNtupletCUDA(const edm::ParameterSet& iConfig)
    : onGPU_(iConfig.getParameter<bool>("onGPU")), tokenField_(esConsumes()), gpuAlgo_(iConfig, consumesCollector()) {
  if (onGPU_) {
    tokenHitGPU_ =
        consumes<cms::cuda::Product<TrackingRecHitSoADevice>>(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"));
    tokenTrackGPU_ = produces<cms::cuda::Product<pixelTrack::TrackSoADevice>>();
  } else {
    tokenHitCPU_ = consumes<TrackingRecHitSoAHost>(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"));
    tokenTrackCPU_ = produces<pixelTrack::TrackSoAHost>();
  }
}

void CAHitNtupletCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<bool>("onGPU", true);
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsPreSplittingCUDA"));

  CAHitNtupletGeneratorOnGPU::fillDescriptions(desc);
  descriptions.add("pixelTracksCUDA", desc);
}

void CAHitNtupletCUDA::beginJob() { gpuAlgo_.beginJob(); }

void CAHitNtupletCUDA::endJob() { gpuAlgo_.endJob(); }

void CAHitNtupletCUDA::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& es) const {
  auto bf = 1. / es.getData(tokenField_).inverseBzAtOriginInGeV();

  if (onGPU_) {
    edm::Handle<cms::cuda::Product<TrackingRecHitSoADevice>> hHits;
    iEvent.getByToken(tokenHitGPU_, hHits);
    // auto hHits = iEvent.getHandle(tokenHitGPU_);

    cms::cuda::ScopedContextProduce ctx{*hHits};
    auto& hits_d = ctx.get(*hHits);
    ctx.emplace(iEvent, tokenTrackGPU_, gpuAlgo_.makeTuplesAsync(hits_d, bf, ctx.stream()));
  } else {
    auto& hits_h = iEvent.get(tokenHitCPU_);
    iEvent.emplace(tokenTrackCPU_, gpuAlgo_.makeTuples(hits_h, bf));
  }
}

DEFINE_FWK_MODULE(CAHitNtupletCUDA);
