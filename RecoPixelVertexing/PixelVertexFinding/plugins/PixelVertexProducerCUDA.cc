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
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousDevice.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousHost.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoAHeterogeneousDevice.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoAHeterogeneousHost.h"

#include "gpuVertexFinder.h"

#undef PIXVERTEX_DEBUG_PRODUCE

class PixelVertexProducerCUDA : public edm::global::EDProducer<> {
public:
  explicit PixelVertexProducerCUDA(const edm::ParameterSet& iConfig);
  ~PixelVertexProducerCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produceOnGPU(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const;
  void produceOnCPU(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const;
  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  bool onGPU_;

  edm::EDGetTokenT<cms::cuda::Product<pixelTrack::TrackSoADevice>> tokenGPUTrack_;
  edm::EDPutTokenT<cms::cuda::Product<zVertex::ZVertexSoADevice>> tokenGPUVertex_;
  edm::EDGetTokenT<pixelTrack::TrackSoAHost> tokenCPUTrack_;
  edm::EDPutTokenT<zVertex::ZVertexSoAHost> tokenCPUVertex_;

  const gpuVertexFinder::Producer gpuAlgo_;

  // Tracking cuts before sending tracks to vertex algo
  const float ptMin_;
  const float ptMax_;
};

PixelVertexProducerCUDA::PixelVertexProducerCUDA(const edm::ParameterSet& conf)
    : onGPU_(conf.getParameter<bool>("onGPU")),
      gpuAlgo_(conf.getParameter<bool>("oneKernel"),
               conf.getParameter<bool>("useDensity"),
               conf.getParameter<bool>("useDBSCAN"),
               conf.getParameter<bool>("useIterative"),
               conf.getParameter<int>("minT"),
               conf.getParameter<double>("eps"),
               conf.getParameter<double>("errmax"),
               conf.getParameter<double>("chi2max")),
      ptMin_(conf.getParameter<double>("PtMin")),  // 0.5 GeV
      ptMax_(conf.getParameter<double>("PtMax"))   // 75. GeV
{
  if (onGPU_) {
    tokenGPUTrack_ =
        consumes<cms::cuda::Product<pixelTrack::TrackSoADevice>>(conf.getParameter<edm::InputTag>("pixelTrackSrc"));
    tokenGPUVertex_ = produces<cms::cuda::Product<zVertex::ZVertexSoADevice>>();
  } else {
    tokenCPUTrack_ = consumes<pixelTrack::TrackSoAHost>(conf.getParameter<edm::InputTag>("pixelTrackSrc"));
    tokenCPUVertex_ = produces<zVertex::ZVertexSoAHost>();
  }
}

void PixelVertexProducerCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // Only one of these three algos can be used at once.
  // Maybe this should become a Plugin Factory
  desc.add<bool>("onGPU", true);
  desc.add<bool>("oneKernel", true);
  desc.add<bool>("useDensity", true);
  desc.add<bool>("useDBSCAN", false);
  desc.add<bool>("useIterative", false);

  desc.add<int>("minT", 2);          // min number of neighbours to be "core"
  desc.add<double>("eps", 0.07);     // max absolute distance to cluster
  desc.add<double>("errmax", 0.01);  // max error to be "seed"
  desc.add<double>("chi2max", 9.);   // max normalized distance to cluster

  desc.add<double>("PtMin", 0.5);
  desc.add<double>("PtMax", 75.);
  desc.add<edm::InputTag>("pixelTrackSrc", edm::InputTag("pixelTracksCUDA"));

  auto label = "pixelVerticesCUDA";
  descriptions.add(label, desc);
}

void PixelVertexProducerCUDA::produceOnGPU(edm::StreamID streamID,
                                           edm::Event& iEvent,
                                           const edm::EventSetup& iSetup) const {
  edm::Handle<cms::cuda::Product<pixelTrack::TrackSoADevice>> hTracks;
  iEvent.getByToken(tokenGPUTrack_, hTracks);

  cms::cuda::ScopedContextProduce ctx{*hTracks};
  auto& tracks = ctx.get(*hTracks);

  ctx.emplace(iEvent, tokenGPUVertex_, gpuAlgo_.makeAsync(ctx.stream(), tracks.view(), ptMin_, ptMax_));
}

void PixelVertexProducerCUDA::produceOnCPU(edm::StreamID streamID,
                                           edm::Event& iEvent,
                                           const edm::EventSetup& iSetup) const {
  auto& tracks = iEvent.get(tokenCPUTrack_);

#ifdef PIXVERTEX_DEBUG_PRODUCE

  auto maxTracks = tracks.view().metadata().size();

  int32_t nt = 0;
  for (int32_t it = 0; it < maxTracks; ++it) {
    auto nHits = pixelTrack::utilities::nHits(tracks.view(), it);
    assert(nHits == int(tracks.view().hitIndices().size(it)));
    if (nHits == 0)
      break;  // this is a guard: maybe we need to move to nTracks...
    nt++;
  }
#endif  // PIXVERTEX_DEBUG_PRODUCE

  iEvent.emplace(tokenCPUVertex_, gpuAlgo_.make(tracks.view(), ptMin_, ptMax_));
}

void PixelVertexProducerCUDA::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  if (onGPU_) {
    produceOnGPU(streamID, iEvent, iSetup);
  } else {
    produceOnCPU(streamID, iEvent, iSetup);
  }
}

DEFINE_FWK_MODULE(PixelVertexProducerCUDA);
