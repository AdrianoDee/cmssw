#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousDevice.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousHost.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoAHeterogeneousHost.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoAHeterogeneousDevice.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

class PixelTrackDumpCUDA : public edm::global::EDAnalyzer<> {
public:
  explicit PixelTrackDumpCUDA(const edm::ParameterSet& iConfig);
  ~PixelTrackDumpCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::StreamID streamID, edm::Event const& iEvent, const edm::EventSetup& iSetup) const override;
  const bool m_onGPU;
  // GPU
  edm::EDGetTokenT<cms::cuda::Product<pixelTrack::TrackSoADevice>> tokenGPUTrack_;
  edm::EDGetTokenT<cms::cuda::Product<zVertex::ZVertexSoADevice>> tokenGPUVertex_;

  // CPU
  edm::EDGetTokenT<pixelTrack::TrackSoAHost> tokenSoATrack_;
  edm::EDGetTokenT<zVertex::ZVertexSoAHost> tokenSoAVertex_;
};

PixelTrackDumpCUDA::PixelTrackDumpCUDA(const edm::ParameterSet& iConfig)
    : m_onGPU(iConfig.getParameter<bool>("onGPU")) {
  if (m_onGPU) {
    tokenGPUTrack_ =
        consumes<cms::cuda::Product<pixelTrack::TrackSoADevice>>(iConfig.getParameter<edm::InputTag>("pixelTrackSrc"));
    tokenGPUVertex_ =
        consumes<cms::cuda::Product<zVertex::ZVertexSoADevice>>(iConfig.getParameter<edm::InputTag>("pixelVertexSrc"));
  } else {
    tokenSoATrack_ = consumes<pixelTrack::TrackSoAHost>(iConfig.getParameter<edm::InputTag>("pixelTrackSrc"));
    tokenSoAVertex_ = consumes<zVertex::ZVertexSoAHost>(iConfig.getParameter<edm::InputTag>("pixelVertexSrc"));
  }
}

void PixelTrackDumpCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<bool>("onGPU", true);
  desc.add<edm::InputTag>("pixelTrackSrc", edm::InputTag("pixelTracksCUDA"));
  desc.add<edm::InputTag>("pixelVertexSrc", edm::InputTag("pixelVerticesCUDA"));
  descriptions.add("pixelTrackDumpCUDA", desc);
}

void PixelTrackDumpCUDA::analyze(edm::StreamID streamID,
                                 edm::Event const& iEvent,
                                 const edm::EventSetup& iSetup) const {
  if (m_onGPU) {
    auto const& hTracks = iEvent.get(tokenGPUTrack_);
    cms::cuda::ScopedContextProduce ctx{hTracks};

    auto const& tracks = ctx.get(hTracks);
    auto const* tsoa = &tracks;
    assert(tsoa);

    auto const& vertices = ctx.get(iEvent.get(tokenGPUVertex_));
    auto const* vsoa = &vertices;
    assert(vsoa);

  } else {
    auto const& tsoa = iEvent.get(tokenSoATrack_);
    assert(tsoa.buffer());

    auto const& vsoa = iEvent.get(tokenSoAVertex_);
    assert(vsoa.buffer());
  }
}

DEFINE_FWK_MODULE(PixelTrackDumpCUDA);
