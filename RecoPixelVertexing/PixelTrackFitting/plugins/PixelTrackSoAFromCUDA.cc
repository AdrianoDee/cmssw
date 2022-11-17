#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/Common/interface/HostProduct.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousHost.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousDevice.h"
#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

// Switch on to enable checks and printout for found tracks
// #define PIXEL_DEBUG_PRODUCE

class PixelTrackSoAFromCUDA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit PixelTrackSoAFromCUDA(const edm::ParameterSet& iConfig);
  ~PixelTrackSoAFromCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  edm::EDGetTokenT<cms::cuda::Product<pixelTrack::TrackSoADevice>> tokenCUDA_;
  edm::EDPutTokenT<pixelTrack::TrackSoAHost> tokenSOA_;

  pixelTrack::TrackSoAHost tracks_h;
};

PixelTrackSoAFromCUDA::PixelTrackSoAFromCUDA(const edm::ParameterSet& iConfig)
    : tokenCUDA_(consumes<cms::cuda::Product<pixelTrack::TrackSoADevice>>(iConfig.getParameter<edm::InputTag>("src"))),
      tokenSOA_(produces<pixelTrack::TrackSoAHost>()) {}

void PixelTrackSoAFromCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("pixelTracksCUDA"));
  descriptions.add("pixelTracksSoA", desc);
}

void PixelTrackSoAFromCUDA::acquire(edm::Event const& iEvent,
                                    edm::EventSetup const& iSetup,
                                    edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::Product<pixelTrack::TrackSoADevice> const& inputDataWrapped = iEvent.get(tokenCUDA_);
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& tracks_d = ctx.get(inputDataWrapped);   // Tracks on device
  tracks_h = pixelTrack::TrackSoAHost(ctx.stream());  // Create an instance of Tracks on Host, using the stream
  cudaCheck(cudaMemcpyAsync(tracks_h.buffer().get(),
                            tracks_d.const_buffer().get(),
                            tracks_d.bufferSize(),
                            cudaMemcpyDeviceToHost,
                            ctx.stream()));  // Copy data from Device to Host
  cudaCheck(cudaGetLastError());
}

void PixelTrackSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  // check that the fixed-size SoA does not overflow
  auto maxTracks = tracks_h.view().metadata().size();
  auto nTracks = tracks_h.view().nTracks();
  assert(nTracks < maxTracks);

  if (nTracks == maxTracks - 1) {
    edm::LogWarning("PixelTracks") << "Unsorted reconstructed pixel tracks truncated to " << maxTracks - 1
                                   << " candidates";
  }

#ifdef PIXEL_DEBUG_PRODUCE
  std::cout << " stride " << maxTracks << std::endl;
  std::cout << "found " << nTracks << std::endl;

  int32_t nt = 0;
  for (int32_t it = 0; it < maxTracks; ++it) {
    auto nHits = pixelTrack::utilities::nHits(tracks_h.view(), it);
    assert(nHits == int(tracks_h.view().hitIndices().size(it)));
    if (nHits == 0)
      break;  // this is a guard: maybe we need to move to nTracks...
    nt++;
  }
  assert(nTracks == nt);
#endif
  // DO NOT  make a copy  (actually TWO....)
  iEvent.emplace(tokenSOA_, std::move(tracks_h));
  assert(!tracks_h.buffer());
}

DEFINE_FWK_MODULE(PixelTrackSoAFromCUDA);
