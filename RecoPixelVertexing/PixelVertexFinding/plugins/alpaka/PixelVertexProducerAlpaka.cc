#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/Product.h"
#include "DataFormats/TrackerCommon/interface/SimplePixelTopology.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "HeterogeneousCore/AlpakaCore/interface/module_backend_config.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"

#include "DataFormats/Track/interface/alpaka/TrackSoADevice.h"
#include "DataFormats/Vertex/interface/alpaka/ZVertexSoADevice.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"

#include "gpuVertexFinder.h"

#undef PIXVERTEX_DEBUG_PRODUCE

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  template <typename TrackerTraits>
  class PixelVertexProducerCUDAT : public global::EDProducer<> {
    using TkSoADevice = TrackSoADevice<TrackerTraits>;
    using GPUAlgo = gpuVertexFinder::Producer<TrackerTraits>;

  public:
    explicit PixelVertexProducerCUDAT(const edm::ParameterSet& iConfig);
    ~PixelVertexProducerCUDAT() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produceOnGPU(edm::StreamID streamID,  // maybe even remove this and leave only produce?
                      device::Event& iEvent,
                      const device::EventSetup& iSetup) const;
    void produce(edm::StreamID streamID, device::Event& iEvent, const device::EventSetup& iSetup) const override;

    bool onGPU_;

    device::EDGetToken<TkSoADevice> tokenGPUTrack_;
    device::EDPutToken<ZVertexDevice> tokenGPUVertex_;

    const GPUAlgo gpuAlgo_;

    // Tracking cuts before sending tracks to vertex algo
    const float ptMin_;
    const float ptMax_;
  };

  template <typename TrackerTraits>
  PixelVertexProducerCUDAT<TrackerTraits>::PixelVertexProducerCUDAT(const edm::ParameterSet& conf)
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
    tokenGPUTrack_ = consumes(conf.getParameter<edm::InputTag>("pixelTrackSrc"));
    tokenGPUVertex_ = produces();
  }

  template <typename TrackerTraits>
  void PixelVertexProducerCUDAT<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
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

    descriptions.addWithDefaultLabel(desc);
  }

  template <typename TrackerTraits>
  void PixelVertexProducerCUDAT<TrackerTraits>::produceOnGPU(edm::StreamID streamID,
                                                             device::Event& iEvent,
                                                             const device::EventSetup& iSetup) const {
    using TracksSoA = TrackSoADevice<TrackerTraits>;
    auto const& hTracks = iEvent.get(tokenGPUTrack_);

    iEvent.emplace(tokenGPUVertex_, gpuAlgo_.makeAsync(iEvent.queue(), hTracks.view(), ptMin_, ptMax_));
  }

  template <typename TrackerTraits>
  void PixelVertexProducerCUDAT<TrackerTraits>::produce(edm::StreamID streamID,
                                                        device::Event& iEvent,
                                                        const device::EventSetup& iSetup) const {
    produceOnGPU(streamID, iEvent, iSetup);
  }

  using PixelVertexProducerCUDA = PixelVertexProducerCUDAT<pixelTopology::Phase1>;
  using PixelVertexProducerCUDAPhase1 = PixelVertexProducerCUDAT<pixelTopology::Phase1>;
  using PixelVertexProducerCUDAPhase2 = PixelVertexProducerCUDAT<pixelTopology::Phase2>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(PixelVertexProducerCUDA);
DEFINE_FWK_ALPAKA_MODULE(PixelVertexProducerCUDAPhase1);
DEFINE_FWK_ALPAKA_MODULE(PixelVertexProducerCUDAPhase2);