#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/Product.h"
#include "DataFormats/Track/interface/alpaka/TrackSoADevice.h"
#include "DataFormats/Track/interface/TrackSoAHost.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "HeterogeneousCore/AlpakaCore/interface/ScopedContext.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ProducerBase.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "CAHitNtupletGeneratorOnGPU.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  template <typename TrackerTraits>
  class CAHitNtupletCUDAT : public stream::EDProducer<> {
    using HitsConstView = TrackingRecHitAlpakaSoAConstView<TrackerTraits>;
    using HitsOnDevice = TrackingRecHitAlpakaDevice<TrackerTraits>;
    using HitsOnHost = TrackingRecHitAlpakaHost<TrackerTraits>;

    using TkSoAHost = TrackSoAHost<TrackerTraits>;
    using TkSoADevice = TrackSoADevice<TrackerTraits>;

    using GPUAlgo = CAHitNtupletGeneratorOnGPU<TrackerTraits>;

  public:
    explicit CAHitNtupletCUDAT(const edm::ParameterSet& iConfig);
    ~CAHitNtupletCUDAT() override = default;
    void produce(device::Event& iEvent, const device::EventSetup& es) override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    bool onGPU_;

    edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tokenField_;
    device::EDGetToken<HitsOnDevice> tokenHit_;
    device::EDPutToken<TkSoADevice> tokenTrack_;

    GPUAlgo gpuAlgo_;
  };

  template <typename TrackerTraits>
  CAHitNtupletCUDAT<TrackerTraits>::CAHitNtupletCUDAT(const edm::ParameterSet& iConfig)
      : onGPU_(iConfig.getParameter<bool>("onGPU")), tokenField_(esConsumes()), gpuAlgo_(iConfig, consumesCollector()) {
    tokenHit_ = consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"));
    tokenTrack_ = produces();
  }

  template <typename TrackerTraits>
  void CAHitNtupletCUDAT<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<bool>("onGPU", true);
    desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsPreSplittingCUDA"));

    GPUAlgo::fillDescriptions(desc);
    descriptions.addWithDefaultLabel(desc);
  }

  template <typename TrackerTraits>
  void CAHitNtupletCUDAT<TrackerTraits>::produce(device::Event& iEvent, const device::EventSetup& es) {
    auto bf = 1. / es.getData(tokenField_).inverseBzAtOriginInGeV();

    auto const& hits = iEvent.get(tokenHit_);

    iEvent.emplace(tokenTrack_, gpuAlgo_.makeTuplesAsync(hits, bf, iEvent.queue()));
  }

  using CAHitNtupletCUDA = CAHitNtupletCUDAT<pixelTopology::Phase1>;
  using CAHitNtupletCUDAPhase1 = CAHitNtupletCUDAT<pixelTopology::Phase1>;
  using CAHitNtupletCUDAPhase2 = CAHitNtupletCUDAT<pixelTopology::Phase2>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"

DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletCUDA);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletCUDAPhase1);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletCUDAPhase2);
