#include <alpaka/alpaka.hpp>


#include "DataFormats/TrackSoA/interface/TrackHostSoA.h"
#include "DataFormats/TrackSoA/interface/alpaka/TrackSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitSoACollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoLocalTracker/Records/interface/PixelCPEFastParamsRecord.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/alpaka/PixelCPEFastParamsCollection.h"

#include "RecoLocalTracker/Records/interface/FrameSoARecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/alpaka/FrameSoACollection.h"

#include "CAHitNtupletGenerator.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {



  template <typename TrackerTraits>
  class CAHitNtupletAlpaka : public stream::EDProducer<> {
    using HitsConstView = TrackingRecHitAlpakaSoAConstView<TrackerTraits>;
    using HitsOnDevice = TrackingRecHitAlpakaCollection<TrackerTraits>;
    using HitsOnHost = TrackingRecHitHost<TrackerTraits>;

    using TkSoAHost = TrackHostSoA<TrackerTraits>;
    using TkSoADevice = TrackSoACollection<TrackerTraits>;

    using Algo = CAHitNtupletGenerator<TrackerTraits>;

  public:
    explicit CAHitNtupletAlpaka(const edm::ParameterSet& iConfig);
    ~CAHitNtupletAlpaka() override = default;
    void produce(device::Event& iEvent, const device::EventSetup& es) override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
 
    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tokenField_;
    // const device::ESGetToken<PixelCPEFastParams<pixelTopology::base_traits_t<TrackerTraits>>, PixelCPEFastParamsRecord> cpeToken_; 
    const device::ESGetToken<FrameSoACollection, FrameSoARecord> frameToken_; 
    const device::EDGetToken<HitsOnDevice> tokenHit_;
    const device::EDPutToken<TkSoADevice> tokenTrack_;

    Algo algo_;
  };

  template <typename TrackerTraits>
  CAHitNtupletAlpaka<TrackerTraits>::CAHitNtupletAlpaka(const edm::ParameterSet& iConfig)
      : tokenField_(esConsumes()),
        frameToken_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("frameSoA")))),
        tokenHit_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
        tokenTrack_(produces()),
        algo_(iConfig) {

  }

  template <typename TrackerTraits>
  void CAHitNtupletAlpaka<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsPreSplittingAlpaka"));

    std::string frame = "FrameSoA";
    frame += TrackerTraits::nameModifier;
    desc.add<std::string>("frameSoA", frame);

    Algo::fillPSetDescription(desc);
    descriptions.addWithDefaultLabel(desc);
  }

  template <typename TrackerTraits>
  void CAHitNtupletAlpaka<TrackerTraits>::produce(device::Event& iEvent, const device::EventSetup& es) {
    auto bf = 1. / es.getData(tokenField_).inverseBzAtOriginInGeV();

    auto const& frame = es.getData(frameToken_);

    auto const& hits = iEvent.get(tokenHit_);

    iEvent.emplace(tokenTrack_, algo_.makeTuplesAsync(hits, frame, bf, iEvent.queue()));

  }

  using CAHitNtupletAlpakaPhase1 = CAHitNtupletAlpaka<pixelTopology::Phase1>;
  using CAHitNtupletAlpakaPhase2 = CAHitNtupletAlpaka<pixelTopology::Phase2>;
  using CAHitNtupletAlpakaHIonPhase1 = CAHitNtupletAlpaka<pixelTopology::HIonPhase1>;
  using CAHitNtupletAlpakaPhase1Strip = CAHitNtupletAlpaka<pixelTopology::Phase1Strip>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"

DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase1);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase2);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaHIonPhase1);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase1Strip);

