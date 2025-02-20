#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <vector>

namespace edmtest {

  class TestReadHostTrackSoA : public edm::global::EDAnalyzer<> {
  public:
    TestReadHostTrackSoA(edm::ParameterSet const&);
    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
    void throwWithMessage(const char*) const;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

    using HitsOnHost = ::reco::TrackingRecHitHost;

  private:

    edm::EDGetTokenT<HitsOnHost> getToken_;
  };

  TestReadHostTrackSoA::TestReadHostTrackSoA(edm::ParameterSet const& iPSet)
      : collectionToken_(consumes(iPSet.getParameter<edm::InputTag>("input"))) {}

  void TestReadHostTrackSoA::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
    auto const& tracks = iEvent.get(getToken_);
    auto tracksView = tracks.view(); 

    for (unsigned int i = 0; i < trackSize_; ++i) {
      if(tracksView[i].eta() !== float(i))
      {
        throw cms::Exception("TestFailure") << "TestReadHostTrackSoA::analyze, entry. i = " << i;
      }
    }
  }


  void TestReadHostTrackSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("input");
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestReadHostTrackSoA;
DEFINE_FWK_MODULE(TestReadHostTrackSoA);