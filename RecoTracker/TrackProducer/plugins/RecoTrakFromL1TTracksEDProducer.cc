#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackFromFTS.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

using namespace reco;
using namespace std;
using namespace edm;

class dso_hidden RecoTrackFromL1TTracksEDProducer : public edm::global::EDProducer<> {
public:
  RecoTrackFromL1TTracksEDProducer(const edm::ParameterSet& cfg);
  ~RecoTrackFromL1TTracksEDProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>> theInputCollectionTag_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;

};

RecoTrackFromL1TTracksEDProducer::RecoTrackFromL1TTracksEDProducer(const edm::ParameterSet& cfg)
    : theInputCollectionTag_(
               consumes<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>(cfg.getParameter<edm::InputTag>("InputCollection"))),
      mfToken_{esConsumes<MagneticField, IdealMagneticFieldRecord>()}
{
  produces< std::vector<TransientTrack> >();
}

void RecoTrackFromL1TTracksEDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputCollection", {"TTTracksFromTrackletEmulation", "Level1TTTracks"});
  descriptions.addWithDefaultLabel(desc);
}


void RecoTrackFromL1TTracksEDProducer::produce(edm::StreamID, edm::Event& ev, const edm::EventSetup& es) const {
  std::unique_ptr<std::vector<TransientTrack>> result(new std::vector<TransientTrack>());

  // TTrack Collection
  auto const& trks = ev.get(theInputCollectionTag_);

  // Mag field
  const auto& mag = es.getData(mfToken_);

  for (auto const& l1 : trks) {
    FreeTrajectoryState fts = trajectoryStateTransform::initialFreeStateL1TTrack(l1, &mag, true);
    auto ttt = TransientTrack(new TransientTrackFromFTS(fts));
    result->emplace_back(ttt);

    std::cout << "prod" << std::endl;

  }  // end loop over L1Tracks

  ev.put(std::move(result));
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTrackFromL1TTracksEDProducer);
