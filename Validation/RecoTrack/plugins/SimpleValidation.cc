// system include files
#include <memory>
#include <string>
#include <iostream>

#include "TTree.h"
#include "TFile.h"


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

#include "SimTracker/Common/interface/TrackingParticleSelector.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.

using reco::TrackCollection;

class SimpleValidation : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit SimpleValidation(const edm::ParameterSet&);
  ~SimpleValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  int global_rt_ = 0;
  int global_at_ = 0;
  int global_st_ = 0;
  int global_dt_ = 0;
  int global_ast_ = 0;

  TrackingParticleSelector tpSelector;
  TTree* output_tree_;
  std::vector<edm::InputTag> trackLabels_;
  edm::EDGetTokenT<ClusterTPAssociation> tpMap_;
//   edm::EDGetTokenT<std::vector<PileupSummaryInfo>>  infoPileUp_;
  std::vector<edm::EDGetTokenT<edm::View<reco::Track>>> trackTokens_;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> trackAssociatorToken_;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleToken_;

//   const double sharingFraction_;
//   const double sharingFractionForTriplets_;


};

SimpleValidation::SimpleValidation(const edm::ParameterSet& iConfig)
    : trackLabels_(iConfig.getParameter<std::vector<edm::InputTag>>("trackLabels")),
      // tpMap_(consumes(iConfig.getParameter<edm::InputTag>("tpMap"))),
    //   infoPileUp_(consumes(iConfig.getParameter< edm::InputTag >("infoPileUp"))),
      trackAssociatorToken_(consumes<reco::TrackToTrackingParticleAssociator>(iConfig.getUntrackedParameter<edm::InputTag>("trackAssociator"))),
      trackingParticleToken_(consumes<TrackingParticleCollection>(iConfig.getParameter< edm::InputTag >("trackingParticles")))
    //   sharingFraction_(iConfig.getUntrackedParameter<double>("sharingFraction")),
    //   sharingFractionForTriplets_(iConfig.getUntrackedParameter<double>("sharingFractionForTriplets"))
{

  for (auto& itag : trackLabels_) {
    trackTokens_.push_back(consumes<edm::View<reco::Track>>(itag));
    // edm::LogPrint("TrackValidator") << itag.label() << "\n";
  }
  tpSelector = TrackingParticleSelector(iConfig.getParameter<double>("ptMinTP"),
                                        iConfig.getParameter<double>("ptMaxTP"),
                                        iConfig.getParameter<double>("minRapidityTP"),
                                        iConfig.getParameter<double>("maxRapidityTP"),
                                        iConfig.getParameter<double>("tipTP"),
                                        iConfig.getParameter<double>("lipTP"),
                                        iConfig.getParameter<int>("minHitTP"),
                                        iConfig.getParameter<bool>("signalOnlyTP"),
                                        iConfig.getParameter<bool>("intimeOnlyTP"),
                                        iConfig.getParameter<bool>("chargedOnlyTP"),
                                        iConfig.getParameter<bool>("stableOnlyTP"),
                                        iConfig.getParameter<std::vector<int>>("pdgIdTP"),
                                        iConfig.getParameter<bool>("invertRapidityCutTP"),
                                        iConfig.getParameter<double>("minPhi"),
                                        iConfig.getParameter<double>("maxPhi"));
  //now do what ever initialization is needed
}

SimpleValidation::~SimpleValidation() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
  // if (trackLabels_[0].label().compare("pixelTracks0") == 0) {
  //   std::cerr << "pixelTracks" << "\n"
  //             << "Total Simulated "<< global_st_ << "\n"
  //             << "Total Reconstructed " << global_rt_ << "\n"
  //             << "Total Associated (recoToSim) " << global_at_ << "\n"
  //             << "Total Fakes " << global_rt_ - global_at_ << "\n"
  //             << "Total Associated (simRoReco) " << global_ast_ << "\n"
  //             << "Total Duplicated " << global_dt_ << "\n";
  // }
}

//
// member functions
//

// ------------ method called for each event  ------------
void SimpleValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

//   auto const& tpClust = iEvent.get(tpMap_);
  auto const& associatorByHits = iEvent.get(trackAssociatorToken_);
  
  TrackingParticleRefVector tpCollection;
  TrackingParticleRefVector selectedTPCollection;
  edm::Handle<TrackingParticleCollection> TPCollectionH;
  iEvent.getByToken(trackingParticleToken_, TPCollectionH);
//   auto const& tp = iEvent.get(trackingParticleToken_);

  for (size_t i = 0, size = TPCollectionH->size(); i < size; ++i) {
    auto tp = TrackingParticleRef(TPCollectionH, i);
    tpCollection.push_back(tp);
    if (tpSelector(*tp)) {
      selectedTPCollection.push_back(tp);
    }
  }
  
  for (const auto& trackToken : trackTokens_) 
  {

    edm::Handle<edm::View<reco::Track>> tracksHandle;
    iEvent.getByToken(trackToken, tracksHandle);
    const edm::View<reco::Track>& tracks = *tracksHandle;
    
    edm::RefToBaseVector<reco::Track> trackRefs;
    for (edm::View<reco::Track>::size_type i = 0; i < tracks.size(); ++i) {
        trackRefs.push_back(tracks.refAt(i));
    }

    reco::RecoToSimCollection recSimColl = associatorByHits.associateRecoToSim(trackRefs, tpCollection);
    reco::SimToRecoCollection simRecColl = associatorByHits.associateSimToReco(trackRefs, selectedTPCollection);

    int rt = 0;
    int at = 0;
    int ast = 0;
    int dt = 0;
    int st = selectedTPCollection.size();
    for (const auto& track : trackRefs) {
      rt++;
      auto foundTP = recSimColl.find(track);
        if (foundTP != recSimColl.end()) {
          const auto& tp = foundTP->val;
          if (!tp.empty()) {
            at++;
          }
          if (simRecColl.find(tp[0].first) != simRecColl.end()) {
            if (simRecColl[tp[0].first].size() > 1) {
              dt++;
            }
          }
      }
    }
    for (const TrackingParticleRef& tpr : selectedTPCollection) {
      auto foundTrack = simRecColl.find(tpr);
      if (foundTrack != simRecColl.end() && !simRecColl[tpr].empty()) {
          ast++;	      
      }
    }
    // if (trackLabels_[0].label().compare("pixelTracks0") == 0) {
    //   LogPrint("TrackValidator") << "Tag " << trackLabels_[0].label() << "\n"
    //                             << "Total Simulated "<< st << "\n"
    //                             << "Total Reconstructed " << rt << "\n"
    //                             << "Total Associated (recoToSim) " << at << "\n"
    //                             << "Total Fakes " << rt - at << "\n"
    //                             << "Total Associated (simRoReco) " << ast << "\n"
    //                             << "Total Duplicated " << dt << "\n";
    // }
    global_rt_ += rt;
    global_st_ += st;
    global_at_ += at;
    global_dt_ += dt;
    global_ast_ += ast;
  }
}

// ------------ method called once each job just before starting event loop  ------------
void SimpleValidation::beginJob() {
  // please remove this method if not needed
  edm::Service<TFileService> fs;
  output_tree_ = fs->make<TTree>("output", "putput params");

  output_tree_->Branch("rt", &global_rt_);
  output_tree_->Branch("at", &global_at_);
  output_tree_->Branch("st", &global_st_);
  output_tree_->Branch("dt", &global_dt_);
  output_tree_->Branch("ast", &global_ast_);
}

// ------------ method called once each job just after ending the event loop  ------------
void SimpleValidation::endJob() {
  // please remove this method if not needed
  output_tree_->Fill();
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SimpleValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);

  //Specify that only 'tracks' is allowed
  //To use, remove the default given above and uncomment below
  //ParameterSetDescription desc;
  //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
  //descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SimpleValidation);
