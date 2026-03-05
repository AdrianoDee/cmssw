// TrackComparisonAnalyzer.cc
// Compares two track collections to debug efficiency differences
// between CA Extension (P-hits only) and Stubs approaches
//
// Key difference:
// - CAExt: pixel hits + P-hits from TOB PS barrel layers 1-2 only
// - Stubs: pixel hits + both inner/outer hits from all OT stubs

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TH1F.h"
#include "TH2F.h"

class TrackComparisonAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit TrackComparisonAnalyzer(const edm::ParameterSet& iConfig);
  ~TrackComparisonAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // Match tracks by kinematics
  bool tracksMatchByKinematics(const reco::Track& t1, const reco::Track& t2) const;

  // Match tracks by shared pixel hits
  int countSharedPixelHits(const reco::Track& t1, const reco::Track& t2) const;

  // Get pixel hit DetIds for a track
  std::set<uint32_t> getPixelHitDetIds(const reco::Track& trk) const;

  // Tokens
  edm::EDGetTokenT<reco::TrackCollection> tracksCAExtToken_;
  edm::EDGetTokenT<reco::TrackCollection> tracksStubsToken_;

  // Matching parameters
  double matchDPtRel_;
  double matchDEta_;
  double matchDPhi_;
  int minSharedPixelHits_;

  // Output file for detailed track dump
  std::string outputFileName_;
  std::ofstream outputFile_;

  // Statistics
  uint64_t nEvents_ = 0;
  uint64_t nTracksCAExt_ = 0;
  uint64_t nTracksStubs_ = 0;
  uint64_t nMatchedTracks_ = 0;
  uint64_t nOnlyCAExt_ = 0;
  uint64_t nOnlyStubs_ = 0;

  // Eta-binned statistics
  // Bins: |eta| < 0.8 (barrel), 0.8-1.5 (transition), 1.5-2.5 (endcap)
  std::map<std::string, uint64_t> etaBinsMatched_;
  std::map<std::string, uint64_t> etaBinsOnlyCAExt_;
  std::map<std::string, uint64_t> etaBinsOnlyStubs_;

  // Histograms
  TH1F* h_nTracksCAExt_;
  TH1F* h_nTracksStubs_;
  TH1F* h_nMatched_;
  TH1F* h_nOnlyCAExt_;
  TH1F* h_nOnlyStubs_;

  TH1F* h_etaOnlyCAExt_;
  TH1F* h_ptOnlyCAExt_;
  TH1F* h_phiOnlyCAExt_;
  TH1F* h_nHitsOnlyCAExt_;

  TH1F* h_etaOnlyStubs_;
  TH1F* h_ptOnlyStubs_;
  TH1F* h_phiOnlyStubs_;
  TH1F* h_nHitsOnlyStubs_;

  TH1F* h_etaMatched_;
  TH1F* h_ptMatched_;

  TH2F* h_etaPt_OnlyCAExt_;
  TH2F* h_etaPt_OnlyStubs_;
  TH2F* h_etaPt_Matched_;
};

TrackComparisonAnalyzer::TrackComparisonAnalyzer(const edm::ParameterSet& iConfig)
    : tracksCAExtToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracksCAExtension"))),
      tracksStubsToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracksStubs"))),
      matchDPtRel_(iConfig.getParameter<double>("matchDPtRel")),
      matchDEta_(iConfig.getParameter<double>("matchDEta")),
      matchDPhi_(iConfig.getParameter<double>("matchDPhi")),
      minSharedPixelHits_(iConfig.getParameter<int>("minSharedPixelHits")),
      outputFileName_(iConfig.getParameter<std::string>("outputFileName")) {
  usesResource(TFileService::kSharedResource);

  // Initialize eta bin counters
  for (const auto& bin : {"barrel", "transition", "endcap"}) {
    etaBinsMatched_[bin] = 0;
    etaBinsOnlyCAExt_[bin] = 0;
    etaBinsOnlyStubs_[bin] = 0;
  }
}

void TrackComparisonAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tracksCAExtension", edm::InputTag("hltPhase2PixelTracksCAExtension"));
  desc.add<edm::InputTag>("tracksStubs", edm::InputTag("hltPhase2PixelTracksWithStubs"));
  desc.add<double>("matchDPtRel", 0.1);   // 10% relative pt difference
  desc.add<double>("matchDEta", 0.02);    // delta eta
  desc.add<double>("matchDPhi", 0.02);    // delta phi
  desc.add<int>("minSharedPixelHits", 3); // minimum shared pixel hits to consider a match
  desc.add<std::string>("outputFileName", "track_comparison.txt");
  descriptions.addWithDefaultLabel(desc);
}

void TrackComparisonAnalyzer::beginJob() {
  // Open output file
  if (!outputFileName_.empty()) {
    outputFile_.open(outputFileName_);
    outputFile_ << "# Track Comparison: CAExtension vs Stubs\n";
    outputFile_ << "# Format: EventID Category pt eta phi nHits chi2 [hit_detIds...]\n";
    outputFile_ << "# Categories: MATCHED, ONLY_CAEXT, ONLY_STUBS\n";
    outputFile_ << "#\n";
  }

  // Book histograms
  edm::Service<TFileService> fs;

  h_nTracksCAExt_ = fs->make<TH1F>("nTracksCAExt", "Number of CAExt tracks per event;N tracks;Events", 100, 0, 200);
  h_nTracksStubs_ = fs->make<TH1F>("nTracksStubs", "Number of Stubs tracks per event;N tracks;Events", 100, 0, 200);
  h_nMatched_ = fs->make<TH1F>("nMatched", "Number of matched tracks per event;N tracks;Events", 100, 0, 200);
  h_nOnlyCAExt_ = fs->make<TH1F>("nOnlyCAExt", "Number of CAExt-only tracks per event;N tracks;Events", 50, 0, 50);
  h_nOnlyStubs_ = fs->make<TH1F>("nOnlyStubs", "Number of Stubs-only tracks per event;N tracks;Events", 50, 0, 50);

  h_etaOnlyCAExt_ = fs->make<TH1F>("etaOnlyCAExt", "Eta of CAExt-only tracks;#eta;Tracks", 50, -2.5, 2.5);
  h_ptOnlyCAExt_ = fs->make<TH1F>("ptOnlyCAExt", "pT of CAExt-only tracks;p_{T} [GeV];Tracks", 50, 0, 50);
  h_phiOnlyCAExt_ = fs->make<TH1F>("phiOnlyCAExt", "Phi of CAExt-only tracks;#phi;Tracks", 50, -M_PI, M_PI);
  h_nHitsOnlyCAExt_ = fs->make<TH1F>("nHitsOnlyCAExt", "nHits of CAExt-only tracks;N hits;Tracks", 20, 0, 20);

  h_etaOnlyStubs_ = fs->make<TH1F>("etaOnlyStubs", "Eta of Stubs-only tracks;#eta;Tracks", 50, -2.5, 2.5);
  h_ptOnlyStubs_ = fs->make<TH1F>("ptOnlyStubs", "pT of Stubs-only tracks;p_{T} [GeV];Tracks", 50, 0, 50);
  h_phiOnlyStubs_ = fs->make<TH1F>("phiOnlyStubs", "Phi of Stubs-only tracks;#phi;Tracks", 50, -M_PI, M_PI);
  h_nHitsOnlyStubs_ = fs->make<TH1F>("nHitsOnlyStubs", "nHits of Stubs-only tracks;N hits;Tracks", 20, 0, 20);

  h_etaMatched_ = fs->make<TH1F>("etaMatched", "Eta of matched tracks;#eta;Tracks", 50, -2.5, 2.5);
  h_ptMatched_ = fs->make<TH1F>("ptMatched", "pT of matched tracks;p_{T} [GeV];Tracks", 50, 0, 50);

  h_etaPt_OnlyCAExt_ =
      fs->make<TH2F>("etaPt_OnlyCAExt", "CAExt-only tracks;#eta;p_{T} [GeV]", 50, -2.5, 2.5, 50, 0, 50);
  h_etaPt_OnlyStubs_ =
      fs->make<TH2F>("etaPt_OnlyStubs", "Stubs-only tracks;#eta;p_{T} [GeV]", 50, -2.5, 2.5, 50, 0, 50);
  h_etaPt_Matched_ = fs->make<TH2F>("etaPt_Matched", "Matched tracks;#eta;p_{T} [GeV]", 50, -2.5, 2.5, 50, 0, 50);
}

std::set<uint32_t> TrackComparisonAnalyzer::getPixelHitDetIds(const reco::Track& trk) const {
  std::set<uint32_t> pixelDetIds;
  for (auto hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++hit) {
    if ((*hit)->isValid()) {
      DetId detId = (*hit)->geographicalId();
      // Only pixel hits (barrel or endcap)
      if (detId.subdetId() == PixelSubdetector::PixelBarrel || detId.subdetId() == PixelSubdetector::PixelEndcap) {
        pixelDetIds.insert(detId.rawId());
      }
    }
  }
  return pixelDetIds;
}

int TrackComparisonAnalyzer::countSharedPixelHits(const reco::Track& t1, const reco::Track& t2) const {
  auto detIds1 = getPixelHitDetIds(t1);
  auto detIds2 = getPixelHitDetIds(t2);

  int nShared = 0;
  for (const auto& id : detIds1) {
    if (detIds2.find(id) != detIds2.end()) {
      nShared++;
    }
  }
  return nShared;
}

bool TrackComparisonAnalyzer::tracksMatchByKinematics(const reco::Track& t1, const reco::Track& t2) const {
  float ptRel = std::abs(t1.pt() - t2.pt()) / t1.pt();
  float dEta = std::abs(t1.eta() - t2.eta());
  float dPhi = std::abs(reco::deltaPhi(t1.phi(), t2.phi()));

  return (ptRel < matchDPtRel_ && dEta < matchDEta_ && dPhi < matchDPhi_);
}

void TrackComparisonAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto const& tracksCAExt = iEvent.get(tracksCAExtToken_);
  auto const& tracksStubs = iEvent.get(tracksStubsToken_);

  nEvents_++;
  nTracksCAExt_ += tracksCAExt.size();
  nTracksStubs_ += tracksStubs.size();

  h_nTracksCAExt_->Fill(tracksCAExt.size());
  h_nTracksStubs_->Fill(tracksStubs.size());

  // Track matching status
  std::vector<bool> caExtMatched(tracksCAExt.size(), false);
  std::vector<bool> stubsMatched(tracksStubs.size(), false);
  std::vector<int> caExtToStubsMatch(tracksCAExt.size(), -1);  // index of matching stubs track

  // Match tracks - prefer shared pixel hits, fall back to kinematics
  for (size_t i = 0; i < tracksCAExt.size(); ++i) {
    const auto& trkCAExt = tracksCAExt[i];
    int bestMatch = -1;
    int bestSharedHits = 0;

    for (size_t j = 0; j < tracksStubs.size(); ++j) {
      if (stubsMatched[j])
        continue;

      const auto& trkStubs = tracksStubs[j];

      // Try to match by shared pixel hits first
      int nShared = countSharedPixelHits(trkCAExt, trkStubs);
      if (nShared >= minSharedPixelHits_ && nShared > bestSharedHits) {
        bestMatch = j;
        bestSharedHits = nShared;
      }
      // If no good hit match, try kinematics
      else if (bestMatch < 0 && tracksMatchByKinematics(trkCAExt, trkStubs)) {
        bestMatch = j;
      }
    }

    if (bestMatch >= 0) {
      caExtMatched[i] = true;
      stubsMatched[bestMatch] = true;
      caExtToStubsMatch[i] = bestMatch;
    }
  }

  // Count and categorize
  int nMatchedEvent = 0;
  int nOnlyCAExtEvent = 0;
  int nOnlyStubsEvent = 0;

  auto getEtaBin = [](float eta) -> std::string {
    float absEta = std::abs(eta);
    if (absEta < 0.8)
      return "barrel";
    else if (absEta < 1.5)
      return "transition";
    else
      return "endcap";
  };

  // Process matched and CAExt-only tracks
  for (size_t i = 0; i < tracksCAExt.size(); ++i) {
    const auto& trk = tracksCAExt[i];
    std::string etaBin = getEtaBin(trk.eta());

    if (caExtMatched[i]) {
      nMatchedEvent++;
      nMatchedTracks_++;
      etaBinsMatched_[etaBin]++;
      h_etaMatched_->Fill(trk.eta());
      h_ptMatched_->Fill(trk.pt());
      h_etaPt_Matched_->Fill(trk.eta(), trk.pt());

      // Write matched track to file
      if (outputFile_.is_open()) {
        outputFile_ << iEvent.id().event() << " MATCHED " << std::fixed << std::setprecision(3) << trk.pt() << " "
                    << trk.eta() << " " << trk.phi() << " " << trk.numberOfValidHits() << " "
                    << trk.normalizedChi2();
        // Write pixel hit DetIds
        for (auto hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++hit) {
          if ((*hit)->isValid()) {
            outputFile_ << " " << (*hit)->geographicalId().rawId();
          }
        }
        outputFile_ << "\n";
      }
    } else {
      nOnlyCAExtEvent++;
      nOnlyCAExt_++;
      etaBinsOnlyCAExt_[etaBin]++;
      h_etaOnlyCAExt_->Fill(trk.eta());
      h_ptOnlyCAExt_->Fill(trk.pt());
      h_phiOnlyCAExt_->Fill(trk.phi());
      h_nHitsOnlyCAExt_->Fill(trk.numberOfValidHits());
      h_etaPt_OnlyCAExt_->Fill(trk.eta(), trk.pt());

      // Write CAExt-only track to file with hit details
      if (outputFile_.is_open()) {
        outputFile_ << iEvent.id().event() << " ONLY_CAEXT " << std::fixed << std::setprecision(3) << trk.pt() << " "
                    << trk.eta() << " " << trk.phi() << " " << trk.numberOfValidHits() << " "
                    << trk.normalizedChi2();
        for (auto hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++hit) {
          if ((*hit)->isValid()) {
            auto gp = (*hit)->globalPosition();
            outputFile_ << " [" << (*hit)->geographicalId().rawId() << ":" << gp.perp() << "," << gp.z() << "]";
          }
        }
        outputFile_ << "\n";
      }
    }
  }

  // Process Stubs-only tracks
  for (size_t j = 0; j < tracksStubs.size(); ++j) {
    if (!stubsMatched[j]) {
      const auto& trk = tracksStubs[j];
      std::string etaBin = getEtaBin(trk.eta());

      nOnlyStubsEvent++;
      nOnlyStubs_++;
      etaBinsOnlyStubs_[etaBin]++;
      h_etaOnlyStubs_->Fill(trk.eta());
      h_ptOnlyStubs_->Fill(trk.pt());
      h_phiOnlyStubs_->Fill(trk.phi());
      h_nHitsOnlyStubs_->Fill(trk.numberOfValidHits());
      h_etaPt_OnlyStubs_->Fill(trk.eta(), trk.pt());

      // Write Stubs-only track to file
      if (outputFile_.is_open()) {
        outputFile_ << iEvent.id().event() << " ONLY_STUBS " << std::fixed << std::setprecision(3) << trk.pt() << " "
                    << trk.eta() << " " << trk.phi() << " " << trk.numberOfValidHits() << " "
                    << trk.normalizedChi2();
        for (auto hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++hit) {
          if ((*hit)->isValid()) {
            auto gp = (*hit)->globalPosition();
            outputFile_ << " [" << (*hit)->geographicalId().rawId() << ":" << gp.perp() << "," << gp.z() << "]";
          }
        }
        outputFile_ << "\n";
      }
    }
  }

  h_nMatched_->Fill(nMatchedEvent);
  h_nOnlyCAExt_->Fill(nOnlyCAExtEvent);
  h_nOnlyStubs_->Fill(nOnlyStubsEvent);

  // Print per-event summary
  edm::LogPrint("TrackComparison") << "Event " << iEvent.id().event() << ": CAExt=" << tracksCAExt.size()
                                   << " Stubs=" << tracksStubs.size() << " Matched=" << nMatchedEvent
                                   << " OnlyCAExt=" << nOnlyCAExtEvent << " OnlyStubs=" << nOnlyStubsEvent;
}

void TrackComparisonAnalyzer::endJob() {
  // Close output file
  if (outputFile_.is_open()) {
    outputFile_.close();
  }

  // Print summary
  edm::LogPrint("TrackComparison") << "\n";
  edm::LogPrint("TrackComparison") << "======================================================================";
  edm::LogPrint("TrackComparison") << "Track Comparison Summary: CAExtension (P-hits) vs Stubs";
  edm::LogPrint("TrackComparison") << "======================================================================";
  edm::LogPrint("TrackComparison") << "Events analyzed: " << nEvents_;
  edm::LogPrint("TrackComparison") << "";
  edm::LogPrint("TrackComparison") << "Total tracks:";
  edm::LogPrint("TrackComparison") << "  CAExtension: " << nTracksCAExt_ << " (" << (float)nTracksCAExt_ / nEvents_
                                   << " per event)";
  edm::LogPrint("TrackComparison") << "  Stubs:       " << nTracksStubs_ << " (" << (float)nTracksStubs_ / nEvents_
                                   << " per event)";
  edm::LogPrint("TrackComparison") << "";
  edm::LogPrint("TrackComparison") << "Matching results:";
  edm::LogPrint("TrackComparison") << "  Matched:     " << nMatchedTracks_;
  edm::LogPrint("TrackComparison") << "  OnlyCAExt:   " << nOnlyCAExt_ << " (MISSED by stubs)";
  edm::LogPrint("TrackComparison") << "  OnlyStubs:   " << nOnlyStubs_ << " (EXTRA in stubs)";
  edm::LogPrint("TrackComparison") << "";

  // Efficiency by eta region
  edm::LogPrint("TrackComparison") << "Efficiency by eta region (tracks MISSED by stubs):";
  for (const auto& bin : {"barrel", "transition", "endcap"}) {
    uint64_t matched = etaBinsMatched_[bin];
    uint64_t missed = etaBinsOnlyCAExt_[bin];
    uint64_t total = matched + missed;
    if (total > 0) {
      float eff = 100.0f * matched / total;
      float missRate = 100.0f * missed / total;
      edm::LogPrint("TrackComparison") << "  " << std::setw(12) << bin << ": " << matched << "/" << total << " = "
                                       << std::fixed << std::setprecision(1) << eff << "% efficiency (" << missed
                                       << " missed, " << missRate << "%)";
    }
  }

  edm::LogPrint("TrackComparison") << "";
  edm::LogPrint("TrackComparison") << "Note: CAExt uses P-hits from TOB PS barrel layers 1-2 only";
  edm::LogPrint("TrackComparison") << "      Stubs uses inner+outer hits from all OT stubs";
  edm::LogPrint("TrackComparison") << "======================================================================";
}

DEFINE_FWK_MODULE(TrackComparisonAnalyzer);
