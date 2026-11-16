// MaskHitCounter (masking_proof study, 2026-11-15): how many hits each CA pass removes from the next one.
// Reads the host copies of the cumulative CA hit masks (TrackingRecHitsMaskingHost; recHitMask = iteration index + 1 of the
// track that masked the hit, 0 = free; each mask starts from the previous one) and, per configured mask, counts the hits
// by mask value, split into pixel (row < number of pixel rechits) and OT rows. Prints per-event averages at endJob.
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsMaskingHost.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class MaskHitCounter : public edm::global::EDAnalyzer<> {
public:
  explicit MaskHitCounter(edm::ParameterSet const& cfg)
      : labels_(cfg.getParameter<std::vector<std::string>>("masks")),
        pixelToken_(consumes(cfg.getParameter<edm::InputTag>("pixelRecHits"))) {
    for (auto const& l : labels_)
      tokens_.push_back(consumes<reco::TrackingRecHitsMaskingHost>(edm::InputTag(l)));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<std::string>>("masks", {});
    desc.add<edm::InputTag>("pixelRecHits", edm::InputTag("siPixelRecHits"));
    descriptions.addWithDefaultLabel(desc);
  }

  void analyze(edm::StreamID, edm::Event const& ev, edm::EventSetup const&) const override {
    uint64_t const nPix = ev.get(pixelToken_).dataSize();
    std::vector<std::map<uint32_t, std::pair<uint64_t, uint64_t>>> counts(labels_.size());
    std::vector<uint64_t> sizes(labels_.size());
    for (size_t i = 0; i < labels_.size(); ++i) {
      auto const& mask = ev.get(tokens_[i]);
      auto const view = mask.const_view();
      int const n = view.metadata().size();
      sizes[i] = n;
      for (int h = 0; h < n; ++h) {
        auto& c = counts[i][view[h].recHitMask()];
        (uint64_t(h) < nPix ? c.first : c.second) += 1;
      }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    ++nEvents_;
    nPixTot_ += nPix;
    for (size_t i = 0; i < labels_.size(); ++i) {
      sizeTot_[i] += sizes[i];
      for (auto const& [v, c] : counts[i]) {
        tot_[i][v].first += c.first;
        tot_[i][v].second += c.second;
      }
    }
  }

  void beginJob() override {
    tot_.resize(labels_.size());
    sizeTot_.assign(labels_.size(), 0);
  }

  void endJob() override {
    if (nEvents_ == 0)
      return;
    double const ne = nEvents_;
    edm::LogPrint out("MaskHitCounter");
    out << "MaskHitCounter: " << nEvents_ << " events, per-event averages; pixel rechits " << nPix(ne) << "\n";
    for (size_t i = 0; i < labels_.size(); ++i) {
      double const all = sizeTot_[i] / ne, pix = nPixTot_ / ne;
      out << "  mask " << labels_[i] << ": rows " << all << " (pixel " << pix << ", OT " << all - pix << ")\n";
      for (auto const& [v, c] : tot_[i])
        out << "    value " << v << (v == 0 ? " (free)" : "") << ": pixel " << c.first / ne << " (" << 100. * c.first / ne / pix
            << "%)  OT " << c.second / ne << " (" << 100. * c.second / ne / (all - pix) << "%)  all " << (c.first + c.second) / ne
            << " (" << 100. * (c.first + c.second) / ne / all << "%)\n";
    }
  }

private:
  double nPix(double ne) const { return nPixTot_ / ne; }
  std::vector<std::string> const labels_;
  std::vector<edm::EDGetTokenT<reco::TrackingRecHitsMaskingHost>> tokens_;
  edm::EDGetTokenT<SiPixelRecHitCollection> const pixelToken_;
  mutable std::mutex mutex_;
  mutable uint64_t nEvents_ = 0, nPixTot_ = 0;
  mutable std::vector<std::map<uint32_t, std::pair<uint64_t, uint64_t>>> tot_;
  mutable std::vector<uint64_t> sizeTot_;
};

DEFINE_FWK_MODULE(MaskHitCounter);
