#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"

#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"

#include "Validation/RecoTrack/interface/ParticleHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "Validation/RecoTrack/interface/SimpleMapHost.h"

#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace {
constexpr uint32_t kExpectedEndianness = 0x01020304;
constexpr uint32_t kFormatVersion      = 1;
constexpr char kMagicPAR[4] = {'P','A','R','1'};
constexpr char kMagicTRH[4] = {'T','R','H','1'};
constexpr char kMagicMAP[4] = {'M','A','P','1'};
}

// =====================================================================================

class ParticleHitSoADumper : public edm::one::EDProducer<> {
public:
  explicit ParticleHitSoADumper(const edm::ParameterSet&);
  ~ParticleHitSoADumper() override = default;

  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

private:
  // Tokens
  edm::EDGetTokenT<TrackingParticleCollection> tpToken_;
  edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHitToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> stripRphiRecHitToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> stripStereoRecHitToken_;
  edm::EDGetTokenT<SiStripMatchedRecHit2DCollection> stripMatchedRecHitToken_;
  edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> phase2OTRecHitToken_;
  edm::EDGetTokenT<ClusterTPAssociation> clusterToTPMapToken_;

  bool dumpHits_;
  std::string fileParticles_, fileHits_, fileMap_;

  std::ofstream outParticles_, outHits_, outMap_;
  uint32_t nEventsWritten_ = 0;

  void writeHeader(std::ofstream&, const char magic[4]);
  void patchEventCount(std::ofstream&);
  void dumpParticles(const TrackingParticleCollection&);
  void dumpHitsAndMap(const edm::Event&, const edm::EventSetup&);

  template <typename T>
  inline void write_column(std::ofstream& out, const T* data, size_t n) {
    out.write(reinterpret_cast<const char*>(data), n * sizeof(T));
  }
};

// =====================================================================================

ParticleHitSoADumper::ParticleHitSoADumper(const edm::ParameterSet& cfg)
    : tpToken_(consumes<TrackingParticleCollection>(cfg.getParameter<edm::InputTag>("srcParticles"))),
      dumpHits_(cfg.getParameter<bool>("dumpHits")),
      fileParticles_(cfg.getParameter<std::string>("fileParticles")),
      fileHits_(cfg.getParameter<std::string>("fileHits")),
      fileMap_(cfg.getParameter<std::string>("fileMap")) {

  const edm::InputTag pixelTag       = cfg.getParameter<edm::InputTag>("pixelRecHits");
  const edm::InputTag stripRphiTag   = cfg.getParameter<edm::InputTag>("stripRphiRecHits");
  const edm::InputTag stripStereoTag = cfg.getParameter<edm::InputTag>("stripStereoRecHits");
  const edm::InputTag stripMatchedTag= cfg.getParameter<edm::InputTag>("stripMatchedRecHits");
  const edm::InputTag phase2Tag      = cfg.getParameter<edm::InputTag>("phase2OTRecHits");
  const edm::InputTag clusterTPTag   = cfg.getParameter<edm::InputTag>("clusterTPMap");

  bool anyHitCollection = false;

  if (dumpHits_) {
    if (!pixelTag.label().empty()) {
      pixelRecHitToken_ = consumes<SiPixelRecHitCollection>(pixelTag);
      anyHitCollection = true;
    }
    if (!stripRphiTag.label().empty()) {
      stripRphiRecHitToken_ = consumes<SiStripRecHit2DCollection>(stripRphiTag);
      anyHitCollection = true;
    }
    if (!stripStereoTag.label().empty()) {
      stripStereoRecHitToken_ = consumes<SiStripRecHit2DCollection>(stripStereoTag);
      anyHitCollection = true;
    }
    if (!stripMatchedTag.label().empty()) {
      stripMatchedRecHitToken_ = consumes<SiStripMatchedRecHit2DCollection>(stripMatchedTag);
      anyHitCollection = true;
    }
    if (!phase2Tag.label().empty()) {
      phase2OTRecHitToken_ = consumes<Phase2TrackerRecHit1DCollectionNew>(phase2Tag);
      anyHitCollection = true;
    }
    if (!clusterTPTag.label().empty())
      clusterToTPMapToken_ = consumes<ClusterTPAssociation>(clusterTPTag);

    if (!anyHitCollection)
      throw cms::Exception("Configuration")
          << "[ParticleHitSoADumper] dumpHits=True but no hit collections configured.";
  }

  // Open output files and write headers
  outParticles_.open(fileParticles_, std::ios::binary);
  if (!outParticles_) throw cms::Exception("FileError") << "Cannot open " << fileParticles_;
  writeHeader(outParticles_, kMagicPAR);

  if (dumpHits_) {
    outHits_.open(fileHits_, std::ios::binary);
    outMap_.open(fileMap_, std::ios::binary);
    if (!outHits_ || !outMap_)
      throw cms::Exception("FileError") << "Cannot open hit or map file";
    writeHeader(outHits_, kMagicTRH);
    writeHeader(outMap_, kMagicMAP);
  }
}

// -------------------------------------------------------------------------------------

void ParticleHitSoADumper::writeHeader(std::ofstream& out, const char magic[4]) {
  out.write(magic, 4);
  out.write(reinterpret_cast<const char*>(&kFormatVersion), sizeof(kFormatVersion));
  out.write(reinterpret_cast<const char*>(&kExpectedEndianness), sizeof(kExpectedEndianness));
  int32_t nEventsPlaceholder = -1;
  out.write(reinterpret_cast<const char*>(&nEventsPlaceholder), sizeof(nEventsPlaceholder));
}

// -------------------------------------------------------------------------------------

void ParticleHitSoADumper::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<TrackingParticleCollection> tpH;
  iEvent.getByToken(tpToken_, tpH);
  if (!tpH.isValid()) return;

  ++nEventsWritten_;
  dumpParticles(*tpH);
  if (dumpHits_) dumpHitsAndMap(iEvent, iSetup);
}

// -------------------------------------------------------------------------------------

void ParticleHitSoADumper::dumpParticles(const TrackingParticleCollection& tps) {
  uint32_t n = tps.size();
  outParticles_.write(reinterpret_cast<const char*>(&n), sizeof(n));

  sim::ParticleHost host(static_cast<int>(n), cms::alpakatools::host());
  auto v = host.view();

  for (uint32_t i = 0; i < n; ++i) {
    const auto& tp = tps[i];
    v.vx()[i] = tp.vx();
    v.vy()[i] = tp.vy();
    v.vz()[i] = tp.vz();
    v.px()[i] = tp.px();
    v.py()[i] = tp.py();
    v.pz()[i] = tp.pz();
    v.energy()[i] = tp.energy();
    v.pt()[i] = tp.pt();
    v.eta()[i] = tp.eta();
    v.phi()[i] = tp.phi();
    v.mass()[i] = tp.mass();
    v.charge()[i] = static_cast<int16_t>(tp.threeCharge() / 3);
    v.pdgID()[i] = tp.pdgId();
    v.partInd()[i] = i;
  }

//   auto W = [&](auto& col) { write_column(outParticles_, col, n); };

  write_column(outParticles_, v.vx(),      n);
  write_column(outParticles_, v.vy(),      n);
  write_column(outParticles_, v.vz(),      n);

  write_column(outParticles_, v.px(),      n);
  write_column(outParticles_, v.py(),      n);
  write_column(outParticles_, v.pz(),      n);
  write_column(outParticles_, v.energy(),  n);

  write_column(outParticles_, v.pt(),      n);
  write_column(outParticles_, v.eta(),     n);
  write_column(outParticles_, v.phi(),     n);
  write_column(outParticles_, v.mass(),    n);

  write_column(outParticles_, v.charge(),  n);
  write_column(outParticles_, v.pdgID(),   n);
  write_column(outParticles_, v.partInd(), n);

//   W(v.vx()); W(v.vy()); W(v.vz());
//   W(v.px()); W(v.py()); W(v.pz()); W(v.energy());
//   W(v.pt()); W(v.eta()); W(v.phi()); W(v.mass());
//   W(v.charge()); W(v.pdgID()); W(v.partInd());
}

// -------------------------------------------------------------------------------------

void ParticleHitSoADumper::dumpHitsAndMap(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<SiPixelRecHitCollection> pixelHits;
  edm::Handle<SiStripRecHit2DCollection> stripRphiHits, stripStereoHits;
  edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
  edm::Handle<Phase2TrackerRecHit1DCollectionNew> phase2Hits;
  edm::Handle<ClusterTPAssociation> clusterTPMapH;

  if (!pixelRecHitToken_.isUninitialized()) iEvent.getByToken(pixelRecHitToken_, pixelHits);
  if (!stripRphiRecHitToken_.isUninitialized()) iEvent.getByToken(stripRphiRecHitToken_, stripRphiHits);
  if (!stripStereoRecHitToken_.isUninitialized()) iEvent.getByToken(stripStereoRecHitToken_, stripStereoHits);
  if (!stripMatchedRecHitToken_.isUninitialized()) iEvent.getByToken(stripMatchedRecHitToken_, matchedHits);
  if (!phase2OTRecHitToken_.isUninitialized()) iEvent.getByToken(phase2OTRecHitToken_, phase2Hits);
  if (!clusterToTPMapToken_.isUninitialized()) iEvent.getByToken(clusterToTPMapToken_, clusterTPMapH);

  const auto& clusterToTPMap = *clusterTPMapH;

  std::vector<const BaseTrackerRecHit*> hits;
  if (pixelHits.isValid()) for (auto const& d : *pixelHits) for (auto const& h : d) hits.push_back(&h);
  if (stripRphiHits.isValid()) for (auto const& d : *stripRphiHits) for (auto const& h : d) hits.push_back(&h);
  if (stripStereoHits.isValid()) for (auto const& d : *stripStereoHits) for (auto const& h : d) hits.push_back(&h);
  if (matchedHits.isValid()) for (auto const& d : *matchedHits) for (auto const& h : d) hits.push_back(&h);
  if (phase2Hits.isValid()) for (auto const& d : *phase2Hits) for (auto const& h : d) hits.push_back(&h);

  uint32_t nHits = hits.size();
  uint32_t nModules = 0;
  outHits_.write(reinterpret_cast<const char*>(&nHits), 4);
  outHits_.write(reinterpret_cast<const char*>(&nModules), 4);
  uint32_t dummy[1] = {0};
  outHits_.write(reinterpret_cast<const char*>(dummy), sizeof(dummy));

  reco::TrackingRecHitHost host(cms::alpakatools::host(), nHits, nModules);
  auto view = host.view<reco::TrackingRecHitSoA>();

  // Also fill map
  SimpleMapHost mapHost(static_cast<int>(nHits), cms::alpakatools::host());
  auto mapView = mapHost.view();

  for (uint32_t i = 0; i < nHits; ++i) {
    const auto* h = hits[i];
    auto gp = h->globalPosition();
    view.xLocal()[i] = h->localPosition().x();
    view.yLocal()[i] = h->localPosition().y();
    view.xerrLocal()[i] = h->localPositionError().xx();
    view.yerrLocal()[i] = h->localPositionError().yy();
    view.xGlobal()[i] = gp.x();
    view.yGlobal()[i] = gp.y();
    view.zGlobal()[i] = gp.z();
    view.rGlobal()[i] = gp.perp();
    view.detectorIndex()[i] = h->geographicalId().rawId();

    mapView.id()[i] = std::numeric_limits<uint32_t>::max();

    // Try truth association
    const OmniClusterRef& cluster = h->firstClusterRef();
    auto range = clusterToTPMap.equal_range(cluster);
    if (range.first != range.second) {
      const TrackingParticleRef& tpRef = range.first->second;
      mapView.id()[i] = tpRef.key();
    }
  }

  // Write both SoAs
//   auto W = [&](auto& out, auto& col) { write_column(out, col.data(), nHits); };

  write_column(outHits_, view.xLocal(),    nHits);
  write_column(outHits_, view.yLocal(),    nHits);
  write_column(outHits_, view.xerrLocal(), nHits);

  write_column(outHits_, view.yerrLocal(), nHits);
  write_column(outHits_, view.xGlobal(),   nHits);
  write_column(outHits_, view.yGlobal(),   nHits);
  write_column(outHits_, view.zGlobal(),   nHits);

  write_column(outHits_, view.rGlobal(),   nHits);
  write_column(outHits_, view.iphi(),      nHits);
  write_column(outHits_, view.chargeAndStatus(), nHits);
  write_column(outHits_, view.clusterSizeX(),    nHits);

  write_column(outHits_, view.clusterSizeY(),  nHits);
  write_column(outHits_, view.detectorIndex(), nHits);

  write_column(outMap_, mapView.id(), nHits);

}

// -------------------------------------------------------------------------------------

void ParticleHitSoADumper::patchEventCount(std::ofstream& out) {
  out.seekp(12);
  out.write(reinterpret_cast<const char*>(&nEventsWritten_), sizeof(int32_t));
}

void ParticleHitSoADumper::endJob() {
  if (outParticles_) { patchEventCount(outParticles_); outParticles_.close(); }
  if (dumpHits_) {
    if (outHits_) { patchEventCount(outHits_); outHits_.close(); }
    if (outMap_)  { patchEventCount(outMap_);  outMap_.close(); }
  }
  edm::LogInfo("ParticleHitSoADumper")
      << "Wrote " << nEventsWritten_ << " events to binary outputs.";
}

DEFINE_FWK_MODULE(ParticleHitSoADumper);
