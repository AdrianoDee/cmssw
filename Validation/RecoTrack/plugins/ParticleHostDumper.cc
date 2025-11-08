#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

#include "Validation/RecoTrack/interface/ParticleHost.h"

#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <iomanip>

namespace {
  // Binary format constants — same as particleReader
  constexpr uint32_t kExpectedEndianness = 0x01020304;
  constexpr uint32_t kFormatVersion      = 1;
  constexpr char     kMagic[4]           = {'P', 'A', 'R', '1'};
}

class ParticleHostBinaryDumper : public edm::one::EDProducer<> {
public:
  explicit ParticleHostBinaryDumper(const edm::ParameterSet&);
  ~ParticleHostBinaryDumper() override = default;

  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

private:
  edm::EDGetTokenT<TrackingParticleCollection> tpToken_;
  std::string outputFileName_;

  std::ofstream out_;
  uint32_t nEventsWritten_ = 0;

  // utility to write vector-like spans
  template <typename T>
  void write_column(const T* data, size_t n) {
    out_.write(reinterpret_cast<const char*>(data), n * sizeof(T));
  }
};

ParticleHostBinaryDumper::ParticleHostBinaryDumper(const edm::ParameterSet& iConfig)
    : tpToken_(consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      outputFileName_(iConfig.getParameter<std::string>("fileName")) {

  // open and write header (we’ll patch nEvents at end)
  out_.open(outputFileName_, std::ios::binary);
  if (!out_.is_open())
    throw cms::Exception("FileError") << "Cannot open output file " << outputFileName_;

  out_.write(kMagic, 4);
  out_.write(reinterpret_cast<const char*>(&kFormatVersion), sizeof(kFormatVersion));
  out_.write(reinterpret_cast<const char*>(&kExpectedEndianness), sizeof(kExpectedEndianness));

  // reserve space for nEvents (patch later)
  int32_t nEventsPlaceholder = -1;
  out_.write(reinterpret_cast<const char*>(&nEventsPlaceholder), sizeof(nEventsPlaceholder));

  edm::LogInfo("ParticleHostBinaryDumper") << "Writing binary output to " << outputFileName_;
}

void ParticleHostBinaryDumper::produce(edm::Event& iEvent, const edm::EventSetup&) {
  edm::Handle<TrackingParticleCollection> handle;
  iEvent.getByToken(tpToken_, handle);
  if (!handle.isValid()) {
    edm::LogWarning("ParticleHostBinaryDumper") << "Missing TrackingParticleCollection";
    return;
  }

  const auto& tps = *handle;
  ++nEventsWritten_;

  uint32_t nParticles = tps.size();
  out_.write(reinterpret_cast<const char*>(&nParticles), sizeof(nParticles));

  // Fill a sim::ParticleHost on host
  sim::ParticleHost host(int(nParticles), cms::alpakatools::host());
  auto view = host.view();

  // Fill the SoA from the TrackingParticleCollection
  for (uint32_t i = 0; i < nParticles; ++i) {
    const auto& tp = tps[i];

    view.vx()[i] = tp.vx();
    view.vy()[i] = tp.vy();
    view.vz()[i] = tp.vz();

    view.px()[i] = tp.px();
    view.py()[i] = tp.py();
    view.pz()[i] = tp.pz();
    view.energy()[i] = tp.energy();

    view.pt()[i] = tp.pt();
    view.eta()[i] = tp.eta();
    view.phi()[i] = tp.phi();
    view.mass()[i] = tp.mass();

    view.charge()[i] = static_cast<int16_t>(tp.threeCharge() / 3); // convert back to ±1 convention
    view.pdgID()[i] = static_cast<int32_t>(tp.pdgId());
    view.partInd()[i] = i; // local index within event
  }

  // write all columns to file (same order as reader)
  write_column(view.vx(),      nParticles);
  write_column(view.vy(),      nParticles);
  write_column(view.vz(),      nParticles);

  write_column(view.px(),      nParticles);
  write_column(view.py(),      nParticles);
  write_column(view.pz(),      nParticles);
  write_column(view.energy(),  nParticles);

  write_column(view.pt(),      nParticles);
  write_column(view.eta(),     nParticles);
  write_column(view.phi(),     nParticles);
  write_column(view.mass(),    nParticles);

  write_column(view.charge(),  nParticles);
  write_column(view.pdgID(),   nParticles);
  write_column(view.partInd(), nParticles);
}

void ParticleHostBinaryDumper::endJob() {
  if (!out_.is_open())
    return;

  // Patch the nEvents field at offset = 12 bytes (4 + 4 + 4)
  out_.seekp(12);
  out_.write(reinterpret_cast<const char*>(&nEventsWritten_), sizeof(int32_t));

  out_.close();

  edm::LogInfo("ParticleHostBinaryDumper")
      << "Closed file " << outputFileName_ << " after writing " << nEventsWritten_ << " events.";
}

DEFINE_FWK_MODULE(ParticleHostBinaryDumper);
