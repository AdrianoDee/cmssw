// #include <fstream>
// #include <filesystem>
// #include <iostream>
// #include <stdexcept>
// #include <vector>
// #include <cstring>
// #include <cassert>
// #include <cmath>

// #include "FWCore/Framework/interface/Event.h"
// #include "FWCore/Framework/interface/EventSetup.h"
// #include "FWCore/Framework/interface/MakerMacros.h"
// #include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
// #include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
// #include "FWCore/Utilities/interface/Exception.h"

// #include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDAnalyzer.h"
// #include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
// #include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"

// namespace ALPAKA_ACCELERATOR_NAMESPACE {

//   class TrackingRecHitHostBinVerifier : public stream::EDAnalyzer<> {
//   public:
//     explicit TrackingRecHitHostBinVerifier(const edm::ParameterSet& iConfig);
//     ~TrackingRecHitHostBinVerifier() override = default;

//     static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

//   private:
//     void analyze(device::Event const& iEvent, device::EventSetup const& iSetup) override;

//     const std::filesystem::path inputFile_;
//     bool checked_ = false;
//   };

//   // ------------------- Constructor -------------------
//   TrackingRecHitHostBinVerifier::TrackingRecHitHostBinVerifier(const edm::ParameterSet& iConfig)
//       : EDAnalyzer(iConfig),
//         inputFile_(iConfig.getParameter<std::string>("inputFile")) {
// #ifdef GPU_DEBUG
//     std::cout << "[TrackingRecHitHostBinVerifier] Verifying file: " << inputFile_ << std::endl;
// #endif
//   }

//   // ------------------- Analyze (run once) -------------------
//   void TrackingRecHitHostBinVerifier::analyze(device::Event const&, device::EventSetup const&) {
//     if (checked_)
//       return;
//     checked_ = true;

//     std::ifstream in(inputFile_, std::ios::binary);
//     if (!in.is_open()) {
//       throw cms::Exception("FileOpenError") << "Cannot open file " << inputFile_ << " for verification.";
//     }

//     constexpr uint32_t expectedEndianness = 0x01020304;
//     constexpr uint32_t expectedVersion = 1;
//     constexpr char expectedMagic[4] = {'T', 'R', 'H', '1'};

//     // ---- Header ----
//     char magic[4];
//     in.read(magic, 4);
//     assert(in.good() && "Failed to read magic");
//     assert(std::memcmp(magic, expectedMagic, 4) == 0 && "Magic tag mismatch (not TRH1)");

//     uint32_t version = 0;
//     in.read(reinterpret_cast<char*>(&version), sizeof(version));
//     assert(in.good() && "Failed to read version");
//     assert(version == expectedVersion && "Unsupported file version");

//     uint32_t endianMarker = 0;
//     in.read(reinterpret_cast<char*>(&endianMarker), sizeof(endianMarker));
//     assert(in.good() && "Failed to read endianness marker");
//     assert(endianMarker == expectedEndianness && "Endianness mismatch");

//     uint32_t headerEvents = 0;
//     in.read(reinterpret_cast<char*>(&headerEvents), sizeof(headerEvents));
//     assert(in.good() && "Failed to read header event count placeholder");
//     // Dumper currently writes 0 here; we just check it's not garbage.
// #ifdef GPU_DEBUG
//     std::cout << "[Verifier] Header OK. Placeholder event count = "
//               << headerEvents << " (ignored).\n";
// #endif

//     uint32_t nEventsRead = 0;
//     uint64_t totalHits = 0;

//     auto read_vec = [&](auto& vec, uint32_t nHits) {
//       using ElemT = typename std::remove_reference_t<decltype(vec)>::value_type;
//       vec.resize(nHits);
//       in.read(reinterpret_cast<char*>(vec.data()), nHits * sizeof(ElemT));
//       assert(in.good() && "I/O error reading column");
//     };

//     // ---- Loop over events until EOF ----
//     while (true) {
//       uint32_t nHits = 0, nModules = 0;

//       in.read(reinterpret_cast<char*>(&nHits), sizeof(nHits));
//       if (!in) {
//         // clean EOF: we tried to read nHits past the last event
//         assert(in.eof() && "I/O error while reading nHits");
//         break;
//       }

//       in.read(reinterpret_cast<char*>(&nModules), sizeof(nModules));
//       assert(in.good() && "I/O error while reading nModules");
//       assert(nHits > 0 && nModules > 0 && "Invalid nHits or nModules");

// #ifdef GPU_DEBUG
//       std::cout << "[Verifier] Event " << nEventsRead << ": "
//                 << nHits << " hits, " << nModules << " modules\n";
// #endif

//       // moduleStart
//       std::vector<uint32_t> moduleStart(nModules + 1);
//       in.read(reinterpret_cast<char*>(moduleStart.data()),
//               (nModules + 1) * sizeof(uint32_t));
//       assert(in.good() && "I/O error reading moduleStart");

//       assert(moduleStart.front() == 0 && "moduleStart[0] must be 0");
//       assert(moduleStart.back() == nHits && "moduleStart last must equal nHits");
//       for (uint32_t i = 1; i <= nModules; ++i) {
//         assert(moduleStart[i] >= moduleStart[i - 1] && "moduleStart must be non-decreasing");
//       }

//       // Columns
//       std::vector<float>   xLocal, yLocal, xerrLocal, yerrLocal;
//       std::vector<float>   xGlobal, yGlobal, zGlobal, rGlobal;
//       std::vector<int16_t> iphi, clusterSizeX, clusterSizeY;
//       std::vector<uint32_t> chargeAndStatus;
//       std::vector<uint16_t> detectorIndex;

//       read_vec(xLocal,        nHits);
//       read_vec(yLocal,        nHits);
//       read_vec(xerrLocal,     nHits);
//       read_vec(yerrLocal,     nHits);
//       read_vec(xGlobal,       nHits);
//       read_vec(yGlobal,       nHits);
//       read_vec(zGlobal,       nHits);
//       read_vec(rGlobal,       nHits);
//       read_vec(iphi,          nHits);
//       read_vec(chargeAndStatus, nHits);
//       read_vec(clusterSizeX,  nHits);
//       read_vec(clusterSizeY,  nHits);
//       read_vec(detectorIndex, nHits);

//       // ---- Basic consistency checks ----

//       // 1) detectorIndex in range
//       for (uint32_t i = 0; i < nHits; ++i) {
//         assert(detectorIndex[i] < nModules && "detectorIndex out of range");
//       }

//       // 2) rGlobal ≈ sqrt(x^2 + y^2)
//       for (uint32_t i = 0; i < nHits; ++i) {
//         float r2 = std::sqrt(xGlobal[i] * xGlobal[i] + yGlobal[i] * yGlobal[i]);
//         // allow small numerical tolerance
//         assert(std::fabs(r2 - rGlobal[i]) < 1e-2f && "rGlobal inconsistent with xGlobal,yGlobal");
//       }

//       // 3) moduleStart bins make sense with detectorIndex sorting
//       //    (this assumes hits are sorted by detectorIndex, as in your converter)
//       for (uint32_t m = 0; m < nModules; ++m) {
//         uint32_t begin = moduleStart[m];
//         uint32_t end   = moduleStart[m + 1];
//         for (uint32_t i = begin; i < end; ++i) {
//           assert(detectorIndex[i] == m && "Hits not grouped by module as expected");
//         }
//       }

// #ifdef GPU_DEBUG
//       if (nHits > 0) {
//         std::cout << "  First hit x,y,z,r = "
//                   << xGlobal[0] << ", "
//                   << yGlobal[0] << ", "
//                   << zGlobal[0] << ", "
//                   << rGlobal[0]
//                   << " detIdx=" << detectorIndex[0] << std::endl;
//       }
// #endif

//       ++nEventsRead;
//       totalHits += nHits;
//     }

//     in.close();

// #ifdef GPU_DEBUG
//     std::cout << "[Verifier] ✅ Verification done: "
//               << nEventsRead << " events, "
//               << totalHits << " total hits.\n";
// #endif

//     // You can add a final assert if you expect at least one event:
//     assert(nEventsRead > 0 && "No events read from file");
//   }

//   // ------------------- fillDescriptions -------------------
//   void TrackingRecHitHostBinVerifier::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
//     edm::ParameterSetDescription desc;
//     desc.add<std::string>("inputFile", "TrackingRecHitHost.bin")
//         ->setComment("Binary file to verify (produced by TrackingRecHitHostDumper)");
//     descriptions.addWithDefaultLabel(desc);
//   }

// }  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// DEFINE_FWK_ALPAKA_MODULE(TrackingRecHitHostBinVerifier);
