// Test analyzer for StackedModuleGeometry validation
// Reads geometry from EventSetup and prints statistics

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/Record/interface/StackedModuleGeometryRecord.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometryHost.h"

#include <iomanip>
#include <algorithm>

class StackedModuleGeometryAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit StackedModuleGeometryAnalyzer(const edm::ParameterSet&);
  ~StackedModuleGeometryAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override {}

  const edm::ESGetToken<reco::StackedModuleGeometryHost, StackedModuleGeometryRecord> geomToken_;
  const int dumpFirstN_;
  const bool checkValues_;
};

StackedModuleGeometryAnalyzer::StackedModuleGeometryAnalyzer(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes<reco::StackedModuleGeometryHost, StackedModuleGeometryRecord, edm::Transition::BeginRun>()),
      dumpFirstN_(iConfig.getParameter<int>("dumpFirstN")),
      checkValues_(iConfig.getParameter<bool>("checkValues")) {}

void StackedModuleGeometryAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("dumpFirstN", 10)->setComment("Number of modules to dump detailed info for (0 = none)");
  desc.add<bool>("checkValues", true)->setComment("Check if values are in reasonable ranges");
  descriptions.add("stackedModuleGeometryAnalyzer", desc);
}

void StackedModuleGeometryAnalyzer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  edm::LogPrint("StackedModuleGeometryAnalyzer") << "\n"
                                                 << "========================================\n"
                                                 << "  Stacked Module Geometry Validation\n"
                                                 << "========================================";

  // Get the geometry
  auto const& geomHost = iSetup.getData(geomToken_);
  auto const& geomView = geomHost.view();
  const uint32_t nModules = geomView.metadata().size();

  edm::LogPrint("StackedModuleGeometryAnalyzer") << "Total stacked modules: " << nModules;

  if (nModules == 0) {
    edm::LogPrint("StackedModuleGeometryAnalyzer") << "WARNING: No stacked modules found!";
    return;
  }

  // Statistics
  uint32_t nPS = 0, nSS = 0;
  uint32_t nBarrel = 0, nEndcap = 0;
  uint32_t nFlat = 0, nTilted = 0;
  uint32_t nFlipped = 0;

  float minSeparation = 999.0f, maxSeparation = 0.0f, sumSeparation = 0.0f;
  float minRadius = 999.0f, maxRadius = 0.0f, sumRadius = 0.0f;
  float minMaxBend = 999.0f, maxMaxBend = 0.0f, sumMaxBend = 0.0f;
  float maxTiltAngle = 0.0f;

  // Separate tilt angle statistics for barrel and endcap
  float minTiltBarrel = 999.0f, maxTiltBarrel = 0.0f, sumTiltBarrel = 0.0f;
  float minTiltEndcap = 999.0f, maxTiltEndcap = 0.0f, sumTiltEndcap = 0.0f;

  // Count per layer, separated by location. Layers are 1-based.
  std::array<uint32_t, 7> layerCountsBarrel = {};
  std::array<uint32_t, 7> layerCountsBarrelFlat = {};
  std::array<uint32_t, 7> layerCountsBarrelTilted = {};
  std::array<uint32_t, 7> layerCountsFwdDisk = {};
  std::array<uint32_t, 7> layerCountsBwdDisk = {};

  // Collect statistics
  for (uint32_t i = 0; i < nModules; ++i) {
    if (geomView[i].isPS())
      nPS++;
    else
      nSS++;

    if (geomView[i].isBarrel())
      nBarrel++;
    else
      nEndcap++;

    if (geomView[i].isFlat())
      nFlat++;
    else
      nTilted++;

    if (geomView[i].isFlipped())
      nFlipped++;

    float sep = geomView[i].sensorSeparation();
    float rad = geomView[i].meanRadius();
    float bend = geomView[i].maxBend();
    float tilt = geomView[i].tiltAngle();
    uint8_t layer = geomView[i].layer();

    minSeparation = std::min(minSeparation, sep);
    maxSeparation = std::max(maxSeparation, sep);
    sumSeparation += sep;

    minRadius = std::min(minRadius, rad);
    maxRadius = std::max(maxRadius, rad);
    sumRadius += rad;

    minMaxBend = std::min(minMaxBend, bend);
    maxMaxBend = std::max(maxMaxBend, bend);
    sumMaxBend += bend;

    maxTiltAngle = std::max(maxTiltAngle, std::abs(tilt));

    // Track tilt angles separately for barrel and endcap
    if (geomView[i].isBarrel()) {
      minTiltBarrel = std::min(minTiltBarrel, std::abs(tilt));
      maxTiltBarrel = std::max(maxTiltBarrel, std::abs(tilt));
      sumTiltBarrel += std::abs(tilt);
    } else {
      minTiltEndcap = std::min(minTiltEndcap, std::abs(tilt));
      maxTiltEndcap = std::max(maxTiltEndcap, std::abs(tilt));
      sumTiltEndcap += std::abs(tilt);
    }

    // Count by location (barrel vs forward/backward disk)
    if (geomView[i].isBarrel()) {
      layerCountsBarrel[layer]++;
      if (geomView[i].isFlat()) {
        layerCountsBarrelFlat[layer]++;
      } else {
        layerCountsBarrelTilted[layer]++;
      }
    } else {
      // Endcap - use isFwdEndcap flag to determine forward vs backward
      if (geomView[i].isFwdEndcap()) {
        layerCountsFwdDisk[layer]++;
      } else {
        layerCountsBwdDisk[layer]++;
      }
    }
  }

  // Print summary
  edm::LogPrint("StackedModuleGeometryAnalyzer") << "\nModule Type Breakdown:";
  edm::LogPrint("StackedModuleGeometryAnalyzer") << "  PS modules:     " << nPS;
  edm::LogPrint("StackedModuleGeometryAnalyzer") << "  SS modules:     " << nSS;

  edm::LogPrint("StackedModuleGeometryAnalyzer") << "\nLocation:";
  edm::LogPrint("StackedModuleGeometryAnalyzer") << "  Barrel:         " << nBarrel;
  edm::LogPrint("StackedModuleGeometryAnalyzer") << "    Flat:         " << nFlat;
  edm::LogPrint("StackedModuleGeometryAnalyzer") << "    Tilted:       " << nTilted;
  edm::LogPrint("StackedModuleGeometryAnalyzer") << "  Endcap:         " << nEndcap;

  edm::LogPrint("StackedModuleGeometryAnalyzer") << "\nOrientation:";
  edm::LogPrint("StackedModuleGeometryAnalyzer") << "  Flipped:        " << nFlipped << " (" << std::fixed
                                                 << std::setprecision(1) << (100.0 * nFlipped / nModules) << "%)";

  edm::LogPrint("StackedModuleGeometryAnalyzer") << "\nSensor Separation (mm):";
  edm::LogPrint("StackedModuleGeometryAnalyzer")
      << "  Min:            " << std::fixed << std::setprecision(2) << minSeparation;
  edm::LogPrint("StackedModuleGeometryAnalyzer") << "  Max:            " << maxSeparation;
  edm::LogPrint("StackedModuleGeometryAnalyzer") << "  Mean:           " << (sumSeparation / nModules);

  edm::LogPrint("StackedModuleGeometryAnalyzer") << "\nMean Radius (cm):";
  edm::LogPrint("StackedModuleGeometryAnalyzer")
      << "  Min:            " << std::fixed << std::setprecision(1) << minRadius;
  edm::LogPrint("StackedModuleGeometryAnalyzer") << "  Max:            " << maxRadius;
  edm::LogPrint("StackedModuleGeometryAnalyzer") << "  Mean:           " << (sumRadius / nModules);

  edm::LogPrint("StackedModuleGeometryAnalyzer") << "\nMax Bend (rad):";
  edm::LogPrint("StackedModuleGeometryAnalyzer")
      << "  Min:            " << std::scientific << std::setprecision(3) << minMaxBend;
  edm::LogPrint("StackedModuleGeometryAnalyzer") << "  Max:            " << maxMaxBend;
  edm::LogPrint("StackedModuleGeometryAnalyzer") << "  Mean:           " << (sumMaxBend / nModules);

  edm::LogPrint("StackedModuleGeometryAnalyzer")
      << "\nMax Tilt Angle: " << std::fixed << std::setprecision(3) << maxTiltAngle << " rad ("
      << (maxTiltAngle * 180.0 / M_PI) << " deg)";

  edm::LogPrint("StackedModuleGeometryAnalyzer") << "\nTilt Angle (Barrel, rad):";
  edm::LogPrint("StackedModuleGeometryAnalyzer") << "  Min:            " << std::fixed << std::setprecision(3)
                                                 << minTiltBarrel << " (" << (minTiltBarrel * 180.0 / M_PI) << " deg)";
  edm::LogPrint("StackedModuleGeometryAnalyzer")
      << "  Max:            " << maxTiltBarrel << " (" << (maxTiltBarrel * 180.0 / M_PI) << " deg)";
  edm::LogPrint("StackedModuleGeometryAnalyzer") << "  Mean:           " << (sumTiltBarrel / nBarrel) << " ("
                                                 << (sumTiltBarrel / nBarrel * 180.0 / M_PI) << " deg)";

  edm::LogPrint("StackedModuleGeometryAnalyzer") << "\nTilt Angle (Endcap, rad):";
  edm::LogPrint("StackedModuleGeometryAnalyzer") << "  Min:            " << std::fixed << std::setprecision(3)
                                                 << minTiltEndcap << " (" << (minTiltEndcap * 180.0 / M_PI) << " deg)";
  edm::LogPrint("StackedModuleGeometryAnalyzer")
      << "  Max:            " << maxTiltEndcap << " (" << (maxTiltEndcap * 180.0 / M_PI) << " deg)";
  edm::LogPrint("StackedModuleGeometryAnalyzer") << "  Mean:           " << (sumTiltEndcap / nEndcap) << " ("
                                                 << (sumTiltEndcap / nEndcap * 180.0 / M_PI) << " deg)";

  // Layer distribution by location
  edm::LogPrint("StackedModuleGeometryAnalyzer") << "\nModules per Layer (Barrel):";
  for (size_t i = 0; i < layerCountsBarrel.size(); ++i) {
    if (layerCountsBarrel[i] > 0) {
      edm::LogPrint("StackedModuleGeometryAnalyzer")
          << "  Layer " << std::setw(2) << i << ": " << layerCountsBarrel[i]
          << " tilted: " << layerCountsBarrelTilted[i] << " flat: " << layerCountsBarrelFlat[i];
    }
  }

  edm::LogPrint("StackedModuleGeometryAnalyzer") << "\nModules per Layer (Forward Disk, +z):";
  for (size_t i = 0; i < layerCountsFwdDisk.size(); ++i) {
    if (layerCountsFwdDisk[i] > 0) {
      edm::LogPrint("StackedModuleGeometryAnalyzer")
          << "  Disk  " << std::setw(2) << i << ": " << layerCountsFwdDisk[i];
    }
  }

  edm::LogPrint("StackedModuleGeometryAnalyzer") << "\nModules per Layer (Backward Disk, -z):";
  for (size_t i = 0; i < layerCountsBwdDisk.size(); ++i) {
    if (layerCountsBwdDisk[i] > 0) {
      edm::LogPrint("StackedModuleGeometryAnalyzer")
          << "  Disk  " << std::setw(2) << i << ": " << layerCountsBwdDisk[i];
    }
  }

  // Value checks
  if (checkValues_) {
    edm::LogPrint("StackedModuleGeometryAnalyzer") << "\nValidation Checks:";
    bool allOK = true;

    // Check separation (should be 1.6-4.5 mm for Phase-2 OT)
    if (minSeparation < 1.0f || maxSeparation > 5.0f) {
      edm::LogPrint("StackedModuleGeometryAnalyzer")
          << "  WARNING: Sensor separation outside expected range [1.6, 4.5] mm";
      allOK = false;
    } else {
      edm::LogPrint("StackedModuleGeometryAnalyzer") << "  ✓ Sensor separation in expected range";
    }

    // Check radius (should be ~20-110 cm for Phase-2 OT)
    if (minRadius < 15.0f || maxRadius > 120.0f) {
      edm::LogPrint("StackedModuleGeometryAnalyzer") << "  WARNING: Module radius outside expected range [20, 110] cm";
      allOK = false;
    } else {
      edm::LogPrint("StackedModuleGeometryAnalyzer") << "  ✓ Module radius in expected range";
    }

    // Check maxBend (should be small but positive)
    if (minMaxBend <= 0.0f || maxMaxBend > 0.1f) {
      edm::LogPrint("StackedModuleGeometryAnalyzer") << "  WARNING: maxBend outside expected range (0, 0.1] rad";
      allOK = false;
    } else {
      edm::LogPrint("StackedModuleGeometryAnalyzer") << "  ✓ maxBend values in expected range";
    }

    // Check endcap tilt angles (should be ~90° or ~-90°, i.e. z-stacked, parallel/anti-parallel to z-axis)
    float meanTiltEndcap = sumTiltEndcap / nEndcap;
    // Check if close to π/2 or -π/2
    bool endcapTiltOK = (std::abs(meanTiltEndcap) - M_PI / 2.) < 0.1f;
    if (!endcapTiltOK) {
      edm::LogPrint("StackedModuleGeometryAnalyzer")
          << "  WARNING: Endcap tilt angles not near 90° or -90° (mean: " << (meanTiltEndcap * 180.0 / M_PI) << " deg)";
      allOK = false;
    } else {
      edm::LogPrint("StackedModuleGeometryAnalyzer") << "  ✓ Endcap tilt angles near 90°/-90° as expected (z-stacked)";
    }

    // Check module counts (Phase-2 OT has thousands of modules)
    if (nModules < 1000) {
      edm::LogPrint("StackedModuleGeometryAnalyzer")
          << "  WARNING: Only " << nModules << " modules found (expected ~13k for Phase-2 OT)";
      allOK = false;
    } else {
      edm::LogPrint("StackedModuleGeometryAnalyzer") << "  ✓ Module count looks reasonable";
    }

    if (allOK) {
      edm::LogPrint("StackedModuleGeometryAnalyzer") << "\n✓ All validation checks passed!";
    } else {
      edm::LogPrint("StackedModuleGeometryAnalyzer") << "\n✗ Some validation checks failed - see warnings above";
    }
  }

  // Dump first N modules in detail""
  if (dumpFirstN_ > 0) {
    edm::LogPrint("StackedModuleGeometryAnalyzer") << "\nDetailed dump of first " << dumpFirstN_ << " modules:";
    edm::LogPrint("StackedModuleGeometryAnalyzer")
        << "  Idx |    DetId(P)[S]    |  GeomIndex | K | Type  | Barrel | Flat | Flipped | Layer | "
           "Sep(mm) | Radius(cm) | Tilt(deg) | maxBend(rad) | LowUpNormX | LowUpNormY | LowUpNormZ | locXInGlbX | "
           "locXInGlbY | locInGlbZ |";
    edm::LogPrint("StackedModuleGeometryAnalyzer")
        << "  "
           "----+-------------------+------------+"
           "---+-------+--------+------+--------"
           "-+-------+---------+------------+-----------+--------------+------------+------------+------------+--------"
           "----+------------+-----------|";

    for (int i = 0; i < std::min(dumpFirstN_, static_cast<int>(nModules)); ++i) {
      edm::LogPrint("StackedModuleGeometryAnalyzer")
          << "  " << std::setw(3) << i << " |  " << geomView[i].detId() << "("
          << static_cast<int64_t>(geomView[i].partnerDetId()) - static_cast<int64_t>(geomView[i].detId()) << ")["
          << static_cast<int64_t>(geomView[i].stackedDetId()) - static_cast<int64_t>(geomView[i].detId()) << "]"
          << " | " << std::setw(10) << geomView[i].geomIndex() << " | " << (geomView[i].moduleType() == 0 ? "P" : "S")
          << " | " << (geomView[i].isPS() ? " PS " : " SS ") << "  |    " << (geomView[i].isBarrel() ? "Y" : "N")
          << "   |   " << (geomView[i].isFlat() ? "Y" : "N") << "  |    " << (geomView[i].isFlipped() ? "Y" : "N")
          << "    |  " << std::setw(2) << static_cast<int>(geomView[i].layer()) << "   |  " << std::fixed
          << std::setprecision(2) << std::setw(5) << geomView[i].sensorSeparation() << "  |   " << std::setw(6)
          << geomView[i].meanRadius() << "   |   " << std::setw(6) << (geomView[i].tiltAngle() * 180.0 / M_PI)
          << "  |   " << std::scientific << std::setprecision(3) << geomView[i].maxBend() << "  | " << std::setw(10)
          << geomView[i].globalLowUpNormX() << " | " << std::setw(10) << geomView[i].globalLowUpNormY() << " | "
          << std::setw(10) << geomView[i].globalLowUpNormZ() << " | " << std::setw(10) << geomView[i].localXInGlobalX()
          << " | " << std::setw(10) << geomView[i].localXInGlobalY() << " | " << std::setw(10)
          << geomView[i].localXInGlobalZ();
    }
  }

  edm::LogPrint("StackedModuleGeometryAnalyzer") << "\n========================================\n";
}

DEFINE_FWK_MODULE(StackedModuleGeometryAnalyzer);
