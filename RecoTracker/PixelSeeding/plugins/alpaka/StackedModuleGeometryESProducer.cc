#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "DataFormats/Portable/interface/PortableCollection.h"

#include "RecoTracker/Record/interface/StackedModuleGeometryRecord.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometryHost.h"

#include <vector>
#include <algorithm>
#include <cmath>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class StackedModuleGeometryESProducer : public ESProducer {
  public:
    StackedModuleGeometryESProducer(edm::ParameterSet const& iConfig)
        : ESProducer(iConfig),
          minPt_(iConfig.getParameter<double>("minPt")),
          magneticField_(iConfig.getParameter<double>("magneticField")) {
      auto c = setWhatProduced(this);
      geomToken_ = c.consumes();
      topoToken_ = c.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<double>("minPt", 2.0)->setComment("Minimum pT threshold for maxBend calculation (GeV)");
      desc.add<double>("magneticField", 3.8)->setComment("Magnetic field strength (T)");
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<reco::StackedModuleGeometryHost> produce(StackedModuleGeometryRecord const& iRecord) {
      auto const& geom = iRecord.get(geomToken_);
      auto const& topo = iRecord.get(topoToken_);

      // Temporary structure to hold module info for sorting
      struct ModuleInfo {
        const GeomDetUnit* detUnit;
        const GeomDetUnit* partnerDetUnit;
        DetId detId;
        DetId partnerDetId;
        DetId stackedDetId;
        uint8_t layer;
        bool isBarrel;
        bool isFwdEndcap;  // z > 0
        uint8_t category;  // 0 = barrel, 1 = backward (z<0), 2 = forward (z>0)
      };

      std::vector<ModuleInfo> modules;
      modules.reserve(15000);  // Approximate number of OT stacked modules

      // First pass: collect all stacked modules
      for (auto const& detUnit : geom.detUnits()) {
        DetId detId = detUnit->geographicalId();

        // Only process lower sensors of stacked modules
        if (topo.stack(detId) == 0 || !topo.isLower(detId))
          continue;

        DetId partnerDetId = topo.partnerDetId(detId);
        DetId stackedDetId = topo.stack(detId);
        auto const* partnerDetUnit = geom.idToDetUnit(partnerDetId);

        if (!partnerDetUnit)
          continue;

        auto const& lowerPos = detUnit->position();
        bool isBarrel = (detId.subdetId() == StripSubdetector::TOB);
        bool isFwdEndcap = (lowerPos.z() > 0);
        uint8_t layer = topo.layer(detId);

        // Determine category for sorting: barrel first, then backward, then forward
        uint8_t category;
        if (isBarrel) {
          category = 0;
        } else if (!isFwdEndcap) {
          category = 1;  // Backward (z < 0)
        } else {
          category = 2;  // Forward (z > 0)
        }

        modules.push_back(
            {detUnit, partnerDetUnit, detId, partnerDetId, stackedDetId, layer, isBarrel, isFwdEndcap, category});
      }

      // Sort modules by category (barrel, backward, forward), then by layer
      // This matches CA module ordering expectations:
      // - CA module 4000+: barrel layer 1, then layer 2, ..., layer 6
      // - Then backward disk 1, 2, 3, 4, 5
      // - Then forward disk 1, 2, 3, 4, 5
      // Use stable_sort to preserve the original ordering within each layer
      // (sensors belonging to the same stacked module should remain consecutive)
      std::stable_sort(modules.begin(), modules.end(), [](const ModuleInfo& a, const ModuleInfo& b) {
        if (a.category != b.category)
          return a.category < b.category;
        return a.layer < b.layer;
      });

      uint32_t nModules = modules.size();

      // Allocate the SoA
      auto hostProduct = std::make_unique<reco::StackedModuleGeometryHost>(cms::alpakatools::host(), nModules);
      auto view = hostProduct->view();

      // Second pass: fill the geometry information in sorted order
      for (uint32_t iModule = 0; iModule < nModules; ++iModule) {
        const auto& mod = modules[iModule];
        auto const& detUnit = mod.detUnit;
        auto const& partnerDetUnit = mod.partnerDetUnit;
        DetId detId = mod.detId;
        DetId partnerDetId = mod.partnerDetId;
        DetId stackedDetId = mod.stackedDetId;

        // Get global positions of both sensors
        auto const& lowerPos = detUnit->position();
        auto const& upperPos = partnerDetUnit->position();

        // Calculate sensor separation (convert from cm to mm)
        float separation = (upperPos - lowerPos).mag() * 10.0f;

        // Mean radius (in cm)
        float meanRadius = lowerPos.perp();

        // Module classification
        bool isBarrel = mod.isBarrel;
        bool isPS = !(geom.getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2SS);
        bool isPSP = (geom.getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP);
        bool isPSS = (geom.getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSS);
        bool isFwdEndcap = mod.isFwdEndcap;

        // Determine isFlipped: true if the "lower" sensor (in stack reference frame) is
        // actually farther from the beam line than the "upper" sensor.
        // For barrel modules, we compare radii (perp()) since the tracking cares about
        // which sensor is closer to the beam line, not closer to IP in 3D.
        // For endcap modules, we compare |z| since layers are at fixed z.
        // Note: Using mag() (3D distance) is WRONG for tilted barrel modules because a sensor
        // at smaller r can have larger mag() if it's at larger |z| due to the tilt.
        // IMPORTANT: isFlipped must be computed BEFORE tiltAngle since tiltAngle depends on it.
        bool isFlipped;
        if (isBarrel) {
          isFlipped = (lowerPos.perp() > upperPos.perp());
        } else {
          isFlipped = (std::abs(lowerPos.z()) > std::abs(upperPos.z()));
        }

        // Calculate tilt angle using PHYSICAL inner->outer direction.
        //
        // The tilt angle is the angle from the +r axis (radial outward) to the
        // physical inner->outer direction vector, measured counterclockwise in the (r,z) plane.
        //
        // IMPORTANT: We use the PHYSICAL inner->outer direction, not the topological
        // lower->upper direction. This ensures the tilt angle has consistent physical
        // meaning regardless of whether the module is flipped:
        //   - For non-flipped: inner=lower, outer=upper
        //   - For flipped: inner=upper, outer=lower
        //
        // Formula: tiltAngle = atan2(dz_phys, dr_phys)
        //   where (dz_phys, dr_phys) is from physical inner to physical outer sensor
        //
        // Physical interpretation (consistent for flipped and non-flipped):
        //   - Flat barrel: tiltAngle ~= 0 (inner->outer vector along +r)
        //   - Endcap at +z: tiltAngle ~= +pi/2 (inner->outer vector along +z)
        //   - Endcap at -z: tiltAngle ~= -pi/2 (inner->outer vector along -z)
        //   - Tilted barrel: intermediate values depending on module orientation
        //
        // By using the physical direction, the parallax correction and dPhiDr
        // calculations give consistent results regardless of flipped status.
        //
        float dz_phys, dr_phys;
        if (isFlipped) {
          // For flipped: physical inner is "upper", physical outer is "lower"
          dz_phys = lowerPos.z() - upperPos.z();
          dr_phys = lowerPos.perp() - upperPos.perp();  // > 0 for barrel
        } else {
          // For non-flipped: physical inner is "lower", physical outer is "upper"
          dz_phys = upperPos.z() - lowerPos.z();
          dr_phys = upperPos.perp() - lowerPos.perp();  // > 0 for barrel
        }
        float tiltAngle = std::atan2(dz_phys, dr_phys);

        // Module axis vector for globalLowUpNorm (physical inner->outer direction)
        // This is used for parallax correction and must be consistent with the tiltAngle.
        // For non-flipped modules: inner=lower, outer=upper, so vector is upper - lower
        // For flipped modules: inner=upper, outer=lower, so vector is lower - upper
        auto moduleAxisVec = isFlipped ? (lowerPos - upperPos) : (upperPos - lowerPos);
        auto moduleAxis = moduleAxisVec.basicVector() / moduleAxisVec.mag();

        auto localXInGlobal = detUnit->surface().toGlobal(LocalVector(1.0, 0.0, 0.0));

        // Diagnostic: check local frame consistency between lower and upper sensors
        // For correct width calculation (lx_lower - lx_upper), local-x must point the
        // same direction on both sensors. Similarly, local-y consistency is needed for
        // the same-sign-y cut.
        {
          auto localXUpper = partnerDetUnit->surface().toGlobal(LocalVector(1.0, 0.0, 0.0));
          auto localYLower = detUnit->surface().toGlobal(LocalVector(0.0, 1.0, 0.0));
          auto localYUpper = partnerDetUnit->surface().toGlobal(LocalVector(0.0, 1.0, 0.0));
          auto localZLower = detUnit->surface().toGlobal(LocalVector(0.0, 0.0, 1.0));
          auto localZUpper = partnerDetUnit->surface().toGlobal(LocalVector(0.0, 0.0, 1.0));

          float dotX = localXInGlobal.x() * localXUpper.x() + localXInGlobal.y() * localXUpper.y() +
                       localXInGlobal.z() * localXUpper.z();
          float dotY = localYLower.x() * localYUpper.x() + localYLower.y() * localYUpper.y() +
                       localYLower.z() * localYUpper.z();
          float dotZ = localZLower.x() * localZUpper.x() + localZLower.y() * localZUpper.y() +
                       localZLower.z() * localZUpper.z();

          if (dotX < 0.9f || dotY < 0.9f) {
            edm::LogPrint("StackedModuleGeometry")
                << "LOCAL FRAME MISMATCH module " << iModule << " detId=" << detId.rawId()
                << " isBarrel=" << isBarrel << " isFlipped=" << isFlipped << " layer=" << (int)mod.layer
                << " dotX=" << dotX << " dotY=" << dotY << " dotZ=" << dotZ
                << " lower=(" << lowerPos.x() << "," << lowerPos.y() << "," << lowerPos.z() << ")"
                << " upper=(" << upperPos.x() << "," << upperPos.y() << "," << upperPos.z() << ")"
                << " localX_lower=(" << localXInGlobal.x() << "," << localXInGlobal.y() << ","
                << localXInGlobal.z() << ")"
                << " localX_upper=(" << localXUpper.x() << "," << localXUpper.y() << "," << localXUpper.z()
                << ")"
                << " localY_lower=(" << localYLower.x() << "," << localYLower.y() << "," << localYLower.z()
                << ")"
                << " localY_upper=(" << localYUpper.x() << "," << localYUpper.y() << "," << localYUpper.z()
                << ")";
          }
        }

        // Determine if module is flat (non-tilted)
        bool isFlat;
        constexpr float threshold = std::cos(0.1f);
        if (isBarrel) {
          isFlat = std::abs(std::cos(tiltAngle)) > threshold;
        } else {
          isFlat = true;
        }

        // Calculate maxBend for pT threshold
        float maxBend = 0.0f;
        if (meanRadius > 0.0f) {
          maxBend = 0.3f * magneticField_ * separation / (minPt_ * meanRadius * 10.0f);
        }

        // Fill the SoA
        view[iModule].detId() = detId.rawId();
        view[iModule].partnerDetId() = partnerDetId.rawId();
        view[iModule].stackedDetId() = stackedDetId.rawId();
        view[iModule].geomIndex() = detUnit->index();
        view[iModule].moduleType() = (isPSP ? 0 : (isPSS ? 1 : 2));
        view[iModule].sensorSeparation() = separation;
        view[iModule].meanRadius() = meanRadius;
        view[iModule].tiltAngle() = tiltAngle;
        view[iModule].sinTilt() = std::sin(tiltAngle);
        view[iModule].cosTilt() = std::cos(tiltAngle);
        view[iModule].isPS() = isPS;
        view[iModule].isFlipped() = isFlipped;
        view[iModule].isBarrel() = isBarrel;
        view[iModule].isFlat() = isFlat;
        view[iModule].isFwdEndcap() = isFwdEndcap;
        view[iModule].layer() = mod.layer;
        view[iModule].maxBend() = maxBend;
        view[iModule].globalLowUpNormX() = moduleAxis.x();
        view[iModule].globalLowUpNormY() = moduleAxis.y();
        view[iModule].globalLowUpNormZ() = moduleAxis.z();
        view[iModule].localXInGlobalX() = localXInGlobal.x();
        view[iModule].localXInGlobalY() = localXInGlobal.y();
        view[iModule].localXInGlobalZ() = localXInGlobal.z();
      }

      // Log module counts by category for debugging
      uint32_t nBarrel = 0, nBackward = 0, nForward = 0;
      for (const auto& mod : modules) {
        if (mod.category == 0)
          nBarrel++;
        else if (mod.category == 1)
          nBackward++;
        else
          nForward++;
      }
      edm::LogInfo("StackedModuleGeometry")
          << "Produced " << nModules << " stacked modules in CA order:\n"
          << "  Barrel: " << nBarrel << " (indices 0-" << (nBarrel - 1) << ")\n"
          << "  Backward: " << nBackward << " (indices " << nBarrel << "-" << (nBarrel + nBackward - 1) << ")\n"
          << "  Forward: " << nForward << " (indices " << (nBarrel + nBackward) << "-" << (nModules - 1) << ")";

      return hostProduct;
    }

  private:
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
    double minPt_;
    double magneticField_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(StackedModuleGeometryESProducer);
