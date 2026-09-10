#ifndef RecoTracker_Record_CAGeometryRecord_h
#define RecoTracker_Record_CAGeometryRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "RecoTracker/Record/interface/StackedModuleGeometryRecord.h"

#include "FWCore/Utilities/interface/mplVector.h"

// EventSetup record for the CA geometry SoA, keyed to the tracker geometry, topology and
// StackedModuleGeometry IOVs. Per-producer cut blocks are configuration, not a condition.
class CAGeometryRecord
    : public edm::eventsetup::DependentRecordImplementation<
          CAGeometryRecord,
          edm::mpl::Vector<TrackerDigiGeometryRecord, TrackerTopologyRcd, StackedModuleGeometryRecord>> {};

#endif  // RecoTracker_Record_CAGeometryRecord_h
