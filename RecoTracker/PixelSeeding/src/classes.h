#include "RecoTracker/PixelSeeding/interface/IntermediateHitTriplets.h"
#include "DataFormats/Common/interface/Wrapper.h"

// Per-built-triplet training-dataset host SoA, the host transcription of the CA_TRIPLET_DUMP device
// product. Type-info only, harmless when CA_TRIPLET_DUMP is off.
#include "RecoTracker/PixelSeeding/interface/TripletDumpHost.h"

// CA-ordered module geometry host collection, needed for the serial backend product and for the
// host transcription of the device product. Type-info only.
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometryHost.h"

#include <vector>

namespace RecoPixelVertexing_PixelTriplets {
  struct dictionary {
    IntermediateHitTriplets iht;
    edm::Wrapper<IntermediateHitTriplets> wiht;
  };
}  // namespace RecoPixelVertexing_PixelTriplets
