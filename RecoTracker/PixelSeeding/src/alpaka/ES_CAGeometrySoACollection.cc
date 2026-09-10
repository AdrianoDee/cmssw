#include "RecoTracker/PixelSeeding/interface/alpaka/CAGeometrySoACollection.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/typelookup.h"

// ESDeviceProduct typelookup for the CA geometry SoA. Must live in an alpaka library
// (PixelSeeding): the macro only emits the device symbol under ALPAKA_ACCELERATOR_NAMESPACE.
TYPELOOKUP_ALPAKA_DATA_REG(reco::CAGeometrySoACollection);
