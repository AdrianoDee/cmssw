// #include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
// #include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsMaskingHost.h"
// #include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
// #include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsMaskingSoACollection.h"
// #include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

// DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(reco::TrackingRecHitHost, reco::TrackingRecHitsSoACollection);
// DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(reco::TrackingRecHitHost, reco::TrackingRecHitsMaskingSoACollection);

#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsMaskingSoACollection.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN_DEVICE(reco::TrackingRecHitsSoACollection);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_DEVICE(reco::TrackingRecHitsMaskingSoACollection);