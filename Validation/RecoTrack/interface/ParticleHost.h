#ifndef Validation_RecoTrack_ParticleHost_H
#define Validation_RecoTrack_ParticleHost_H

#include "Validation/RecoTrack/interface/ParticleSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace sim {
    using ParticleHost = PortableHostCollection<ParticleSoA>;
}

#endif  // Validation_RecoTrack_ParticleHost_H


