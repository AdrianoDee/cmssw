#ifndef Validation_RecoTrack_ParticleSoA_H
#define Validation_RecoTrack_ParticleSoA_H

#include <alpaka/alpaka.hpp>
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace sim {

  GENERATE_SOA_LAYOUT(ParticleLayout,
    SOA_COLUMN(float, vx),
    SOA_COLUMN(float, vy),
    SOA_COLUMN(float, vz),

    SOA_COLUMN(float, px),
    SOA_COLUMN(float, py),
    SOA_COLUMN(float, pz),
    SOA_COLUMN(float, energy),

    SOA_COLUMN(float, pt),
    SOA_COLUMN(float, eta),
    SOA_COLUMN(float, phi),
    SOA_COLUMN(float, mass),

    SOA_COLUMN(int16_t, charge),
    SOA_COLUMN(int32_t, pdgID),

    SOA_COLUMN(uint32_t, partInd)
    );

    using ParticleSoA = ParticleLayout<>;
    using ParticleSoAView = ParticleSoA::View;
    using ParticleSoAConstView = ParticleSoA::ConstView;

}


#endif  // Validation_RecoTrack_ParticleSoA_H
