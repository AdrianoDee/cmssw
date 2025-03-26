#ifndef DataFormats_CandidateSoA_interface_CandidatesSoA_h
#define DataFormats_CandidateSoA_interface_CandidatesSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(CandidatesLayout,
                    SOA_COLUMN(uint32_t, candidateIndex),
                    SOA_COLUMN(float, px),
                    SOA_COLUMN(float, py),
                    SOA_COLUMN(float, pz),
                    SOA_COLUMN(float, pt),
                    SOA_COLUMN(float, eta),
                    SOA_COLUMN(float, phi))

using CandidatesSoA = CandidatesLayout<>;
using CandidatesSoAView = CandidatesSoA::View;
using CandidatesSoAConstView = CandidatesSoA::ConstView;

#endif  // DataFormats_CandidateSoA_interface_CandidatesSoA_h
