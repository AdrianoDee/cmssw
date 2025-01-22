#ifndef DataFormats_CandidateSoA_interface_CandidateLayout_h
#define DataFormats_CandidateSoA_interface_CandidateLayout_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(CandidateLayout,
                    SOA_COLUMN(uint32_t, candidateIndex),
                    SOA_COLUMN(float, px),
                    SOA_COLUMN(float, py),
                    SOA_COLUMN(float, pz),
                    SOA_COLUMN(float, pt),
                    SOA_COLUMN(float, eta),
                    SOA_COLUMN(float, phi))

using CandidateSoA = CandidateLayout<>;
using CandidateSoAView = CandidateSoA::View;
using CandidateSoAConstView = CandidateSoA::ConstView;

#endif  // DataFormats_CandidateSoA_interface_CandidateLayout_h
