#ifndef CalibFormats_SiStripObjects_test_alpaka_SiStripClusterizerConditionsSoA_test_h
#define CalibFormats_SiStripObjects_test_alpaka_SiStripClusterizerConditionsSoA_test_h

#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditionsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testStripCondsSoA {

  void runKernels(::reco::StripConditionsDetSoAView conds_view,
                                  Queue& queue);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testStripCondsSoA

#endif  // CalibFormats_SiStripObjects_test_alpaka_SiStripClusterizerConditionsSoA_test_h