#ifndef RecoLocalTracker_ClusterParameterEstimator_interface_alpaka_FrameSoACollection_h
#define RecoLocalTracker_ClusterParameterEstimator_interface_alpaka_FrameSoACollection_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "RecoLocalTracker/ClusterParameterEstimator/interface/FrameSoAHost.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/FrameSoADevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  using FrameSoACollection = FrameSoAHost;
#else
  using FrameSoACollection = FrameSoADevice<Device>;
#endif

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalTracker_ClusterParameterEstimator_interface_alpaka_FrameSoACollection_h
