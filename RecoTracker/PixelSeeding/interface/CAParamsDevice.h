#ifndef RecoTracker_PixelSeeding_interface_CAParamsDevice_H
#define RecoTracker_PixelSeeding_interface_CAParamsDevice_H

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "RecoTracker/PixelSeeding/interface/CAParamsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

template <typename TDev>
using CAParamsDevice = PortableDeviceMultiCollection<TDev, reco::CALayersSoA, reco::CACellsSoA, reco::CARegionsSoA>;

#endif  // DataFormats_VertexSoA_CAParamsDevice_H