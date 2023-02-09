#ifndef DataFormats_SiPixelMappingSoA_interface_SiPixelClustersDevice_h
#define DataFormats_SiPixelMappingSoA_interface_SiPixelClustersDevice_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/alpaka/PortableHostCollection.h"
#include "SiPixelMappingLayout.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using SiPixelMappingDevice = PortableHostCollection<SiPixelMappingLayout<>>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_SiPixelMappingSoA_interface_SiPixelClustersDevice_h
