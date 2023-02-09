#ifndef DataFormats_SiPixelMappingSoA_interface_SiPixelClustersDevice_h
#define DataFormats_SiPixelMappingSoA_interface_SiPixelClustersDevice_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SiPixelMappingSoA/interface/SiPixelMappingLayout.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using SiPixelMappingDevice = PortableCollection<SiPixelMappingLayout<>>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_SiPixelMappingSoA_interface_SiPixelClustersDevice_h
