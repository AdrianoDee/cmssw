#ifndef DataFormats_SiPixelGainCalibrationForHLTSoA_interface_SiPixelClustersDevice_h
#define DataFormats_SiPixelGainCalibrationForHLTSoA_interface_SiPixelClustersDevice_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "SiPixelGainCalibrationForHLTLayout.h"

using SiPixelGainCalibrationForHLTHost = PortableHostCollection<SiPixelGainCalibrationForHLTLayout<>>;
// class SiPixelFedCablingMap;

// class SiPixelGainCalibrationForHLTHost : public PortableHostCollection<SiPixelGainCalibrationForHLTLayout<>> {
// public:
//   SiPixelGainCalibrationForHLTHost() = default;
//   ~SiPixelGainCalibrationForHLTHost() = default;

//   template <typename TQueue>
//   explicit SiPixelGainCalibrationForHLTHost(size_t maxModules, SiPixelFedCablingMap const& cablingMap, bool hasQuality, TQueue queue)
//       : PortableHostCollection<SiPixelGainCalibrationForHLTLayout<>>(maxModules + 1, queue), hasQuality_(hasQuality),
//        cablingMap_(&cablingMap)
//        {}

//   SiPixelGainCalibrationForHLTHost(SiPixelGainCalibrationForHLTHost &&) = default;
//   SiPixelGainCalibrationForHLTHost &operator=(SiPixelGainCalibrationForHLTHost &&) = default;

//   bool hasQuality() const { return hasQuality_; }

// private:
//   bool hasQuality_;
//   const SiPixelFedCablingMap *cablingMap_; //this is the cabling map that is ALWAYS on Host
// };

#endif  // DataFormats_SiPixelGainCalibrationForHLTSoA_interface_SiPixelClustersDevice_h
