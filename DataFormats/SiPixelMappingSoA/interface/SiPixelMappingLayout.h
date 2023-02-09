#ifndef DataFormats_SiPixelMappingSoA_interface_SiPixelMappingLayout_h
#define DataFormats_SiPixelMappingSoA_interface_SiPixelMappingLayout_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelROCsStatusAndMapping.h"

// Layout con
// modToUnpDefault (unsigned char sempre di MAX_SIZE)
// hasQuality_ (scalar bool)
// cablingMap che è una struct SiPixelROCsStatusAndMapping
//
// Nel costruttore ho bisogno della SiPixelFedCablingMap da assegnare ad un pointer generico tanto non la uso quasi mai.
//
// Nel costruttore in pratica riempio la cablingMap e basta.
//
// Poi voglio un metodo
// - getGPUProductAsync che mi restituisce sta cabling map on device
// - getModToUnpAllAsync che mi da il vettore di sopra
// - getModToUnpRegionalAsync che mi da il vettore di sopra “skimmed” regionally


GENERATE_SOA_LAYOUT(SiPixelMappingLayout,
                    SOA_SCALAR(SiPixelROCsStatusAndMapping, cablingMap),
                    SOA_SCALAR(std::array<unsigned char, pixelgpudetails::MAX_SIZE>, modToUnpDefault),
                    SOA_SCALAR(bool, hasQuality))

using SiPixelMappingLayoutSoA = SiPixelMappingLayout<>;
using SiPixelMappingLayoutSoAView = SiPixelMappingLayout<>::View;
using SiPixelMappingLayoutSoAConstView = SiPixelMappingLayout<>::ConstView;


#endif  // DataFormats_SiPixelMappingoA_interface_SiPixelMappingLayout_h
