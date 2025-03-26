#ifndef DataFormats_ClusterGeometrySoA_interface_ClusterGeometrysSoA_h
#define DataFormats_ClusterGeometrySoA_interface_ClusterGeometrysSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(ClusterGeometrysLayout,
                    SOA_COLUMN(uint32_t, clusterIds),
                    SOA_COLUMN(float, pitchX),
                    SOA_COLUMN(float, pitchY),
                    SOA_COLUMN(float, thickness),
                    SOA_COLUMN(float, tanLorentzAngles),
                    SOA_COLUMN(float, transformXX),
                    SOA_COLUMN(float, transformXY),
                    SOA_COLUMN(float, transformXZ),
                    SOA_COLUMN(float, transformYX),
                    SOA_COLUMN(float, transformYY),
                    SOA_COLUMN(float, transformYZ),
                    SOA_COLUMN(float, transformZX),
                    SOA_COLUMN(float, transformZY),
                    SOA_COLUMN(float, transformZZ),
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_COLUMN(uint32_t, sizeX),
                    SOA_COLUMN(uint32_t, sizeY),
                    SOA_COLUMN(uint32_t, clusterOffset),
                    SOA_COLUMN(uint32_t, moduleId))

using ClusterGeometrysSoA = ClusterGeometrysLayout<>;
using ClusterGeometrysSoAView = ClusterGeometrysSoA::View;
using ClusterGeometrysSoAConstView = ClusterGeometrysSoA::ConstView;

#endif  // DataFormats_ClusterGeometrySoA_interface_ClusterGeometrysSoA_h
