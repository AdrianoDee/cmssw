#ifndef DataFormats_ClusterGeometrySoA_interface_ClusterGeometryLayout_h
#define DataFormats_ClusterGeometrySoA_interface_ClusterGeometryLayout_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(ClusterGeometryLayout,
                    SOA_COLUMN(uint32_t, clusterIds),
                    SOA_COLUMN(float, pitchX),
                    SOA_COLUMN(float, pitchY),
                    SOA_COLUMN(float, thickness),
                    SOA_COLUMN(float, tanLorentzAngles))

using ClusterGeometrySoA = ClusterGeometryLayout<>;
using ClusterGeometrySoAView = ClusterGeometrySoA::View;
using ClusterGeometrySoAConstView = ClusterGeometrySoA::ConstView;

#endif  // DataFormats_ClusterGeometrySoA_interface_ClusterGeometryLayout_h
