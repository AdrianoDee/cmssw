#ifndef CUDADataFormats_Track_TrackForPVHeterogeneous_h
#define CUDADataFormats_Track_TrackForPVHeterogeneous_h

#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"
#include "CUDADataFormats/Track/interface/TrackForPVSoAT.h"

using TrackForPVHeterogeneous = HeterogeneousSoA<TrackForPV::TrackForPVSoA>;
using VertexForPVHeterogeneous = HeterogeneousSoA<TrackForPV::VertexForPVSoA>;

#endif  // #ifndef CUDADataFormats_Track_TrackForPVHeterogeneous_h
