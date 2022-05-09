#ifndef trackFilterCUDA_h
#define trackFilterCUDA_h
#include "CUDADataFormats/Track/interface/TrackForPVHeterogeneous.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include <cstddef>
#include <cstdint>
#include "CUDADataFormats/Vertex/interface/ZVertexHeterogeneous.h"

namespace trackFilterCUDA {
  struct filterParameters {
    double maxSignificance;
    double maxdxyError;
    double maxdzError;
    double minpAtIP;
    double maxetaAtIP;
    double maxchi2;
    int minpixelHits;
    int mintrackerHits;
    double vertexSize;
    double d0CutOff;
  };
void sorterWrapper(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, cudaStream_t stream);
 // void filterWrapper(unsigned int ntracks, TrackForPV::TrackForPVSoA* tracks, filterParameters params, double* osumtkwt, cudaStream_t stream);
}

#endif
