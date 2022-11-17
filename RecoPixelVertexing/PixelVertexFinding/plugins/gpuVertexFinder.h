#ifndef RecoPixelVertexing_PixelVertexFinding_plugins_gpuVertexFinder_h
#define RecoPixelVertexing_PixelVertexFinding_plugins_gpuVertexFinder_h

#include <cstddef>
#include <cstdint>

#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoAHeterogeneousHost.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoAHeterogeneousDevice.h"
#include "CUDADataFormats/Vertex/interface/ZVertexUtilities.h"
#include "WorkSpaceUtilities.h"
#include "WorkSpaceSoAHeterogeneousHost.h"
#include "WorkSpaceSoAHeterogeneousDevice.h"

namespace gpuVertexFinder {

  using VtxSoAView = zVertex::ZVertexSoAView;
  using TkSoAConstView = pixelTrack::TrackSoAConstView;
  using WsSoAView = gpuVertexFinder::workSpace::WorkSpaceSoAView;

  __global__ void init(VtxSoAView pdata, WsSoAView pws) {
    zVertex::utilities::init(pdata);
    gpuVertexFinder::workSpace::utilities::init(pws);
  }

  class Producer {
  public:
    Producer(bool oneKernel,
             bool useDensity,
             bool useDBSCAN,
             bool useIterative,
             int iminT,      // min number of neighbours to be "core"
             float ieps,     // max absolute distance to cluster
             float ierrmax,  // max error to be "seed"
             float ichi2max  // max normalized distance to cluster
             )
        : oneKernel_(oneKernel && !(useDBSCAN || useIterative)),
          useDensity_(useDensity),
          useDBSCAN_(useDBSCAN),
          useIterative_(useIterative),
          minT(iminT),
          eps(ieps),
          errmax(ierrmax),
          chi2max(ichi2max) {}

    ~Producer() = default;

    zVertex::ZVertexSoADevice makeAsync(cudaStream_t stream, TkSoAConstView tracks_view, float ptMin, float ptMax) const;
    zVertex::ZVertexSoAHost make(TkSoAConstView tracks_view, float ptMin, float ptMax) const;

  private:
    const bool oneKernel_;
    const bool useDensity_;
    const bool useDBSCAN_;
    const bool useIterative_;

    int minT;       // min number of neighbours to be "core"
    float eps;      // max absolute distance to cluster
    float errmax;   // max error to be "seed"
    float chi2max;  // max normalized distance to cluster
  };

}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_plugins_gpuVertexFinder_h
