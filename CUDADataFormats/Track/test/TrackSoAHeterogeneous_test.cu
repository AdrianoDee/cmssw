#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousDevice.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousHost.h"
#include "HeterogeneousCore/CUDAUtilities/interface/OneToManyAssoc.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

namespace testTrackSoAHeterogeneousT {

  __global__ void fill(pixelTrack::TrackSoAView tracks_view) {
    int i = threadIdx.x;
    if (i == 0) {
      tracks_view.nTracks() = 420;
    }

    for (int j = i; j < tracks_view.metadata().size(); j += blockDim.x) {
      tracks_view[j].pt() = (float)j;
      tracks_view[j].eta() = (float)j;
      tracks_view[j].chi2() = (float)j;
      tracks_view[j].quality() = (pixelTrack::Quality)(j % 256);
      tracks_view[j].nLayers() = j % 128;
      tracks_view.hitIndices().off[j] = j;
    }
  }

  // TODO: Use TrackSoAConstView when https://github.com/cms-sw/cmssw/pull/39919 is merged
  __global__ void verify(pixelTrack::TrackSoAView tracks_view) {
    int i = threadIdx.x;

    if (i == 0) {
      printf("SoA size: % d, block dims: % d\n", tracks_view.metadata().size(), blockDim.x);
      assert(tracks_view.nTracks() == 420);
    }
    for (int j = i; j < tracks_view.metadata().size(); j += blockDim.x) {
      assert(abs(tracks_view[j].pt() - (float)j) < .0001);
      assert(abs(tracks_view[j].eta() - (float)j) < .0001);
      assert(abs(tracks_view[j].chi2() - (float)j) < .0001);
      assert(tracks_view[j].quality() == (pixelTrack::Quality)(j % 256));
      assert(tracks_view[j].nLayers() == j % 128);
      assert(tracks_view.hitIndices().off[j] == j);
    }
  }

  void runKernels(pixelTrack::TrackSoAView tracks_view, cudaStream_t stream) {
    fill<<<1, 1024, 0, stream>>>(tracks_view);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    verify<<<1, 1024, 0, stream>>>(tracks_view);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
  }

}  // namespace testTrackSoAHeterogeneousT
