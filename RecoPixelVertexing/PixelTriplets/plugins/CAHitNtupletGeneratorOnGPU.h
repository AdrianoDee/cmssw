#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h

#include <cuda_runtime.h>
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SimpleVector.h"

#include "CAHitNtupletGeneratorKernels.h"
#include "HelixFitOnGPU.h"

#include "GPUCACell.h"

#define GPU_DEBUG 1
namespace edm {
  class Event;
  class EventSetup;
  class ParameterSetDescription;
}  // namespace edm

template <typename TrackerTraits>
class CAHitNtupletGeneratorOnGPUT {
public:

  using PixelTrackHeterogeneous = PixelTrackHeterogeneousT<TrackerTraits>;

  using HitsView = TrackingRecHit2DSOAViewT<TrackerTraits>;
  using HitsOnGPU = TrackingRecHit2DGPUT<TrackerTraits>;
  using HitsOnCPU = TrackingRecHit2DCPUT<TrackerTraits>;
  using hindex_type = typename HitsView::hindex_type;

  using HitToTuple = pixelTopology::HitToTupleT<TrackerTraits>;
  using TupleMultiplicity = pixelTopology::TupleMultiplicityT<TrackerTraits>;
  using OuterHitOfCell = pixelTopology::OuterHitOfCellT<TrackerTraits>;

  using GPUCACell = GPUCACellT<TrackerTraits>;
  using OutputSoA = pixelTrack::TrackSoAT<TrackerTraits>;
  using HitContainer = typename OutputSoA::HitContainer;
  using Tuple = HitContainer;

  using CellNeighborsVector = pixelTopology::CellNeighborsVectorT<TrackerTraits>;
  using CellTracksVector = pixelTopology::CellTracksVectorT<TrackerTraits>;

  using Quality = pixelTrack::Quality;

  using QualityCuts = cAHitNtupletGenerator::QualityCuts;
  using Params = cAHitNtupletGenerator::Params;
  using Counters = cAHitNtupletGenerator::Counters;

public:
  CAHitNtupletGeneratorOnGPUT(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC)
      : CAHitNtupletGeneratorOnGPUT(cfg, iC) {}
  CAHitNtupletGeneratorOnGPUT(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  ~CAHitNtupletGeneratorOnGPUT();

  static void fillDescriptions(edm::ParameterSetDescription& desc);
  //static const char* fillDescriptionsLabel() { return "caHitNtupletOnGPU"; }


  PixelTrackHeterogeneous makeTuplesAsync(HitsOnGPU const& hits_d, float bfield, cudaStream_t stream) const;

  PixelTrackHeterogeneous makeTuples(HitsOnCPU const& hits_d, float bfield) const;

private:
  void buildDoublets(HitsOnGPU const& hh, cudaStream_t stream) const;

  void hitNtuplets(HitsOnGPU const& hh, const edm::EventSetup& es, bool useRiemannFit, cudaStream_t cudaStream);

  void launchKernels(HitsOnGPU const& hh, bool useRiemannFit, cudaStream_t cudaStream) const;

  Params m_params;

  Counters* m_counters = nullptr;
};


#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h
