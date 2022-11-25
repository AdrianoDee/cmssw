#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h

#include <cuda_runtime.h>
// #include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousHost.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousDevice.h"

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitsUtilities.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoAHost.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoADevice.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SimpleVector.h"

#include "CAHitNtupletGeneratorKernels.h"
#include "HelixFitOnGPU.h"

#include "GPUCACell.h"

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSetDescription;
}  // namespace edm

class CAHitNtupletGeneratorOnGPU {
public:
  using HitsView = trackingRecHitSoA::HitSoAView;
  using HitsConstView = trackingRecHitSoA::HitSoAConstView;
  using HitsOnGPU = trackingRecHit::TrackingRecHitSoADevice;
  using HitsOnCPU = trackingRecHit::TrackingRecHitSoAHost;
  // using hindex_type = TrackingRecHit2DSOAView::hindex_type;

  using Quality = pixelTrack::Quality;
  using OutputSoAHost = pixelTrack::TrackSoAHost;
  using OutputSoADevice = pixelTrack::TrackSoADevice;
  using HitContainer = pixelTrack::HitContainer;
  using Tuple = HitContainer;

  using QualityCuts = cAHitNtupletGenerator::QualityCuts;
  using Params = cAHitNtupletGenerator::Params;
  using Counters = cAHitNtupletGenerator::Counters;

public:
  CAHitNtupletGeneratorOnGPU(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC)
      : CAHitNtupletGeneratorOnGPU(cfg, iC) {}
  CAHitNtupletGeneratorOnGPU(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  static void fillDescriptions(edm::ParameterSetDescription& desc);
  static const char* fillDescriptionsLabel() { return "caHitNtupletOnGPU"; }

  void beginJob();
  void endJob();

  // On GPU
  pixelTrack::TrackSoADevice makeTuplesAsync(trackingRecHit::TrackingRecHitSoADevice const& hits_d,
                                             float bfield,
                                             cudaStream_t stream) const;

  // On CPU
  pixelTrack::TrackSoAHost makeTuples(trackingRecHit::TrackingRecHitSoAHost const& hits_h, float bfield) const;

private:
  void buildDoublets(HitsConstView hh, cudaStream_t stream) const;

  void hitNtuplets(HitsConstView hh, const edm::EventSetup& es, bool useRiemannFit, cudaStream_t cudaStream);

  void launchKernels(HitsConstView hh, bool useRiemannFit, cudaStream_t cudaStream) const;

  Params m_params;

  Counters* m_counters = nullptr;
};

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h
