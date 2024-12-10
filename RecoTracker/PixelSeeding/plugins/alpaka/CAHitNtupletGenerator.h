#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGenerator_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGenerator_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/alpaka/FrameSoACollection.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/CAParamsSoACollection.h"

#include "CACell.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "HelixFit.h"

namespace edm {
  class ParameterSetDescription;
}  // namespace edm

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  class CAHitNtupletGenerator {
  public:
    using HitsView = TrackingRecHitSoAView<TrackerTraits>;
    using HitsConstView = TrackingRecHitSoAConstView<TrackerTraits>;
    using HitsOnDevice = TrackingRecHitsSoACollection<TrackerTraits>;
    using HitsOnHost = TrackingRecHitHost<TrackerTraits>;
    using hindex_type = typename TrackingRecHitSoA<TrackerTraits>::hindex_type;

    using HitToTuple = caStructures::HitToTupleT<TrackerTraits>;
    using TupleMultiplicity = caStructures::TupleMultiplicityT<TrackerTraits>;
    using OuterHitOfCell = caStructures::OuterHitOfCellT<TrackerTraits>;

    using CACell = CACellT<TrackerTraits>;
    using TkSoAHost = TracksHost<TrackerTraits>;
    using TkSoADevice = TracksSoACollection<TrackerTraits>;
    using HitContainer = typename ::reco::TrackSoA<TrackerTraits>::HitContainer;
    using Tuple = HitContainer;

    using CellNeighborsVector = caStructures::CellNeighborsVectorT<TrackerTraits>;
    using CellTracksVector = caStructures::CellTracksVectorT<TrackerTraits>;

    using Quality = ::pixelTrack::Quality;

    using QualityCuts = ::pixelTrack::QualityCutsT<TrackerTraits>;
    using Params = caHitNtupletGenerator::ParamsT<TrackerTraits>;
    using Counters = caHitNtupletGenerator::Counters;

    using FrameOnDevice = FrameSoACollection;
    using CAParamsOnDevice = reco::CAParamsSoACollection;
  public:
    CAHitNtupletGenerator(const edm::ParameterSet& cfg);

    static void fillPSetDescription(edm::ParameterSetDescription& desc);

    // NOTE: beginJob and endJob were meant to be used
    // to fill the statistics. This is still not implemented in Alpaka
    // since we are missing the begin/endJob functionality for the Alpaka
    // producers.
    //
    // void beginJob();
    // void endJob();

    TkSoADevice makeTuplesAsync(HitsOnDevice const& hits_d,
                                FrameOnDevice const& frame_d,
                                CAParamsOnDevice const& params_d,
                                float bfield,
                                Queue& queue) const;

  private:
    void prepareHits(const HitsConstView& hh, const ::reco::CALayersSoAConstView& ll, Queue& queue) const;

    void buildDoublets(const HitsConstView& hh, const ::reco::CALayersSoAConstView& ll, uint32_t offsetBPIX2, Queue& queue) const;

    void hitNtuplets(const HitsConstView& hh, const edm::EventSetup& es, bool useRiemannFit, Queue& queue);

    void launchKernels(const HitsConstView& hh, bool useRiemannFit, Queue& queue) const;

    Params m_params;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGenerator_h
