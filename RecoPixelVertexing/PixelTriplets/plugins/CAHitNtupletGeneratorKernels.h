#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h

// #define GPU_DEBUG

#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"
#include "GPUCACell.h"
#include "gpuPixelDoublets.h"
// #define DUMP_GPU_TK_TUPLES

namespace caHitNtupletGenerator {

  struct ParamsCommon {
    ParamsCommon(
           bool onGPU,
           uint32_t minHitsPerNtuplet,
           // uint32_t maxNumberOfDoublets,
           uint16_t minHitsForSharingCuts,
           bool useRiemannFit,
           bool fitNas4,
           bool includeJumpingForwardDoublets,
           bool earlyFishbone,
           bool lateFishbone,
           bool doStats,
           // bool doClusterCut,
           // bool doZ0Cut,
           // bool doPtCut,
           bool doSharedHitCut,
           bool dupPassThrough,
           bool useSimpleTripletCleaner,
           float ptmin,
           float CAThetaCutBarrel,
           float CAThetaCutForward,
           float hardCurvCut,
           float dcaCutInnerTriplet,
           float dcaCutOuterTriplet)
        : onGPU_(onGPU),
          minHitsPerNtuplet_(minHitsPerNtuplet),
          // maxNumberOfDoublets_(maxNumberOfDoublets),
          minHitsForSharingCut_(minHitsForSharingCuts),
          useRiemannFit_(useRiemannFit),
          fitNas4_(fitNas4),
          includeJumpingForwardDoublets_(includeJumpingForwardDoublets),
          earlyFishbone_(earlyFishbone),
          lateFishbone_(lateFishbone),
          doStats_(doStats),
          // doClusterCut_(doClusterCut),
          // doZ0Cut_(doZ0Cut),
          // doPtCut_(doPtCut),
          doSharedHitCut_(doSharedHitCut),
          dupPassThrough_(dupPassThrough),
          useSimpleTripletCleaner_(useSimpleTripletCleaner),
          ptmin_(ptmin),
          CAThetaCutBarrel_(CAThetaCutBarrel),
          CAThetaCutForward_(CAThetaCutForward),
          hardCurvCut_(hardCurvCut),
          dcaCutInnerTriplet_(dcaCutInnerTriplet),
          dcaCutOuterTriplet_(dcaCutOuterTriplet)
          {}

    const bool onGPU_;
    const uint32_t minHitsPerNtuplet_;
    // const uint32_t maxNumberOfDoublets_;
    const uint16_t minHitsForSharingCut_;
    const bool useRiemannFit_;
    const bool fitNas4_;
    const bool includeJumpingForwardDoublets_;
    const bool earlyFishbone_;
    const bool lateFishbone_;
    const bool doStats_;
    // const bool doClusterCut_;
    // const bool doZ0Cut_;
    // const bool doPtCut_;
    const bool doSharedHitCut_;
    const bool dupPassThrough_;
    const bool useSimpleTripletCleaner_;
    const float ptmin_;
    const float CAThetaCutBarrel_;
    const float CAThetaCutForward_;
    const float hardCurvCut_;
    const float dcaCutInnerTriplet_;
    const float dcaCutOuterTriplet_;

    /// Compute the number of pairs
    inline uint32_t nPairs() const {
      return 0;
    }
  };

  template<typename TrackerTraits>
  struct ParamsT : public ParamsCommon {};

  template<>
  struct ParamsT<pixelTopology::Phase1> : public ParamsCommon {

    using QualityCuts = pixelTrack::QualityCutsT<pixelTopology::Phase1>;
    using CellCuts = gpuPixelDoublets::CellCutsT<pixelTopology::Phase1>;

    ParamsT(bool onGPU,
           uint32_t minHitsPerNtuplet,
           uint32_t maxNumberOfDoublets,
           uint16_t minHitsForSharingCuts,
           bool useRiemannFit,
           bool fitNas4,
           bool includeJumpingForwardDoublets,
           bool earlyFishbone,
           bool lateFishbone,
           bool idealConditions,
           bool doStats,
           bool doClusterCut,
           bool doZ0Cut,
           bool doPtCut,
           bool doSharedHitCut,
           bool dupPassThrough,
           bool useSimpleTripletCleaner,
           float ptmin,
           float CAThetaCutBarrel,
           float CAThetaCutForward,
           float hardCurvCut,
           float dcaCutInnerTriplet,
           float dcaCutOuterTriplet,
           QualityCuts const& cuts)
        : ParamsCommon(onGPU, minHitsPerNtuplet, /*maxNumberOfDoublets,*/ minHitsForSharingCuts, useRiemannFit,
          fitNas4, includeJumpingForwardDoublets, earlyFishbone, lateFishbone,
          doStats, /*doClusterCut, doZ0Cut, doPtCut,*/
          doSharedHitCut, dupPassThrough, useSimpleTripletCleaner, ptmin,
          CAThetaCutBarrel, CAThetaCutForward, hardCurvCut, dcaCutInnerTriplet,
          dcaCutOuterTriplet),
          // idealConditions_(idealConditions),
          cellCuts_(CellCuts{{maxNumberOfDoublets, doClusterCut, doZ0Cut, doPtCut}, idealConditions}),
          qualityCuts_(cuts)
          {}

    // quality cuts
    // const bool idealConditions_;

    CellCuts cellCuts_;

    QualityCuts qualityCuts_{// polynomial coefficients for the pT-dependent chi2 cut
                      {0.68177776, 0.74609577, -0.08035491, 0.00315399},
                      // max pT used to determine the chi2 cut
                      10.,
                      // chi2 scale factor: 30 for broken line fit, 45 for Riemann fit
                      30.,
                      // regional cuts for triplets
                      {
                          0.3,  // |Tip| < 0.3 cm
                          0.5,  // pT > 0.5 GeV
                          12.0  // |Zip| < 12.0 cm
                      },
                      // regional cuts for quadruplets
                      {
                          0.5,  // |Tip| < 0.5 cm
                          0.3,  // pT > 0.3 GeV
                          12.0  // |Zip| < 12.0 cm
                      }};

      /// Compute the number of pairs
      inline uint32_t nPairs() const {
        // take all layer pairs into account
        uint32_t nActualPairs = pixelTopology::Phase1::nPairs;
        if (not includeJumpingForwardDoublets_) {
          // exclude forward "jumping" layer pairs
          nActualPairs = pixelTopology::Phase1::nPairsForTriplets;
        }
        if (minHitsPerNtuplet_ > 3) {
          // for quadruplets, exclude all "jumping" layer pairs
          nActualPairs = pixelTopology::Phase1::nPairsForQuadruplets;
        }

        return nActualPairs;
      }

  };  // Params Phase1

  template<>
  struct ParamsT<pixelTopology::Phase2> : public ParamsCommon {

    using QualityCuts = pixelTrack::QualityCutsT<pixelTopology::Phase2>;
    using CellCuts = gpuPixelDoublets::CellCutsT<pixelTopology::Phase2>;

    ParamsT(bool onGPU,
           uint32_t minHitsPerNtuplet,
           uint32_t maxNumberOfDoublets,
           uint16_t minHitsForSharingCuts,
           bool useRiemannFit,
           bool fitNas4,
           bool includeJumpingForwardDoublets,
           bool includeFarForwards,
           bool earlyFishbone,
           bool lateFishbone,
           bool doStats,
           bool doClusterCut,
           bool doZ0Cut,
           bool doPtCut,
           bool doSharedHitCut,
           bool dupPassThrough,
           bool useSimpleTripletCleaner,
           float ptmin,
           float CAThetaCutBarrel,
           float CAThetaCutForward,
           float hardCurvCut,
           float dcaCutInnerTriplet,
           float dcaCutOuterTriplet,
           QualityCuts const& cuts)
        : ParamsCommon(onGPU, minHitsPerNtuplet, /*maxNumberOfDoublets,*/ minHitsForSharingCuts, useRiemannFit,
          fitNas4, includeJumpingForwardDoublets, earlyFishbone, lateFishbone,
          doStats, /*doClusterCut, doZ0Cut, doPtCut,*/
          doSharedHitCut, dupPassThrough, useSimpleTripletCleaner, ptmin,
          CAThetaCutBarrel, CAThetaCutForward, hardCurvCut, dcaCutInnerTriplet,
          dcaCutOuterTriplet),
          includeFarForwards_(includeFarForwards),
          cellCuts_(CellCuts{{maxNumberOfDoublets, doClusterCut, doZ0Cut, doPtCut}}),
          qualityCuts_(cuts)
          {}


    const bool includeFarForwards_;
    // quality cuts
    CellCuts cellCuts_;

    QualityCuts qualityCuts_{ 50.0f, /*chi2*/ 0.9f, /* pT in Gev*/  0.4f, /*zip in cm*/ 12.0f /*tip in cm*/};

    inline uint32_t nPairs() const {
      // take all layer pairs into account
      uint32_t nActualPairs = pixelTopology::Phase2::nPairs;
      if (includeFarForwards_) {
        // considera far forwards (> 11 & > 23)
        nActualPairs = pixelTopology::Phase2::nPairsFarForwards;
      }
      if (includeJumpingForwardDoublets_) {
        // include jumping forwards
        nActualPairs = pixelTopology::Phase2::nPairsForJumpingForwards;
      }

      return nActualPairs;
    }

  };  // Params Phase1

  // counters
  struct Counters {
    unsigned long long nEvents;
    unsigned long long nHits;
    unsigned long long nCells;
    unsigned long long nTuples;
    unsigned long long nFitTracks;
    unsigned long long nLooseTracks;
    unsigned long long nGoodTracks;
    unsigned long long nUsedHits;
    unsigned long long nDupHits;
    unsigned long long nFishCells;
    unsigned long long nKilledCells;
    unsigned long long nEmptyCells;
    unsigned long long nZeroTrackCells;
  };

  using Quality = pixelTrack::Quality;

}  // namespace caHitNtupletGenerator

template <typename TTraits, typename TTTraits>
class CAHitNtupletGeneratorKernelsBaseT {
public:
  using Traits = TTraits;
  using TrackerTraits = TTTraits;
  using QualityCuts = pixelTrack::QualityCutsT<TrackerTraits>;
  using Params = caHitNtupletGenerator::ParamsT<TrackerTraits>;
  using Counters = caHitNtupletGenerator::Counters;

  template <typename T>
  using unique_ptr = typename Traits::template unique_ptr<T>;

  using HitsView = TrackingRecHit2DSOAViewT<TrackerTraits>;
  using HitsOnCPU = TrackingRecHit2DHeterogeneousT<Traits,TrackerTraits>;

  using HitToTuple = caStructures::HitToTupleT<TrackerTraits>;
  using TupleMultiplicity = caStructures::TupleMultiplicityT<TrackerTraits>;
  using CellNeighborsVector = caStructures::CellNeighborsVectorT<TrackerTraits>;
  using CellNeighbors = caStructures::CellNeighborsT<TrackerTraits>;
  using CellTracksVector = caStructures::CellTracksVectorT<TrackerTraits>;
  using CellTracks = caStructures::CellTracksT<TrackerTraits>;
  using OuterHitOfCellContainer = caStructures::OuterHitOfCellContainerT<TrackerTraits>;
  using OuterHitOfCell = caStructures::OuterHitOfCellT<TrackerTraits>;

  using GPUCACell = GPUCACellT<TrackerTraits>;

  using Quality = pixelTrack::Quality;
  using TkSoA = pixelTrack::TrackSoAT<TrackerTraits>;
  using HitContainer = pixelTrack::HitContainerT<TrackerTraits>;

  CAHitNtupletGeneratorKernelsBaseT(Params const& params)
      : params_(params), paramsMaxDoubletes3Quarters_(3 * params.cellCuts_.maxNumberOfDoublets_ / 4) {}
  ~CAHitNtupletGeneratorKernelsBaseT() = default;

  TupleMultiplicity const* tupleMultiplicity() const { return device_tupleMultiplicity_.get(); }

  void launchKernels(HitsOnCPU const& hh, TkSoA* tuples_d, cudaStream_t cudaStream);

  void classifyTuples(HitsOnCPU const& hh, TkSoA* tuples_d, cudaStream_t cudaStream);

  void buildDoublets(HitsOnCPU const& hh, cudaStream_t stream);
  void allocateOnGPU(int32_t nHits, cudaStream_t stream);
  void cleanup(cudaStream_t cudaStream);

  static void printCounters(Counters const* counters);
  void setCounters(Counters* counters) { counters_ = counters; }

protected:
  Counters* counters_ = nullptr;

  // workspace
  unique_ptr<unsigned char[]> cellStorage_;
  unique_ptr<CellNeighborsVector> device_theCellNeighbors_;
  CellNeighbors* device_theCellNeighborsContainer_;
  unique_ptr<CellTracksVector> device_theCellTracks_;
  CellTracks* device_theCellTracksContainer_;

  unique_ptr<GPUCACell[]> device_theCells_;
  unique_ptr<OuterHitOfCellContainer[]> device_isOuterHitOfCell_;
  OuterHitOfCell isOuterHitOfCell_;
  uint32_t* device_nCells_ = nullptr;

  unique_ptr<HitToTuple> device_hitToTuple_;
  unique_ptr<uint32_t[]> device_hitToTupleStorage_;
  typename
  HitToTuple::View hitToTupleView_;

  cms::cuda::AtomicPairCounter* device_hitToTuple_apc_ = nullptr;

  cms::cuda::AtomicPairCounter* device_hitTuple_apc_ = nullptr;

  unique_ptr<TupleMultiplicity> device_tupleMultiplicity_;

  unique_ptr<cms::cuda::AtomicPairCounter::c_type[]> device_storage_;
  // params
  Params const& params_;
  /// Intermediate result avoiding repeated computations.
  const uint32_t paramsMaxDoubletes3Quarters_;
  /// Compute the number of doublet blocks for block size
  inline uint32_t nDoubletBlocks(uint32_t blockSize) {
    // We want (3 * params_.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize, but first part is pre-computed.
    return (paramsMaxDoubletes3Quarters_ + blockSize - 1) / blockSize;
  }

  /// Compute the number of quadruplet blocks for block size
  inline uint32_t nQuadrupletBlocks(uint32_t blockSize) {
    // pixelTopology::maxNumberOfQuadruplets is a constexpr, so the compiler will pre compute the 3*max/4
    return (3 * TrackerTraits::maxNumberOfQuadruplets / 4 + blockSize - 1) / blockSize;
  }

};

template <typename Traits,typename TrackerTraits>
class CAHitNtupletGeneratorKernelsGPUT : public CAHitNtupletGeneratorKernelsBaseT<Traits,TrackerTraits>{};

template <typename TrackerTraits>
class CAHitNtupletGeneratorKernelsGPUT<cms::cudacompat::GPUTraits,TrackerTraits> : public CAHitNtupletGeneratorKernelsBaseT<cms::cudacompat::GPUTraits,TrackerTraits>
{
  using CAHitNtupletGeneratorKernelsBaseT<cms::cudacompat::GPUTraits,TrackerTraits>::CAHitNtupletGeneratorKernelsBaseT;
  using HitsOnCPU = TrackingRecHit2DHeterogeneousT<cms::cudacompat::GPUTraits,TrackerTraits>;
  using TkSoA = pixelTrack::TrackSoAT<TrackerTraits>;
  using Counters = caHitNtupletGenerator::Counters;
  using HitContainer = pixelTrack::HitContainerT<TrackerTraits>;
  using CellNeighborsVector = caStructures::CellNeighborsVectorT<TrackerTraits>;
  using HitToTuple = caStructures::HitToTupleT<TrackerTraits>;
  using CellTracksVector = caStructures::CellTracksVectorT<TrackerTraits>;
  using TupleMultiplicity = caStructures::TupleMultiplicityT<TrackerTraits>;

  public:

    void launchKernels(HitsOnCPU const& hh, TkSoA* tuples_d, cudaStream_t cudaStream);
    void classifyTuples(HitsOnCPU const& hh, TkSoA* tuples_d, cudaStream_t cudaStream);
    void buildDoublets(HitsOnCPU const& hh, cudaStream_t stream);
    void allocateOnGPU(int32_t nHits, cudaStream_t stream);
    static void printCounters(Counters const* counters);

};


template <typename Traits,typename TrackerTraits>
class CAHitNtupletGeneratorKernelsCPUT : public CAHitNtupletGeneratorKernelsBaseT<Traits,TrackerTraits> {};

template <typename TrackerTraits>
class CAHitNtupletGeneratorKernelsCPUT<cms::cudacompat::CPUTraits,TrackerTraits> : public CAHitNtupletGeneratorKernelsBaseT<cms::cudacompat::CPUTraits,TrackerTraits>
{
  using CAHitNtupletGeneratorKernelsBaseT<cms::cudacompat::CPUTraits,TrackerTraits>::CAHitNtupletGeneratorKernelsBaseT;
  using HitsOnCPU = TrackingRecHit2DHeterogeneousT<cms::cudacompat::CPUTraits,TrackerTraits>;
  using TkSoA = pixelTrack::TrackSoAT<TrackerTraits>;
  using Counters = caHitNtupletGenerator::Counters;
  using CellNeighborsVector = caStructures::CellNeighborsVectorT<TrackerTraits>;
  using HitToTuple = caStructures::HitToTupleT<TrackerTraits>;
  using CellTracksVector = caStructures::CellTracksVectorT<TrackerTraits>;
  using TupleMultiplicity = caStructures::TupleMultiplicityT<TrackerTraits>;

  public:

    void launchKernels(HitsOnCPU const& hh, TkSoA* tuples_d, cudaStream_t cudaStream);
    void classifyTuples(HitsOnCPU const& hh, TkSoA* tuples_d, cudaStream_t cudaStream);
    void buildDoublets(HitsOnCPU const& hh, cudaStream_t stream);
    void allocateOnGPU(int32_t nHits, cudaStream_t stream);
    static void printCounters(Counters const* counters);

};

template<typename TrackerTraits>
using CAHitNtupletGeneratorKernelsGPU = CAHitNtupletGeneratorKernelsGPUT<cms::cudacompat::GPUTraits,TrackerTraits>;

template<typename TrackerTraits>
using CAHitNtupletGeneratorKernelsCPU = CAHitNtupletGeneratorKernelsCPUT<cms::cudacompat::CPUTraits,TrackerTraits>;

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h
