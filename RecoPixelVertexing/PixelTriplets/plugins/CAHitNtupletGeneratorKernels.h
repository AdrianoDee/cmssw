#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h

// #define GPU_DEBUG

#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"
#include "GPUCACell.h"

// #define DUMP_GPU_TK_TUPLES

namespace cAHitNtupletGenerator {

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
    unsigned long long nKilledCells;
    unsigned long long nEmptyCells;
    unsigned long long nZeroTrackCells;
  };

  // using HitsView = TrackingRecHit2DSOAView;
  // using HitsOnGPU = TrackingRecHit2DSOAView;

  // using HitToTuple = pixelTopology::HitToTuple;
  // using TupleMultiplicity = pixelTopology::TupleMultiplicity;

  using Quality = pixelTrack::Quality;
  using TkSoA = pixelTrack::TrackSoA;
  using HitContainer = pixelTrack::HitContainer;

  struct QualityCuts {
    // chi2 cut = chi2Scale * (chi2Coeff[0] + pT/GeV * (chi2Coeff[1] + pT/GeV * (chi2Coeff[2] + pT/GeV * chi2Coeff[3])))
    float chi2Coeff[4];
    float chi2MaxPt;  // GeV
    float chi2Scale;

    struct Region {
      float maxTip;  // cm
      float minPt;   // GeV
      float maxZip;  // cm
    };

    Region triplet;
    Region quadruplet;
  };

  // params (FIXME: thi si a POD: so no constructor no traling _  and no const as params_ is already const)
  struct Params {
    Params(bool onGPU,
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
           bool isPhase2,
           float ptmin,
           float CAThetaCutBarrel,
           float CAThetaCutForward,
           float hardCurvCut,
           float dcaCutInnerTriplet,
           float dcaCutOuterTriplet,

           QualityCuts const& cuts)
        : onGPU_(onGPU),
          minHitsPerNtuplet_(minHitsPerNtuplet),
          maxNumberOfDoublets_(maxNumberOfDoublets),
          minHitsForSharingCut_(minHitsForSharingCuts),
          useRiemannFit_(useRiemannFit),
          fitNas4_(fitNas4),
          includeJumpingForwardDoublets_(includeJumpingForwardDoublets),
          earlyFishbone_(earlyFishbone),
          lateFishbone_(lateFishbone),
          idealConditions_(idealConditions),
          doStats_(doStats),
          doClusterCut_(doClusterCut),
          doZ0Cut_(doZ0Cut),
          doPtCut_(doPtCut),
          doSharedHitCut_(doSharedHitCut),
          dupPassThrough_(dupPassThrough),
          useSimpleTripletCleaner_(useSimpleTripletCleaner),
          isPhase2_(isPhase2),
          ptmin_(ptmin),
          CAThetaCutBarrel_(CAThetaCutBarrel),
          CAThetaCutForward_(CAThetaCutForward),
          hardCurvCut_(hardCurvCut),
          dcaCutInnerTriplet_(dcaCutInnerTriplet),
          dcaCutOuterTriplet_(dcaCutOuterTriplet),
          cuts_(cuts) {}

    const bool onGPU_;
    const uint32_t minHitsPerNtuplet_;
    const uint32_t maxNumberOfDoublets_;
    const uint16_t minHitsForSharingCut_;
    const bool useRiemannFit_;
    const bool fitNas4_;
    const bool includeJumpingForwardDoublets_;
    const bool earlyFishbone_;
    const bool lateFishbone_;
    const bool idealConditions_;
    const bool doStats_;
    const bool doClusterCut_;
    const bool doZ0Cut_;
    const bool doPtCut_;
    const bool doSharedHitCut_;
    const bool dupPassThrough_;
    const bool useSimpleTripletCleaner_;
    const bool isPhase2_;
    const float ptmin_;
    const float CAThetaCutBarrel_;
    const float CAThetaCutForward_;
    const float hardCurvCut_;
    const float dcaCutInnerTriplet_;
    const float dcaCutOuterTriplet_;

    // quality cuts
    QualityCuts cuts_{// polynomial coefficients for the pT-dependent chi2 cut
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

  };  // Params

}  // namespace cAHitNtupletGenerator

template <typename TTraits, typename TTTraits>
class CAHitNtupletGeneratorKernelsBaseT {
public:
  using Traits = TTraits;
  using TrackerTraits = TTTraits;
  using QualityCuts = cAHitNtupletGenerator::QualityCuts;
  using Params = cAHitNtupletGenerator::Params;
  using Counters = cAHitNtupletGenerator::Counters;

  template <typename T>
  using unique_ptr = typename Traits::template unique_ptr<T>;

  using HitsView = TrackingRecHit2DSOAViewT<TrackerTraits>;
  using HitsOnCPU = TrackingRecHit2DHeterogeneousT<Traits,TrackerTraits>;

  using HitToTuple = pixelTopology::HitToTupleT<TrackerTraits>;
  using TupleMultiplicity = pixelTopology::TupleMultiplicityT<TrackerTraits>;
  using CellNeighborsVector = pixelTopology::CellNeighborsVectorT<TrackerTraits>;
  using CellNeighbors = pixelTopology::CellNeighborsT<TrackerTraits>;
  using CellTracksVector = pixelTopology::CellTracksVectorT<TrackerTraits>;
  using CellTracks = pixelTopology::CellTracksT<TrackerTraits>;
  using OuterHitOfCellContainer = pixelTopology::OuterHitOfCellContainerT<TrackerTraits>;
  using OuterHitOfCell = pixelTopology::OuterHitOfCellT<TrackerTraits>;

  using GPUCACell = GPUCACellT<TrackerTraits>;

  using Quality = pixelTrack::Quality;
  using TkSoA = pixelTrack::TrackSoAT<TrackerTraits>;
  using HitContainer = pixelTrack::HitContainerT<TrackerTraits>;

  CAHitNtupletGeneratorKernelsBaseT(Params const& params)
      : params_(params), paramsMaxDoubletes3Quarters_(3 * params.maxNumberOfDoublets_ / 4) {}
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
  using Counters = cAHitNtupletGenerator::Counters;
  using HitContainer = pixelTrack::HitContainerT<TrackerTraits>;
  using CellNeighborsVector = pixelTopology::CellNeighborsVectorT<TrackerTraits>;
  using HitToTuple = pixelTopology::HitToTupleT<TrackerTraits>;
  using CellTracksVector = pixelTopology::CellTracksVectorT<TrackerTraits>;
  using TupleMultiplicity = pixelTopology::TupleMultiplicityT<TrackerTraits>;

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
  using Counters = cAHitNtupletGenerator::Counters;
  using QualityCuts = cAHitNtupletGenerator::QualityCuts;
  using CellNeighborsVector = pixelTopology::CellNeighborsVectorT<TrackerTraits>;
  using HitToTuple = pixelTopology::HitToTupleT<TrackerTraits>;
  using CellTracksVector = pixelTopology::CellTracksVectorT<TrackerTraits>;
  using TupleMultiplicity = pixelTopology::TupleMultiplicityT<TrackerTraits>;

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

// using CAHitNtupletGeneratorKernelsGPU = CAHitNtupletGeneratorKernelsT<cms::cudacompat::GPUTraits,pixelTopology::Phase1>;
// using CAHitNtupletGeneratorKernelsCPU = CAHitNtupletGeneratorKernelsT<cms::cudacompat::CPUTraits,pixelTopology::Phase1>;
//
// using CAHitNtupletGeneratorKernelsGPUPhase2 = CAHitNtupletGeneratorKernelsT<cms::cudacompat::GPUTraits,pixelTopology::Phase1>;
// using CAHitNtupletGeneratorKernelsCPUPhase2 = CAHitNtupletGeneratorKernelsT<cms::cudacompat::CPUTraits,pixelTopology::Phase1>;

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h
