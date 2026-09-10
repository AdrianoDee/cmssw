#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGenerator_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGenerator_h

#include <alpaka/alpaka.hpp>

#include <optional>
#include <memory>

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/OTRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/StubsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsMaskingSoACollection.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/CAGeometrySoACollection.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/StackedModuleGeometrySoACollection.h"

#include "CACell.h"
#include "CAExtensionKernels.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "CATripletDumpMacro.h"
#include "HelixFit.h"

namespace edm {
  class ParameterSetDescription;
}  // namespace edm

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  class CAHitNtupletGenerator {
  public:
    using HitsView = ::reco::TrackingRecHitView;
    using HitsConstView = ::reco::TrackingRecHitConstView;
    using HitsOnDevice = reco::TrackingRecHitsSoACollection;
    using HitsOnHost = ::reco::TrackingRecHitHost;
    using MapToHit = reco::TrackingRecHitsMaskingSoACollection;

    using MapToHitConstView = ::reco::TrackingRecHitsMaskingConstView;

    using TkSoADevice = reco::TracksSoACollection;
    using Quality = ::pixelTrack::Quality;

    using QualityCuts = ::pixelTrack::QualityCutsT<TrackerTraits>;
    using Params = caHitNtupletGenerator::ParamsT<TrackerTraits>;
    using Counters = caHitNtupletGenerator::Counters;

    using CAGeometryOnDevice = reco::CAGeometrySoACollection;

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

    // State crossing the acquire->produce seam: acquire() runs beginTuplesAsync (hit prep, doublets,
    // CA kernels, one async D2H of the tuple-multiplicity per-N-bin offsets), produce() runs
    // finishTuplesAsync (both fit passes, classification, product). The framework waits for that copy
    // at the seam, so everything mirrored here has landed before finishTuplesAsync runs.
    struct PendingTuples {
      // unique_ptr, not optional: the kernels object is not move-constructible (reference and const
      // members) while PendingTuples must move across the seam.
      std::unique_ptr<CAHitNtupletGeneratorKernels<TrackerTraits>> kernels;
      std::optional<TkSoADevice> tracks;
      // Pinned host mirror of the tuple-multiplicity per-N-bin offsets.
      std::optional<cms::alpakatools::host_buffer<uint32_t[]>> offsetsHost;
      // Pinned host mirror of the 5-word extraStorage counter block, filled only when
      // delayAllocations is on. Word [0] is the tuple AtomicPairCounter, decoded to size the
      // hit->track storage without a host wait.
      std::optional<cms::alpakatools::host_buffer<cms::alpakatools::AtomicPairCounter::DoubleWord[]>> countsHost;
      uint32_t maxTuples = 0;
      uint32_t maxDoublets = 0;
      float bfield = 0.f;
      const float* rhoMapDevice = nullptr;
      // Normalized (Bz,Br) r-z field map, device-resident; null selects the scalar-field fallback.
      const float* bMapDevice = nullptr;
      bool built = false;  // false = early-out (too few hits): tracks holds the empty collection
    };

    // Two handles onto the CA geometry SoA, split by ownership: geometry_d carries the `layers` and
    // `modules` conditions blocks, cuts_d the per-iteration `graph`, `doubletCuts`, `tripletCuts` and
    // `ntupletCuts` blocks. The same object may be passed twice. finishTuplesAsync reads `modules` only.
    PendingTuples beginTuplesAsync(HitsOnDevice const& hits_d,
                                   CAGeometryOnDevice const& geometry_d,
                                   CAGeometryOnDevice const& cuts_d,
                                   float bfield,
                                   uint32_t maxDoublets,
                                   uint32_t maxTuples,
                                   // Optional per-iteration hit mask; a default-constructed (null,
                                   // zero-row) view disables masking.
                                   MapToHitConstView const& maskView,
                                   Queue& queue,
                                   const float* rhoMapDevice,
                                   const float* bMapDevice) const;

    TkSoADevice finishTuplesAsync(PendingTuples&& pending,
                                  HitsOnDevice const& hits_d,
                                  CAGeometryOnDevice const& geometry_d,
                                  Queue& queue) const;

    // Always-on overflow reporting: the capacity guards in the build kernels truncate silently.
    // Per-stream device accumulator (kOvfWords words, layout in Kernel_overflowSentinel; slots 0-5
    // in use, 6-7 reserved), mirrored to pinned host by an async D2H enqueued after classifyTuples;
    // by endStream every event queue has drained, so the mirror holds the final totals. Each event
    // writes the mirror slot its parity selects, so the in-flight copies of consecutive events never
    // touch the same bytes; reportOverflows() takes the element-wise maximum of the two slots, which
    // is exact because the device counters only grow.
    static constexpr uint32_t kOvfWords = 8u;
    mutable std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> ovfAccum_;
    mutable std::optional<cms::alpakatools::host_buffer<uint32_t[]>> ovfHost_;
    mutable uint32_t ovfSlot_ = 0;
    void reportOverflows(std::string const& moduleLabel) const;

#ifdef CA_TRIPLET_DUMP
    // Per-built-triplet training capture: the kernels object owning the buffer dies with
    // PendingTuples, so finishTuplesAsync moves it here for the producer to emplace as the
    // 'Triplet' nano product.
    mutable std::optional<TripletDumpSoACollection> device_tripletDump_;
    std::optional<TripletDumpSoACollection>& tripletDumpBuffer() const { return device_tripletDump_; }
#endif

  private:
    Params m_params;
    // One-shot post-fit device dump of the first fitted tracks; config parameter verboseBLFit.
    bool m_verboseBLDump;
  };

  // class CAHitMaskingAndMerger {
  // public:
  //   using MapToHit = reco::TrackingRecHitsMaskingCollection;
  //   using TkSoADevice = reco::TracksSoACollection;

  // public:
  //   CAHitMaskingAndMerger() = default;
  //   ~CAHitMaskingAndMerger() = default;

  //   CAHitMaskingAndMerger(const CAHitMaskingAndMerger&) = delete;
  //   CAHitMaskingAndMerger(CAHitMaskingAndMerger&&) = delete;
  //   CAHitMaskingAndMerger& operator=(const CAHitMaskingAndMerger&) = delete;
  //   CAHitMaskingAndMerger& operator=(CAHitMaskingAndMerger&&) = delete;

  //   MapToHit makeMaskingAsync(MapToHit const& mask_d,
  //                             TkSoADevice const& tracks_d,
  //                             const pixelTrack::Quality minQuality,
  //                             uint32_t const& iterationIndex,
  //                             Queue& queue,
  //                             bool applyMasking = true,
  //                             bool maskAttachedHits = false) const;

  //   // Device-side gather/compact: reads each input's nTracks() and hitOffsets() on device and packs
  //   // the track and trackHits columns into a dense merged layout, writing the merged nTracks and the
  //   // shifted hitOffsets on device. No host readback in the sizing or in the copy.
  //   void mergeGather(TkSoADevice& outTracks,
  //                    TkSoADevice const& inp0Tracks,
  //                    TkSoADevice const& inp1Tracks,
  //                    int nInputs,
  //                    int32_t* armBuf,
  //                    const int32_t arm0,
  //                    const int32_t arm1,
  //                    Queue& queue) const;

  //   TkSoADevice makeFilteredTracks(int const& nTracks,
  //                                  int const& nHits,
  //                                  TkSoADevice const& inpTracks,
  //                                  pixelTrack::Quality const& minQuality,
  //                                  double const& matchFraction,
  //                                  Queue& queue,
  //                                  bool twinMerge = false,
  //                                  const int32_t* armOfTrack = nullptr,
  //                                  float twinMergeDeltaEta = 0.03f,
  //                                  float twinMergeDeltaPhi = 0.03f,
  //                                  int twinMergeMinSharedHits = 1,
  //                                  bool twinMergeTier2 = false,
  //                                  float twinMergeTier2DeltaEta = 0.01f,
  //                                  float twinMergeTier2DeltaPhi = 0.01f,
  //                                  float twinMergeNSigma2 = -1.f,
  //                                  int twinMergeMinSharedFwd = 1,
  //                                  bool twinMergeRefit = false,
  //                                  bool refitAllTracks = false,
  //                                  int32_t* unitedWinnerMask = nullptr,
  //                                  // Per-track arm, input order in and merged-SoA order out; both null
  //                                  // when no arm is tracked.
  //                                  const uint8_t* pocketArmIn = nullptr,
  //                                  uint8_t* pocketArmIdOut = nullptr) const;

  //   // Merger-side GBL refit of the twin-united winners: re-fits each winner's post-union hit list
  //   // (pixel plus absorbed OT extras) and overwrites its state, covariance, chi2 and ndof in place.
  //   // unitedWinnerMask is >= 0 for winners; a null otHits means pixel-only.
  //   void refitUnitedTracks(TkSoADevice& tracks,
  //                          const int32_t* unitedWinnerMask,
  //                          reco::CAGeometrySoACollection const& geometry,
  //                          reco::TrackingRecHitsSoACollection const& hits,
  //                          reco::OTRecHitsSoACollection const* otHits,
  //                          reco::StackedModuleGeometrySoACollection const* stackedGeom,
  //                          const float* rhoMapDevice,
  //                          // Normalized (Bz,Br) r-z field-map device buffer, or null to use the scalar
  //                          // field at the origin.
  //                          const float* bFieldMapDevice,
  //                          float bfield,
  //                          Queue& queue,
  //                          // When true the refit's single-outlier drop is removed from the emitted hit
  //                          // list (nHits == nMeasFit); when false the dropped node stays on the track.
  //                          bool dropOutlierFromHitList = false,
  //                          // Drop nothing when the largest-pull measured node is an original pixel-core
  //                          // hit; when false that node is dropped whatever its provenance.
  //                          bool outlierCoreProtect = false,
  //                          // Take the curvature->pT conversion field from the fit's own curvature weights
  //                          // instead of the hit-count average of B_bend; inert without bFieldMapDevice.
  //                          bool fieldKernelWeights = false,
  //                          // Charge-symmetric corrections: signed arc of the node-0 -> PCA step and the
  //                          // bending-field profile offset.
  //                          bool chargeSymmetric = false,
  //                          // Reference-trajectory corrections: node-0 path length, arc->azimuth sign of a
  //                          // measurement-less node, and the field's B_r lambda row.
  //                          bool trajectoryCorrections = false,
  //                          // Evaluate Highland's logarithm at the track's total material and apportion the
  //                          // variance to the gaps by thickness: theta0^2 does not add over a chain.
  //                          bool scatteringLogAtTotal = false,
  //                          // Typical loss of the charged column taken as the single-column Landau law at
  //                          // the accumulated thickness (the family is stable under convolution); when
  //                          // false each lump is charged its own MPV.
  //                          bool cumulativeEloss = false) const;

  //   // Post-refit de-dup of the merged tracks: shared-hit co-occurrence pairing, a covariance-scaled
  //   // 3-parameter gate and a 0-shared forward fallback, keeping the higher-quality member of a pair.
  //   // Returns a fresh compacted SoA of the survivors; the input is unchanged. nHits and nOTHits size
  //   // the hit-id key space.
  //   TkSoADevice finalDedupTracks(TkSoADevice const& refinedTracks,
  //                                uint32_t nHits,
  //                                uint32_t nOTHits,
  //                                Queue& queue,
  //                                // Merger GBL-refit inputs for the 0-shared merge-or-keep-both confirm;
  //                                // null makes the confirm inert.
  //                                const MergerDedupConfirmInputs* confirm = nullptr) const;
  // };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAHitNtupletGenerator_h
