#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAITExtend_CAITExtendKernels_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAITExtend_CAITExtendKernels_h

// Host-visible declarations for the IT inward-extension stage.  Method bodies
// (which contain alpaka::exec launches) live in CAITExtendKernels.dev.cc; this
// header is safe to include from .cc translation units (no static_assert).

#include <cstdint>
#include <optional>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "../CAStructures.h"
#include "../HelixFit.h"
#include "CAITCandidateCollector.h"  // brings CandidateSlot type
#include "CAITExtendDoublets.h"       // brings ExtendedChain type
#include "CAITExtendLayout.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::caITExtend {

  // POD of all parameters needed by the IT-extension stage. Host-side, populated
  // from the Python config of the iteration that runs the extension.
  struct ExtensionConfig {
    bool enable;
    ::pixelTrack::Iteration sourceIteration;
    ::pixelTrack::Iteration extendedIteration;
    ::pixelTrack::Quality minQuality;
    float nSigmaPhi;
    float nSigmaZ;
    float floorDPhi;
    float floorDZ;
    float maxDPhi;
    float maxDZ;
    float kappaSigmaCut;
    float scoreFloor2Phi;
    float scoreFloor2Z;
    float materialDensity;
    uint16_t maxLayersPerTrack;
    uint16_t maxNewLayers;
    uint16_t refitMinNewHits;
    bool dropOnEmptyExtension;
    bool doRefit;
    bool verbose;  // when true: host-side counter readback + log line after each event
  };

  // Diagnostic counters.  Indexed:
  //   0  = candidate hits accepted into per-slot lists (collector)
  //   1  = candidate hits dropped because slot was at cap (collector)
  //   2  = chain steps rejected by kappa-significance (selector)
  //   3  = tracks that gained >=1 IT hit (extended)
  //   4  = tracks tagged by source iteration but ended with 0 IT hits (fallback)
  //   5  = tracks extended with exactly 1 IT hit
  //   6  = tracks extended with exactly 2 IT hits
  //   7  = tracks extended with exactly 3 IT hits
  //   8  = tracks extended with exactly 4 IT hits
  //   9  = total IT hits attached (sum of chain.nHits across all extended tracks)
  //   10 = valid tracks rejected by iteration filter        (funnel: top)
  //   11 = valid tracks rejected by quality filter          (funnel: mid)
  //   12 = valid tracks that entered the per-layer scan     (funnel: passed)
  //        Invariant: counter[12] == counter[3] + counter[4].
  //   13 = tracks where at least one disk crossing was valid + box-OK
  //   14 = tracks where at least one barrel crossing was valid + box-OK
  constexpr int kNExtCounters = 15;

  // ============================================================================
  // Launcher class -- method bodies defined in CAITExtendKernels.dev.cc.
  // ============================================================================

  template <typename TrackerTraits>
  class Kernels {
   public:
    using LayoutTraits = ::caITExtend::Phase2OTStubsTag;
    using TkSoAView = ::reco::TrackSoAView;
    using TkSoAConstView = ::reco::TrackSoAConstView;
    using TkHitSoAView = ::reco::TrackHitSoAView;
    using TkHitSoAConstView = ::reco::TrackHitSoAConstView;
    using HitsConstView = ::reco::TrackingRecHitConstView;
    using PhiBinner = caStructures::PhiBinnerT<TrackerTraits>;
    using SequentialContainer = caStructures::SequentialContainer;
    using SequentialContainerStorage = caStructures::SequentialContainerStorage;
    using SequentialContainerOffsets = caStructures::SequentialContainerOffsets;
    using GenericContainer = caStructures::GenericContainer;
    using GenericContainerStorage = caStructures::GenericContainerStorage;
    using GenericContainerOffsets = caStructures::GenericContainerOffsets;

    Kernels(ExtensionConfig const& cfg, uint32_t nTracks, uint32_t offsetStubs, Queue& queue);

    // Phase A: collect candidates + walk chains.  Writes chains_ on device.
    void buildChains(TkSoAConstView tracks,
                     HitsConstView hits,
                     PhiBinner const* phiBinner,
                     float bField,
                     Queue& queue);

    // Phase B: rebuild Tuples + Multiplicity for ALL tracks, refit through the
    // existing HelixFit instance, then refresh TrackHitSoA and stamp iteration.
    //
    // The TracksSoACollection backing `tracks`/`trackHits` MUST have been
    // allocated with headroom = nTracks * (avgHitsPerTrack + cfg.maxNewLayers)
    // hit slots so the post-refit content fits without overlap.
    void runRefit(HelixFit<TrackerTraits>& fitter,
                  TkSoAView tracks,
                  TkHitSoAView trackHits,
                  TkHitSoAConstView origTrackHits,
                  HitsConstView hits,
                  ::reco::CAModulesConstView modules,
                  ::reco::OTRecHitsConstView otRecHits,
                  ::reco::StubsConstView stubs,
                  uint32_t nHits,
                  uint32_t maxNumberOfTuples,
                  uint16_t maxHitsOnTrack,
                  Queue& queue);

    // Setter for the original avgHitsPerTrack -- needed for sizing the refit
    // content buffer.  Producer should call this before runRefit() with the
    // same value used to allocate the TracksSoACollection.
    void setOrigAvgHitsPerTrack(uint32_t h) { origAvgHitsPerTrack_ = h; }

    ExtendedChain const* chainsDevice() const { return chains_->data(); }
    uint32_t const* countersDevice() const { return counters_->data(); }

    // Host-side counter readback + log via edm::LogInfo.  Forces a queue sync
    // -- call only when debugging (controlled by ExtensionConfig::verbose).
    void logCounters(Queue& queue) const;

   private:
    ExtensionConfig cfg_;
    uint32_t nTracks_;
    uint32_t offsetStubs_;
    uint32_t origAvgHitsPerTrack_ = 0;
    std::optional<cms::alpakatools::device_buffer<Device, CandidateSlot[]>> slots_;
    std::optional<cms::alpakatools::device_buffer<Device, uint8_t[]>> slotLayer_;
    std::optional<cms::alpakatools::device_buffer<Device, ExtendedChain[]>> chains_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> counters_;

    std::optional<cms::alpakatools::device_buffer<Device, SequentialContainer>> extTuples_;
    std::optional<cms::alpakatools::device_buffer<Device, SequentialContainerStorage[]>> extTuplesStorage_;
    std::optional<cms::alpakatools::device_buffer<Device, SequentialContainerOffsets[]>> extTuplesOffsets_;
    std::optional<cms::alpakatools::device_buffer<Device, GenericContainer>> extMult_;
    std::optional<cms::alpakatools::device_buffer<Device, GenericContainerStorage[]>> extMultStorage_;
    std::optional<cms::alpakatools::device_buffer<Device, GenericContainerOffsets[]>> extMultOffsets_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::caITExtend

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAITExtend_CAITExtendKernels_h
