// #define GPU_DEBUG
// #define DUMP_GPU_TK_TUPLES

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <optional>
#include <type_traits>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "CAHitNtupletGenerator.h"
#include "CAHitNtupletGeneratorKernels.h"
#include "CAPixelDoublets.h"
#include "CAPixelDoubletsAlgos.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace {

    using namespace caHitNtupletGenerator;
    using namespace caPixelDoublets;
    using namespace caStructures;
    using namespace pixelTopology;
    using namespace pixelTrack;

    template <typename T>
    T sqr(T x) {
      return x * x;
    }

    // Common Params
    // Per-topology defaults. Phase2OTStubs defaults to the working point of the HLT configuration, so
    // its two HLT cfis need only state what differs between the prompt and the displaced iteration.
    // Every other topology keeps the upstream defaults with the stub-seeding features off.
    //
    // Generic (topology-independent) cut parameters live at the top level of the module PSet, where
    // the Run-3 / HIon / Phase-2 menus set them. They are consumed by filling the corresponding
    // CAGeometry SoA scalars/columns in CAHitNtuplet.cc:
    //   ptmin       -> tripletCuts.ptmin           (scalar)
    //   hardCurvCut -> tripletCuts.maxCurv         (scalar)
    //   cellZ0Cut   -> doubletCuts.maxZ0           (broadcast to every layer pair)
    //   dzdrFact    -> doubletCuts.dzdrFact        (scalar)
    //   minYsizeB1  -> doubletCuts.minInnerSizeB1  (scalar)
    //   minYsizeB2  -> doubletCuts.minInnerSizeB2  (scalar)
    //   maxDYsize12 -> doubletCuts.maxDSizeB1      (scalar)
    //   maxDYsize   -> doubletCuts.maxDSize        (scalar)
    //   maxDYPred   -> doubletCuts.maxDSizePred    (scalar)
    // The optional per-layer-pair entries of the `geometry` PSet override the same values: when present
    // they win, when absent the generic scalar above is used. Only the stub-seeded configurations set them.
    template <typename TrackerTraits>
    void fillDescriptionsCommon(edm::ParameterSetDescription& desc) {
      constexpr bool kStubs = std::is_same_v<TrackerTraits, pixelTopology::Phase2OTStubs>;
      constexpr bool kPhase1 = std::is_same_v<TrackerTraits, pixelTopology::Phase1>;

      // ---------------------------------------------
      // Generic cut parameters
      // ---------------------------------------------
      desc.add<double>("cellZ0Cut", TrackerTraits::cellZ0Cut)->setComment("Z0 cut for cells");

      //// Pixel Cluster Cuts (@cell level)
      desc.add<double>("dzdrFact", TrackerTraits::dzdrFact);
      desc.add<int>("minYsizeB1", TrackerTraits::minInnerSizeB1)
          ->setComment("Cut on inner hit cluster size (in global z / local y) for barrel-forward cells. Barrel 1 cut.");
      desc.add<int>("minYsizeB2", TrackerTraits::minInnerSizeB2)
          ->setComment(
              "Cut on inner hit cluster size (in global z / local y) for barrel-forward cells. Anything but Barrel 1 "
              "cut.");
      desc.add<int>("maxDYsize12", TrackerTraits::maxDSizeB1)
          ->setComment(
              "Cut on cluster size differences (in global z / local y) for barrel-forward cells. Barrel 1-2 cells.");
      desc.add<int>("maxDYsize", TrackerTraits::maxDSize)
          ->setComment(
              "Cut on cluster size differences (in global z / local y) for barrel-forward cells. Other barrel cells.");
      desc.add<int>("maxDYPred", TrackerTraits::maxDSizePred)
          ->setComment(
              "Maximum difference between actual and expected cluster size of inner RecHit. Barrel-forward cells.");

      // nTuplet Cuts and Params
      desc.add<double>("ptmin", 0.9f)->setComment("Cut on minimum pt");
      //// p [GeV/c] = B [T] * R [m] * 0.3 (factor from conversion from J to GeV and q = e = 1.6 * 10e-19 C)
      //// 87 cm/GeV = 1/(3.8T * 0.3)
      //// take less than radius given by the hardPtCut and reject everything below
      desc.add<double>("hardCurvCut", TrackerTraits::maxCurv)
          ->setComment("Cut on minimum curvature, used in DCA ntuplet selection");

      // OT-stub extension triplet scalars.
      desc.add<double>("maxPhiResid", -1.0)
          ->setComment(
              "OT-stub extension. Max |phi residual| per connection during chain extension [rad]. "
              "Negative = disabled. Typical value when enabled: 0.01.");
      desc.add<bool>("sameDPhiSign", kStubs)
          ->setComment(
              "OT-stub extension. If true, require dPhi12 * dPhi23 > 0 in a triplet. Off on every "
              "topology but Phase2OTStubs.");

      // Layer- and layer-pair-dependent configuration: the `geometry` PSet. The CA SoA is organised by
      // purpose (layers / graph / doubletCuts / tripletCuts / ntupletCuts) and indexes the two triplet
      // cuts by layer pair rather than by layer; CAHitNtuplet.cc broadcasts each per-layer value onto the
      // pairs whose inner layer it belongs to, which is the indexing the kernels read back (caDCACuts at
      // the triplet's innermost layer, caThetaCuts at its middle layer). The OT-stub extension parameters
      // at the end of the block are optional and inert when absent.
      edm::ParameterSetDescription geometryParams;

      // ---- layers params ----
      geometryParams
          .add<std::vector<double>>(
              "caDCACuts",
              std::vector<double>(TrackerTraits::dcaCuts, TrackerTraits::dcaCuts + TrackerTraits::numberOfLayers))
          ->setComment("Cut on RZ alignement. One per layer, the layer being the middle one for a triplet.");
      geometryParams
          .add<std::vector<double>>(
              "caThetaCuts",
              std::vector<double>(TrackerTraits::thetaCuts, TrackerTraits::thetaCuts + TrackerTraits::numberOfLayers))
          ->setComment("Cut on origin radius. One per layer, the layer being the innermost one for a triplet.");
      // The list of pair ids the ntuplet building may start from. The topology tables carry the same
      // information as a per-pair flag array sized to the topology's own pair bound
      // (nPairsForQuadruplets), so the default id list is obtained by scanning that array.
      std::vector<unsigned int> startingPairsDefault;
      for (int pairId = 0; pairId < TrackerTraits::nPairsForQuadruplets; ++pairId)
        if (TrackerTraits::startingPairs[pairId])
          startingPairsDefault.push_back(static_cast<unsigned int>(pairId));
      geometryParams.add<std::vector<unsigned int>>("startingPairs", startingPairsDefault)
          ->setComment("The list of the ids of pairs from which the CA ntuplets building may start.");
      geometryParams
          .add<std::vector<double>>("startMaxInnerR", std::vector<double>(TrackerTraits::numberOfLayers, 99.0))
          ->setComment(
              "The maximum allowed r coordinate of the inner hit of a doublet to use it as a starting point for "
              "ntuplet building.");
      /*
      Cut on quadruplets (two triplets sharing a doublet) using the curvatures Ci, Co of the triplets:
      |Co - Ci| < (|Co| + |Ci|)/2 * maxDCurv + floorDCurv
      */
      geometryParams.add<std::vector<double>>("maxDCurv", std::vector<double>(TrackerTraits::numberOfLayers, 99.))
          ->setComment("Cut on curvature difference between two consecutive triplets.");
      geometryParams.add<std::vector<double>>("floorDCurv", std::vector<double>(TrackerTraits::numberOfLayers, 99.))
          ->setComment("Offset for the cut on curvature difference between two consecutive triplets.");
      geometryParams
          .add<std::vector<double>>("fishboneCuts", std::vector<double>(TrackerTraits::numberOfLayers, 0.99999f))
          ->setComment(
              "Threshold for merging aligned doublets in fishbone cleaning. Depends on the layer of the outer RecHit. "
              "Warning: this will be a float in the final algorithm, therefore 0.9999999 will become 1 == no merging!");

      // ---- cells params ----
      geometryParams
          .add<std::vector<unsigned int>>(
              "pairGraph",
              std::vector<unsigned int>(TrackerTraits::layerPairs,
                                        TrackerTraits::layerPairs + (TrackerTraits::nPairsForQuadruplets * 2)))
          ->setComment("CA graph (layer pairs used for building doublets/cells)");
      geometryParams
          .add<std::vector<unsigned int>>("skipsLayers",
                                          std::vector<unsigned int>(TrackerTraits::nPairsForQuadruplets, 0U))
          ->setComment(
              "List of bools idicating whether layer pairs are skipping layers or not (0 means non-skipping, 1 means "
              "skipping). This is relevant for the N-tuplet building as non-skipping ones are prioritized.");
      geometryParams
          .add<std::vector<int>>(
              "phiCuts",
              std::vector<int>(TrackerTraits::maxDPhi, TrackerTraits::maxDPhi + TrackerTraits::nPairsForQuadruplets))
          ->setComment("Cuts in dphi for cells");
      geometryParams
          .add<std::vector<double>>(
              "ptCuts",
              std::vector<double>(TrackerTraits::minPt, TrackerTraits::minPt + TrackerTraits::nPairsForQuadruplets))
          ->setComment("Cuts in pt for cells");
      geometryParams
          .add<std::vector<double>>("minInner",
                                    std::vector<double>(TrackerTraits::minInner,
                                                        TrackerTraits::minInner + TrackerTraits::nPairsForQuadruplets))
          ->setComment("Cuts on inner hit's z (for barrel) or r (for endcap) for cells (min value)");
      geometryParams
          .add<std::vector<double>>("maxInner",
                                    std::vector<double>(TrackerTraits::maxInner,
                                                        TrackerTraits::maxInner + TrackerTraits::nPairsForQuadruplets))
          ->setComment("Cuts on inner hit's z (for barrel) or r (for endcap) for cells (max value)");
      geometryParams
          .add<std::vector<double>>("minOuter",
                                    std::vector<double>(TrackerTraits::minOuter,
                                                        TrackerTraits::minOuter + TrackerTraits::nPairsForQuadruplets))
          ->setComment("Cuts on outer hit's z (for barrel) or r (for endcap) for cells (min value)");
      geometryParams
          .add<std::vector<double>>("maxOuter",
                                    std::vector<double>(TrackerTraits::maxOuter,
                                                        TrackerTraits::maxOuter + TrackerTraits::nPairsForQuadruplets))
          ->setComment("Cuts on outer hit's z (for barrel) or r (for endcap) for cells (max value)");
      geometryParams
          .add<std::vector<double>>(
              "maxDR",
              std::vector<double>(TrackerTraits::maxDR, TrackerTraits::maxDR + TrackerTraits::nPairsForQuadruplets))
          ->setComment("Cuts in max dr for cells");
      geometryParams
          .add<std::vector<double>>(
              "minDZ",
              std::vector<double>(TrackerTraits::minDZ, TrackerTraits::minDZ + TrackerTraits::nPairsForQuadruplets))
          ->setComment("Cuts in minimum dz between hits for cells");
      geometryParams
          .add<std::vector<double>>(
              "maxDZ",
              std::vector<double>(TrackerTraits::maxDZ, TrackerTraits::maxDZ + TrackerTraits::nPairsForQuadruplets))
          ->setComment("Cuts in maximum dz between hits for cells");

      // OT-stub extension. All entries below are optional: the three "...PerPair" vectors fall back to
      // the broadcast of their per-layer or scalar source, the stub-only cuts to the sentinel -1.
      geometryParams.addOptional<std::vector<double>>("caDCACutsPerPair")
          ->setComment(
              "OT-stub extension. Per-layer-pair override of caDCACuts: one entry per layer pair, replacing the "
              "broadcast of the per-layer caDCACuts. Absent = use caDCACuts.");
      geometryParams.addOptional<std::vector<double>>("caThetaCutsPerPair")
          ->setComment(
              "OT-stub extension. Per-layer-pair override of caThetaCuts: one entry per layer pair, replacing the "
              "broadcast of the per-layer caThetaCuts. Absent = use caThetaCuts.");
      geometryParams.addOptional<std::vector<double>>("cellZ0CutPerPair")
          ->setComment(
              "OT-stub extension. Per-layer-pair override of the scalar cellZ0Cut: one entry per layer pair. "
              "Absent = use cellZ0Cut for every pair.");
      geometryParams.addOptional<std::vector<double>>("floorDCACuts")
          ->setComment(
              "OT-stub extension. Additive DCA floor for high-pT tracks, one entry per layer pair "
              "(dcaThreshold = caDCACut * curvature + floorDCACut), which prevents the threshold from "
              "collapsing at high pT. Negative = disabled. Absent = disabled on every pair.");
      geometryParams.addOptional<std::vector<double>>("maxStubCurvSigma")
          ->setComment(
              "OT-stub extension. Stub-stub pairwise curvature-significance cut, one entry per layer pair. "
              "Negative = disabled. Barrel flat-flat: kappa-corrected significance; forward: dPhiDr significance.");
      geometryParams.addOptional<std::vector<double>>("maxStubGeomCurvSigma")
          ->setComment(
              "OT-stub extension. Geometric-vs-stub kappa significance cut, one entry per layer pair. "
              "Negative = disabled (no stubs on that pair).");
      geometryParams.addOptional<std::vector<double>>("maxStubInnerDoubletDCurv")
          ->setComment(
              "OT-stub extension. Cut on the phi residual at the middle hit [rad], one entry per layer pair. "
              "Negative = disabled. Requires nStubs >= 2 for a reliable stub-kappa prediction.");

      desc.add<edm::ParameterSetDescription>("geometry", geometryParams)
          ->setComment("Layer-dependent cuts and settings of the CA");

      // Container sizes
      //
      // maxNumberOfDoublets and maxNumberOfTuples may be defined at runtime depending on the number of hits.
      // This is done via a FormulaEvaluator expecting 'x' as nHits.
      // e.g. : maxNumberOfDoublets = cms.string( '0.00022*pow(x,2)  + 0.53*x + 10000' )
      // will compute maxNumberOfDoublets for each event as
      //
      //  	maxNumberOfDoublets = 2.2e-4 * nHits^2 + 0.53 * nHits + 10000
      //
      // this may also be simply a constant (as for the default parameters)
      //
      // 	 maxNumberOfDoublets = cms.string(str(512*1024))
      //
      desc.add<std::string>("maxNumberOfDoublets", std::to_string(TrackerTraits::maxNumberOfDoublets))
          ->setComment(
              "Max nummber of doublets (cells) as a string. The string will be parsed to a TFormula, depending on "
              "nHits (labeled 'x'), \n and evaluated for each event. May also be a constant.");
      desc.add<std::string>("maxNumberOfTuples", std::to_string(TrackerTraits::maxNumberOfTuples))
          ->setComment("Max nummber of tuples as a string. Same behavior as maxNumberOfDoublets.");
      desc.add<uint32_t>("minNumberOfDoublets", kStubs ? 23552 : 0)
          ->setComment(
              "Per-event floor on the doublet capacity: the effective capacity is max(formula, floor). Sized "
              "per iteration from the quiet-event demand (the hit-dependent formulas undershoot at low "
              "occupancy); demand beyond it is truncated and counted by the overflow sentinel. 0 removes the "
              "floor, leaving the formula alone to size the container.");
      desc.add<uint32_t>("minNumberOfTuples", kStubs ? 13312 : 0)
          ->setComment("Per-event floor on the tuple capacity. Same behavior as minNumberOfDoublets.");
      desc.add<double>("avgHitsPerTrack", double(TrackerTraits::avgHitsPerTrack))
          ->setComment("Number of hits per track. Average per track.");
      desc.add<double>("avgCellsPerHit", TrackerTraits::avgCellsPerHit)
          ->setComment(
              "Number of cells for which a hit is the outer hit. Average per hit. Informative only: the hit->cell "
              "association holds exactly one entry per cell, so it is sized from the cell bound itself (the exact "
              "count when useExactAllocations is on, maxNumberOfDoublets otherwise). The parameter is read by the "
              "sizing report and kept for configuration compatibility.");
      desc.add<double>("avgCellsPerCell", TrackerTraits::avgCellsPerCell)
          ->setComment("Number of cells connected to another cell. Average per cell.");
      desc.add<double>("avgTracksPerCell", TrackerTraits::avgTracksPerCell)
          ->setComment("Number of tracks to which a cell belongs. Average per cell.");

      desc.add<bool>("earlyFishbone", true);
      desc.add<bool>("lateFishbone", false);
      desc.add<bool>("onlySameLayersFishbone", kStubs);
      desc.add<bool>("fillStatistics", false);
      desc.add<unsigned int>("minHitsPerNtuplet", 4)
          ->setComment("Minimum number of layers an N-tuplet must span to be kept.");
      desc.add<unsigned int>("minHitsForSharingCut", kStubs ? 1 : 10)
          ->setComment("Maximum number of hits in a tuple to clean also if the shared hit is on bpx1");

      desc.add<bool>("fitNas4", false)
          ->setComment(
              "obsolete: the serial per-bin fit ladder it selected has been removed, only the fused ladder "
              "remains. Kept so the deployed menus still validate; setting it true is a configuration error.");
      desc.add<bool>("verboseBLFit", false)
          ->setComment("one-shot device dump of the first fitted tracks at the end of each BL-fit launch (debug)");
      desc.add<bool>("useRiemannFit", false)->setComment("true for Riemann, false for BrokenLine");
      desc.add<bool>("useFitCorrections", kStubs)
          ->setComment(
              "Correctness package of the factorized fast BrokenLine fit. true applies it as ONE unit: "
              "one thin scatterer per gap charged at that gap's arrival node (with the rigid-node guard for "
              "a material-free gap), the Karimaki-consistent Fisher basis in the covariance blend, the pion "
              "1/beta factor and no pt cap in the scattering variance, trapezoid quadrature in the material "
              "line integral, and the covariance blend completed to the full 3x3; false is the plain "
              "upstream algebra. It is the CA main fit's own material model, so the merger's General Broken "
              "Lines refit and the extender walk are unaffected. It moves the whole "
              "emitted covariance and the chi2, so every covariance-derived input of the high-purity "
              "selector shifts and covPhiDxy changes sign on part of the population: the selector's working "
              "point is NOT comparable across this switch, and flipping it requires retraining the selector. "
              "MUST agree with the merger's extPredFitCorrections. Defaults to true on Phase2OTStubs and "
              "false on every other topology.");
      desc.add<bool>("doSharedHitCut", true)->setComment("Sharing hit nTuples cleaning");
      desc.add<bool>("dupPassThrough", false)->setComment("Do not reject duplicate");
      desc.add<bool>("useSimpleTripletCleaner", true)->setComment("use alternate implementation");
      desc.add<bool>("doTripletCleaner", true)
          ->setComment(
              "Disable the triplet cleaner entirely.");  // FIXME this should be implemented as an automatic check (simple if) that disables if minHitsPerNtuplet > 3
      desc.add<bool>("doFastDuplicateRemover", true)->setComment("Disable the fastDuplicateRemover");
      desc.add<bool>("doEarlyDuplicateRemover", true)->setComment("Disable the earlyDuplicateRemover");

      // Inline per-triplet DNN gate (Phase2OTStubs only)
      desc.add<bool>("useTripletDNN", kStubs)
          ->setComment(
              "Enable the inline per-triplet DNN gate (CATripletDNN.h, weights compiled into the plugin) "
              "on triplets accepted by the cut ladder in Kernel_connect. It suppresses combinatorial "
              "triplets before they are grown into tracks, at the cost of one small MLP evaluation per "
              "accepted triplet; false leaves the cut ladder as the only selection. Phase2OTStubs only "
              "(true there by default, false on every other topology).");
      desc.add<double>("tripletDNNThreshold", -1.0)
          ->setComment(
              "DNN gate threshold: a triplet is kept iff its score >= threshold. Negative selects the "
              "threshold shipped with the SELECTED bank's weights (its kDefaultThreshold, the recall point "
              "that bank was trained for; the bank follows iterationName). Raise it to reject more "
              "combinatorial triplets at the cost of "
              "real-triplet recall; lower it to keep more.");
      desc.add<std::string>("tripletDumpIteration", "")
          ->setComment(
              "Restricts the built-triplet dump to the iteration with this name; empty dumps every iteration. "
              "Read only in builds that enable the dump (CA_TRIPLET_DUMP), ignored otherwise.");
      // Classify-embedded track classifier (Phase2OTStubs only)
      desc.add<bool>("useTrackDNN", kStubs)
          ->setComment(
              "Enable the track classifier embedded in Kernel_classifyTracks (CATrackDNN.h, weights "
              "compiled into the plugin): its score replaces the chi2-based promotion of a track from "
              "loose to tight, which keeps efficiency on displaced tracks that a chi2 cut alone loses. "
              "false restores the chi2 gate. Phase2OTStubs only (true there by default, false on every "
              "other topology).");
      desc.add<double>("trackDNNThreshold", -1.0)
          ->setComment(
              "Threshold of the loose->tight track classifier: a candidate is promoted to tight iff its "
              "score >= threshold. Negative selects the threshold shipped with the SELECTED bank's weights "
              "(its kDefaultThreshold, the high-recall point that bank was trained for; the bank follows "
              "iterationName). Keep it "
              "low to retain efficiency here; fake rejection for displaced tracks is the job of the "
              "separate post-reco high-purity selector, not of this gate.");

      // Device-memory allocation strategy
      desc.add<bool>("useExactAllocations", false)
          ->setComment(
              "Size the event-dependent device buffers from the measured per-event occupancy instead of "
              "from the configured caps: a count-only doublet pass plus one device readback per event "
              "determines the exact allocation sizes for the doublet store, the hit->cell association and "
              "the cell-derived containers, and the hit->track storage is sized from the actual "
              "hits-in-tracks count delivered through the acquire/produce boundary. Costs one host "
              "synchronization per event plus the count pass; adapts automatically to occupancy beyond "
              "the calibrated caps. false allocates everything up-front from the configured caps with "
              "zero synchronizations. On the CPU backend the count pass is skipped and the caps size the "
              "doublet store (identical results, no serial-time cost).");

      desc.add<double>("fastDupNSigma2", kStubs ? 2.5 : (kPhase1 ? 25.0 : 5.0))
          ->setComment(
              "CA duplicate-removal parameter-covariance gate width nSigma^2: two tracks are duplicates "
              "when every compared fitted parameter satisfies dp^2 <= fastDupNSigma2*(cov_i+cov_j) "
              "(Kernel_fastDuplicateRemover + Kernel_rejectDuplicate). It is a width in units of the "
              "fitted covariance, so it tracks the fit's covariance convention. Lower tightens (fewer "
              "merges); higher merges more aggressively. The default reproduces, topology by topology, "
              "the constant those kernels hard-wire upstream: 5 for the five-parameter check used by "
              "every Phase-2 topology, 25 for the two-parameter check of the Phase-1 specializations "
              "(which read their own nSigma2Phase1 constant and ignore this parameter, so the value is "
              "only informative there). Phase2OTStubs uses 2.5, matching the tighter covariance its "
              "fit emits.");
    }

    AlgoParams makeCommonParams(edm::ParameterSet const& cfg) {
      // Designated initializers: the field set of AlgoParams is shared with upstream and grows there,
      // so a positional aggregate initialization would silently shift every value the next time a
      // member is inserted. Keep this list keyed by name.
      if (cfg.getParameter<bool>("fitNas4"))
        throw cms::Exception("Configuration")
            << "fitNas4 is no longer supported: the serial per-bin BrokenLine fit ladder has been removed.";
      return AlgoParams{
          // Container sizes
          .avgHitsPerTrack_ = (float)cfg.getParameter<double>("avgHitsPerTrack"),
          .avgCellsPerHit_ = (float)cfg.getParameter<double>("avgCellsPerHit"),
          .avgCellsPerCell_ = (float)cfg.getParameter<double>("avgCellsPerCell"),
          .avgTracksPerCell_ = (float)cfg.getParameter<double>("avgTracksPerCell"),

          // Algo params
          .minHitsPerNtuplet_ = (uint16_t)cfg.getParameter<unsigned int>("minHitsPerNtuplet"),
          .minHitsForSharingCut_ = (uint16_t)cfg.getParameter<unsigned int>("minHitsForSharingCut"),

          // Flags
          .useRiemannFit_ = cfg.getParameter<bool>("useRiemannFit"),
          .useFitCorrections_ = cfg.getParameter<bool>("useFitCorrections"),
          .fitNas4_ = cfg.getParameter<bool>("fitNas4"),
          .earlyFishbone_ = cfg.getParameter<bool>("earlyFishbone"),
          .lateFishbone_ = cfg.getParameter<bool>("lateFishbone"),
          .onlySameLayersFishbone_ = cfg.getParameter<bool>("onlySameLayersFishbone"),
          .doStats_ = cfg.getParameter<bool>("fillStatistics"),
          .doSharedHitCut_ = cfg.getParameter<bool>("doSharedHitCut"),
          .dupPassThrough_ = cfg.getParameter<bool>("dupPassThrough"),
          .useSimpleTripletCleaner_ = cfg.getParameter<bool>("useSimpleTripletCleaner"),
          .doTripletCleaner_ = cfg.getParameter<bool>("doTripletCleaner"),
          .doFastDuplicateRemover_ = cfg.getParameter<bool>("doFastDuplicateRemover"),
          .doEarlyDuplicateRemover_ = cfg.getParameter<bool>("doEarlyDuplicateRemover"),

          // Inline triplet DNN gate (scores every accepted triplet)
          .useTripletDNN_ = cfg.getParameter<bool>("useTripletDNN"),
          .tripletDNNThreshold_ = (float)cfg.getParameter<double>("tripletDNNThreshold"),

          // Classify-embedded track classifier (loose/tight selector)
          .useTrackDNN_ = cfg.getParameter<bool>("useTrackDNN"),
          .trackDNNThreshold_ = (float)cfg.getParameter<double>("trackDNNThreshold"),

          // The single useExactAllocations switch drives both the deferred demand-based allocations
          // (delayAllocations_) and the count-only doublet pass (countDoubletsFirst_). The count pass is
          // forced off on the CPU backend, where it costs serial time and only trims an allocation. The
          // accepted doublet set is the same either way, so the physics output does not depend on it.
          .delayAllocations_ = cfg.getParameter<bool>("useExactAllocations"),
          .countDoubletsFirst_ =
              cfg.getParameter<bool>("useExactAllocations") && !std::is_same_v<Device, alpaka::DevCpu>,

          // CA fast-duplicate / shared-hit covariance gate width, in units of the fitted covariance
          .fastDupNSigma2_ = (float)cfg.getParameter<double>("fastDupNSigma2"),
          // Iteration label
          .iterationName_ = pixelTrack::iterationByName(cfg.getParameter<std::string>("iterationName")),
      };
    }

    //This is needed to have the partial specialization for isPhase1Topology/isPhase2Topology
    template <typename TrackerTraits, typename Enable = void>
    struct TopologyCuts {};

    template <typename TrackerTraits>
    struct TopologyCuts<TrackerTraits, isPhase1Topology<TrackerTraits>> {
      static constexpr ::pixelTrack::QualityCutsT<TrackerTraits> makeQualityCuts(edm::ParameterSet const& pset) {
        auto coeff = pset.getParameter<std::array<double, 2>>("chi2Coeff");
        auto ptMax = pset.getParameter<double>("chi2MaxPt");

        coeff[1] = (coeff[1] - coeff[0]) / log2(ptMax);
        return ::pixelTrack::QualityCutsT<TrackerTraits>{// polynomial coefficients for the pT-dependent chi2 cut
                                                         {(float)coeff[0], (float)coeff[1], 0.f, 0.f},
                                                         // max pT used to determine the chi2 cut
                                                         (float)ptMax,
                                                         // chi2 scale factor: 8 for broken line fit, ?? for Riemann fit
                                                         (float)pset.getParameter<double>("chi2Scale"),
                                                         // regional cuts for triplets
                                                         {(float)pset.getParameter<double>("tripletMaxTip"),
                                                          (float)pset.getParameter<double>("tripletMinPt"),
                                                          (float)pset.getParameter<double>("tripletMaxZip")},
                                                         // regional cuts for quadruplets
                                                         {(float)pset.getParameter<double>("quadrupletMaxTip"),
                                                          (float)pset.getParameter<double>("quadrupletMinPt"),
                                                          (float)pset.getParameter<double>("quadrupletMaxZip")}};
      }
    };

    template <typename TrackerTraits>
    struct TopologyCuts<TrackerTraits, isPhase2Topology<TrackerTraits>> {
      static constexpr ::pixelTrack::QualityCutsT<TrackerTraits> makeQualityCuts(edm::ParameterSet const& pset) {
        return ::pixelTrack::QualityCutsT<TrackerTraits>{
            static_cast<float>(pset.getParameter<double>("maxChi2")),
            static_cast<float>(pset.getParameter<double>("maxChi2TripletsOrQuadruplets")),
            static_cast<float>(pset.getParameter<double>("maxChi2Quintuplets")),
            static_cast<float>(pset.getParameter<double>("minPt")),
            static_cast<float>(pset.getParameter<double>("maxTip")),
            static_cast<float>(pset.getParameter<double>("maxZip")),
            static_cast<float>(pset.getParameter<double>("maxNtupletStubChi2")),
        };
      }
    };

  }  // namespace

  using namespace std;

  template <typename TrackerTraits>
  CAHitNtupletGenerator<TrackerTraits>::CAHitNtupletGenerator(const edm::ParameterSet& cfg)
      : m_params(makeCommonParams(cfg),
                 TopologyCuts<TrackerTraits>::makeQualityCuts(cfg.getParameterSet("trackQualityCuts")),
                 cfg.getParameter<std::string>("tripletDumpIteration")),
        m_verboseBLDump(cfg.getParameter<bool>("verboseBLFit")) {
#ifdef DUMP_GPU_TK_TUPLES
    printf("TK: %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n",
           "tid",
           "qual",
           "nh",
           "nl",
           "charge",
           "pt",
           "eta",
           "phi",
           "tip",
           "zip",
           "chi2",
           "h1",
           "h2",
           "h3",
           "h4",
           "h5",
           "hn");
#endif
  }

  template <typename TrackerTraits>
  void CAHitNtupletGenerator<TrackerTraits>::fillPSetDescription(edm::ParameterSetDescription& desc) {
    static_assert(sizeof(TrackerTraits) == 0,
                  "Note: this fillPSetDescription is a dummy one. Please specialise it for the correct version of "
                  "CAHitNtupletGenerator<TrackerTraits>.");
  }

  template <>
  void CAHitNtupletGenerator<pixelTopology::Phase1>::fillPSetDescription(edm::ParameterSetDescription& desc) {
    fillDescriptionsCommon<pixelTopology::Phase1>(desc);

    edm::ParameterSetDescription trackQualityCuts;
    trackQualityCuts.add<double>("chi2MaxPt", 10.)->setComment("max pT used to determine the pT-dependent chi2 cut");
    trackQualityCuts.add<std::vector<double>>("chi2Coeff", {0.9, 1.8})->setComment("chi2 at 1GeV and at ptMax above");
    trackQualityCuts.add<double>("chi2Scale", 8.)
        ->setComment(
            "Factor to multiply the pT-dependent chi2 cut (currently: 8 for the broken line fit, ?? for the Riemann "
            "fit)");
    trackQualityCuts.add<double>("tripletMinPt", 0.5)->setComment("Min pT for triplets, in GeV");
    trackQualityCuts.add<double>("tripletMaxTip", 0.3)->setComment("Max |Tip| for triplets, in cm");
    trackQualityCuts.add<double>("tripletMaxZip", 12.)->setComment("Max |Zip| for triplets, in cm");
    trackQualityCuts.add<double>("quadrupletMinPt", 0.3)->setComment("Min pT for quadruplets, in GeV");
    trackQualityCuts.add<double>("quadrupletMaxTip", 0.5)->setComment("Max |Tip| for quadruplets, in cm");
    trackQualityCuts.add<double>("quadrupletMaxZip", 12.)->setComment("Max |Zip| for quadruplets, in cm");
    desc.add<edm::ParameterSetDescription>("trackQualityCuts", trackQualityCuts)
        ->setComment(
            "Quality cuts based on the results of the track fit:\n  - apply a pT-dependent chi2 cut;\n  - apply "
            "\"region cuts\" based on the fit results (pT, Tip, Zip).");
  }

  template <>
  void CAHitNtupletGenerator<pixelTopology::HIonPhase1>::fillPSetDescription(edm::ParameterSetDescription& desc) {
    fillDescriptionsCommon<pixelTopology::HIonPhase1>(desc);

    edm::ParameterSetDescription trackQualityCuts;
    trackQualityCuts.add<double>("chi2MaxPt", 10.)->setComment("max pT used to determine the pT-dependent chi2 cut");
    trackQualityCuts.add<std::vector<double>>("chi2Coeff", {0.9, 1.8})->setComment("chi2 at 1GeV and at ptMax above");
    trackQualityCuts.add<double>("chi2Scale", 8.)
        ->setComment(
            "Factor to multiply the pT-dependent chi2 cut (currently: 8 for the broken line fit, ?? for the Riemann "
            "fit)");
    trackQualityCuts.add<double>("tripletMinPt", 0.0)->setComment("Min pT for triplets, in GeV");
    trackQualityCuts.add<double>("tripletMaxTip", 0.1)->setComment("Max |Tip| for triplets, in cm");
    trackQualityCuts.add<double>("tripletMaxZip", 6.)->setComment("Max |Zip| for triplets, in cm");
    trackQualityCuts.add<double>("quadrupletMinPt", 0.0)->setComment("Min pT for quadruplets, in GeV");
    trackQualityCuts.add<double>("quadrupletMaxTip", 0.5)->setComment("Max |Tip| for quadruplets, in cm");
    trackQualityCuts.add<double>("quadrupletMaxZip", 6.)->setComment("Max |Zip| for quadruplets, in cm");

    desc.add<edm::ParameterSetDescription>("trackQualityCuts", trackQualityCuts)
        ->setComment(
            "Quality cuts based on the results of the track fit:\n  - apply a pT-dependent chi2 cut;\n  - apply "
            "\"region cuts\" based on the fit results (pT, Tip, Zip).");
  }

  template <>
  void CAHitNtupletGenerator<pixelTopology::Phase2>::fillPSetDescription(edm::ParameterSetDescription& desc) {
    fillDescriptionsCommon<pixelTopology::Phase2>(desc);

    edm::ParameterSetDescription trackQualityCuts;
    trackQualityCuts.add<double>("maxChi2", 5.)->setComment("Max normalized chi2 for tracks with 6 or more hits");
    trackQualityCuts.add<double>("maxChi2TripletsOrQuadruplets", 5.)
        ->setComment("Max normalized chi2 for tracks with 4 or less hits");
    trackQualityCuts.add<double>("maxChi2Quintuplets", 5.)->setComment("Max normalized chi2 for tracks with 5 hits");
    trackQualityCuts.add<double>("minPt", 0.5)->setComment("Min pT in GeV");
    trackQualityCuts.add<double>("maxTip", 0.3)->setComment("Max |Tip| in cm");
    trackQualityCuts.add<double>("maxZip", 12.)->setComment("Max |Zip|, in cm");
    trackQualityCuts.add<double>("maxNtupletStubChi2", -1.)
        ->setComment(
            "Ntuplet-wide stub-curvature consistency: max reduced-chi2 of the per-stub curvatures of all stubs on a "
            "track around their inverse-variance-weighted mean. Demotes inconsistent (combinatorial) tracks below "
            "tight. Negative = disabled.");
    desc.add<edm::ParameterSetDescription>("trackQualityCuts", trackQualityCuts)
        ->setComment(
            "Quality cuts based on the results of the track fit:\n  - apply cuts based on the fit results (pT, Tip, "
            "Zip).");
  }

  template <>
  void CAHitNtupletGenerator<pixelTopology::Phase2OT>::fillPSetDescription(edm::ParameterSetDescription& desc) {
    fillDescriptionsCommon<pixelTopology::Phase2OT>(desc);

    edm::ParameterSetDescription trackQualityCuts;
    trackQualityCuts.add<double>("maxChi2", 5.)->setComment("Max normalized chi2 for tracks with 6 or more hits");
    trackQualityCuts.add<double>("maxChi2TripletsOrQuadruplets", 1.)
        ->setComment("Max normalized chi2 for tracks with 4 or less hits");
    trackQualityCuts.add<double>("maxChi2Quintuplets", 3.)->setComment("Max normalized chi2 for tracks with 5 hits");
    trackQualityCuts.add<double>("minPt", 0.9)->setComment("Min pT in GeV");
    trackQualityCuts.add<double>("maxTip", 0.3)->setComment("Max |Tip| in cm");
    trackQualityCuts.add<double>("maxZip", 12.)->setComment("Max |Zip|, in cm");
    trackQualityCuts.add<double>("maxNtupletStubChi2", -1.)
        ->setComment(
            "Ntuplet-wide stub-curvature consistency: max reduced-chi2 of the per-stub curvatures of all stubs on a "
            "track around their inverse-variance-weighted mean. Demotes inconsistent (combinatorial) tracks below "
            "tight. Negative = disabled.");
    desc.add<edm::ParameterSetDescription>("trackQualityCuts", trackQualityCuts)
        ->setComment(
            "Quality cuts based on the results of the track fit:\n  - apply cuts based on the fit results (pT, Tip, "
            "Zip).");
  }

  template <>
  void CAHitNtupletGenerator<pixelTopology::Phase2OTStubs>::fillPSetDescription(edm::ParameterSetDescription& desc) {
    fillDescriptionsCommon<pixelTopology::Phase2OTStubs>(desc);

    // Defaults follow the HLT configuration (both HLT cfis set the same three chi2 values).
    edm::ParameterSetDescription trackQualityCuts;
    trackQualityCuts.add<double>("maxChi2", 7.)->setComment("Max normalized chi2 for tracks with 6 or more hits");
    trackQualityCuts.add<double>("maxChi2TripletsOrQuadruplets", 7.)
        ->setComment("Max normalized chi2 for tracks with 4 or less hits");
    trackQualityCuts.add<double>("maxChi2Quintuplets", 7.)->setComment("Max normalized chi2 for tracks with 5 hits");
    trackQualityCuts.add<double>("minPt", 0.9)->setComment("Min pT in GeV");
    trackQualityCuts.add<double>("maxTip", 0.3)->setComment("Max |Tip| in cm");
    trackQualityCuts.add<double>("maxZip", 12.)->setComment("Max |Zip|, in cm");
    trackQualityCuts.add<double>("maxNtupletStubChi2", -1.)
        ->setComment(
            "Ntuplet-wide stub-curvature consistency: max reduced-chi2 of the per-stub curvatures of all stubs on a "
            "track around their inverse-variance-weighted mean. Demotes inconsistent (combinatorial) tracks below "
            "tight. Negative = disabled.");
    desc.add<edm::ParameterSetDescription>("trackQualityCuts", trackQualityCuts)
        ->setComment(
            "Quality cuts based on the results of the track fit:\n  - apply cuts based on the fit results (pT, Tip, "
            "Zip).");
  }

  template <typename TrackerTraits>
  typename CAHitNtupletGenerator<TrackerTraits>::PendingTuples CAHitNtupletGenerator<TrackerTraits>::beginTuplesAsync(
      HitsOnDevice const& hits_d,
      CAGeometryOnDevice const& geometry_d,
      CAGeometryOnDevice const& cuts_d,
      float bfield,
      uint32_t nDoublets,
      uint32_t nTracks,
      MapToHitConstView const& maskView,
      Queue& queue,
      const float* rhoMapDevice,
      const float* bMapDevice) const {
    using GPUKernels = CAHitNtupletGeneratorKernels<TrackerTraits>;

    PendingTuples pending;
    pending.bfield = bfield;
    pending.rhoMapDevice = rhoMapDevice;
    pending.bMapDevice = bMapDevice;
    pending.maxTuples = nTracks;
    pending.maxDoublets = nDoublets;

    // Output trackHits rows: the same expression the kernels use for the internal per-track hit
    // container, so the SoA the hits are copied into is never smaller than the container feeding it.
    const uint32_t nHitRows = caHitNtupletGenerator::nHitRowsForTuples(nTracks, m_params.algoParams_.avgHitsPerTrack_);
    pending.tracks.emplace(queue, nTracks, nHitRows);

    auto tracks = pending.tracks->view().tracks();

    auto trackingHits = hits_d.view().trackingHits();
    auto hitModules = hits_d.view().hitModules();

    // Geometry blocks from the geometry handle, configuration blocks from the cuts handle. The two are
    // the same object for every topology that does not consume the shared EventSetup geometry; when they
    // differ, a block not owned by a handle has zero rows in it, so taking it from the wrong one gives an
    // empty view.
    auto layers = geometry_d.view().layers();
    auto modules = geometry_d.view().modules();
    auto graph = cuts_d.view().graph();
    auto doubletCuts = cuts_d.view().doubletCuts();
    auto tripletCuts = cuts_d.view().tripletCuts();
    auto ntupletCuts = cuts_d.view().ntupletCuts();

    // Don't bother if less than 2 hits: return the empty (built=false) pending state; finish
    // hands the zeroed collection through.
    if (trackingHits.metadata().size() < 2) {
      const auto device = alpaka::getDev(queue);
      auto ntracks_d = cms::alpakatools::make_device_view(device, tracks.nTracks());
      alpaka::memset(queue, ntracks_d, 0);
      return pending;
    }
    uint32_t nOTHitsDomain = 0u;
    pending.kernels = std::make_unique<GPUKernels>(m_params,
                                                   hits_d.nHits(),
                                                   nOTHitsDomain,
                                                   hits_d.offsetBPIX2(),
                                                   nDoublets,
                                                   nTracks,
                                                   layers.metadata().size(),
                                                   queue);
    auto& kernels = *pending.kernels;

    // Lazy per-stream init of the overflow accumulator (kOvfWords device words) and of its pinned mirror
    // (two kOvfWords slots), then arm the kernels object so classifyTuples launches the sentinel into it.
    if (!ovfAccum_) {
      ovfAccum_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, kOvfWords));
      ovfHost_.emplace(cms::alpakatools::make_host_buffer<uint32_t[]>(queue, 2u * kOvfWords));
      alpaka::memset(queue, *ovfAccum_, 0);
      std::fill_n(ovfHost_->data(), 2u * kOvfWords, 0u);
    }
    kernels.ovfAccum_ = ovfAccum_->data();

    // With useExactAllocations off, all buffers are allocated up-front in the kernels constructor and
    // there is no device-to-host synchronization here.
    const bool delay = m_params.algoParams_.delayAllocations_;
    // The count pass already reads the doublet count back inside buildDoublets, to size the cells buffer
    // and the hit->cell storage; that value is reused as the allocateAfterDoublets basis, so the scalar
    // crosses the bus exactly once per event.
    const bool countFirst = m_params.algoParams_.countDoubletsFirst_;

    kernels.prepareHits(trackingHits, hitModules, layers, queue);

    const uint32_t nCellsCounted =
        kernels.buildDoublets(trackingHits, graph, layers, doubletCuts, hits_d.offsetBPIX2(), maskView, queue);

    if (delay) {
      // Size the cell-derived buffers from the actual number of doublets. With countDoubletsFirst on
      // the count pass already brought that number across (nCellsCounted): reuse it, no second sync.
      // Without it, the number only exists on the device and one D2H sync is unavoidable here.
      const uint32_t nCells = countFirst ? nCellsCounted : kernels.readbackNCells(queue);
      kernels.allocateAfterDoublets(nCells, queue);
    }

    kernels.launchKernels(trackingHits,
                          hits_d.offsetBPIX2(),
                          layers.metadata().size(),
                          pending.tracks->view(),
                          graph,
                          tripletCuts,
                          ntupletCuts,
                          queue);

    // Size the hit->track storage from the actual hits-in-tracks count. Under delayAllocations the count
    // is not read back here: the counter block is enqueued as an async device-to-host copy into the
    // pending buffer, the framework seam guarantees it has landed before finishTuplesAsync runs, and the
    // allocation happens there, still ahead of its first consumer (classifyTuples).
    if (delay) {
      pending.countsHost.emplace(
          cms::alpakatools::make_host_buffer<cms::alpakatools::AtomicPairCounter::DoubleWord[]>(queue, 5u));
      kernels.enqueueCountsReadback(*pending.countsHost, queue);
    } else {
      kernels.allocateAfterNtuplets(0u, queue);
    }

    // One async device-to-host copy of the tuple-multiplicity per-N-bin offsets into the pinned pending
    // buffer. No wait here: the framework's acquire->produce boundary guarantees the copy has landed
    // before finishTuplesAsync consumes it, and the fit passes then get the launch-elision information.
    if (const uint32_t* offDev = kernels.tupleMultiplicityOffsets(); offDev != nullptr) {
      constexpr uint32_t kNOff = TrackerTraits::maxHitsOnTrack + 2u;
      pending.offsetsHost.emplace(cms::alpakatools::make_host_buffer<uint32_t[]>(queue, std::size_t(kNOff)));
      auto offSrc =
          cms::alpakatools::make_device_view(alpaka::getDev(queue), const_cast<uint32_t*>(offDev), std::size_t(kNOff));
      alpaka::memcpy(queue, *pending.offsetsHost, offSrc);
    }
    pending.built = true;
    return pending;
  }

  template <typename TrackerTraits>
  reco::TracksSoACollection CAHitNtupletGenerator<TrackerTraits>::finishTuplesAsync(
      PendingTuples&& pending, HitsOnDevice const& hits_d, CAGeometryOnDevice const& geometry_d, Queue& queue) const {
    using HelixFit = HelixFit<TrackerTraits>;

    if (!pending.built)
      return std::move(*pending.tracks);

    auto& kernels = *pending.kernels;

    // Deferred hit->track storage sizing (delayAllocations): the counter block enqueued in
    // beginTuplesAsync has landed (framework seam guarantee). Word [0] is the tuple AtomicPairCounter,
    // low half tuple count and high half hits-in-tracks total; every track has at least one hit, so the
    // larger half is the hits-in-tracks total. Allocated here, ahead of classifyTuples.
    if (pending.countsHost) {
      const uint64_t apcRaw = static_cast<uint64_t>(pending.countsHost->data()[0]);
      const uint32_t apcLo = static_cast<uint32_t>(apcRaw & 0xFFFFFFFFull);
      const uint32_t apcHi = static_cast<uint32_t>(apcRaw >> 32);
      kernels.allocateAfterNtuplets(std::max(apcLo, apcHi), queue);
    }

    auto tracks = pending.tracks->view().tracks();
    auto trackingHits = hits_d.view().trackingHits();
    // `modules` is the only geometry block the fit passes read; geometry_d is the geometry handle.
    auto modules = geometry_d.view().modules();
    const uint32_t nTracks = pending.maxTuples;
    const float bfield = pending.bfield;

    HelixFit fitter(bfield);
    fitter.setVerboseDump(m_verboseBLDump);
    fitter.setMaterialMap(pending.rhoMapDevice);  // device BLMaterialMap (EventSetup condition)
    // Device BLBFieldMap: the same (Bz,Br) r-z condition the merger's GBL refit reads. Consumed by the
    // fit only under useFitCorrections, otherwise inert.
    fitter.setBFieldMap(pending.bMapDevice);
    fitter.allocate(kernels.tupleMultiplicity(), tracks, kernels.hitContainer());
    // The per-N-bin offsets were read back once in beginTuplesAsync and have landed (framework seam), so
    // the fit consumes host values. If the container was absent, fall back to the cap-bounded fit.
    if (pending.offsetsHost)
      fitter.setHostTupleMultiplicityOffsets(pending.offsetsHost->data());
    if (m_params.algoParams_.useRiemannFit_) {
      fitter.launchRiemannKernels(
          trackingHits, modules, trackingHits.metadata().size(), TrackerTraits::maxNumberOfQuadruplets, queue);
    } else {
      // The CA main fit is the factorized fast BrokenLine fit (circle + line, 5 params, factorized cov
      // and chi2); the General Broken Lines fit runs once per track downstream, in the merger. The
      // per-tuple work bound is the runtime tuple capacity nTracks rather than the compile-time
      // maxNumberOfQuadruplets: tuple ids are always < nTracks, so the extra chunks would be empty.
      fitter.setFitCorrections(m_params.algoParams_.useFitCorrections_);
      fitter.launchBrokenLineKernels(trackingHits, modules, trackingHits.metadata().size(), nTracks, queue);
    }
    kernels.classifyTuples(trackingHits, tracks, queue, nullptr);

    // Refresh the pinned overflow mirror with the running totals. Ordered after classifyTuples, which
    // enqueues the overflow sentinel on this same queue, so the snapshot includes this event; consumed at
    // endStream once every event queue has drained. The destination alternates between the mirror's two
    // slots, so that the copies of two consecutive events, which can overlap, never write the same bytes.
    const uint32_t ovfSlot = ovfSlot_;
    ovfSlot_ ^= 1u;
    auto ovfHostSlot = cms::alpakatools::make_host_view(ovfHost_->data() + ovfSlot * kOvfWords, kOvfWords);
    alpaka::memcpy(queue, ovfHostSlot, *ovfAccum_);
#ifdef CA_SIZING_DUMP
    // Per-event per-container demand dump, for re-fitting the cap curves. One line per event with
    // the actual per-container demand and the caps they were sized against. The readback is a
    // single D2H sync of the 5-word extraStorage (APC + nCells + nTriplets + nCellTracks).
    {
      uint32_t dumpNTracks, dumpNHitsInTracks, dumpNCells, dumpNTriplets, dumpNCellTracks;
      kernels.readbackAllCounts(queue, dumpNTracks, dumpNHitsInTracks, dumpNCells, dumpNTriplets, dumpNCellTracks);
      // The cell->cell edge count is the triplet count, in deviceTriplets_ sized from
      // nCells * avgCellsPerCell; the cell->track edge count is nCellTracks, in deviceTracksCells_
      // sized from nCells * avgTracksPerCell.
      const auto dumpIterName = ::pixelTrack::iterationName[uint8_t(m_params.algoParams_.iterationName_)];
      printf(
          "[CA Sizing] iter=%.*s nHits=%u nCells=%u capCells=%u nTriplets=%u capTrips=%u "
          "nTracks=%u capTuples=%u nHitsInTracks=%u capHitCont=%u nCellTracks=%u capCellTrk=%u\n",
          int(dumpIterName.size()),
          dumpIterName.data(),
          hits_d.nHits(),
          dumpNCells,
          pending.maxDoublets,
          dumpNTriplets,
          kernels.tripletsN(),
          dumpNTracks,
          nTracks,
          dumpNHitsInTracks,
          // The binding per-track-hit bound: the hitContainer content capacity, which is also the
          // output trackHits row count (both come from nHitRowsForTuples).
          caHitNtupletGenerator::nHitRowsForTuples(nTracks, m_params.algoParams_.avgHitsPerTrack_),
          dumpNCellTracks,
          kernels.tracksCellsN());
    }
#endif

#ifdef CA_TRIPLET_DUMP
    // Surface the per-built-triplet dump out of the function-local kernels object before it is destroyed:
    // its device buffer moves into the generator member so the producer can emplace it. nValid was
    // stamped device-side in Kernel_connect and the buffer is sized to the full triplet capacity.
    device_tripletDump_ = std::move(kernels.tripletDumpBuffer());
#endif

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "finished building pixel tracks on GPU" << std::endl;
#endif

    return std::move(*pending.tracks);
  }

  reco::TrackingRecHitsMaskingCollection CAHitMaskingAndMerger::makeMaskingAsync(MapToHit const& mask_d,
                                                                                 TkSoADevice const& tracks_d,
                                                                                 const pixelTrack::Quality minQuality,
                                                                                 uint32_t const& iterationIndex,
                                                                                 Queue& queue,
                                                                                 bool applyMasking,
                                                                                 bool maskAttachedHits) const {
    const int nHits = mask_d.view().metadata().size();

    reco::TrackingRecHitsMaskingCollection mask(queue, static_cast<uint32_t>(nHits));

    alpaka::memcpy(queue,
                   cms::alpakatools::make_device_view(queue, mask.view().recHitMask(), nHits),
                   cms::alpakatools::make_device_view(queue, mask_d.view().recHitMask(), nHits));

    // When masking is deactivated, return a pass-through copy of the input mask: no hits from this
    // iteration's tracks are added, so the next iteration sees all hits, and the module stays wired.
    if (applyMasking) {
      CAHitMaskingAndMergerKernels kernels;

      auto tracksd_view = tracks_d.view().tracks();
      auto tracks_hitsd_view = tracks_d.view().trackHits();

      kernels.updateMasking(
          mask.view(), tracksd_view, tracks_hitsd_view, minQuality, iterationIndex, queue, maskAttachedHits);
    }
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "finished updating pixel masking on GPU (applyMasking=" << applyMasking << ")" << std::endl;
#endif

    return mask;
  }

  void CAHitMaskingAndMerger::mergeGather(TkSoADevice& outTracks,
                                          TkSoADevice const& inp0Tracks,
                                          TkSoADevice const& inp1Tracks,
                                          int nInputs,
                                          int32_t* armBuf,
                                          const int32_t arm0,
                                          const int32_t arm1,
                                          Queue& queue) const {
    CAHitMaskingAndMergerKernels kernels;

    auto outTrack_view = outTracks.view().tracks();
    auto outHit_view = outTracks.view().trackHits();
    auto inp0Track_view = inp0Tracks.view().tracks();
    auto inp0Hit_view = inp0Tracks.view().trackHits();
    auto inp1Track_view = inp1Tracks.view().tracks();
    auto inp1Hit_view = inp1Tracks.view().trackHits();

    kernels.mergeGather(outTrack_view,
                        outHit_view,
                        inp0Track_view,
                        inp0Hit_view,
                        inp1Track_view,
                        inp1Hit_view,
                        nInputs,
                        armBuf,
                        arm0,
                        arm1,
                        queue);
  }

  reco::TracksSoACollection CAHitMaskingAndMerger::makeFilteredTracks(int const& nTracks,
                                                                      int const& nHits,
                                                                      TkSoADevice const& inpTracks,
                                                                      pixelTrack::Quality const& minQuality,
                                                                      double const& matchFraction,
                                                                      Queue& queue,
                                                                      bool twinMerge,
                                                                      const int32_t* armOfTrack,
                                                                      float twinMergeDeltaEta,
                                                                      float twinMergeDeltaPhi,
                                                                      int twinMergeMinSharedHits,
                                                                      bool twinMergeTier2,
                                                                      float twinMergeTier2DeltaEta,
                                                                      float twinMergeTier2DeltaPhi,
                                                                      float twinMergeNSigma2,
                                                                      int twinMergeMinSharedFwd,
                                                                      bool twinMergeRefit,
                                                                      bool refitAllTracks,
                                                                      int32_t* unitedWinnerMask,
                                                                      const uint8_t* pocketArmIn,
                                                                      uint8_t* pocketArmIdOut) const {
    CAHitMaskingAndMergerKernels kernels;

    reco::TracksSoACollection tracks(queue, nTracks, nHits);

    auto tracksd_view = tracks.view().tracks();
    auto tracks_hitsd_view = tracks.view().trackHits();
    auto inptracksd_view = inpTracks.view().tracks();
    auto inptracks_hitsd_view = inpTracks.view().trackHits();

    // Strict cross-arm twin merge: compute per-track twin pairing and winner/loser bookkeeping, then let
    // filterTracks drop losers and unite their hits onto the winners. The scratch buffers are
    // stream-ordered, so they may destruct at function scope after the async launches. armOfTrack must be
    // a device pointer with nTracks entries (0 = prompt-side, 1 = displaced-side).
    const int32_t* loserOf_d = nullptr;
    const int32_t* isLoser_d = nullptr;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> bestTwin, loserOf, isLoser;
    if (twinMerge && nTracks > 0 && armOfTrack != nullptr) {
      bestTwin.emplace(cms::alpakatools::make_device_buffer<int32_t[]>(queue, nTracks));
      loserOf.emplace(cms::alpakatools::make_device_buffer<int32_t[]>(queue, nTracks));
      isLoser.emplace(cms::alpakatools::make_device_buffer<int32_t[]>(queue, nTracks));
      // isLoser is written cross-thread in Kernel_twinConfirm -> zero-initialise first.
      alpaka::memset(queue, *isLoser, 0);
      kernels.twinMerge(inptracksd_view,
                        inptracks_hitsd_view,
                        armOfTrack,
                        twinMergeDeltaEta,
                        twinMergeDeltaPhi,
                        twinMergeMinSharedHits,
                        twinMergeTier2,
                        twinMergeTier2DeltaEta,
                        twinMergeTier2DeltaPhi,
                        twinMergeNSigma2,
                        twinMergeMinSharedFwd,
                        minQuality,
                        bestTwin->data(),
                        loserOf->data(),
                        isLoser->data(),
                        nTracks,
                        queue);
      loserOf_d = loserOf->data();
      isLoser_d = isLoser->data();
    }

    kernels.filterTracks(tracksd_view,
                         tracks_hitsd_view,
                         inptracksd_view,
                         inptracks_hitsd_view,
                         minQuality,
                         matchFraction,
                         queue,
                         loserOf_d,
                         isLoser_d,
                         twinMergeRefit,
                         refitAllTracks,
                         unitedWinnerMask,
                         pocketArmIn,
                         pocketArmIdOut);
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "finished filtering track SoAs on GPU" << std::endl;
#endif

    return tracks;
  }

  void CAHitMaskingAndMerger::refitUnitedTracks(TkSoADevice& tracks,
                                                const int32_t* unitedWinnerMask,
                                                reco::CAGeometrySoACollection const& geometry,
                                                reco::TrackingRecHitsSoACollection const& hits,
                                                reco::OTRecHitsSoACollection const* otHits,
                                                reco::StackedModuleGeometrySoACollection const* stackedGeom,
                                                const float* rhoMapDevice,
                                                const float* bFieldMapDevice,
                                                float bfield,
                                                Queue& queue,
                                                bool dropOutlierFromHitList,
                                                bool outlierCoreProtect,
                                                bool fieldKernelWeights,
                                                bool chargeSymmetric,
                                                bool trajectoryCorrections,
                                                bool scatteringLogAtTotal,
                                                bool cumulativeEloss) const {
    using Fitter = HelixFit<pixelTopology::Phase2OTStubs>;
    // Refit population is bounded by the merged-collection capacity; the fast-fit scan skips every
    // slot whose mask entry is < 0 (non-winners + unused tail), so no host readback of nTracks is
    // needed and the garbage hitOffsets of the unused tail are never dereferenced.
    const uint32_t nTracksCap = uint32_t(tracks.view().tracks().metadata().size());
    if (nTracksCap == 0 || unitedWinnerMask == nullptr)
      return;

    // Minimal OT source: refitMergedTwins reads only otHits + per-module stacked frames for the
    // tagged (bit30) OT extras -- no phi binner / stub mask / ownership (those are attach-only).
    caExtension::OTHitsSource otSrc{};
    const caExtension::OTHitsSource* otSrcPtr = nullptr;
    if (otHits != nullptr && stackedGeom != nullptr && otHits->nHits() > 0) {
      otSrc.otHits = otHits->const_view().otRecHits();
      otSrc.otHitModules = otHits->const_view().otHitModules();
      otSrc.stackedGeometry = stackedGeom->const_view();
      otSrc.nOTHits = otHits->nHits();
      otSrcPtr = &otSrc;
    }

    Fitter fitter(bfield);
    fitter.setMaterialMap(rhoMapDevice);   // device BLMaterialMap (same EventSetup condition the CA uses)
    fitter.setBFieldMap(bFieldMapDevice);  // device BLBFieldMap (Bz,Br) r-z map; null => the scalar field
    fitter.setDropOutlierFromHitList(dropOutlierFromHitList);  // merger final-refit only, default off
    fitter.setOutlierCoreProtect(outlierCoreProtect);          // abstain instead of drop; final refit only
    fitter.setFieldKernelWeights(fieldKernelWeights);          // fit-consistent conversion field, default off
    fitter.setChargeSymmetric(chargeSymmetric);                // charge-symmetric corrections, default off
    fitter.setTrajectoryCorrections(trajectoryCorrections);    // reference-trajectory corrections, default off
    fitter.setScatteringLogAtTotal(scatteringLogAtTotal);      // Highland log at the track total, default off
    fitter.setCumulativeEloss(cumulativeEloss);                // cumulative-column typical loss, default off
    fitter.refitMergedTwins(hits.view().trackingHits(),
                            geometry.view().modules(),
                            tracks.view().tracks(),
                            tracks.view().trackHits(),
                            unitedWinnerMask,
                            otSrcPtr,
                            nTracksCap,
                            queue);
  }

  CAHitMaskingAndMerger::TkSoADevice CAHitMaskingAndMerger::finalDedupTracks(
      TkSoADevice const& refinedTracks,
      uint32_t nHits,
      uint32_t nOTHits,
      Queue& queue,
      const MergerDedupConfirmInputs* confirm) const {
    CAHitMaskingAndMergerKernels kernels;
    // The output can hold at most the input's tracks + hits (de-dup only removes) -> reuse the input
    // capacities so the compaction never overflows.
    const int nTracksCap = int(refinedTracks.view().tracks().metadata().size());
    const int nHitsCap = int(refinedTracks.view().trackHits().metadata().size());
    reco::TracksSoACollection out(queue, nTracksCap, nHitsCap);
    auto out_view = out.view().tracks();
    auto outHit_view = out.view().trackHits();
    auto in_view = refinedTracks.view().tracks();
    auto inHit_view = refinedTracks.view().trackHits();
    kernels.finalDedup(out_view, outHit_view, in_view, inHit_view, nTracksCap, nHits, nOTHits, queue, confirm);
    return out;
  }

  template <typename TrackerTraits>
  void CAHitNtupletGenerator<TrackerTraits>::reportOverflows(std::string const& moduleLabel) const {
    if (!ovfHost_)
      return;  // stream never processed an event
    // Element-wise maximum of the mirror's two slots (the device counters only grow, so the
    // larger of the two snapshots is the later one). Only the six written words are examined;
    // kOvfWords - 6 trailing words are reserved and never touched by the sentinel.
    auto const* v = ovfHost_->data();
    constexpr uint32_t kUsed = 6u;
    uint32_t tot[kUsed];
    bool any = false;
    for (uint32_t i = 0; i < kUsed; ++i) {
      tot[i] = std::max(v[i], v[i + kOvfWords]);
      any = any || (tot[i] != 0u);
    }
    if (any) {
      edm::LogWarning("CAHitNtupletGenerator")
          << moduleLabel << ": observed container overflow in this stream: output was truncated"
          << " (tracks/doublets/hits were dropped): tuples=" << tot[0] << " cells=" << tot[1]
          << " cellToCell=" << tot[2] << " cellToTrack=" << tot[3] << " hitContent=" << tot[4]
          << " hitToTuple=" << tot[5]
          << ". The container capacities (maxNumberOfDoublets/maxNumberOfTuples/avg* ratios) need review.";
    }
  }

  template class CAHitNtupletGenerator<pixelTopology::Phase1>;
  template class CAHitNtupletGenerator<pixelTopology::Phase2>;
  template class CAHitNtupletGenerator<pixelTopology::Phase2OT>;
  template class CAHitNtupletGenerator<pixelTopology::Phase2OTStubs>;
  template class CAHitNtupletGenerator<pixelTopology::HIonPhase1>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
