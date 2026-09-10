"""Track-level training dataset: one row per reconstructed track, with the features the
selectors see and the truth the track validation uses.

Adds tables to an existing reconstruction process (the base config; see _import_chassis) and
writes them as nanoAOD. Two row spaces per iteration, joined offline by (pt, phi):
  Trk<X>Full  : every track in the SoA, before selection -> fitted features
  Trk<X>CA    : the same tracks' hit and stub features
  Trk<X>Truth : legacy conversion at quality NANO_MINQUALITY -> matched/duplicate/truth kinematics
With NANO_MERGED=1 the same three tables exist for X = Merged (the merged collection both
iterations produce together). Feeds the track DNN and both final HP selectors. Multithreaded.

  NANO_SAMPLE, NANO_NEVT, NANO_OUT, NANO_SKIP, NANO_FILES, NANO_THREADS: see below.
  NANO_MINQUALITY  'loose' (track DNN population) | 'tight' (default, final selector)
  NANO_CHASSIS     base reconstruction config module (make_base_config.sh generates one)
  NANO_MERGED=1     adds TrkMerged* over the merged collection (both arms' HP output, twin-merged,
                    OT-extended, GBL-refitted, de-duplicated) -- the population a final HP forest
                    decides on. Needs a two-iteration chassis. Truth always carries BOTH labels.
  NANO_CLUSTER=1     adds 8 pixel-cluster charge/shape columns to the MERGED CA table. Off by
                    default (changes schema). NANO_CLUSTER_QLOW overrides the low-charge threshold.
  NANO_MTV_LABEL=1  adds MTV-true columns matchedAny/duplicateAny to the PROMPT/DISPLACED
                    truth tables. Off by default (changes schema). matched keeps its meaning.

Chain-state overrides (one set per iteration; default = inherit the producer config):
NANO_PROMPT_GATE / _GATEOFF / _TRACKDNN / _TRACKTHR / _CAPS and NANO_DISP_* equivalents.
Every dump prints a '[trackNano CHAIN-STATE]' line per iteration; a dataset is only valid for
the state on that line. The retraining entry scripts set the right state for each step.

Iterations: prompt (hltPhase2PixelTracksSoA) and displaced
(hltPhase2PixelTracksSoADisplaced or hltPhase2PixelTracksSoALowPt -- either label
resolves). The displaced truth uses a displaced-friendly TP selection (tip < 60 cm, no in-time req).
"""
import glob
import os

# Eigen must be in the cling interpreter before any module is constructed:
# SimplePixelTrackSoATabFlatTableProducer's expression parser autoparses PixelTrackSoATab.h ->
# SoA headers, which require Eigen/Core to be visible first (else FatalRootError).
try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gInterpreter.Declare("#include <Eigen/Core>")
except Exception as _e:  # pragma: no cover
    print("WARNING: Eigen cling preload failed:", _e)

import FWCore.ParameterSet.Config as cms

# NANO_SAMPLE is a label: it picks the built-in input list and names the output. NANO_FILES
# overrides the input list with an explicit one.
_sample = os.environ.get("NANO_SAMPLE", "ttbarPU")


def _import_chassis(candidates):
    """Import the base reconstruction config this dump adds its tables to.

    This config builds no reconstruction of its own. NANO_CHASSIS names the module to use;
    otherwise the first importable name in `candidates` is taken. Generate one with

        ./make_base_config.sh --files <input.root,...> --out <dir>/base_reco_cfg.py

    and either pass NANO_CHASSIS=base_reco_cfg with <dir> on PYTHONPATH, or let the retraining
    entry scripts do it. Resolving by name (rather than a fixed import) means a renamed or
    missing base config gives a clear error listing what was tried. The validation and
    monitoring paths of the base config are pruned below, so a validation config is a fine
    base."""
    import importlib
    forced = os.environ.get("NANO_CHASSIS")
    names = [forced] if forced else list(candidates)
    tried = []
    for _n in names:
        try:
            _mod = importlib.import_module(_n)
        except ImportError as _e:
            tried.append("%s (%s)" % (_n, _e))
            continue
        print("trackNano: base config = %s" % _n)
        return _mod.process
    raise ImportError(
        "trackNano_config: no base reconstruction config found. Tried: " + "; ".join(tried)
        + ".\n  Generate one, from the events you want to reconstruct, with:\n"
        "    RecoTracker/PixelSeeding/test/models/make_base_config.sh \\\n"
        "        --files <a.root,b.root|dir|list.txt> --out <dir>/base_reco_cfg.py\n"
        "  then run this config with NANO_CHASSIS=base_reco_cfg and <dir> on PYTHONPATH.\n"
        "  The retraining entry scripts (retrain_prompt.sh / retrain_displaced.sh) do all of\n"
        "  that for you when they are given --input <event files>.")


# Only the generated chassis is a candidate: a private configuration picked up from the checkout
# would reconstruct on whatever geometry, era and aging scenario it happens to carry. Name one
# explicitly with NANO_CHASSIS if that is really what is wanted.
process = _import_chassis(["base_reco_cfg"])


# The displaced CA module label differs between menus (SoADisplaced / SoALowPt). Resolve it once
# (it is also what the InputTags / clone(trackSrc=) need).
def _resolve_label(*names):
    for _n in names:
        if hasattr(process, _n):
            return _n
    raise ImportError("trackNano_config: none of %r found on the chassis process" % (names,))


_dispLabel = _resolve_label("hltPhase2PixelTracksSoADisplaced", "hltPhase2PixelTracksSoALowPt")

# Built-in input list for the no-pile-up displaced sample. It is a convenience for one site;
# NANO_FILES (or a base config generated with the files already in it) replaces it, and the
# base config's own list is kept if the directory is not there.
if _sample == "displacedNoPU" and not os.environ.get("NANO_FILES"):
    _dir = ("/eos/cms/store/relval/CMSSW_16_1_0_pre4/RelValDisplacedSUSY_14TeV/GEN-SIM-DIGI-RAW/"
            "150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU_20260420_140846-v1/2590000/")
    _files = sorted(glob.glob(_dir + "*.root"))
    if _files:
        process.source.fileNames = cms.untracked.vstring(["file:" + f for f in _files])
    else:
        print("trackNano: the built-in displacedNoPU input list is not available here; keeping "
              "the base config's own input files. Pass NANO_FILES to choose them explicitly.")

_files = os.environ.get("NANO_FILES")
if _files:
    process.source.fileNames = cms.untracked.vstring(
        [(f if f.startswith("file:") or f.startswith("/store") else "file:" + f)
         for f in _files.split(",") if f])

process.maxEvents.input = cms.untracked.int32(int(os.environ.get("NANO_NEVT", "1000")))
process.source.skipEvents = cms.untracked.uint32(int(os.environ.get("NANO_SKIP", "0")))

# NANO_DISPLOOSEN scales the displaced iteration's longitudinal doublet cuts, so a dataset can be
# dumped on the looser population a retrained chain is meant to reach. It scales what the producer
# carries, so it stays correct when those cuts are retuned.
_dl = os.environ.get("NANO_DISPLOOSEN")
if _dl is not None:
    s = float(_dl); _dc = getattr(process, _dispLabel).doubletCuts
    _dc.maxZ0 = cms.vdouble([v * s for v in _dc.maxZ0])
    _dc.maxDZ = cms.vdouble([v * s for v in _dc.maxDZ])
    _dc.minDZ = cms.vdouble([v * s for v in _dc.minDZ])
    print("trackNano: displaced z0/dz cuts x%g" % s)

# NANO_DOSTATS=1 turns on the occupancy summary, which reports an overflowed buffer instead of
# letting it drop tracks silently. NANO_PROMPT_CAPS / NANO_DISP_CAPS below enlarge the buffers.
if os.environ.get("NANO_DOSTATS") == "1":
    process.hltPhase2PixelTracksSoA.fillStatistics = cms.bool(True)
    getattr(process, _dispLabel).fillStatistics = cms.bool(True)
    print("trackNano: buffer occupancy summary on")

# Chain-state hooks, one set per iteration. A training dump must be produced with the upstream
# models in the state the retrained chain will run in: tripletDNNThreshold at the deployed gate,
# useTrackDNN off for a track-DNN dataset and on at the deployed threshold for a final-HP one.
# By default the producer configuration is inherited; NANO_PROMPT_GATE / _GATEOFF / _TRACKDNN /
# _TRACKTHR / _CAPS and their NANO_DISP_* equivalents override it and are printed.


def _scale_str_cap(mod, name, factor):
    """Scale a cap written as a formula in x (the hit count) by `factor`, by wrapping it as
    'factor*(expr)'. That is correct for any form -- a plain number, a linear or a quadratic
    formula -- whereas ASSIGNING a fixed formula can silently shrink a cap the arm has since
    outgrown. No-op if the parameter is absent."""
    if not hasattr(mod, name):
        return
    old = getattr(mod, name).value()
    setattr(mod, name, cms.string("%g*(%s)" % (factor, old)))


def _scale_dbl_cap(mod, name, factor):
    if not hasattr(mod, name):
        return
    setattr(mod, name, cms.double(float(getattr(mod, name).value()) * factor))


def _apply_chain_overrides(mod, prefix):
    """Apply the <prefix>_* overrides to one CA producer; return what was applied."""
    ov = []
    if os.environ.get(prefix + "_CAPS") == "big":
        # doublets are not scaled: the doublet count does not depend on the (switched-off) track DNN,
        # and x4 pushes Kernel_connect past the 65535-block grid limit on GPU at PU200.
        # <prefix>_CAPS_FACTOR (default 4) sets the enlargement; a training dump must not lose
        # items to container truncation (the dropped population would bias the model).
        _f = float(os.environ.get(prefix + "_CAPS_FACTOR", "4"))
        _scale_str_cap(mod, "maxNumberOfTuples", _f)
        _scale_dbl_cap(mod, "avgCellsPerCell", _f)
        _scale_dbl_cap(mod, "avgTracksPerCell", _f)
        mod.fillStatistics = cms.bool(True)
        ov.append("caps=tuples/cells x%g" % _f)
    if os.environ.get(prefix + "_GATE"):
        mod.tripletDNNThreshold = cms.double(float(os.environ[prefix + "_GATE"]))
        ov.append("gate=%s" % os.environ[prefix + "_GATE"])
    if os.environ.get(prefix + "_GATEOFF") == "1":
        mod.useTripletDNN = cms.bool(False)
        ov.append("gateOFF")
    if os.environ.get(prefix + "_TRACKDNN") is not None:
        mod.useTrackDNN = cms.bool(os.environ[prefix + "_TRACKDNN"] == "1")
        ov.append("trackDNN=%s" % os.environ[prefix + "_TRACKDNN"])
    if os.environ.get(prefix + "_TRACKTHR"):
        mod.trackDNNThreshold = cms.double(float(os.environ[prefix + "_TRACKTHR"]))
        ov.append("trackThr=%s" % os.environ[prefix + "_TRACKTHR"])
    return ov


def _shown(mod, name):
    """Value of a producer parameter for the provenance line. A parameter the python
    configuration does not set takes the default declared in the module's C++ fillDescriptions,
    which cannot be read from here -- say so instead of guessing or crashing."""
    p = getattr(mod, name, None)
    return p.value() if p is not None else "<C++ default>"


_pm = process.hltPhase2PixelTracksSoA
_dm = getattr(process, _dispLabel)
_ovPrompt = _apply_chain_overrides(_pm, "NANO_PROMPT")
_ovDisp = _apply_chain_overrides(_dm, "NANO_DISP")

# Third row space: the merged collection (twin merge, OT-hit attach walk, GBL refit and dedup over
# both arms' HP-selected outputs). It exists only on a two-iteration chassis, hence the guard on
# the module being there as well as on the switch.
_MERGED = os.environ.get("NANO_MERGED", "0") == "1"
# Pixel-cluster charge/shape aggregates on the merged CA table (schema change -> opt-in).
_CLUSTER = os.environ.get("NANO_CLUSTER", "0") == "1"
_CLUSTER_QLOW = float(os.environ.get("NANO_CLUSTER_QLOW", "7000"))
if _MERGED and not hasattr(process, "hltPhase2PixelTracksSoAMerger"):
    print("trackNano: NANO_MERGED=1 but this chassis has no hltPhase2PixelTracksSoAMerger "
          "(single-iteration menu?) -- the merged tables are NOT emitted.")
    _MERGED = False

# The dump's provenance line, one per iteration. Read it in the log before using a dataset: the
# dataset is only valid for the chain state printed here.
for _label, _mod, _ov in (("prompt", _pm, _ovPrompt), ("displaced", _dm, _ovDisp)):
    print("[trackNano CHAIN-STATE] %s: useTripletDNN=%s tripletDNNThreshold=%s | useTrackDNN=%s "
          "trackDNNThreshold=%s | maxNumberOfTuples=%s avgCellsPerCell=%s avgTracksPerCell=%s | "
          "minQuality=%s | overrides=[%s]"
          % (_label, _shown(_mod, "useTripletDNN"), _shown(_mod, "tripletDNNThreshold"),
             _shown(_mod, "useTrackDNN"), _shown(_mod, "trackDNNThreshold"),
             _shown(_mod, "maxNumberOfTuples"), _shown(_mod, "avgCellsPerCell"),
             _shown(_mod, "avgTracksPerCell"), os.environ.get("NANO_MINQUALITY", "tight"),
             ",".join(_ov) if _ov else "none (inheriting the configuration)"))

# The merged row space has no combinatorics knobs of its own: it inherits both lines above plus the
# two HP selectors it consumes; its own provenance is the collections it merges and the operating
# point of the attach/refit/dedup stages.
if _MERGED:
    _mg = process.hltPhase2PixelTracksSoAMerger
    print("[trackNano CHAIN-STATE] merged: hltPhase2PixelTracksSoAMerger inputTkSoAs=%s | "
          "mergerMinQuality=%s twinMergeNSigma2=%s extHostMaxChi2Ndof=%s extMaxWalkLayers=%s | "
          "minQuality=%s | row space = post-HP on BOTH arms (twin merge -> OT attach -> GBL refit "
          "-> dedup); valid only for the two lines above"
          % ([str(_t) for _t in _mg.inputTkSoAs], _shown(_mg, "minQuality"),
             _shown(_mg, "twinMergeNSigma2"), _shown(_mg, "extHostMaxChi2Ndof"),
             _shown(_mg, "extMaxWalkLayers"), os.environ.get("NANO_MINQUALITY", "tight")))

# Prune validation/DQM paths: only the HLT reconstruction is needed.
# Also prune the base config's own output EndPath (<eventcontent>output_step) and drop every
# output module it declares: this dump writes its own NanoAOD file below, and a chassis made
# without cmsDriver's '--output={}' would otherwise write a full GEN-SIM-DIGI-RAW file next to
# every table -- tens of GB and a large share of the wall time, silently.
_keep = []
for _p in process.schedule:
    _lbl = _p.label_()
    if any(k in _lbl for k in ("alidation", "DQM", "dqm")) or _lbl.endswith("output_step"):
        continue
    _keep.append(_p)
process.schedule = cms.Schedule(*_keep)
for _om in list(process.outputModules_()):
    delattr(process, _om)

# Associator: the same modules the validation uses.
from Validation.RecoTrack.associators_cff import hltTPClusterProducer, hltTrackAssociatorByHits
process.hltTPClusterProducer = hltTPClusterProducer
process.hltTrackAssociatorByHits = hltTrackAssociatorByHits

from HLTrigger.NGTScouting.hltTracks_cfi import (hltPixelTrackSoATable, hltPixelTrackTable,
                                                 pixelTrackAssoc, tpSelectorPixelTracks)
from HLTrigger.Configuration.HLT_75e33.modules.hltPhase2PixelTrackSoATableProducer_cfi import (
    hltPhase2PixelTrackSoATableProducer)

# Legacy conversions of the pre-HP SoAs (truth row space; minQuality tight = the population the
# HP/tight selection decides on).
_convCommon = dict(
    beamSpot=cms.InputTag("hltOnlineBeamSpot"),
    pixelRecHitLegacySrc=cms.InputTag("hltSiPixelRecHits"),
    outerTrackerRecHitSrc=cms.InputTag("hltSiPhase2RecHits"),
    otRecHitsSoASrc=cms.InputTag("hltPixelSeedingOTRecHitsSoA"),
    stubsSoASrc=cms.InputTag("hltOTStubProducer"),
    minNumberOfHits=cms.int32(0),
    minQuality=cms.string(os.environ.get("NANO_MINQUALITY", "tight")),  # 'loose' = full candidate population (track-DNN dataset)
    useOTExtension=cms.bool(True),
    expandStubs=cms.bool(True),
    requireQuadsFromConsecutiveLayers=cms.bool(False),
)
process.trainPromptTracks = cms.EDProducer("PixelTrackProducerFromSoAAlpaka",
                                           trackSrc=cms.InputTag("hltPhase2PixelTracksSoA"),
                                           **_convCommon)
process.trainDispTracks = cms.EDProducer("PixelTrackProducerFromSoAAlpaka",
                                         trackSrc=cms.InputTag(_dispLabel),
                                         **_convCommon)

# Displaced-friendly TP selection: keep displaced TPs (tip<60cm), in/out-of-time PU real.
tpSelectorDisplaced = tpSelectorPixelTracks.clone(
    tip=cms.double(60.0),
    lip=cms.double(80.0),
    ptMin=cms.double(0.4),
    intimeOnly=cms.bool(False),
)

# All-inclusive TP selection, i.e. the MTV fake definition: MTV calls a track fake only when it is
# associated to no TrackingParticle at all, while the narrow selections above are the efficiency
# denominators. A track matched to a real particle that fails one of their cuts is real by MTV and
# fake by the narrow label, so the truth tables carry both `matched` (narrow, the efficiency class)
# and `matchedAny` (inclusive, the training target). Only charge is cut here, as the hit associator
# needs it.
tpSelectorAnyTP = tpSelectorPixelTracks.clone(
    ptMin=cms.double(0.0),
    tip=cms.double(1000.0),
    lip=cms.double(1000.0),
    minRapidity=cms.double(-10.0),
    maxRapidity=cms.double(10.0),
    intimeOnly=cms.bool(False),
    signalOnly=cms.bool(False),
    stableOnly=cms.bool(False),
    chargedOnly=cms.bool(True),
    minHit=cms.int32(0),
)

# NANO_MTV_LABEL=1 adds the inclusive association (and the matchedAny/duplicateAny columns) to the
# prompt and displaced truth tables too; off by default because it changes those tables' schema.
# The merged arm always carries both labels.
_MTV_LABEL = os.environ.get("NANO_MTV_LABEL", "0") == "1"

process.trainPromptAssoc = pixelTrackAssoc.clone(trackCollection="trainPromptTracks")
process.trainDispAssoc = pixelTrackAssoc.clone(trackCollection="trainDispTracks",
                                               tpSelectorPSet=tpSelectorDisplaced)
if _MTV_LABEL:
    process.trainPromptAssocAny = pixelTrackAssoc.clone(trackCollection="trainPromptTracks",
                                                        tpSelectorPSet=tpSelectorAnyTP)
    process.trainDispAssocAny = pixelTrackAssoc.clone(trackCollection="trainDispTracks",
                                                      tpSelectorPSet=tpSelectorAnyTP)

# Feature tables: all SoA tracks, exact deployment ABI.
process.trainPromptSoATab = hltPhase2PixelTrackSoATableProducer.clone(trackSrc="hltPhase2PixelTracksSoA")
process.trainDispSoATab = hltPhase2PixelTrackSoATableProducer.clone(trackSrc=_dispLabel)
process.trainPromptFullTable = hltPixelTrackSoATable.clone(src="trainPromptSoATab", name="TrkPromptFull")
process.trainDispFullTable = hltPixelTrackSoATable.clone(src="trainDispSoATab", name="TrkDispFull")

# CA-features table: the hit/stub features of the in-kernel track DNN plus the extra columns,
# host-computed via the shared CATrackFeatures.h; same row space as TrkDispFull.
process.trainDispCATable = cms.EDProducer(
    "CATrackFeaturesTableProducer",
    tableName=cms.string("TrkDispCA"),
    trackSrc=cms.InputTag(_dispLabel),
    mergedHitsSrc=cms.InputTag("hltPhase2PixelRecHitsStubsMerger"),
    # Set explicitly rather than left to the C++ default: an empty/wrong OT SoA silently NaNs every
    # OT-extended row (caTrackFeatures::fill returns false with no view) instead of raising.
    otRecHitsSoASrc=cms.InputTag("hltPixelSeedingOTRecHitsSoA"),
    # Attach-purity: also emit a per-(track,hit) truth table (isOTExtra/isStub/isTrueForOwnTP).
    emitHitTruth=cms.bool(True),
    hitTableName=cms.string("TrkDispCAHit"),
)

# Same CA hit/stub features for the prompt iteration, with the same 12-feature ABI as the displaced
# one. Index-aligned to TrkPromptFull.
process.trainPromptCATable = cms.EDProducer(
    "CATrackFeaturesTableProducer",
    tableName=cms.string("TrkPromptCA"),
    trackSrc=cms.InputTag("hltPhase2PixelTracksSoA"),
    mergedHitsSrc=cms.InputTag("hltPhase2PixelRecHitsStubsMerger"),
    otRecHitsSoASrc=cms.InputTag("hltPixelSeedingOTRecHitsSoA"),  # explicit; see TrkDispCA above
    # Attach-purity: also emit a per-(track,hit) truth table (isOTExtra/isStub/isTrueForOwnTP).
    emitHitTruth=cms.bool(True),
    hitTableName=cms.string("TrkPromptCAHit"),
)

# Truth tables on the legacy conversions (kinematics for the join + truth columns).
# assocAny (optional): a second association run with tpSelectorAnyTP, contributing matchedAny /
# duplicateAny. `matched` keeps its meaning (the efficiency-selected match) so nothing downstream
# shifts; matchedAny is the MTV-true label a fake-rejection model should be trained against, and by
# construction matchedAny >= matched row-wise.
def _truthTable(src, name, assoc, assocAny=None):
    t = hltPixelTrackTable.clone(src=src, name=name)
    for _v in t.externalVariables.parameterNames_():
        getattr(t.externalVariables, _v).src = cms.InputTag(assoc, getattr(t.externalVariables, _v).src.getProductInstanceLabel())
    if assocAny:
        t.externalVariables.matchedAny = cms.PSet(
            src=cms.InputTag(assocAny, "matched"),
            doc=cms.string("1 if matched to ANY TrackingParticle (the MTV-true label: MTV calls a "
                           "track fake iff this is 0). Training target for fake rejection."),
            type=cms.string("uint8"))
        t.externalVariables.duplicateAny = cms.PSet(
            src=cms.InputTag(assocAny, "duplicate"),
            doc=cms.string("1 if multiple reco tracks map to the same TP, under the all-inclusive "
                           "TP selection"),
            type=cms.string("uint8"))
    return t

process.trainPromptTruthTable = _truthTable("trainPromptTracks", "TrkPromptTruth", "trainPromptAssoc",
                                            "trainPromptAssocAny" if _MTV_LABEL else None)
process.trainDispTruthTable = _truthTable("trainDispTracks", "TrkDispTruth", "trainDispAssoc",
                                          "trainDispAssocAny" if _MTV_LABEL else None)

# Merged row space (NANO_MERGED=1): TrkMergedFull / TrkMergedCA / TrkMergedTruth over
# hltPhase2PixelTracksSoAMerger. It differs from the two per-arm row spaces in that the population
# is already HP-selected once per arm, chi2/ndof are post-GBL-refit and nHits post-extension, and
# `iteration` is meaningful only here (hence emitMergedProvenance). The TP selection is the
# displaced one: the prompt selector's tip=2.5cm would label every genuine displaced track a fake.
if _MERGED:
    _mergedLabel = "hltPhase2PixelTracksSoAMerger"
    process.trainMergedTracks = cms.EDProducer("PixelTrackProducerFromSoAAlpaka",
                                               trackSrc=cms.InputTag(_mergedLabel),
                                               **_convCommon)
    process.trainMergedAssoc = pixelTrackAssoc.clone(trackCollection="trainMergedTracks",
                                                     tpSelectorPSet=tpSelectorDisplaced)
    # Always both labels on this arm: matched (displaced efficiency class) and matchedAny (MTV).
    process.trainMergedAssocAny = pixelTrackAssoc.clone(trackCollection="trainMergedTracks",
                                                        tpSelectorPSet=tpSelectorAnyTP)
    process.trainMergedSoATab = hltPhase2PixelTrackSoATableProducer.clone(trackSrc=_mergedLabel)
    process.trainMergedFullTable = hltPixelTrackSoATable.clone(src="trainMergedSoATab",
                                                               name="TrkMergedFull")
    process.trainMergedCATable = cms.EDProducer(
        "CATrackFeaturesTableProducer",
        tableName=cms.string("TrkMergedCA"),
        trackSrc=cms.InputTag(_mergedLabel),
        mergedHitsSrc=cms.InputTag("hltPhase2PixelRecHitsStubsMerger"),
        # A missing OT SoA does not fail loudly: it NaNs the whole feature row of every track
        # carrying a tagged OT extra (caTrackFeatures::fill returns false with no view).
        otRecHitsSoASrc=cms.InputTag("hltPixelSeedingOTRecHitsSoA"),
        emitHitTruth=cms.bool(False),  # heavy (TrackerHitAssociator); the merged forest does not need it
        emitMergedProvenance=cms.bool(True),  # iteration / ndof / nOTExtra / nAttached
        # Pixel-cluster charge/shape block (NANO_CLUSTER=1). Off by default so an existing merged
        # dataset's schema is unchanged; the threshold is a plain parameter, so re-picking it from
        # the measured distribution costs a re-dump but never a rebuild.
        emitClusterFeatures=cms.bool(_CLUSTER),
        lowChargeThreshold=cms.double(_CLUSTER_QLOW),
        hitTableName=cms.string("TrkMergedCAHit"),
    )
    process.trainMergedTruthTable = _truthTable("trainMergedTracks", "TrkMergedTruth",
                                                "trainMergedAssoc", "trainMergedAssocAny")

process.trainNanoTask = cms.Task(
    process.hltTPClusterProducer,
    process.hltTrackAssociatorByHits,
    process.trainPromptTracks,
    process.trainDispTracks,
    process.trainPromptAssoc,
    process.trainDispAssoc,
    process.trainPromptSoATab,
    process.trainDispSoATab,
)
process.trainNanoPath = cms.Path(
    process.trainPromptFullTable
    + process.trainDispFullTable
    + process.trainPromptCATable  # also emits per-hit purity table TrkPromptCAHit
    + process.trainDispCATable    # also emits per-hit purity table TrkDispCAHit
    + process.trainPromptTruthTable
    + process.trainDispTruthTable,
    process.trainNanoTask,
)
if _MTV_LABEL:
    process.trainNanoTask.add(process.trainPromptAssocAny, process.trainDispAssocAny)
if _MERGED:
    process.trainNanoTask.add(process.trainMergedTracks,
                              process.trainMergedAssoc,
                              process.trainMergedAssocAny,
                              process.trainMergedSoATab)
    process.trainNanoPath += (process.trainMergedFullTable
                              + process.trainMergedCATable
                              + process.trainMergedTruthTable)
process.schedule.append(process.trainNanoPath)

_out = os.environ.get("NANO_OUT", "trackNano_%s.root" % _sample)
process.trainNanoOut = cms.OutputModule(
    "NanoAODOutputModule",
    fileName=cms.untracked.string(_out),
    outputCommands=cms.untracked.vstring(
        "drop *",
        "keep nanoaodFlatTable_train*_*_*",
    ),
)
process.trainNanoEnd = cms.EndPath(process.trainNanoOut)
process.schedule.append(process.trainNanoEnd)

# Thread/stream count: the nano tables carry the event structure (no ordering constraint), so
# allow many threads via NANO_THREADS for fast parallel dump jobs.
_nthr = int(os.environ.get("NANO_THREADS", "1"))
process.options.numberOfThreads = _nthr
process.options.numberOfStreams = _nthr

print("trackNano: sample=%s nevt=%d out=%s paths=%d threads=%d" % (
    _sample, process.maxEvents.input.value(), _out, len(list(process.schedule)), _nthr))
