import FWCore.ParameterSet.Config as cms

# CA tracking seeding of the Phase-2 offline iterative tracking (trackingGPUOffline modifier).
#
# The pixel-seeded steps are replaced by a single initialStep, seeded by the
# tracks of a chain of Alpaka CA sub-iterations: each sub-iteration runs on the
# hits not used by the tracks of the previous ones (recHit masking), and the
# tracks of all the sub-iterations are merged in SoA and converted to seeds.
# Only the pixelPairStep and the muon-seeded steps are kept from the legacy
# iterations (see iterativeTkConfig).
#
# The high-pT quadruplets are always used; the other sub-iterations are added to
# the seeds by their own modifier (trackingGPUOffline<SubIteration>). The chain
# order is fixed, so the modifiers are meant to be added in the chain order: a
# sub-iteration still runs (on demand) all the previous ones for the masking.

from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
from Configuration.ProcessModifiers.splitClustersInPhase2Pixel_cff import splitClustersInPhase2Pixel
from Configuration.ProcessModifiers.trackingGPUOffline_cff import trackingGPUOffline
from Configuration.ProcessModifiers.trackingGPUOfflineLowPtQuads_cff import trackingGPUOfflineLowPtQuads
from Configuration.ProcessModifiers.trackingGPUOfflineTriplets_cff import trackingGPUOfflineTriplets
from Configuration.ProcessModifiers.trackingGPUOfflineDetachedQuads_cff import trackingGPUOfflineDetachedQuads

import RecoTracker.IterativeTracking.InitialStepPreSplitting_cff as _initialStepPreSplitting
import RecoTracker.IterativeTracking.InitialStep_cff as _initialStep

from RecoLocalTracker.Phase2TrackerRecHits.Phase2TrackerRecHits_cfi import siPhase2RecHits
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import phase2OTRecHitsSoAConverter
from RecoLocalTracker.SiPixelClusterizer.siPixelDigisClustersFromSoAAlpakaPhase2_cfi import siPixelDigisClustersFromSoAAlpakaPhase2 as _siPixelDigisClustersFromSoAAlpakaPhase2
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromSoAAlpaka_cfi import siPixelRecHitFromSoAAlpaka as _siPixelRecHitFromSoAAlpaka
from RecoTracker.PixelTrackFitting.PixelTracks_cff import pixelTracksHighPtAlpaka as _pixelTracksHighPtAlpaka
from RecoTracker.PixelTrackFitting.PixelTracks_cff import pixelTracksHighPtMaskingSoA as _pixelTracksHighPtMaskingSoA
from RecoTracker.PixelTrackFitting.PixelTracks_cff import pixelTracksLowPtAlpakaPhase2Extended as _pixelTracksLowPtAlpakaPhase2Extended
from RecoTracker.PixelTrackFitting.PixelTracks_cff import pixelTracksSoA as _pixelTracksSoA
from RecoTracker.PixelTrackFitting.pixelTrackProducerFromSoAAlpaka_cfi import pixelTrackProducerFromSoAAlpaka as _pixelTrackProducerFromSoAAlpaka
from RecoTracker.TkSeedGenerator.SeedGeneratorFromProtoTracksEDProducer_cfi import SeedGeneratorFromProtoTracksEDProducer as _seedGeneratorFromProtoTracksEDProducer

import RecoTracker.IterativeTracking.CATracksHighPtParameters_cff as _highPt
import RecoTracker.IterativeTracking.CATracksLowPtParameters_cff as _lowPt

# the per layer pair cuts of the parameters files have to follow the pairGraph of the templates
def _checkLayerPairs(template, parameters):
    pairs = list(zip(template.geometry.pairGraph[::2], template.geometry.pairGraph[1::2]))
    if pairs != [(lp[0], lp[1]) for lp in parameters.layerPairs]:
        raise RuntimeError("the layerPairs of %s do not match the pairGraph of the CA template" % parameters.__name__)

_checkLayerPairs(_pixelTracksHighPtAlpaka, _highPt)
_checkLayerPairs(_pixelTracksLowPtAlpakaPhase2Extended, _lowPt)

# the legacy pixel clusters and rechits used by the iterative tracking are
# converted from the Alpaka SoA, so that the hits of the CA seeds match the ones
# used by the CKF and by the cluster removal
(trackingPhase2PU140 & ~splitClustersInPhase2Pixel & trackingGPUOffline).toReplaceWith(_initialStepPreSplitting.siPixelClusters, _siPixelDigisClustersFromSoAAlpakaPhase2.clone(
    clusterThreshold_layer1 = 4000,
    clusterThreshold_otherLayers = 4000,
    src = "siPixelClustersPreSplittingAlpaka",
    storeDigis = False,
    produceDigis = False
))
(trackingPhase2PU140 & ~splitClustersInPhase2Pixel & trackingGPUOffline).toReplaceWith(_initialStepPreSplitting.siPixelRecHits, _siPixelRecHitFromSoAAlpaka.clone(
    pixelRecHitSrc = "siPixelRecHitsPreSplittingAlpaka",
    src = "siPixelClusters"
))

# tracks of the CA sub-iterations, merged in SoA (the sub-iterations are added below)
initialStepCATracksSoA = _pixelTracksSoA.clone(
    inputTracks = []
)

# legacy tracks from the merged CA tracks, used as seeds
initialStepCATracks = _pixelTrackProducerFromSoAAlpaka.clone(
    pixelRecHitLegacySrc = "siPixelRecHits",
    beamSpot = "offlineBeamSpot",
    minNumberOfHits = 0,
    minNumberOfPixelHits = 3, # needed by the electron seeds
    minQuality = "tight",
    trackSrc = "initialStepCATracksSoA",
    outerTrackerRecHitSrc = "siPhase2RecHits",
    outerTrackerRecHitSoAConverterSrc = "phase2OTRecHitsSoAConverter",
    useOTExtension = True,
    requireQuadsFromConsecutiveLayers = True
)

initialStepCATask = cms.Task(siPhase2RecHits,
                             phase2OTRecHitsSoAConverter,
                             initialStepCATracksSoA,
                             initialStepCATracks)

def _caSubIteration(name, ca, previous = None, modifier = None):
    """Defines the CA sub-iteration initialStepCA<name>, on the hits not masked by
    the sub-iteration <previous>, and initialStepCA<name>Mask, with the hits of
    this and of the previous sub-iterations masked, input of the next one.
    The tracks are added to the seeds always if modifier is None, otherwise only
    when the modifier is active."""
    label = "initialStepCA" + name
    if previous is None:
        mask = _pixelTracksHighPtMaskingSoA.clone(
            trackSoA = label
        )
    else:
        ca = ca.clone(
            hitMask = "initialStepCA" + previous + "Mask"
        )
        mask = _pixelTracksHighPtMaskingSoA.clone(
            trackSoA = label,
            oldMask = "initialStepCA" + previous + "Mask",
            hitSoAs = []
        )
    globals().update({label: ca, label + "Mask": mask})
    initialStepCATask.add(ca, mask)
    if modifier is None:
        initialStepCATracksSoA.inputTracks.append(cms.InputTag(label))
    else:
        modifier.toModify(initialStepCATracksSoA, lambda merger: merger.inputTracks.append(cms.InputTag(label)))

# CA sub-iterations, in the order in which they mask the hits
# (pT thresholds and phi windows from CATracksHighPtParameters_cff and CATracksLowPtParameters_cff)
_caSubIteration("HighPtQuads", _pixelTracksHighPtAlpaka.clone(
                    **_highPt.caTracksHighPtParameters
                ))

_caSubIteration("LowPtQuads", _pixelTracksLowPtAlpakaPhase2Extended.clone(
                    **_lowPt.caTracksLowPtParameters
                ),
                previous = "HighPtQuads",
                modifier = trackingGPUOfflineLowPtQuads)

_caSubIteration("HighPtTriplets", _pixelTracksHighPtAlpaka.clone(
                    minHitsPerNtuplet = 3,
                    iterationName = "tripletsHighPt",
                    **_highPt.caTracksHighPtParameters
                ),
                previous = "LowPtQuads",
                modifier = trackingGPUOfflineTriplets)

_caSubIteration("LowPtTriplets", _pixelTracksLowPtAlpakaPhase2Extended.clone(
                    minHitsPerNtuplet = 3,
                    iterationName = "tripletsLowPt",
                    **_lowPt.caTracksLowPtParameters
                ),
                previous = "HighPtTriplets",
                modifier = trackingGPUOfflineTriplets)

# starting point: transverse compatibility with the beamspot opened to the
# originRadius of the legacy detachedQuadStep region (0.9 cm), and the same for
# the tip cut of the tight quality (required by the merger and by the converter)
_caSubIteration("DetachedQuads", _pixelTracksLowPtAlpakaPhase2Extended.clone(
                    iterationName = "detachedStep",
                    trackQualityCuts = dict(maxTip = 1.0),
                    geometry = dict(caDCACuts = [1.0] * len(_pixelTracksLowPtAlpakaPhase2Extended.geometry.caDCACuts))
                ),
                previous = "LowPtTriplets",
                modifier = trackingGPUOfflineDetachedQuads)

# initialStep seeded by the CA
(trackingPhase2PU140 & trackingGPUOffline).toReplaceWith(_initialStep.initialStepSeeds, _seedGeneratorFromProtoTracksEDProducer.clone(
    InputCollection = "initialStepCATracks",
    useProtoTrackKinematics = True,
    includeFourthHit = False,
))

_InitialStepTask_trackingPhase2_GPUOffline = _initialStep.InitialStepTask.copyAndExclude([_initialStep.initialStepSeedLayers,
                                                                                        _initialStep.initialStepTrackingRegions,
                                                                                        _initialStep.initialStepHitDoublets,
                                                                                        _initialStep.initialStepHitQuadruplets])
_InitialStepTask_trackingPhase2_GPUOffline.add(initialStepCATask)
(trackingPhase2PU140 & trackingGPUOffline).toReplaceWith(_initialStep.InitialStepTask, _InitialStepTask_trackingPhase2_GPUOffline)
