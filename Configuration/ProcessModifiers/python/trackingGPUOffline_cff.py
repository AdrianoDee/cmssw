import FWCore.ParameterSet.Config as cms

# This modifier replaces the pixel-seeded steps of the Phase-2 offline tracking
# with an initialStep seeded by the Alpaka CA (with recHit masking): only the
# initialStep, the pixelPairStep and the muon-seeded steps are run.
# Further CA sub-iterations of the initialStep are added by the
# trackingGPUOffline<SubIteration> modifiers, see
# RecoTracker/IterativeTracking/python/CATracksIterations_cff.py
# It has to be used together with the alpaka modifier.

trackingGPUOffline = cms.Modifier()
