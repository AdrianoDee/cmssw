import FWCore.ParameterSet.Config as cms

# This modifier adds the Alpaka CA sub-iteration(s) for the low-pT quadruplets to the seeds of
# the initialStep, see RecoTracker/IterativeTracking/python/CATracksIterations_cff.py
# It has to be used together with the trackingGPUOffline modifier.

trackingGPUOfflineLowPtQuads = cms.Modifier()
