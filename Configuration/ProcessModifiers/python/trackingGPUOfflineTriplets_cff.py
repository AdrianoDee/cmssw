import FWCore.ParameterSet.Config as cms

# This modifier adds the Alpaka CA sub-iteration(s) for the high-pT and the low-pT triplets to the seeds of
# the initialStep, see RecoTracker/IterativeTracking/python/CATracksIterations_cff.py
# It has to be used together with the trackingGPUOffline modifier.

trackingGPUOfflineTriplets = cms.Modifier()
