import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.ClusterSplitterProducer.go_cfi import *
from RecoLocalTracker.ClusterSplitterGPU.customizeForAlpakaSplitter_cfi import *

SplitterTask = cms.Task(process.trial, trialProducer)
LocalResult = cms.Sequence(SplitterTask)

