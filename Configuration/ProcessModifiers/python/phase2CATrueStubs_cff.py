import FWCore.ParameterSet.Config as cms

# Overlay modifier: use with phase2CAStubs to swap in TrueStubProducer
# for upper-bound performance studies with truth-matched stubs.
# Usage: --procModifiers phase2CAStubs,phase2CATrueStubs
phase2CATrueStubs = cms.Modifier()
