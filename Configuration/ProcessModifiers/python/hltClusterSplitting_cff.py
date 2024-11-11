import FWCore.ParameterSet.Config as cms

# This modifier enables
# - saving Z pixel vertices coordinates at HLT;
# - using those coordinates in input for the jet core cluster splitting;

hltClusterSplitting =  cms.Modifier()

