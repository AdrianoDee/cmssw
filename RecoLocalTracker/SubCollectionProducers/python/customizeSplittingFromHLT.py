import FWCore.ParameterSet.Config as cms

def customiseSplittingFromHLT(process):

    if not hasattr(process, 'hltTrimmedPixelVertices'):
        return process
    
    process.hltTrimmedPixelVertices.dumpCoordinates = cms.bool(True)

    return process
