import FWCore.ParameterSet.Config as cms

# Alpaka ESProducer - framework automatically selects backend (CPU serial, CUDA, ROCm)
# based on runtime configuration
stackedModuleGeometryESProducer = cms.ESProducer('StackedModuleGeometryESProducer@alpaka',
    minPt = cms.double(2.0),           # Minimum pT threshold for maxBend calculation (GeV)
    magneticField = cms.double(3.8),   # Magnetic field strength (Tesla)
    appendToDataLabel = cms.string(''),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')  # Empty string = use default backend
    )
)
