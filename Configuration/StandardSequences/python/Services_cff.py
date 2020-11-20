import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

# random numbers initialization service
from IOMC.RandomEngine.IOMC_cff import *

# DQM store service
from DQMServices.Core.DQMStore_cfi import *

# load CUDA services when the "gpu" modifier is enabled
def _addCUDAServices(process):
     process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")

from Configuration.ProcessModifiers.gpu_cff import gpu
from Configuration.ProcessModifiers.gpuTracks_cff import gpuTracks
modifyConfigurationStandardSequencesServicesAddCUDAServices_ = gpu.makeProcessModifier(_addCUDAServices)
modifyConfigurationStandardSequencesServicesAddCUDAServices_ = gpuTracks.makeProcessModifier(_addCUDAServices)
