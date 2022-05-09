#!/bin/bash


eval `scramv1 runtime -sh`

cmsRun vertexTest.py
#cmsRun vertexTest.py gpu=False

harvestTrackValidationPlots.py test_dqm_gpu.root -o gpu.root
#harvestTrackValidationPlots.py test_dqm_cpu.root -o cpu.root

rm -r plots/
rm -r ~/plots/

makeTrackValidationPlots.py gpu.root cpu.root --png --extended

cp -r plots/ /nfshome0/gpizzati/

