#!/bin/bash


eval `scramv1 runtime -sh`
cmsRun oldCudaTest_cfg.py > output_gpu.txt
#cmsRun oldCpuTest_cfg.py > output_cpu.txt

cp output_gpu.txt /nfshome0/gpizzati
cp output_cpu.txt /nfshome0/gpizzati
