#!/bin/bash
eval `scramv1 runtime -sh`
#cmsenv
nvprof -o profile.nvprof -f -- cmsRun cudaTest_cfg.py
nvprof --csv --log-file profiler_output.txt cmsRun cudaTest_cfg.py
