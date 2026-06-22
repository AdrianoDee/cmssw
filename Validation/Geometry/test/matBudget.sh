#!/bin/bash

set -x

# Check if CMSSW envs are setup
: ${CMSSW_BASE:?'You need to set CMSSW environemnt first.'}

# DEFAULTS

events=5000
geometry=ExtendedRun4D121
era=Phase2C22I13M9
# ARGUMENT PARSING

while getopts ":n:g:" opt; do
  case $opt in
    n)
      echo "Generating $OPTARG events" >&1
      events=${OPTARG}
      ;;
    g)
      echo "Using $OPTARG geometry" >&1
      geometry=${OPTARG}
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

. runMaterialDumpFunctions

#  cmsDriver.py TTbar_14TeV_TuneCP5_cfi  -s GEN,SIM -n 10 --conditions auto:phase2_realistic_T35 --beamspot DBrealisticHLLHC --datatier GEN-SIM --eventcontent FEVTDEBUG --geometry ExtendedRun4D121 --era Phase2C22I13M9 --relval 9000,100 --fileout file:step1.root  > step1_TTbar_14TeV+Run4D121.log  2>&1

# GEN-SIM goes first
if checkFile SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseII.root ; then
  cmsDriver.py SingleMuPt10_pythia8_cfi \
-s GEN,SIM \
--conditions auto:phase2_realistic_T35 \
-n ${events} \
--era ${era} \
--eventcontent FEVTDEBUG \
--datatier GEN-SIM \
--beamspot NoSmear \
--nThreads=8 \
--customise Validation/Geometry/customiseForDumpMaterialAnalyser_ForPhaseII.customiseForMaterialAnalyser_ForPhaseII \
--geometry ${geometry} \
--fileout file:SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseII.root \
--python_filename SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseII.py > SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseII.log 2>&1

  if [ $? -ne 0 ]; then
    echo "Error executing the GEN-SIM step, aborting."
    exit 1
  fi
fi

#  cmsDriver.py step2  -s DIGI:pdigi_valid,L1TrackTrigger,L1,L1P2GT,DIGI2RAW,HLT:@relvalRun4 --conditions auto:phase2_realistic_T35 --datatier GEN-SIM-DIGI-RAW -n 10 --eventcontent FEVTDEBUGHLT --geometry ExtendedRun4D121 --era Phase2C22I13M9 --filein  file:step1.root  --fileout file:step2.root  > step2_TTbar_14TeV+Run4D121.log  2>&1


# DIGI comes next
if checkFile SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseII.root ; then
  cmsDriver.py step2   \
-s DIGI:pdigi_valid,L1TrackTrigger,L1,DIGI2RAW,HLT:@fake2  \
--conditions auto:phase2_realistic_T35 \
-n -1  \
--era ${era}   \
--eventcontent FEVTDEBUGHLT \
--datatier GEN-SIM-DIGI-RAW  \
--nThreads=32 \
--geometry ${geometry}  \
--filein file:SingleMuPt10_pythia8_cfi_GEN_SIM_PhaseII.root  \
--fileout file:SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseII.root \
--python_filename SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseII.py > SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseII.log 2>&1

  if [ $? -ne 0 ]; then
    echo "Error executing the DIGI step, aborting."
    exit 1
  fi
fi

#  cmsDriver.py step3  -s RAW2DIGI,RECO,RECOSIM,PAT,VALIDATION:@phase2Validation+@miniAODValidation,DQM:@phase2+@miniAODDQM --conditions auto:phase2_realistic_T35 --datatier GEN-SIM-RECO,MINIAODSIM,DQMIO -n 10 --eventcontent FEVTDEBUGHLT,MINIAODSIM,DQM --geometry ExtendedRun4D121 --era Phase2C22I13M9 --filein  file:step2.root  --fileout file:step3.root  > step3_TTbar_14TeV+Run4D121.log  2>&1

# Reco and special customization
if checkFile SingleMuPt10_step3_RECO_DQM_PhaseII.root ; then
  cmsDriver.py step3 \
-s RAW2DIGI,RECO,RECOSIM,PAT,VALIDATION:@phase2Validation+@miniAODValidation,DQM:@phase2+@miniAODDQM \
--conditions auto:phase2_realistic_T35 \
-n -1  \
--era ${era}   \
--eventcontent FEVTDEBUGHLT,MINIAODSIM,DQM  \
--datatier  GEN-SIM-RECO,MINIAODSIM,DQMIO  \
--nThreads=32 \
--geometry ${geometry}  \
--filein file:SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT_PhaseII.root  \
--fileout file:SingleMuPt10_step3_RECO_DQM_PhaseII.root \
--python_filename SingleMuPt10_step2_RECO_DQM_PhaseII.py > SingleMuPt10_step3_RECO_DQM_PhaseII.log 2>&1

  if [ $? -ne 0 ]; then
    echo "Error executing the RECO step, aborting."
    exit 1
  fi
fi

# HARVESTING
if checkFile DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root ; then
  cmsDriver.py step4  \
-s HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM  \
--conditions auto:phase2_realistic \
-n -1   \
--era ${era}   \
--scenario pp  \
--filetype DQM  \
--geometry ${geometry}  \
--mc  \
--filein file:SingleMuPt10_step3_RECO_DQM_PhaseII_inDQM.root  \
--python_filename SingleMuPt10_step4_HARVESTING_PhaseII.py > SingleMuPt10_step4_HARVESTING_PhaseII.log 2>&1

  if [ $? -ne 0 ]; then
    echo "Error executing the HARVESTING step, aborting."
    exit 1
  fi
fi

# Neutrino Particle gun

if checkFile single_neutrino_random.root ; then
  cmsRun ../python/single_neutrino_cfg.py
  if [ $? -ne 0 ]; then
    echo "Error generating single neutrino gun, aborting."
    exit 1
  fi
  if [ ! -e Images ]; then
    mkdir Images
  fi
fi

# Make material map for each subdetector from simulation

for t in BeamPipe Tracker Phase2PixelBarrel Phase2OTBarrel Phase2PixelEndcap Phase2OTForward; do
  if [ ! -e matbdg_${t}.root ]; then
    python3 runP_Tracker.py geom=${geometry} label=$t >& /dev/null &
  fi
done

waitPendingJobs

# Always run the comparison at this stage, since you are guaranteed that all the ingredients are there

for t in BeamPipe Tracker TrackerSumPhaseII Phase2PixelBarrel Phase2OTBarrel Phase2PixelEndcap Phase2OTForward; do
  python3 MaterialBudget.py -s -d ${t}
  if [ $? -ne 0 ]; then
    echo "Error while producing simulation material for ${t}, aborting"
    exit 1
  fi
done

python MaterialBudget.py -c -r DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root -l PhaseIIDetector > MaterialBudget_Simul_vs_Reco_PhaseII.log 2>&1

