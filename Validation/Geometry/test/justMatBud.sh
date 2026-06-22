geometry=ExtendedRun4D121
era=Phase2C22I13M9

. runMaterialDumpFunctions

for t in Phase2PixelBarrel Phase2OTBarrel Phase2PixelEndcap Phase2OTForward; do
  if [ ! -e matbdg_${t}.root ]; then
    python3 runP_Tracker.py geom=${geometry} label=$t >& /dev/null &
  fi
done

waitPendingJobs

# Always run the comparison at this stage, since you are guaranteed that all the ingredients are there

for t in BeamPipe Tracker TrackerSumPhaseII Phase2PixelBarrel Phase2OTBarrel Phase2PixelEndcap Phase2OTForward; do
  python3 MaterialBudget.py -s -d ${t} -g ${geometry}
  if [ $? -ne 0 ]; then
    echo "Error while producing simulation material for ${t}, aborting"
    exit 1
  fi
done

python MaterialBudget.py -c -r DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root -l PhaseIIDetector > MaterialBudget_Simul_vs_Reco_PhaseII.log 2>&1


