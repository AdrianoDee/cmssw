#!/bin/bash
# =============================================================================
# retrain_displaced.sh -- retrain the machine-learned models of the DISPLACED
# pixel track iteration (hltPhase2PixelTracksSoADisplacedWithStubs).
# =============================================================================
# Same steps and workers as retrain_prompt.sh. RETRAIN THE PROMPT ITERATION
# FIRST: the displaced iteration reconstructs the hits the prompt iteration did
# not use, so its input population changes whenever a prompt model changes.
#
#   step      model                        deployed as                     action
#   --------  ---------------------------  ------------------------------  --------------
#   triplet   per-triplet gate (in kernel) CATripletDNNWeights_displaced.h rebuild
#   gate      track DNN, loose->tight      CATrackDNNWeights_displaced.h   rebuild
#   hp        final high-purity selector   data/.../disp_tree31_<tag>_<date>.bin   file swap
#
# USAGE
#   ./retrain_displaced.sh <step> [options]
#
#   steps
#     dump      produce a training dataset            (--for triplet|gate|hp)
#     triplet   train + bake the per-triplet gate     [rebuild afterwards]
#     gate      train + bake the track DNN            [rebuild afterwards]
#     hp        train + stage the high-purity model   [file swap, no rebuild]
#               the 31-feature forest the chain runs, with the recipe of the deployed file
#     all       triplet, then the instructions to continue
#
#   options
#     --work DIR        where datasets, models and logs are written        (required)
#     --input LIST      event files to reconstruct when a dataset has to be produced:
#                       a comma-separated list, a directory, or a text file of paths.
#                       Per-sample inputs: --input name=files,name2=files2, or one
#                       --input per --sample name, paired in order. With several
#                       samples and a single --input every sample is dumped from the
#                       same files.
#     --dataset FILE    use an already-produced dataset instead (repeatable / comma list)
#     --sample NAMES    dataset label(s), several in one quoted string; the files are
#                       trackNano_<sample>_<quality>.root / tripletNano_<sample>.root in
#                       --work, and a training step trains on their union
#                                                          (default "displacedPU ttbarPU")
#     --events N        events per dataset; applies to the dump it is on
#     --threshold X     bake this decision threshold; 'rule' = the value the trainer
#                       derives                     (default: the value the chain runs today)
#     --bake-recall R   (gate) bake the derived point at this displacement-weighted recall,
#                       0.99 or 0.995; implies --threshold rule (see retrain_track_dnn.sh)
#     --label L         (gate) truth-label definition, mtv or legacy (see retrain_track_dnn.sh)
#     --threads N       reconstruction threads for the dumps                    (default 8)
#     --device DEV      torch device for training                         (default cuda:0)
#     --chassis MODULE  base reconstruction config to dump on top of; by default one is
#                       generated from --input with make_base_config.sh
#     --deploy          (hp) copy the staged model into the data directory
#     --dry-run         print the commands without running them
#
# EXAMPLES
#   ./retrain_displaced.sh all  --work /w --input /data/susy --events 1500
#   ./retrain_displaced.sh gate --work /w --dataset /w/trackNano_displacedPU_loose.root
#   ./retrain_displaced.sh hp   --work /w --input /data/susy --deploy
#   # three samples from three inputs, one dump per stage (see README.md, "Inputs"):
#   ./retrain_displaced.sh dump --for gate --work /w --sample "ttbarPU displacedPU qcdPU" \
#       --input ttbarPU=/data/ttbar,displacedPU=/data/susy,qcdPU=/data/qcd --events 300
#
# ENVIRONMENT (tuning; the defaults are the ones in use)
#   hp step: WP_RULE (uniform | global | profile, see train_merged_forest.py --wp-rule),
#            RECALL (0.995), LABEL (legacy: the deployed file was trained against the narrow
#            efficiency-selected label; mtv = matched to any TrackingParticle, as the other
#            two selectors), MIN_DXY (0.5, the |dxyBS| cut of the training rows), DXY_ALPHA
#            (0, displacement re-weighting of the true tracks), TAG (wp, the tag in the file
#            name), XGB_THREADS (16), MODEL_NAME (the date in the file name, today),
#            TRAIN_EXTRA (further train_merged_forest.py options, e.g. "--ntrees 200")
#   triplet step: TRAIN_EV, TEST_SKIP, TEST_EV, VARIANT, THRESHOLD_RULE, SURVIVAL_FLOOR,
#                 REF_SURVIVAL, EPOCHS, WORKERS  -- see retrain_triplet_gate.sh
#   gate step:    NAME  -- see retrain_track_dnn.sh
#   everywhere:   MEM_GUARD_GB (300)
#
# The final selector of this iteration uses two thresholds (near-beamline and displaced, joined
# by a ramp). Both move together when the model is retrained; the hp step prints them.
# =============================================================================
set -euo pipefail
# shellcheck source=retrain_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/retrain_common.sh"

rt_usage() { sed -n '/^# USAGE/,/^# ====/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//;$d'; }

rt_parse_args "$@"
STEP=$(rt_canonical_step "${RT_STEP:-}")
case "${STEP:-}" in
  dump|triplet|gate|hp|all) ;;
  "") rt_die "give a step: dump | triplet | gate | hp | all  (see --help)" ;;
  *)  rt_die "unknown step '$STEP' (dump | triplet | gate | hp | all)" ;;
esac

BANK=displaced
SAMPLES=${RT_SAMPLE:-${SAMPLES:-"displacedPU ttbarPU"}}
CA_CFI=$(rt_ca_cfi $BANK)
HP_CFI=$(rt_hp_cfi $BANK)
MODEL_TAG=${MODEL_NAME:-$(date +%Y%m%d)}

rt_require_work

track_nano() { local q=$1 s; for s in $SAMPLES; do echo "$RT_WORK/trackNano_${s}_${q}.root"; done; }
triplet_nano() { local s; for s in $SAMPLES; do echo "$RT_WORK/tripletNano_${s}.root"; done; }

# Datasets given with --dataset win; otherwise use the standard names in the
# working directory and produce the ones that are not there yet. The result is
# left in DATASETS.
DATASETS=()
resolve_datasets() {  # $1 = kind (triplet|track), $2 = quality (track only)
  local kind=$1 quality=${2:-} f s i
  if [ ${#RT_DATASETS[@]} -gt 0 ]; then DATASETS=("${RT_DATASETS[@]}"); return 0; fi
  if [ "$kind" = triplet ]; then mapfile -t DATASETS < <(triplet_nano)
  else mapfile -t DATASETS < <(track_nano "$quality"); fi
  local missing=0
  for f in "${DATASETS[@]}"; do [ -f "$f" ] || missing=1; done
  if [ "$missing" -eq 1 ]; then
    # A chassis carries its own input file list, so it is enough on its own.
    rt_can_produce ||
      rt_die "no dataset: pass --dataset <file>, or --input <event files> (or --chassis <module>, which carries its own) so one can be produced"
    if [ "$kind" = triplet ]; then rt_require_dump_build; fi
    rt_resolve_chassis
    i=0
    for s in $SAMPLES; do
      if [ ! -f "${DATASETS[$i]}" ]; then
        if [ "$kind" = triplet ]; then rt_dump_triplet_nano $BANK "$s" "${DATASETS[$i]}" "$i"
        else rt_dump_track_nano $BANK "$quality" "$s" "${DATASETS[$i]}" "$i"; fi
      fi
      i=$((i + 1))
    done
  fi
  return 0
}

# ---- steps ------------------------------------------------------------------
step_dump() {
  local what=${RT_FOR:-}
  case "$what" in
    triplet|gate|hp) ;;
    "") rt_die "say which dataset to produce: --for triplet | gate | hp" ;;
    *)  rt_die "--for must be triplet, gate or hp (got '$what')" ;;
  esac
  rt_require_no_cmsrun
  rt_cmsenv
  rt_resolve_chassis
  local s i
  case "$what" in
    triplet)
      rt_require_dump_build
      i=0; for s in $SAMPLES; do rt_dump_triplet_nano $BANK "$s" "$RT_WORK/tripletNano_${s}.root" "$i"; i=$((i + 1)); done
      rt_report "Triplet-gate dataset ready"
      rt_r_produced "$(triplet_nano | tr '\n' ' ')"
      rt_r_next "./retrain_displaced.sh triplet --work $RT_WORK"
      ;;
    gate)
      i=0; for s in $SAMPLES; do rt_dump_track_nano $BANK loose "$s" "$RT_WORK/trackNano_${s}_loose.root" "$i"; i=$((i + 1)); done
      rt_report "Track-DNN dataset ready (quality 'loose', track DNN switched off at dump time)"
      rt_r_produced "$(track_nano loose | tr '\n' ' ')"
      rt_r_next "./retrain_displaced.sh gate --work $RT_WORK"
      ;;
    hp)
      i=0; for s in $SAMPLES; do rt_dump_track_nano $BANK tight "$s" "$RT_WORK/trackNano_${s}_tight.root" "$i"; i=$((i + 1)); done
      rt_report "High-purity dataset ready (quality 'tight', full deployed chain)"
      rt_r_produced "$(track_nano tight | tr '\n' ' ')"
      rt_r_next "./retrain_displaced.sh hp --work $RT_WORK"
      ;;
  esac
}

step_triplet() {
  rt_require_no_cmsrun
  rt_cmsenv
  resolve_datasets triplet; local -a nanos=("${DATASETS[@]}")
  local -a args=(--bank $BANK --work "$RT_WORK" --device "$RT_DEVICE")
  [ -n "$RT_THRESHOLD" ] && args+=(--threshold "$RT_THRESHOLD")
  [ "$RT_DRYRUN" -eq 1 ] && args+=(--dry-run)
  local n; for n in "${nanos[@]}"; do args+=(--dataset "$n"); done
  "$RT_MODELS/retrain_triplet_gate.sh" "${args[@]}"
}

step_gate() {
  rt_require_no_cmsrun
  rt_cmsenv
  resolve_datasets track loose; local -a nanos=("${DATASETS[@]}")
  local -a args=(--bank $BANK --work "$RT_WORK" --device "$RT_DEVICE")
  [ -n "$RT_THRESHOLD" ] && args+=(--threshold "$RT_THRESHOLD")
  [ -n "$RT_BAKE_RECALL" ] && args+=(--bake-recall "$RT_BAKE_RECALL")
  [ -n "$RT_LABEL" ] && args+=(--label "$RT_LABEL")
  [ "$RT_DRYRUN" -eq 1 ] && args+=(--dry-run)
  local n; for n in "${nanos[@]}"; do args+=(--dataset "$n"); done
  "$RT_MODELS/retrain_track_dnn.sh" "${args[@]}"
}

# The displaced selector is a 31-feature forest (PixelTrackForestHighPuritySelector,
# useHitFeatures=True, PixelTrackFeaturesSoA columns 1-31) trained on the tracks with
# |dxyBS| above MIN_DXY. The hp step trains exactly that, with the recipe of the
# deployed file: the per-arm cache in the selector's column order, the trainer's
# default recipe, 16-bit values in the exported file, staged under the trainer's
# own name disp_tree31_<tag>_<date>.bin.
step_hp() {
  rt_require_no_cmsrun
  rt_cmsenv
  resolve_datasets track tight; local -a nanos=("${DATASETS[@]}")

  local cache="$RT_WORK/disp31_cache.npz" outdir="$RT_WORK/disp_forest"
  local label="${LABEL:-legacy}" tag="${TAG:-wp}"
  local guard; guard=$(rt_start_guard "$RT_WORK/retrain_displaced_ram.log" "displaced hp")
  trap "kill $guard 2>/dev/null || true" EXIT
  local deployed hpmod thr
  deployed=$(rt_deployed_model "$HP_CFI")
  # Scope the readback to the producer the stub chain runs (see rt_hp_module).
  hpmod=$(rt_hp_module "$HP_CFI")
  thr=$(rt_cfi_double "$HP_CFI" scoreThreshold "$hpmod")

  echo ">>> [1/3] build the 31-feature cache (|dxyBS| > ${MIN_DXY:-0.5}) -> $cache"
  rt_run python3 "$RT_MODELS/nano_loader.py" cache-arm "$cache" "${nanos[@]}" \
    --prefix TrkDisp --min-dxy "${MIN_DXY:-0.5}" --label "$label"

  echo ">>> [2/3] train the forest -> $outdir"
  rt_wp_args disp "$cache" "$RT_DATA/${deployed:-none}" "$thr" "$RT_WORK/disp_reference_profile.json" \
    --label "$label"
  # The displaced recipe: the trainer's defaults (depth 12, up to 450 trees, two-pass tail
  # re-weight, early stopping on the working-point rule) and 16-bit values in the exported
  # file (no --no-fp16).
  local -a train=(--cache "$cache" --out "$outdir" --feats 31 --label "$label"
                  --recall "${RECALL:-0.995}" --arm disp --tag "$tag" --date "$MODEL_TAG"
                  --threads "${XGB_THREADS:-16}" "${RT_WP_ARGS[@]}")
  [ -n "${DXY_ALPHA:-}" ] && train+=(--dxy-alpha "$DXY_ALPHA")
  # shellcheck disable=SC2206
  [ -n "${TRAIN_EXTRA:-}" ] && train+=(${TRAIN_EXTRA})
  rt_run python3 "$RT_TRAINER" "${train[@]}"

  echo ">>> [3/3] stage the compact model"
  local out; out=$(rt_forest_bin "$outdir" disp 31 "$tag" "$MODEL_TAG")
  [ "$RT_DEPLOY" -eq 1 ] && rt_run cp "$out" "$RT_DATA/"

  rt_report "Displaced high-purity selector retrained"
  rt_r_produced "$out" \
                "training summary: $outdir/result.json   threshold map: $outdir/thrmap.txt" \
                "cache: $cache"
  rt_r_forest_deploy "$out" "$deployed"
  rt_r_update "$HP_CFI" \
    "  model                = cms.FileInPath('RecoTracker/FinalTrackSelectors/data/" \
    "                           PixelTrackTorchHighPuritySelector/$(basename "$out")')" \
    "  scoreThreshold       -- displaced arm, applied above the ramp knee; ${thr:-?} today" \
    "                          the trainer's starting point: $(rt_forest_wp "$outdir")" \
    "  scoreThresholdLowDxy -- near-beam-line arm; $(rt_cfi_double "$HP_CFI" scoreThresholdLowDxy "$hpmod") today" \
    "  dxyRampKnee          -- where the two meet, in cm; $(rt_cfi_double "$HP_CFI" dxyRampKnee "$hpmod") today" \
    "  All three belong together: a retrained model changes the score" \
    "  distribution, so re-scan them as a pair over the displaced region rather" \
    "  than copying the number the trainer printed."
}

step_all() {
  step_triplet
  echo
  echo "============================================================================"
  echo " The remaining steps need the new triplet gate compiled in and the datasets"
  echo " re-produced underneath it. Continue with:"
  echo "   scram b code-format && scram b -j"
  echo "   ./retrain_displaced.sh gate --work $RT_WORK --input <event files>"
  echo "   scram b code-format && scram b -j"
  echo "   ./retrain_displaced.sh hp   --work $RT_WORK --input <event files>"
  echo "============================================================================"
}

case "$STEP" in
  dump)    step_dump ;;
  triplet) step_triplet ;;
  gate)    step_gate ;;
  hp)      step_hp ;;
  all)     step_all ;;
  *) rt_die "unknown step '$STEP' (dump | triplet | gate | hp | all)" ;;
esac
