#!/bin/bash
# =============================================================================
# retrain_merged.sh -- retrain the final high-purity selector of the MERGED
# pixel-track collection (hltPhase2PixelTrackHighPuritySelectorMerged, which
# scores the output of hltPhase2PixelTracksSoAMerger).
# =============================================================================
# The merged collection is produced by both iterations together, so this comes
# LAST: retrain the prompt iteration, then the displaced one, then this
# selector, with the datasets produced under the models already deployed and
# built.
#
#   step      model                        deployed as                            action
#   --------  ---------------------------  -------------------------------------  ---------
#   hp        final high-purity selector   data/.../merged_tree42_<tag>_<date>.bin  file swap
#
# USAGE
#   ./retrain_merged.sh <step> [options]
#
#   steps
#     dump      produce the merged training dataset: the tight dataset plus the TrkMerged*
#               tables over hltPhase2PixelTracksSoAMerger, with the pixel-cluster columns
#               and both iterations' combinatorics buffers enlarged
#     hp        train + stage the high-purity model   [file swap, no rebuild]
#               the 42-feature forest the chain runs, with the recipe of the deployed file
#
#   options
#     --work DIR        where datasets, models and logs are written        (required)
#     --input LIST      event files to reconstruct when a dataset has to be produced:
#                       a comma-separated list, a directory, or a text file of paths.
#                       Per-sample inputs: --input name=files,name2=files2, or one
#                       --input per --sample name, paired in order.
#     --dataset FILE    use an already-produced dataset instead (repeatable / comma list)
#     --sample NAMES    dataset label(s), several in one quoted string; the files are
#                       trackNano_<sample>_merged.root in --work, and the training step
#                       trains on their union             (default "ttbarPU displacedPU qcdPU")
#     --events N        events per dataset; applies to the dump it is on
#     --threads N       reconstruction threads for the dumps                    (default 8)
#     --chassis MODULE  base reconstruction config to dump on top of; by default one is
#                       generated from --input with make_base_config.sh. It must run both
#                       iterations and the merger.
#     --deploy          (hp) copy the staged model into the data directory
#     --dry-run         print the commands without running them
#
# EXAMPLES
#   ./retrain_merged.sh dump --work /w --sample "ttbarPU displacedPU qcdPU" \
#       --input ttbarPU=/data/ttbar,displacedPU=/data/susy,qcdPU=/data/qcd --events 1000
#   ./retrain_merged.sh hp   --work /w --sample "ttbarPU displacedPU qcdPU"
#   ./retrain_merged.sh hp   --work /w --dataset /w/trackNano_ttbarPU_merged.root --deploy
#
# ENVIRONMENT (tuning; the defaults are the ones in use)
#   hp step: WP_RULE (uniform | global | profile, see train_merged_forest.py --wp-rule),
#            RECALL (0.995), NTREES (900), FAKE_REGION_WEIGHT ("abseta:1.2:1.8:1.5,absdxy:1:inf:1.5",
#            the training-loss up-weighting of fakes in the |eta| band and at large |dxyBS|),
#            TAG (wp, the tag in the file name), XGB_THREADS (16), MODEL_NAME (the date in the
#            file name, today), TRAIN_EXTRA (further train_merged_forest.py options)
#   dump:    RT_PROMPT_CAPS_FACTOR, RT_DISP_CAPS_FACTOR (4: buffer enlargement of the two arms)
#   everywhere: MEM_GUARD_GB (300)
#
# WORKING POINT. The merged selector uses two thresholds joined by a ramp in |dxyBS|:
# scoreThresholdLowDxy at zero impact parameter, scoreThreshold from dxyRampKnee outwards. The
# trainer prints one offline starting point; both values are chosen by re-scanning the full
# reconstruction and the track validation over the displaced region.
# =============================================================================
set -euo pipefail
# shellcheck source=retrain_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/retrain_common.sh"

rt_usage() { sed -n '/^# USAGE/,/^# ====/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//;$d'; }

rt_parse_args "$@"
STEP=$(rt_canonical_step "${RT_STEP:-}")
case "${STEP:-}" in
  dump|hp) ;;
  "") rt_die "give a step: dump | hp  (see --help)" ;;
  *)  rt_die "unknown step '$STEP' (dump | hp)" ;;
esac

BANK=merged
SAMPLES=${RT_SAMPLE:-${SAMPLES:-"ttbarPU displacedPU qcdPU"}}
HP_CFI=$(rt_hp_cfi $BANK)
MODEL_TAG=${MODEL_NAME:-$(date +%Y%m%d)}
# The seven pixel-cluster columns the merged selector reads after the 31 + 4 provenance
# columns (PixelTrackFeaturesSoA columns 36-42), in that order.
CLUSTER_COLS=minCharge,meanCharge,minChargeNorm,maxSizeY,meanSizeY,maxSizeX,nLowCharge
ABI_ORDER=nAttached,nOTExtra,iteration,ndof

rt_require_work

# ---- dataset locations ------------------------------------------------------
merged_nano() { local s; for s in $SAMPLES; do echo "$RT_WORK/trackNano_${s}_merged.root"; done; }

# Datasets given with --dataset win; otherwise use the standard names in the
# working directory and produce the ones that are not there yet. The result is
# left in DATASETS.
DATASETS=()
resolve_datasets() {
  local f s i
  if [ ${#RT_DATASETS[@]} -gt 0 ]; then DATASETS=("${RT_DATASETS[@]}"); return 0; fi
  mapfile -t DATASETS < <(merged_nano)
  local missing=0
  for f in "${DATASETS[@]}"; do [ -f "$f" ] || missing=1; done
  if [ "$missing" -eq 1 ]; then
    rt_can_produce ||
      rt_die "no dataset: pass --dataset <file>, or --input <event files> (or --chassis <module>, which carries its own) so one can be produced"
    rt_resolve_chassis
    i=0
    for s in $SAMPLES; do
      [ -f "${DATASETS[$i]}" ] || rt_dump_track_nano merged tight "$s" "${DATASETS[$i]}" "$i"
      i=$((i + 1))
    done
  fi
  return 0
}

# ---- steps ------------------------------------------------------------------
step_dump() {
  [ -z "${RT_FOR:-}" ] || [ "$RT_FOR" = hp ] || [ "$RT_FOR" = merged ] ||
    rt_die "this script produces only the merged dataset (--for is not needed)"
  rt_require_no_cmsrun
  rt_cmsenv
  rt_resolve_chassis
  local s i
  i=0; for s in $SAMPLES; do rt_dump_track_nano merged tight "$s" "$RT_WORK/trackNano_${s}_merged.root" "$i"; i=$((i + 1)); done
  rt_report "Merged-collection dataset ready (quality 'tight', TrkMerged* over hltPhase2PixelTracksSoAMerger, pixel-cluster columns on)"
  rt_r_produced "$(merged_nano | tr '\n' ' ')"
  rt_r_next "./retrain_merged.sh hp --work $RT_WORK"
}

# The merged selector is a 42-feature forest (PixelTrackForestHighPuritySelector,
# useHitFeatures=True, PixelTrackFeaturesSoA columns 1-42: the 31 per-track
# features, the 4 provenance columns in the selector's order, the 7 pixel-cluster
# columns). The hp step trains exactly that, with the recipe of the deployed file,
# and stages it under the trainer's own name merged_tree42_<tag>_<date>.bin.
step_hp() {
  rt_require_no_cmsrun
  rt_cmsenv
  resolve_datasets; local -a nanos=("${DATASETS[@]}")

  local cache="$RT_WORK/merged43_cache.npz" outdir="$RT_WORK/merged_forest" tag="${TAG:-wp}"
  local guard; guard=$(rt_start_guard "$RT_WORK/retrain_merged_ram.log" "merged hp")
  trap "kill $guard 2>/dev/null || true" EXIT
  local deployed hpmod thr thr_lo
  deployed=$(rt_deployed_model "$HP_CFI")
  hpmod=$(rt_hp_module "$HP_CFI")
  thr=$(rt_cfi_double "$HP_CFI" scoreThreshold "$hpmod")
  thr_lo=$(rt_cfi_double "$HP_CFI" scoreThresholdLowDxy "$hpmod")

  echo ">>> [1/3] build the merged cache (31 + 4 provenance + 8 pixel-cluster columns) -> $cache"
  rt_run python3 "$RT_MODELS/nano_loader.py" cache-merged "$cache" "${nanos[@]}"

  echo ">>> [2/3] train the forest -> $outdir"
  # A reference profile (WP_RULE=profile) measures the deployed model at one threshold; with the
  # ramp enabled that is its near-beam-line value, the tighter of the two.
  local ref_thr="$thr"
  if [ -n "$thr_lo" ] && python3 -c "import sys; sys.exit(0 if float('$thr_lo') >= 0 else 1)"; then ref_thr="$thr_lo"; fi
  rt_wp_args merged "$cache" "$RT_DATA/${deployed:-none}" "$ref_thr" "$RT_WORK/merged_reference_profile.json" \
    --feats 35 --abi-order "$ABI_ORDER" --extra-cols "$CLUSTER_COLS"
  # The merged recipe: 42 columns in the selector's order; duplicates count as negatives in
  # the loss and are left out of the recall axis; fakes in the |eta| ~ 1.5 band and at large
  # |dxyBS| weigh 1.5x in the loss; up to 900 trees; full-precision values in the exported file.
  local -a train=(--cache "$cache" --out "$outdir"
                  --feats 35 --abi-order "$ABI_ORDER" --extra-cols "$CLUSTER_COLS"
                  --label mtv --recall "${RECALL:-0.995}" --arm merged
                  --recall-axis A_nodup --dup-as-fake 1.0
                  --fake-region-weight "${FAKE_REGION_WEIGHT:-abseta:1.2:1.8:1.5,absdxy:1:inf:1.5}"
                  --ntrees "${NTREES:-900}" --no-fp16 --seed 0
                  --tag "$tag" --date "$MODEL_TAG" --threads "${XGB_THREADS:-16}"
                  "${RT_WP_ARGS[@]}")
  # shellcheck disable=SC2206
  [ -n "${TRAIN_EXTRA:-}" ] && train+=(${TRAIN_EXTRA})
  rt_run python3 "$RT_TRAINER" "${train[@]}"

  echo ">>> [3/3] stage the compact model"
  local out; out=$(rt_forest_bin "$outdir" merged 42 "$tag" "$MODEL_TAG")
  [ "$RT_DEPLOY" -eq 1 ] && rt_run cp "$out" "$RT_DATA/"

  rt_report "Merged high-purity selector retrained"
  rt_r_produced "$out" \
                "training summary: $outdir/result.json   threshold map: $outdir/thrmap.txt" \
                "cache: $cache"
  rt_r_forest_deploy "$out" "$deployed"
  rt_r_update "$HP_CFI" \
    "  model                = cms.FileInPath('RecoTracker/FinalTrackSelectors/data/" \
    "                           PixelTrackTorchHighPuritySelector/$(basename "$out")')" \
    "  scoreThreshold       -- displaced arm, applied above the ramp knee; ${thr:-?} today" \
    "  scoreThresholdLowDxy -- near-beam-line arm; ${thr_lo:-?} today" \
    "  dxyRampKnee          -- where the two meet, in cm; $(rt_cfi_double "$HP_CFI" dxyRampKnee "$hpmod") today" \
    "  the trainer's starting point: $(rt_forest_wp "$outdir")" \
    "  All three belong together: a retrained model changes the score" \
    "  distribution, so re-scan them as a pair over the displaced region rather" \
    "  than copying the number the trainer printed."
}

case "$STEP" in
  dump) step_dump ;;
  hp)   step_hp ;;
  *) rt_die "unknown step '$STEP' (dump | hp)" ;;
esac
