#!/usr/bin/env bash
set -uo pipefail
R=${MOSPR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${PY:-python}
RES=${MOSPR_RESULTS_ROOT:-$R/results/run}
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
n_cache(){ ls "$RES"/micro_cache_*_fpsplit_p512_fold*.npz 2>/dev/null | wc -l; }

log "MoSPR first: waiting for the filtered caches ($(n_cache)/12)"
while [ "$(n_cache)" -lt 12 ]; do
  pgrep -u "$(id -u)" -f "build_microstate_cache.py" >/dev/null 2>&1 || {
    log "no cache-building process ($(n_cache)/12) - waiting 10 more minutes"; sleep 600
    [ "$(n_cache)" -lt 12 ] && { log "aborting"; exit 1; }; }
  sleep 120
done
log "caches 12/12"

export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MOSPR_TAU=0
export MOSPR_DATASET_ROOT=${MOSPR_DATASET_ROOT:-$R/data}

log "MoSPR stage 1 (fit on train, select q/lambda on val)"
for c in BRCA KIRC LUAD; do
  setsid "$PY" "$R/code/mospr/train_mospr.py" --cohort "$c" \
    --cache_tag fpsplit_p512 --out_tag "${c}_fpsplit" \
    > "$RES/_mospr_fps_${c}.log" 2>&1 &
done
wait; log "stage 1 done"

log "MoSPR final (refit on train+val, stage-1 hyperparameters fixed)"
for c in BRCA KIRC LUAD; do
  setsid "$PY" "$R/code/mospr/train_mospr.py" --cohort "$c" \
    --cache_tag fpsplit_p512 --fit_on trainval \
    --hp_from "$RES/cohort_${c}_fpsplit_ours.csv" --out_tag "${c}_fpsplit_tv" \
    > "$RES/_mospr_fpstv_${c}.log" 2>&1 &
done
wait; log "MoSPR done"

log "ablation (filtered patches + patient split)"
for c in BRCA KIRC LUAD; do
  setsid "$PY" "$R/code/mospr/ablation.py" --cohort "$c" \
    --cache_tag fpsplit_p512 > "$RES/_abl2_fps_${c}.log" 2>&1 &
done
wait; log "===== all MoSPR runs done ====="
