#!/usr/bin/env bash
set -uo pipefail
R=${MOSPR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
PY=${PY:-python}
LOG=${MOSPR_LOGS:-$R/results/run/logs}/c2l_psplit; mkdir -p "$LOG"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
if [ "${FILTERED:-0}" = 1 ]; then
  export MOSPR_DATASET_ROOT=${MOSPR_DATASET_ROOT:-$R/data}
  DSDIR=${MOSPR_DATASET_ROOT:-$R/data}; PFX=fps
else
  DSDIR=${MOSPR_DATASET_ROOT:-$R/data}; PFX=ps
fi
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
pick_gpu(){
  while :; do
    best=-1; bestutil=999
    for g in 0 1 2 3 4 5; do
      read -r fr ut <<< "$(nvidia-smi --query-gpu=memory.free,utilization.gpu \
                           --format=csv,noheader,nounits -i "$g" | tr -d ',')"
      [ "$fr" -ge ${NEED:-8000} ] || continue
      [ "$ut" -lt "$bestutil" ] && { bestutil=$ut; best=$g; }
    done
    [ "$best" -ge 0 ] && { echo "$best"; return; }
    sleep 120
  done
}
MAXJOBS=${MAXJOBS:-6}
running_c2l(){   # $1=cohort $2=fold - is the same job already running?
  for pid in $(pgrep -u "$(id -u)" -f "cell2location_deconv.py" 2>/dev/null); do
    local _cl; _cl=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
    case "$_cl" in *"--cohort $1 "*"--fold $2"*) return 0 ;; esac
  done
  return 1
}
count_c2l(){ pgrep -u "$(id -u)" -f "cell2location_deconv.py" 2>/dev/null | wc -l; }
throttle_c2l(){ while [ "$(count_c2l)" -ge "$MAXJOBS" ]; do sleep 60; done; }

cd "$R/code/mospr"
for c in ${COHORTS:-BRCA KIRC LUAD}; do
  for f in 0 1 2 3; do
    out=$DSDIR/${c}-paper-digital_slide/${PFX}${f}/fine_parameter_dict.pkl
    [ -s "$out" ] && { log "· $c ps$f already present"; continue; }
    running_c2l "$c" "$f" && { log "· $c ps$f already running"; continue; }
    throttle_c2l
    gpu=$(pick_gpu)
    log "▶ $c fold$f → ${PFX}$f (GPU$gpu)"
    env MOSPR_SPLIT_FILE=split_patient.pkl MOSPR_C2L_OUTDIR=${PFX}${f} CUDA_VISIBLE_DEVICES=$gpu \
      setsid "$PY" cell2location_deconv.py --cohort "$c" --variant paper --stage deconv --fold "$f" \
      > "$LOG/${c}_ps${f}.log" 2>&1 < /dev/null &
    sleep 12
  done
done
log "all launched, waiting"; wait; log "===== deconvolution done ====="
