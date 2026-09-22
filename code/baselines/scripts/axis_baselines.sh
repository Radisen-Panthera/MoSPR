#!/usr/bin/env bash
set -uo pipefail
R=${MOSPR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
REPO=${MOSPR_CPNN_REPO:-$R/external/CPNN}
DATA=${MOSPR_DATASET_ROOT:-$R/data}; DTAG=""
if [ "${FILTERED:-0}" = 1 ]; then DATA=${MOSPR_DATASET_ROOT:-$R/data}; DTAG=f; fi
COND=${CONDITION:-trainval}
case "$COND" in
  trainval)  LOG=${MOSPR_LOGS:-$R/results/run/logs}/seeded_trainval;   SFX=_tv;    SPLIT=split.pkl;         TV=1
             STAGE1="" ;;
  psplit)    LOG=${MOSPR_LOGS:-$R/results/run/logs}/seeded_psplit;     SFX=_ps;    SPLIT=split_patient.pkl; TV=0
             STAGE1="" ;;
  psplit_tv) LOG=${MOSPR_LOGS:-$R/results/run/logs}/seeded_psplit_tv;  SFX=_pstv;  SPLIT=split_patient.pkl; TV=1
             STAGE1=_ps ;;
  dsplit)    LOG=${MOSPR_LOGS:-$R/results/run/logs}/seeded_dsplit;     SFX=_ds;    SPLIT=split_dup.pkl;     TV=0
             STAGE1="" ;;
  dsplit_tv) LOG=${MOSPR_LOGS:-$R/results/run/logs}/seeded_dsplit_tv;  SFX=_dstv;  SPLIT=split_dup.pkl;     TV=1
             STAGE1=_ds ;;
  *) echo "unknown CONDITION=$COND"; exit 1 ;;
esac
case "$COND" in
  psplit_tv) BEST=${MOSPR_RESULTS_ROOT:-$R/results/run}/best_epochs_psplit.csv ;;
  *)         BEST=${MOSPR_RESULTS_ROOT:-$R/results/run}/best_epochs.csv ;;
esac
STD=${PY:-python}
MAM=${PY_MAMBA:-${PY:-python}}
SEED=${CPNN_SEED:-2021}
MAXJOBS=${MAXJOBS:-12}
NEED=${NEED:-25000}
mkdir -p "$LOG"; cd "$REPO"
SFX="${SFX}${DTAG}"
[ -n "$STAGE1" ] && STAGE1="${STAGE1}${DTAG}"
LOG="${LOG}${DTAG}"; mkdir -p "$LOG"
export CPNN_DATA_DIR=$DATA
export CPNN_TRAINVAL=$TV CPNN_RUN_SUFFIX=$SFX CPNN_SPLIT_FILE=$SPLIT
export CPNN_STAGE1_SUFFIX=${STAGE1:-}
export CPNN_FT_LR=${CPNN_FT_LR:-1e-4} CPNN_FT_EPOCHS=${CPNN_FT_EPOCHS:-8}
case "$COND" in
  psplit|psplit_tv) export CPNN_C2L_PREFIX="${DTAG}ps" ;;
  dsplit|dsplit_tv) export CPNN_C2L_PREFIX="dup" ;;
  *)                unset CPNN_C2L_PREFIX ;;
esac
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
declare -A OUTDIM=([BRCA]=14042 [KIRC]=14295 [LUAD]=14514)

JOBS=(
  "cpnn|ProtoSum|DeconvExp|1reg_mse_reg_1e3|std"
  "abmil|AbMIL|||std"              "abmil_max|AbMIL||max|std"
  "abmil_mean|AbMIL||mean|std"     "ilra|ILRA|||std"
  "s4model|S4Model||stop_sampling|std"
  "mambamil|MambaMILvanira||stop_sampling|mamba"
  "srmamba|SRMambaMIL||stop_sampling|mamba"
  "mamba2d|MambaMIL_2D|Mamba2DTrainer|stop_sampling|mamba"
  "abreg|AbRegMIL|||std"           "he2rna|HE2RNA|ComparisonTrainer||std"
  "mosby|SumExpModel||MOSBY|std"   "sequoia_vis|SEQUOIA_VIS|ComparisonTrainer||std"
  "trnasformer|tRNAsformer|ComparisonTrainer||std"
)
if [ "${EXTRA_MODELS:-0}" = 1 ]; then
  JOBS+=("clam_mb|CLAM_MB|||std" "dsmil|DSMIL|||std")
fi
if [ "${ILSE:-0}" = 1 ]; then
  JOBS+=("ilse_mean|AbMIL||ilse_mean|std" "ilse_max|AbMIL||ilse_max|std")
fi
if [ -n "${ONLY:-}" ]; then
  _keep=(); for j in "${JOBS[@]}"; do
    case " $ONLY " in *" ${j%%|*} "*) _keep+=("$j") ;; esac; done
  JOBS=("${_keep[@]}")
fi

best_epoch(){
  awk -F, -v c="$1" -v t="$2" -v f="$3" \
    'NR>1 && $1==c && $2==t && $3==f && $4==1 && $5!="" {print $5; exit}' "$BEST"
}
declare -A RUNNAME=(
  [cpnn]="ProtoSum_1reg_mse_reg_1e3%DeconvExptsfine"
  [abmil]="AbMIL%ts"            [abmil_max]="AbMIL_max%ts"
  [abmil_mean]="AbMIL_mean%ts"  [ilra]="ILRA%ts"
  [s4model]="S4Model_stop_sampling%ts"
  [mambamil]="MambaMILvanira_stop_sampling%ts"
  [srmamba]="SRMambaMIL_stop_sampling%ts"
  [mamba2d]="MambaMIL_2D_stop_sampling%Mamba2DTrainerts"
  [abreg]="AbRegMIL%ts"         [he2rna]="HE2RNA%ComparisonTrainerts"
  [mosby]="SumExpModel_MOSBY%ts"
  [sequoia_vis]="SEQUOIA_VIS%ComparisonTrainerts"
  [trnasformer]="tRNAsformer%ComparisonTrainerts"
  [clam_mb]="CLAM_MB%ts"        [dsmil]="DSMIL%ts"
  [ilse_mean]="AbMIL_ilse_mean%ts" [ilse_max]="AbMIL_ilse_max%ts"
)
done_run(){
  local rn=${RUNNAME[$3]:-}
  [ -z "$rn" ] && return 1
  [ -s "$REPO/outputs/$2/$1-${rn/\%/$2}$SFX.txt" ]
}

running(){
  for pid in $(pgrep -u "$(id -u)" -f "main.py --method" 2>/dev/null); do
    [ "$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ')" = "$pid" ] || continue
    local _cl; _cl=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
    case "$_cl" in
      *"--method $1 "*"--dataset $2 "*"--version $4 --fold $3 "*) ;;
      *) continue ;;
    esac
    local _sfx
    _sfx=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
           | sed -n 's/^CPNN_RUN_SUFFIX=//p')
    [ "${_sfx:-}" = "$SFX" ] && return 0
  done
  return 1
}
count_jobs(){
  local n=0
  for pid in $(pgrep -u "$(id -u)" -f "main.py --method" 2>/dev/null); do
    [ "$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ')" = "$pid" ] && n=$((n+1))
  done
  echo "$n"
}
throttle(){ while :; do [ "$(count_jobs)" -lt "$MAXJOBS" ] && break; sleep 120; done; }
pick_gpu(){
  while :; do
    best=-1; bestutil=999
    for g in 0 1 2 3 4 5; do
      read -r fr ut <<< "$(nvidia-smi --query-gpu=memory.free,utilization.gpu \
                           --format=csv,noheader,nounits -i "$g" | tr -d ',')"
      [ "$fr" -ge "$NEED" ] || continue
      [ "$ut" -lt "$bestutil" ] && { bestutil=$ut; best=$g; }
    done
    [ "$best" -ge 0 ] && { echo "$best"; return; }
    sleep 180
  done
}

log "===== $COND (data=$DATA, split=$SPLIT, trainval=$TV, suffix=$SFX, jobs=$MAXJOBS) ====="
skipped=0
for c in ${COHORTS:-BRCA KIRC LUAD}; do
  for f in ${FOLDS:-0 1 2 3}; do
    for j in "${JOBS[@]}"; do
      IFS='|' read -r tag method trainer version env <<< "$j"
      out="$LOG/${c}-paper_${tag}_fold${f}.log"
      [ -s "$out" ] && grep -aq test_loss "$out" 2>/dev/null && continue
      done_run "${c}-paper" "$f" "$tag" && { log "· $c $tag f$f already finished (.txt)"; continue; }
      running "$method" "${c}-paper" "$f" "$version" && { log "· $c $tag f$f already running"; continue; }
      e=""
      if [ "$TV" = 1 ]; then
        rn=${RUNNAME[$tag]:-}; rn=${rn/\%/$f}
        ls "$REPO/ckpts/exps/$f/${c}-paper-${rn}${STAGE1}"/*epoch=*.ckpt >/dev/null 2>&1 || {
          log "· $c $tag f$f skipped - no stage-1 checkpoint"; skipped=$((skipped+1)); continue; }
      fi
      py=$STD; [ "$env" = mamba ] && py=$MAM
      throttle; gpu=$(pick_gpu)
      log "▶ $c $tag f$f${e:+ (E*=$e)} → GPU$gpu"
      env PYTHONHASHSEED=$SEED CPNN_SEED=$SEED CPNN_OUT_DIM=${OUTDIM[$c]} \
        CUDA_VISIBLE_DEVICES=$gpu \
        setsid "$py" main.py --method "$method" --dataset "${c}-paper" \
        --trainer "$trainer" --resolution "" --version "$version" \
        --fold "$f" --data_type ts > "$out" 2>&1 < /dev/null &
      sleep 20
    done
  done
done
log "all launched (skipped $skipped without E*), waiting"; wait; log "===== done ====="
