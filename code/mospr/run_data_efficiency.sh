#!/usr/bin/env bash
set -uo pipefail
R=${MOSPR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${PY:-python}
RES=${MOSPR_RESULTS_ROOT:-$R/results/run}
COHORT=${COHORT:-BRCA}
FRACS=${FRACS:-"10 25 50 75"}
PAR=${PAR:-4}                     # concurrent cache builds
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

export MOSPR_DATASET_ROOT=${MOSPR_DATASET_ROOT:-$R/data}
export MOSPR_TAU=0
export OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
cd "$R/code/mospr" || exit 1

log "===== 1) build a cache per fraction ====="
for fr in $FRACS; do
  for f in 0 1 2 3; do
    out=$RES/micro_cache_${COHORT}_fpsf${fr}_p512_fold${f}.npz
    [ -f "$out" ] && { log "· f${fr} fold${f} already present"; continue; }
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 10; done
    log "-> cache f${fr} fold${f}"
    MOSPR_SPLIT_FILE=split_patient_f${fr}.pkl \
      "$PY" build_microstate_cache.py --cohort "$COHORT" --variant paper --pca 512 \
      --suffix "_${COHORT}_fpsf${fr}_p512" --folds "$f" \
      > "$RES/_deff_lf_cache_${COHORT}_f${fr}_${f}.log" 2>&1 &
  done
done
wait; log "caches done"

log "===== 2) refit MoSPR per fraction ====="
for fr in $FRACS; do
  frac=$(awk "BEGIN{printf \"%.2f\", $fr/100}")
  log "▶ MoSPR frac=$frac"
  "$PY" data_efficiency_mospr.py --cohort "$COHORT" \
    --cache_tag "fpsf${fr}_p512" --out_tag "lf_f${fr}" \
    --folds 0 1 2 3 --fractions "$frac" \
    > "$RES/_deff_lf_mospr_${COHORT}_f${fr}.log" 2>&1
done
log "refits done"

log "===== 3) merge (frac=1.0 reuses the existing rows) ====="
"$PY" - "$COHORT" "$FRACS" <<'EOF'
import sys, pandas as pd, pathlib
RES = pathlib.Path(str(_p.RESULTS))
cohort, fracs = sys.argv[1], sys.argv[2].split()
old = pd.read_csv(RES / f"deff_mospr_{cohort}_fpsplit_tv.csv")
parts = [pd.read_csv(RES / f"deff_mospr_{cohort}_lf_f{fr}.csv") for fr in fracs]
parts.append(old[old.frac >= 1.0])          # the full-data point is leak-free by definition
df = pd.concat(parts).sort_values(["fold", "frac"]).reset_index(drop=True)
out = RES / f"deff_mospr_{cohort}_fpsplit_tv_leakfree.csv"
df.to_csv(out, index=False)
cmp = (df.groupby("frac")[["scc", "hall_scc"]].mean()
       .join(old.groupby("frac")[["scc", "hall_scc"]].mean(), rsuffix="_old"))
cmp["d_scc"] = cmp.scc - cmp.scc_old
cmp["d_hall"] = cmp.hall_scc - cmp.hall_scc_old
print(f"written: {out}\n")
print(cmp.round(4).to_string())
EOF
log "===== end ====="
