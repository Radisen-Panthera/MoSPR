#!/usr/bin/env bash
set -euo pipefail

SLIDES=""; OUT=""; ENCODER="conch"; PATCH=256; LEVEL=0; BATCH=256; WORKERS=8; DEVICE="cuda"
TPM=""; COUNTS=""; GENES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --slides)  SLIDES=$2; shift 2 ;;
    --out)     OUT=$2; shift 2 ;;
    --encoder) ENCODER=$2; shift 2 ;;
    --patch)   PATCH=$2; shift 2 ;;
    --level)   LEVEL=$2; shift 2 ;;
    --batch)   BATCH=$2; shift 2 ;;
    --workers) WORKERS=$2; shift 2 ;;
    --device)  DEVICE=$2; shift 2 ;;
    --tpm)     TPM=$2; shift 2 ;;
    --counts)  COUNTS=$2; shift 2 ;;
    --genes)   GENES=$2; shift 2 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done
[ -n "$SLIDES" ] && [ -n "$OUT" ] || { echo "usage: $0 --slides DIR --out DIR [--encoder conch|exaone|uni|gigapath] [--tpm CSV --genes TXT [--counts CSV]]"; exit 1; }
PY=${PY:-python}
mkdir -p "$OUT/patches" "$OUT/sample_pair_feature_${ENCODER}"

echo "[1/3] tissue segmentation and patching (${PATCH}px, level ${LEVEL})"
$PY "${CLAM_REPO:-external/CLAM}/create_patches_fp.py" \
    --source "$SLIDES" --save_dir "$OUT/patches" \
    --patch_size "$PATCH" --step_size "$PATCH" --patch_level "$LEVEL" \
    --seg --patch --stitch

echo "[2/3] ${ENCODER} features"
$PY code/features/encode_patches.py \
    --patches "$OUT/patches/patches" --slides "$SLIDES" \
    --out "$OUT/sample_pair_feature_${ENCODER}" --encoder "$ENCODER" \
    --patch_size "$PATCH" --level "$LEVEL" \
    --batch "$BATCH" --workers "$WORKERS" --device "$DEVICE"

if [ -n "$TPM" ]; then
  echo "[3/3] attaching matched bulk expression"
  $PY code/features/attach_expression.py \
      --features "$OUT/sample_pair_feature_${ENCODER}" --tpm "$TPM" --genes "$GENES" \
      ${COUNTS:+--counts "$COUNTS"}
else
  echo "[3/3] skipped (no --tpm); attach bulk expression before training"
fi

echo "done: $OUT/sample_pair_feature_${ENCODER}"
