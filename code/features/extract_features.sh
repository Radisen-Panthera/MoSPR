#!/usr/bin/env bash
# Patching follows the tissue-detection + boundary-clustered tiling pipeline in
# code/features/tile_extraction/ (vendored from Sungmin Lee's
# Pathology-WSI-Tile-Sampling-System; see that directory's README for credit/citation
# and for exactly what --min_tiles / --is_normalized do), not CLAM.
set -euo pipefail

SLIDES=""; OUT=""; ENCODER="conch"; PATCH=256; LEVEL=0; BATCH=256; WORKERS=8; DEVICE="cuda"
MIN_TILES=5; NORMALIZE=True
TPM=""; COUNTS=""; GENES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --slides)     SLIDES=$2; shift 2 ;;
    --out)        OUT=$2; shift 2 ;;
    --encoder)    ENCODER=$2; shift 2 ;;
    --patch)      PATCH=$2; shift 2 ;;
    --level)      LEVEL=$2; shift 2 ;;
    --batch)      BATCH=$2; shift 2 ;;
    --workers)    WORKERS=$2; shift 2 ;;
    --device)     DEVICE=$2; shift 2 ;;
    --min_tiles)  MIN_TILES=$2; shift 2 ;;
    --normalize)  NORMALIZE=$2; shift 2 ;;
    --tpm)        TPM=$2; shift 2 ;;
    --counts)     COUNTS=$2; shift 2 ;;
    --genes)      GENES=$2; shift 2 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done
[ -n "$SLIDES" ] && [ -n "$OUT" ] || { echo "usage: $0 --slides DIR --out DIR [--encoder conch|exaone|uni|gigapath] [--tpm CSV --genes TXT [--counts CSV]]"; exit 1; }
PY=${PY:-python}
mkdir -p "$OUT/patches" "$OUT/sample_pair_feature_${ENCODER}"

echo "[1/3] tissue segmentation and patching (${PATCH}px, min_tiles=${MIN_TILES}, normalize=${NORMALIZE})"
$PY code/features/tile_extraction/tile_processing.py \
    --root "$SLIDES" --output_dir "$OUT/patches" --endswith svs \
    --tile_size "$PATCH" --min_tiles "$MIN_TILES" --is_normalized "$NORMALIZE" \
    --save_thumb True

echo "[2/3] ${ENCODER} features"
$PY code/features/encode_patches.py \
    --patches "$OUT/patches" --slides "$SLIDES" \
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
