#!/usr/bin/env bash
# Patch extraction + foundation-model features for one cohort.
#
#   bash code/features/extract_features.sh --slides /path/to/svs --out data/BRCA --encoder conch
#
# Output, one .h5 per slide, is what the rest of the pipeline reads:
#   <out>/sample_pair_<encoder>/<slide_id>.h5
#       coord  (N, 2)  int64   top-left pixel of each patch at level 0
#       feat   (N, D)  float32 patch embedding, L2 layout as produced by the encoder
#       tpm    (G,)    float32 matched bulk expression, log1p(TPM / sum * 1e4)
#
# Patching follows CLAM (Lu et al., Nat Biomed Eng 2021): tissue is segmented on a
# downsampled thumbnail, then non-overlapping patches are taken inside the mask.
#
# Encoders (choose with --encoder; all are gated on Hugging Face and need `huggingface-cli login`):
#   conch     MahmoodLab/CONCH        512-d   Lu et al., Nat Med 2024          <- used in the paper
#   exaone    LGAI-DILAB/EXAONEPath   768-d   EXAONEPath, LG AI Research 2024
#   uni       MahmoodLab/UNI          1024-d  Chen et al., Nat Med 2024
#   gigapath  prov-gigapath/prov-gigapath  1536-d  Xu et al., Nature 2024
#
# Changing the encoder changes the feature dimension; set --pca in build_microstate_cache.py
# accordingly (the paper uses 512 with CONCH, i.e. a full-rank rotation).
set -euo pipefail

SLIDES=""; OUT=""; ENCODER="conch"; PATCH=256; LEVEL=0; BATCH=256; WORKERS=8; DEVICE="cuda"
EXPR=""                      # optional: CSV/h5ad of matched bulk expression to attach as `tpm`
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
    --expr)    EXPR=$2; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done
[ -n "$SLIDES" ] && [ -n "$OUT" ] || { echo "usage: $0 --slides DIR --out DIR [--encoder conch|exaone|uni|gigapath]"; exit 1; }
PY=${PY:-python}
mkdir -p "$OUT/patches" "$OUT/sample_pair_${ENCODER}"

echo "[1/3] tissue segmentation and patching (${PATCH}px, level ${LEVEL})"
# CLAM: https://github.com/mahmoodlab/CLAM
$PY "${CLAM_REPO:-external/CLAM}/create_patches_fp.py" \
    --source "$SLIDES" --save_dir "$OUT/patches" \
    --patch_size "$PATCH" --step_size "$PATCH" --patch_level "$LEVEL" \
    --seg --patch --stitch

echo "[2/3] ${ENCODER} features"
$PY code/features/encode_patches.py \
    --patches "$OUT/patches/patches" --slides "$SLIDES" \
    --out "$OUT/sample_pair_${ENCODER}" --encoder "$ENCODER" \
    --patch_size "$PATCH" --level "$LEVEL" \
    --batch "$BATCH" --workers "$WORKERS" --device "$DEVICE"

if [ -n "$EXPR" ]; then
  echo "[3/3] attaching matched bulk expression"
  $PY code/features/attach_expression.py \
      --features "$OUT/sample_pair_${ENCODER}" --expression "$EXPR"
else
  echo "[3/3] skipped (no --expr); attach bulk expression before training"
fi

echo "done: $OUT/sample_pair_${ENCODER}"
