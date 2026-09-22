# Tile extraction

Patch extraction and filtering follow G2L:

```bibtex
@article{cho2025g2l,
  title={G2L: From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Knowledge Distillation},
  author={Cho, Yesung and Lee, Sungmin and Lee, Geongyu and Lee, Minkyung and Park, Jongbae and Shin, Dongmyung},
  journal={arXiv preprint arXiv:2510.11176},
  year={2025}
}
```

The output is one `.h5` per slide with a `coords` dataset (`(N, 2)`, level-0 top-left pixel
of each patch), read by `code/features/encode_patches.py`.

## Files

- `ForegroundMasking.py` — tissue/background separation (YUV + HSV color filtering),
  fat, pen and artifact removal, morphological cleanup.
- `criterion.py` — thumbnail-level heuristics (saturation, contrast, intensity
  histogram) deciding whether a slide needs contrast adjustment before masking.
- `enhanceStain.py` — ImageJ-style thumbnail contrast enhancement.
- `stainNorm.py` — optional stain normalization (`--is_normalized True`); not used.
- `Tiling.py` — tile-coordinate extraction over the foreground mask. A tile is kept
  when at least 50% of its footprint on the thumbnail mask is foreground, its centre
  pixel is foreground and at least two of its border pixels are foreground. Contours
  smaller than 0.05% of the foreground (in tile units) are dropped, and tiles are then
  grouped by contour; groups with fewer than `min_tiles` tiles are dropped.
- `TileSampling.py` — `WSITileSampler`: loads each slide, runs masking and tiling, and
  writes `<slide_id>.h5`.
- `tile_processing.py` — command-line entry point.

## Parameters (BRCA, KIRC, LUAD)

| Parameter | Value |
|---|---|
| `tile_size` | 256 (level 0) |
| `overlap` | 0 |
| `min_tiles` | 5 |
| `is_normalized` | False |

These are the defaults in `code/features/extract_features.sh`.

## Usage

Called from `extract_features.sh`. Direct invocation:

```bash
python code/features/tile_extraction/tile_processing.py \
    --root /path/to/svs --output_dir data/BRCA/patches \
    --tile_size 256 --min_tiles 5 --is_normalized False --save_thumb True
```

Writes `<output_dir>/<slide_id>.h5` (plus `<output_dir>/Thumbnails/` with `--save_thumb True`).
