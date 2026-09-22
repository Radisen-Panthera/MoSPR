# Tile extraction

Tissue detection and patch-coordinate extraction actually used to produce the paper's
results, in place of CLAM. Output format is unchanged from the rest of the pipeline: one
`.h5` per slide with a `coords` dataset (`(N, 2)`, full-resolution top-left pixel per
patch), so `code/features/encode_patches.py` reads it exactly as it would read CLAM's
output.

## Reference

The tiling procedure follows G2L. If you use this code, please cite:

```bibtex
@article{cho2025g2l,
  title={G2L: From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Knowledge Distillation},
  author={Cho, Yesung and Lee, Sungmin and Lee, Geongyu and Lee, Minkyung and Park, Jongbae and Shin, Dongmyung},
  journal={arXiv preprint arXiv:2510.11176},
  year={2025}
}
```

## What it does

- `ForegroundMasking.py` — tissue/background separation (YUV + HSV color filtering),
  fat and artifact removal, morphological cleanup.
- `criterion.py` — thumbnail-level heuristics (saturation, contrast, intensity
  histogram) deciding whether a slide needs color/stain adjustment before masking.
- `enhanceStain.py` — ImageJ-style thumbnail contrast enhancement.
- `stainNorm.py` — TCGA-BRCA stain normalization (fit/export/load a normalization
  template). The template is loaded at start-up but only applied when
  `--is_normalized True`; it was not applied for the reported results.
- `Tiling.py` — tile-coordinate extraction over the foreground mask. A tile is kept
  when at least 50% of its footprint on the thumbnail mask is foreground, its centre
  pixel is foreground and at least two of its border pixels are foreground. Contours
  smaller than 0.05% of the foreground (in tile units) are dropped, and tiles are then
  grouped by contour; groups with fewer than `min_tiles` tiles are dropped.
- `TileSampling.py` — `WSITileSampler`, the orchestration class: loads each slide,
  runs masking + tiling, and writes `<slide_id>.h5`.
- `tile_processing.py` — command-line entry point.

## Differences from the original defaults

The copy here is the one that produced the coordinates behind the reported results.
It differs from the original defaults in three places:

- the thumbnail and the coordinate scaler are read from the same pyramid level;
- the tile foreground threshold is 50% of the mask footprint (originally 70%);
- red pen-mark removal uses a YUV V &ge; 225 threshold with a 3&times;3 cleanup kernel
  (originally V &ge; 140 with a 7&times;7 kernel and dilation).

## Parameters used for BRCA / KIRC / LUAD

| Parameter | Value |
|---|---|
| `tile_size` | 256 (level 0) |
| `overlap` | 0 |
| `min_tiles` | 5 |
| `is_normalized` | False |

These are the values in the `settings.txt` written next to the original coordinate
files of all three cohorts, and the defaults in `code/features/extract_features.sh`.
Run with them, the tool reproduces the coordinates used for the results exactly
(checked on 20 randomly drawn BRCA slides). With `--is_normalized True`, every slide
gains about 0.5% extra tiles on the tissue border.

## Usage

Called from `extract_features.sh`; not normally run by hand. Direct invocation:

```bash
python code/features/tile_extraction/tile_processing.py \
    --root /path/to/svs --output_dir data/BRCA/patches \
    --tile_size 256 --min_tiles 5 --is_normalized False --save_thumb True
```

Writes `<output_dir>/<slide_id>.h5` (plus `<output_dir>/Thumbnails/` when
`--save_thumb True`).
