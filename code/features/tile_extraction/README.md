# Tile extraction

Tissue detection and patch-coordinate extraction actually used to produce the paper's
results, in place of CLAM. Output format is unchanged from the rest of the pipeline: one
`.h5` per slide with a `coords` dataset (`(N, 2)`, full-resolution top-left pixel per
patch), so `code/features/encode_patches.py` reads it exactly as it would read CLAM's
output.

## Origin

Vendored (with minor path fixes, no logic changes) from Sungmin Lee's
[Pathology-WSI-Tile-Sampling-System](https://github.com/CocoSungMin/Pathology-WSI-Tile-Sampling-System)
(MIT license per that repository). If you use this tiling code, please cite it:

```bibtex
@software{pathology_tile_sampling,
  title={Pathology WSI Tile Sampling System},
  author={Sungmin Lee},
  year={2025},
  url={https://github.com/CocoSungMin/Pathology-WSI-Tile-Sampling-System}
}
```

## What it does

- `ForegroundMasking.py` — tissue/background separation (YUV + HSV color filtering),
  fat and artifact removal, morphological cleanup.
- `criterion.py` — thumbnail-level heuristics (saturation, contrast, intensity
  histogram) deciding whether a slide needs color/stain adjustment before masking.
- `enhanceStain.py` — ImageJ-style thumbnail contrast enhancement.
- `stainNorm.py` — TCGA-BRCA stain normalization (fit/export/load a normalization
  template; only `load_template` + apply is used at inference time here).
- `Tiling.py` — tile-coordinate extraction over the foreground mask, boundary-based
  region clustering, and per-tile tissue-fraction filtering (keeps tiles with
  &ge;20% valid tissue pixels).
- `TileSampling.py` — `WSITileSampler`, the orchestration class: loads each slide,
  runs masking + tiling, and writes `<slide_id>.h5`.
- `tile_processing.py` — command-line entry point.

## Parameters actually used for BRCA / KIRC / LUAD

```
tile_size   = 256
overlap     = 0
min_tiles   = 5      # minimum tiles per boundary region, post-filtering
is_normalized = True  # applies the TCGA-BRCA stain-normalization template below
                       # to every cohort (no organ-specific template was fit)
thumb_level = -1
```

These match `settings.txt` in the upstream tool at the commit vendored here, and are
the values wired into `code/features/extract_features.sh`.

## Usage

Called from `extract_features.sh`; not normally run by hand. Direct invocation:

```bash
python code/features/tile_extraction/tile_processing.py \
    --root /path/to/svs --output_dir data/BRCA/patches \
    --tile_size 256 --min_tiles 5 --is_normalized True --save_thumb True
```

Writes `<output_dir>/<slide_id>.h5` (plus `<output_dir>/Thumbnail/` when
`--save_thumb True`).
