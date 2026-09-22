# MoSPR

Predicting bulk transcriptomes from H&E whole-slide images with a state-structured linear model.

## Results

Patient-level 4-fold cross-validation on TCGA; mean over folds. Per-fold values, 95% CIs and paired
tests are in `results/tables/`.

**Gene level** (PCC / SCC across slides, averaged over genes)

| Method | BRCA PCC | BRCA SCC | KIRC PCC | KIRC SCC | LUAD PCC | LUAD SCC |
|---|---|---|---|---|---|---|
| Max | 0.294 | 0.283 | 0.236 | 0.244 | 0.258 | 0.278 |
| Mean | 0.336 | 0.338 | 0.285 | 0.293 | 0.292 | 0.315 |
| AbMIL | 0.370 | 0.363 | 0.293 | 0.299 | 0.308 | 0.324 |
| HE2RNA | 0.300 | 0.293 | 0.256 | 0.273 | 0.276 | 0.297 |
| AbReg | 0.355 | 0.349 | 0.289 | 0.293 | 0.296 | 0.311 |
| tRNAformer | 0.341 | 0.332 | 0.312 | 0.319 | 0.316 | 0.336 |
| ILRA | 0.238 | 0.271 | 0.191 | 0.248 | 0.280 | 0.304 |
| S4MIL | 0.328 | 0.330 | 0.269 | 0.274 | 0.299 | 0.315 |
| MambaMIL | 0.363 | 0.356 | 0.290 | 0.290 | 0.319 | 0.335 |
| SRMambaMIL | 0.361 | 0.352 | 0.292 | 0.294 | 0.311 | 0.326 |
| MOSBY | 0.343 | 0.349 | 0.264 | 0.289 | 0.289 | 0.308 |
| SEQUOIA VIS | 0.353 | 0.337 | 0.310 | 0.314 | 0.317 | 0.331 |
| 2DMamba | 0.352 | 0.348 | 0.287 | 0.292 | 0.300 | 0.313 |
| CPNN | 0.360 | 0.356 | 0.293 | 0.310 | 0.305 | 0.325 |
| **MoSPR** | **0.413** | **0.411** | **0.334** | **0.348** | **0.358** | **0.376** |

**Pathway level** (median over gene sets of the per-set SCC)

| Method | BRCA Hallmark | BRCA GO-BP | BRCA KEGG | KIRC Hallmark | KIRC GO-BP | KIRC KEGG | LUAD Hallmark | LUAD GO-BP | LUAD KEGG |
|---|---|---|---|---|---|---|---|---|---|
| Max | 0.356 | 0.337 | 0.321 | 0.281 | 0.284 | 0.309 | 0.420 | 0.379 | 0.388 |
| Mean | 0.469 | 0.448 | 0.415 | 0.367 | 0.379 | 0.415 | 0.498 | 0.446 | 0.452 |
| AbMIL | 0.500 | 0.476 | 0.446 | 0.378 | 0.388 | 0.418 | 0.519 | 0.465 | 0.474 |
| HE2RNA | 0.333 | 0.329 | 0.300 | 0.299 | 0.321 | 0.340 | 0.388 | 0.349 | 0.365 |
| AbReg | 0.481 | 0.453 | 0.425 | 0.397 | 0.389 | 0.420 | 0.507 | 0.443 | 0.462 |
| tRNAformer | 0.412 | 0.381 | 0.345 | 0.367 | 0.368 | 0.404 | 0.467 | 0.415 | 0.424 |
| ILRA | 0.385 | 0.358 | 0.330 | 0.301 | 0.317 | 0.340 | 0.463 | 0.418 | 0.430 |
| S4MIL | 0.434 | 0.414 | 0.383 | 0.301 | 0.329 | 0.343 | 0.481 | 0.428 | 0.436 |
| MambaMIL | 0.479 | 0.458 | 0.423 | 0.308 | 0.326 | 0.345 | **0.529** | 0.481 | 0.487 |
| SRMambaMIL | 0.463 | 0.448 | 0.411 | 0.306 | 0.315 | 0.338 | 0.514 | 0.455 | 0.472 |
| MOSBY | 0.400 | 0.381 | 0.354 | 0.257 | 0.263 | 0.293 | 0.348 | 0.312 | 0.324 |
| SEQUOIA VIS | 0.444 | 0.413 | 0.385 | 0.384 | 0.388 | 0.420 | 0.488 | 0.441 | 0.450 |
| 2DMamba | 0.460 | 0.439 | 0.408 | 0.381 | 0.370 | 0.388 | 0.501 | 0.443 | 0.457 |
| CPNN | 0.391 | 0.377 | 0.327 | 0.282 | 0.293 | 0.349 | 0.427 | 0.380 | 0.388 |
| **MoSPR** | **0.551** | **0.520** | **0.486** | **0.402** | **0.415** | **0.447** | 0.528 | **0.482** | **0.491** |

## Repository layout

```
.
├── code/
│   ├── features/        patch extraction and foundation-model features
│   ├── mospr/           the model and the scoring pipeline (paths.py resolves every path)
│   ├── baselines/       comparison methods: upstream code, patches, runners, environment
│   ├── tables/          table scripts and make_all_tables.sh
│   ├── figures/         figure scripts
│   ├── splits/          patient-level split construction and checks
│   ├── config/          configs used for the reported runs
│   ├── tests/           end-to-end check against the shipped results
│   ├── reproduce_results.py
│   └── fetch_checkpoints.py
├── results/
│   ├── per_cohort/{BRCA,KIRC,LUAD}/   per-fold scores and the 4-fold splits
│   └── tables/                        every table (.tex) and the CSVs behind it
├── resources/processed/ evaluated gene list per cohort
├── figures/final/       manuscript figures
├── checkpoints/         empty in this submission
└── env/                 pyproject.toml, uv.lock, requirements.txt
```

## Setup

```bash
uv sync --project env            # or: pip install -r env/requirements.txt
```

The comparison methods need their own environment, see `code/baselines/README.md`.
All paths go through `code/mospr/paths.py` and can be redirected with `MOSPR_ROOT`,
`MOSPR_DATASET_ROOT`, `MOSPR_RESULTS_ROOT`, `MOSPR_TABLES_ROOT`, `MOSPR_GENESETS`,
`MOSPR_PROCESSED`, `MOSPR_CPNN_REPO` and `MOSPR_LOGS`.

## Data and features

Slides and expression are not included. The evaluated gene lists are in `resources/processed/`
(14,042 BRCA, 14,295 KIRC, 14,514 LUAD genes).

```bash
bash code/features/extract_features.sh --slides /path/to/svs --out data/BRCA-paper-digital_slide \
    --encoder conch --tpm tpm_BRCA.csv --counts counts_BRCA.csv \
    --genes resources/processed/eval_genes_paper_BRCA.txt
python code/features/build_pathway_targets.py --cohort BRCA
```

The first command extracts patches with CLAM, encodes them and writes one `.h5` per slide with
`coord`, `feat`, `tpm` and `raw_count` (untransformed, in the order of the gene list). The second
builds the pathway-level targets: each Hallmark set with at least 10 measured genes is the sum of
its member genes, written to `data/BRCA-paper-pathway-digital_slide/`. `--encoder` takes `conch`
(512-d, used for the results above), `exaone`, `uni` or `gigapath`; with a different dimension, set
`--pca` in `build_microstate_cache.py` to match. Gene sets are read from `MOSPR_GENESETS`
(MSigDB `h.all.v2025.1.Hs.symbols.gmt`).
<!-- TODO: patch filtering script and criteria -->

## Splits

The patient-level 4-fold splits are `results/per_cohort/{cohort}/split/split_patient_4fold.csv`.
The pipeline reads a pickle of paths, rebuilt locally with

```bash
python code/splits/csv_to_pkl.py --cohort BRCA --data data
```

## Reproducing

1. Microstate cache: `python code/mospr/build_microstate_cache.py --cohort BRCA --variant paper --pca 512 --suffix _BRCA_fpsplit_p512`
2. MoSPR: `bash code/mospr/run_mospr.sh`
3. Comparison methods: see `code/baselines/README.md`
4. Tables and figures from the per-fold results: `bash code/tables/make_all_tables.sh`

`python code/reproduce_results.py --check` rebuilds the reported numbers from the shipped per-fold
CSVs without data or models and compares them with the tables. `code/tests/test_pipeline.py` runs
one fold end to end and compares every step with `results/`.

## Checkpoints

Due to storage constraints, pretrained checkpoints are not included in the anonymous submission
repository. The repository contains the complete training and evaluation pipeline, including
configurations and data splits. Pretrained weights will be released publicly upon publication, and
`python code/fetch_checkpoints.py --url "<LINK>"` will download them into `checkpoints/`.

## Environment used for the results

Ubuntu 24.04, 6 x NVIDIA RTX PRO 6000 Blackwell (96 GB), CUDA 13.0. MoSPR runs on CPU with the
versions pinned in `env/`; one fold takes about a minute after the microstate cache (about 6 min
per fold). The comparison methods ran on single GPUs, 20-30 min per model and fold. Seed 2021
throughout.
