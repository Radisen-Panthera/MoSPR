# MoSPR

Predicting bulk transcriptomes from H&E whole-slide images with a state-structured linear model.
Patches are quantised into microstates, microstates are grouped into a small number of tissue
macrostates, and per-macrostate mean features feed a closed-form low-rank ridge regression.

## Repository layout

```
.
├── code/
│   ├── features/                 patching and foundation-model features
│   │   ├── extract_features.sh       one cohort end to end (CLAM patches -> encoder -> .h5)
│   │   ├── encode_patches.py         CONCH / EXAONEPath / UNI / Prov-GigaPath
│   │   └── attach_expression.py      attach matched bulk expression as `tpm`
│   ├── mospr/                    the model and the scoring pipeline
│   │   ├── paths.py                  every path in the repository resolves through this
│   │   ├── build_microstate_cache.py microstates, PCA, adjacency, spectral embedding
│   │   ├── design_blocks.py          design-matrix blocks [M | S]
│   │   ├── train_mospr.py            the model (ridge on low-rank targets)
│   │   ├── ablation.py               component ablation (Table 3)
│   │   ├── score_pathways.py         pathway scoring for every method (Table 2)
│   │   ├── macrostate_enrichment.py  gene-set enrichment of macrostate loadings
│   │   ├── macrostate_hallmark_folds.py  macrostate x Hallmark, per fold
│   │   ├── figure2_slide_pathway.py  Figure 2 (slide overlay + enrichment bars)
│   │   ├── refit_gene_scores.py      stage 1 vs stage 2, gene axis
│   │   ├── refit_pathway_scores.py   stage 1 vs stage 2, pathway axis
│   │   ├── scorer_robustness.py      rescoring with ssGSEA and GSVA
│   │   ├── data_efficiency_mospr.py, data_efficiency_score_baselines.py
│   │   ├── cell2location_deconv.py   cell-type deconvolution (CPNN baseline only)
│   │   ├── prototype_hybrid.py       shared fitting and metric helpers
│   │   ├── run_mospr.sh              end-to-end MoSPR run
│   │   ├── run_data_efficiency.sh    data-efficiency runs
│   │   ├── cohorts.py                per-cohort dataset paths
│   │   └── spatial_proteome/         library: adjacency, spectral, state features, pathways
│   ├── baselines/                comparison methods (authors' code + our patches)
│   │   ├── README.md                 upstream repository, commit, how to apply the patches
│   │   ├── env/                      conda environment and Dockerfile for the baselines
│   │   ├── patches/                  10 patches against naivete5656/CPNN
│   │   └── scripts/                  axis_baselines.sh (both stages), run_ilse.sh,
│   │                                  best_epochs.py, c2l_psplit.sh, build helpers
│   ├── splits/                   patient-level splits
│   │   ├── make_patient_split.py     builds the 4-fold split used in the paper
│   │   ├── make_fraction_splits.py   subsampled splits for the data-efficiency runs
│   │   ├── csv_to_pkl.py             rebuilds the split pickle from the shipped CSV
│   │   └── check_split.py            compares a split against a reference
│   ├── tests/test_pipeline.py    runs the pipeline and compares with the released numbers
│   ├── config/                   configs used for the reported runs
│   ├── notebooks/                every table and figure in the paper
│   │   ├── tables_1_2_3.ipynb
│   │   ├── tables_statistics.ipynb
│   │   ├── tables_refit_and_robustness.ipynb
│   │   ├── figure2_macrostate_hallmark.ipynb
│   │   └── figure3_data_efficiency.ipynb
│   ├── reproduce_results.py      rebuilds the reported numbers from the per-fold CSVs
│   └── fetch_checkpoints.py      downloads the weights once released
├── results/
│   ├── tables/
│   │   ├── final/alpha0/         Tables 1-3 (.tex) and their source CSVs
│   │   ├── statistics/           fold mean / 95% CI / SD, fold-wise raw, appendix tables
│   │   ├── refit/                stage 1 vs stage 2, per fold and summary
│   │   ├── data_efficiency/      per fold x fraction, MoSPR and baselines
│   │   ├── macrostate/           per-fold Hallmark / GO-BP / KEGG enrichment
│   │   └── reproduced_summary.csv
│   └── per_cohort/{BRCA,KIRC,LUAD}/
│       ├── results/              per-fold and summary scores
│       └── split/                split_patient_4fold.csv, fold counts, provenance
├── figures/final/                Figure 2 and the data-efficiency curves
├── checkpoints/                  empty in this submission (see below)
└── env/                          pyproject.toml, uv.lock, .python-version
```

## Setup

```bash
uv sync --project env            # Python 3.12, PyTorch cu128
```

Baseline training needs a second environment; `code/baselines/env/environment.yml` and
`code/baselines/env/Dockerfile` build it, and `code/baselines/README.md` lists the three methods
that need an extra step (MambaMIL/SRMambaMIL, 2DMamba, CPNN).

Paths are resolved by `code/mospr/paths.py` and can be redirected with environment variables:
`MOSPR_ROOT`, `MOSPR_DATASET_ROOT`, `MOSPR_RESULTS_ROOT`, `MOSPR_TABLES_ROOT`, `MOSPR_GENESETS`,
`MOSPR_CPNN_REPO`, `MOSPR_LOGS`.

## Data and features

Slides and expression are not included. Starting from whole-slide images and matched bulk
expression:

```bash
bash code/features/extract_features.sh \
    --slides /path/to/svs --out data/BRCA --encoder conch --expr bulk_BRCA.csv
```

This segments tissue and extracts patches with CLAM, encodes them, and writes one `.h5` per slide
with `coord`, `feat` and `tpm` (expression as log1p(TPM / sum * 1e4), the space all metrics use).
`--encoder` accepts `conch` (512-d, used in the paper), `exaone`, `uni` and `gigapath`; a different
encoder changes the feature dimension, so set `--pca` in `build_microstate_cache.py` to match.

## Splits

The 4-fold patient-level splits are shipped as slide-id CSVs, one per cohort:
`results/per_cohort/{cohort}/split/split_patient_4fold.csv` (fold, split, slide, patient, sample).
The pipeline reads a pickle of paths, which is rebuilt locally from the CSV:

```bash
python code/splits/csv_to_pkl.py --cohort BRCA --data data
```

`code/splits/make_patient_split.py` is the script that produced the split in the first place
(patient-level, seed 2021, no patient shared between train, validation and test; asserted).

## Testing

```bash
python code/tests/test_pipeline.py --data data --cohort BRCA --fold 0 \
    --genesets refdata/genesets --processed data/processed --weights checkpoints
```

It rebuilds the design matrix, runs both MoSPR stages, the ablation and the macrostate enrichment
on one fold, and compares each number with the values shipped in `results/`. Add `--with-cache` to
rebuild the microstate cache first. One fold runs on CPU in a few minutes.

## Weights and predictions

Due to storage constraints, pretrained checkpoints are not included in the anonymous submission
repository. The repository contains the complete training and evaluation pipeline, including
configurations and data splits. Pretrained weights will be released publicly upon publication.

Everything needed to trace how the reported numbers were produced is in the repository:

| Item | Where |
|---|---|
| Feature extraction | `code/features/` |
| Training and evaluation code | `code/mospr/`, `code/baselines/` |
| Configs actually used | `code/config/` (`train_cfg.yaml`, `ProtoSum.yaml`, `cohorts.py`) |
| 4-fold splits, and the code that builds them | `results/per_cohort/{cohort}/split/split_patient_4fold.csv`, `code/splits/` |
| Seed | 2021 throughout (`GENERAL.seed`, `CPNN_SEED`, `PYTHONHASHSEED`); recorded in `results/per_cohort/*/split/provenance.json` |
| Environment | `env/pyproject.toml`, `env/uv.lock`, `env/.python-version` |
| Final tables | `results/tables/final/alpha0/` (.tex and source CSVs) |
| Per-fold raw results | `results/per_cohort/{cohort}/results/`, `results/tables/statistics/stats_foldwise_*.csv` |
| Aggregation script | `code/reproduce_results.py` |

```bash
python code/reproduce_results.py            # rebuild the reported numbers from the per-fold CSVs
python code/reproduce_results.py --check    # and verify they match the .tex tables (exit 1 on mismatch)
```

It runs no model and needs no data: it reads the shipped per-fold scores, aggregates them the way
the manuscript does (fold as the repeated unit, Student's t with df=3, paired tests against MoSPR),
writes `results/tables/reproduced_summary.csv` and, with `--check`, diffs the means against
`results/tables/final/alpha0/table1_gene.tex`.

Once released, fetch the weights with the link provided at that time:

```bash
python code/fetch_checkpoints.py --url "<LINK>"                     # checkpoints
python code/fetch_checkpoints.py --url "<LINK>" --what predictions  # per-fold test predictions
```

Per cohort that is 56 baseline `.ckpt` (14 methods x 4 folds) and 8 MoSPR `.npz`. The `.npz` files
are closed-form fits rather than network checkpoints: `W_q`, `U`, `b_q`, the design-matrix
standardisation (`xmu`/`xsd`), the target standardisation (`ymu`/`ysd`) and `genes`.

## Reproducing the paper

1. Features, as above, into `data/{cohort}/sample_pair_conch/`.
2. Microstate cache, which everything downstream reads:
   ```bash
   python code/mospr/build_microstate_cache.py --cohort BRCA --variant paper --pca 512 \
       --suffix _BRCA_fpsplit_p512
   ```
3. MoSPR, both stages (stage 1 selects q and lambda on validation, stage 2 refits on train+val):
   ```bash
   bash code/mospr/run_mospr.sh
   ```
4. Baselines: clone the upstream repository, apply `code/baselines/patches/`, then
   ```bash
   FILTERED=1 CONDITION=psplit    bash code/baselines/scripts/axis_baselines.sh
   FILTERED=1 CONDITION=psplit_tv bash code/baselines/scripts/axis_baselines.sh
   ```
5. Tables and figures: run the notebooks in `code/notebooks/`.

## Notes

Seed 2021 is fixed for Python, NumPy, PyTorch and the dataloader workers. `PYTHONHASHSEED` has to
be exported by the launcher; the code warns when it is not. The upstream benchmark defines a seed
helper but never calls it, so repeated runs of the same configuration drifted (measured: mean-pool
BRCA fold 0 gave SCC 0.2036 / 0.2379 / 0.2398 over three runs); the patch in
`code/baselines/patches/main.py.patch` fixes that.

