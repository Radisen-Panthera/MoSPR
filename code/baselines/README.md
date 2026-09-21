# Baselines — CPNN benchmark code + our patches

The comparison methods are trained with the authors' released code, not a fork of it.
Clone it, apply the patches in `patches/`, and run the scripts in `scripts/`.

```bash
git clone https://github.com/naivete5656/CPNN.git
cd CPNN
git checkout bcdf7d05299361e4203627f062cf95e78f175477
for p in ../patches/*.patch; do patch -p1 < "$p"; done
```

## What the patches change

| Patch | Change |
|---|---|
| `model_comparisons_abmil.py.patch` | adds `ilse_mean` / `ilse_max`: pooling after the fc+ReLU embedding, as in the cited designs (the released `mean` / `max` pool raw features straight into a linear head) |
| `dataloader_dataset.py.patch` | `CPNN_SPLIT_FILE` (which split to read), `CPNN_TRAINVAL` (fold in the validation slides for stage 2) |
| `dataloader_our_dataset.py.patch` | `CPNN_C2L_PREFIX` — per-split cell2location output folder, used by CPNN only |
| `main.py.patch` | stage-2 fine-tuning from a stage-1 checkpoint, seeding, and a `[condition]` log line recording split / data dir / epochs |
| `model_build_model.py.patch` | `CPNN_OUT_DIM` (gene count per cohort) |
| `config_*.yaml.patch` | data root, workers, logging off |
| `model_comparisons_{ilra,mamba_mil2d,__init__}.py.patch` | environment fixes (sm_120 build, optional Mamba imports) |

Replace `<DATA_ROOT>` and `<REPO_ROOT>` in the patched files with your own paths.
Comments inside the patches are in Korean; they state why each hook exists.

## Running

```bash
FILTERED=1 CONDITION=psplit    bash scripts/axis_baselines.sh   # stage 1: train, early stop on val
FILTERED=1 CONDITION=psplit_tv bash scripts/axis_baselines.sh   # stage 2: fine-tune on train+val
ILSE=1 ONLY="ilse_mean ilse_max" bash scripts/run_ilse.sh       # pooling baselines, cited design
bash scripts/c2l_psplit.sh                                      # cell2location, CPNN only
```

`scripts/collect_release.py` gathers the per-fold metrics, predictions and checkpoints.
