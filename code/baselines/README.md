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

## Environment

```bash
conda env create -f env/environment.yml && conda activate mospr-baselines
# or
docker build -t mospr-baselines -f code/baselines/env/Dockerfile .
```

That covers AbMIL (all pooling variants), AbReg, ILRA, S4MIL, MOSBY, HE2RNA, SEQUOIA VIS and
tRNAformer. Three methods need one extra step each, because they compile CUDA kernels or pull a
large probabilistic stack:

| Method | Extra step | Docker |
|---|---|---|
| MambaMIL, SRMambaMIL | `bash scripts/install_mambamil_fork.sh` (mamba-ssm fork + causal-conv1d 1.1.1) | `--build-arg WITH_MAMBA=1` |
| 2DMamba | `bash scripts/build_2dmamba_kernel.sh` (compiles pscan for your architecture) | `--build-arg WITH_MAMBA=1`, then run the script |
| CPNN (ProtoSum) | `pip install cell2location==0.1.5 scvi-tools==1.4.2`, then `bash scripts/c2l_psplit.sh` | `--build-arg WITH_CELL2LOCATION=1` |

Without them those three are skipped and the code says so on import; the other methods are
unaffected. Reference machine: RTX PRO 6000 Blackwell (sm_120), CUDA 13.0, torch 2.13.0+cu130,
Python 3.11.15. The environment files pin the public cu128 wheels, which we did not build and
test end to end; they describe the dependency set rather than a byte-identical copy of our run.

## Running

```bash
FILTERED=1 CONDITION=psplit    bash scripts/axis_baselines.sh   # stage 1: train, early stop on val
FILTERED=1 CONDITION=psplit_tv bash scripts/axis_baselines.sh   # stage 2: fine-tune on train+val
ILSE=1 ONLY="ilse_mean ilse_max" bash scripts/run_ilse.sh       # pooling baselines, cited design
bash scripts/c2l_psplit.sh                                      # cell2location, CPNN only
```

Each run writes its metrics to `outputs/{fold}/` and its checkpoint to `ckpts/exps/{fold}/` inside
the patched clone; the run name encodes the condition, so the two stages never overwrite each other.

## What the two behavioural patches do

`patches/main.py.patch` seeds Python, NumPy, PyTorch and the dataloader workers from
`GENERAL.seed` (2021, override with `CPNN_SEED`), and warns when `PYTHONHASHSEED` is unset, since
that variable only takes effect when the launcher exports it. It also adds the stage-2
fine-tuning entry point and a `[condition]` log line recording split, data root and epochs.

`patches/model_comparisons_abmil.py.patch` adds the versions `ilse_mean` and `ilse_max`, which
apply mean or max pooling after the fc(512->512)+ReLU embedding, as in the cited designs (Wang et
al., *Revisiting Multiple Instance Neural Networks*, 2018; Ilse et al., *Attention-based Deep
MIL*, 2018), and change nothing else. The Max and Mean rows of our tables use these versions; the
released `mean` and `max` versions, which pool raw features straight into a linear layer, are left
untouched and can still be selected.
