# Checkpoints

Not included in this repository. Pretrained weights will be released publicly upon publication;
`code/fetch_checkpoints.py --url "<LINK>"` downloads them into this folder.

Layout once fetched, per cohort:

```
checkpoints/{BRCA,KIRC,LUAD}/
  weights_gene_fold{0..3}.npz       MoSPR, gene space
  weights_pathway_fold{0..3}.npz    MoSPR, pathway space
  <method>_fold{0..3}.ckpt          14 comparison methods x 4 folds
```

The `.npz` files are closed-form fits rather than network checkpoints. They hold `W_q` (low-rank
projection), `U` (low-rank basis), `b_q`, the design-matrix standardisation `xmu`/`xsd`, the target
standardisation `ymu`/`ysd`, `genes`, and the shapes `K`, `d`, `n_M`, `n_S`. A prediction is
replayed from a microstate cache with

```python
import numpy as np
w = np.load("checkpoints/BRCA/weights_gene_fold0.npz", allow_pickle=True)
# build X = [M | S] from the cache first (design_blocks.build_blocks, MOSPR_TAU=0)
Xz = (X - w["xmu"]) / w["xsd"]
pred = (Xz @ w["W_q"].T) @ w["U"] * w["ysd"] + w["ymu"]
```

All weights come from the final axis: filtered patches, patient-level split, train+val
fine-tuning, seed 2021, and for MoSPR K=8, d=512, tau=0, ALPHA_SMOOTH=0. Stage-1 checkpoints are
not shared. With the optional `predictions/` archive the tables can be rebuilt without the
checkpoints (`code/reproduce_results.py`, `code/notebooks/`).
