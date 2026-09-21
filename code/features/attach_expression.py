#!/usr/bin/env python
"""Attach matched bulk expression to the per-slide feature files as `tpm`.

    python code/features/attach_expression.py --features data/BRCA/sample_pair_conch \
        --expression bulk.csv [--id_len 16]

`--expression` is a CSV with genes in columns and one row per sample, or an .h5ad with the
same layout; the row index must be the sample barcode that prefixes the slide id
(TCGA-XX-XXXX-01A by default, i.e. the first 16 characters). Values are stored as
log1p(TPM / sum * 1e4), the space in which every metric in the paper is computed.
"""
import argparse
import pathlib

import h5py
import numpy as np
import pandas as pd


def log_cpm(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.sum()
    return np.log1p(x / s * 1e4).astype(np.float32) if s > 0 else np.zeros_like(x, dtype=np.float32)


def main(a):
    p = pathlib.Path(a.expression)
    if p.suffix == ".h5ad":
        import anndata
        ad = anndata.read_h5ad(p)
        expr = pd.DataFrame(ad.X.toarray() if hasattr(ad.X, "toarray") else ad.X,
                            index=ad.obs_names, columns=ad.var_names)
    else:
        expr = pd.read_csv(p, index_col=0)
    expr.index = expr.index.astype(str)

    n_ok = n_miss = 0
    for f in sorted(pathlib.Path(a.features).glob("*.h5")):
        sample = f.stem[:a.id_len]
        if sample not in expr.index:
            n_miss += 1
            continue
        with h5py.File(f, "a") as h:
            if "tpm" in h:
                del h["tpm"]
            h.create_dataset("tpm", data=log_cpm(expr.loc[sample].to_numpy()))
        n_ok += 1
    print(f"attached: {n_ok}, no matching expression: {n_miss}, genes: {expr.shape[1]}")
    print("gene order is fixed by the column order of --expression; keep it identical across cohorts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--expression", required=True)
    ap.add_argument("--id_len", type=int, default=16, help="characters of the slide id that name the sample")
    main(ap.parse_args())
