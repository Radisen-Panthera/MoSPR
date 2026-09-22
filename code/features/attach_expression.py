import argparse
import pathlib
import sys

import h5py
import numpy as np
import pandas as pd


def read_table(path):
    p = pathlib.Path(path)
    if p.suffix == ".h5ad":
        import anndata
        ad = anndata.read_h5ad(p)
        x = ad.X.toarray() if hasattr(ad.X, "toarray") else ad.X
        return pd.DataFrame(x, index=ad.obs_names.astype(str), columns=ad.var_names.astype(str))
    t = pd.read_csv(p, index_col=0)
    t.index = t.index.astype(str)
    return t


def main(a):
    genes = pd.read_csv(a.genes, header=None)[0].astype(str).tolist()
    tables = {"tpm": read_table(a.tpm)}
    if a.counts:
        tables["raw_count"] = read_table(a.counts)
    for key, t in tables.items():
        missing = [g for g in genes if g not in t.columns]
        if missing:
            sys.exit(f"{key}: {len(missing)} of {len(genes)} genes missing, e.g. {missing[:5]}")
        tables[key] = t[genes]

    n_ok = n_miss = 0
    for f in sorted(pathlib.Path(a.features).glob("*.h5")):
        sample = f.stem[:a.id_len]
        if sample not in tables["tpm"].index:
            n_miss += 1
            continue
        with h5py.File(f, "a") as h:
            for key, t in tables.items():
                if key in h:
                    del h[key]
                h.create_dataset(key, data=t.loc[sample].to_numpy(dtype=np.float32))
        n_ok += 1
    print(f"slides with expression: {n_ok}, without: {n_miss}, genes: {len(genes)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="folder of per-slide .h5 feature files")
    ap.add_argument("--tpm", required=True)
    ap.add_argument("--counts", default="")
    ap.add_argument("--genes", required=True, help="evaluated gene list, one symbol per line")
    ap.add_argument("--id_len", type=int, default=16, help="characters of the slide id that name the sample")
    main(ap.parse_args())
