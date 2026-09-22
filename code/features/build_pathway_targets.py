import argparse
import pathlib
import pickle
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "mospr"))
import paths as _p
from spatial_proteome import pathways


def project(mat, idx_list):
    return np.stack([np.asarray(mat[i]).sum(axis=0) for i in idx_list])


def main(a):
    gene_root = _p.DATA / f"{a.cohort}-paper-digital_slide"
    path_root = _p.DATA / f"{a.cohort}-paper-pathway-digital_slide"
    gdir = _p.PROCESSED if a.cohort == "BRCA" else _p.PROCESSED / f"processed_{a.cohort}"
    genes = pd.read_csv(gdir / f"eval_genes_paper_{a.cohort}.txt", header=None)[0].values
    idx = pathways.member_index(pathways.parse_gmt(_p.GENESETS / a.gmt), genes,
                                min_members=a.min_members)
    names, idx_list = list(idx), [idx[n] for n in idx]
    print(f"{a.cohort}: {len(names)} pathways with >= {a.min_members} measured genes")

    src = gene_root / "sample_pair_feature_conch"
    dst = path_root / "sample_pair_feature_conch"
    dst.mkdir(parents=True, exist_ok=True)
    slides = sorted(src.glob("*.h5"))
    for f in slides:
        out = dst / f.name
        if out.exists() and out.stat().st_mtime >= f.stat().st_mtime and not a.force:
            continue
        with h5py.File(f, "r") as h, h5py.File(out, "w") as o:
            o.create_dataset("feat", data=h["feat"][:])
            o.create_dataset("coord", data=h["coord"][:])
            for key in ("tpm", "raw_count"):
                if key in h:
                    v = h[key][:]
                    o.create_dataset(key, data=np.array([v[i].sum() for i in idx_list], dtype=np.float32))
    pd.Series(names).to_csv(path_root / "pathway_names.txt", index=False, header=False)
    print(f"pathway slide files: {len(slides)}")

    split_src = gene_root / a.split
    if split_src.exists():
        with open(split_src, "rb") as f:
            split = pickle.load(f)
        split = {fold: {k: [path_root / "sample_pair_feature" / pathlib.Path(p).name for p in v]
                        for k, v in d.items()} for fold, d in split.items()}
        with open(path_root / a.split, "wb") as f:
            pickle.dump(split, f)
        print(f"split repointed: {path_root / a.split}")

    n = 0
    for fold_dir in sorted(p for p in gene_root.iterdir() if p.is_dir()):
        src_pd = fold_dir / f"{a.resolution}_parameter_dict.pkl"
        if not src_pd.exists():
            continue
        with open(src_pd, "rb") as f:
            params = pickle.load(f)
        theta = params["theta"]
        assert list(theta.index) == list(genes), "prototype gene order differs from the gene list"
        out = dict(params)
        out["theta"] = pd.DataFrame(project(theta.values, idx_list), index=names, columns=theta.columns)
        out["m_g"] = np.array([np.asarray(params["m_g"]).ravel()[i].mean() for i in idx_list], dtype=np.float32)
        out["s_eg"] = np.array([np.asarray(params["s_eg"]).ravel()[i].mean() for i in idx_list], dtype=np.float32)
        out["mask"] = pd.DataFrame(project(params["mask"].values.astype(float), idx_list) == 0,
                                   index=names, columns=params["mask"].columns)
        (path_root / fold_dir.name).mkdir(parents=True, exist_ok=True)
        with open(path_root / fold_dir.name / f"{a.resolution}_parameter_dict.pkl", "wb") as f:
            pickle.dump(out, f)
        n += 1
    print(f"CPNN prototypes projected: {n} folds")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="BRCA")
    ap.add_argument("--gmt", default="h.all.v2025.1.Hs.symbols.gmt", help="gene-set file under MOSPR_GENESETS")
    ap.add_argument("--min_members", type=int, default=10)
    ap.add_argument("--split", default="split_patient.pkl")
    ap.add_argument("--resolution", default="fine")
    ap.add_argument("--force", action="store_true", help="rebuild files that already exist")
    main(ap.parse_args())
