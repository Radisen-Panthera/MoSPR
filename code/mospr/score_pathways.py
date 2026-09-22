import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import paths as _p
import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import rankdata

sys.path.insert(0, str(_p.CODE))
import cohorts
from spatial_proteome import pathways

import os as _os
_R = _p.ROOT
RES = _p.RESULTS
DS_ROOT = _p.DATA
SUFFIX = _os.environ.get("MOSPR_RUN_SUFFIX", "")
CPNN_OUT = _p.CPNN_REPO / "outputs"
GENESETS = _p.GENESETS

COLLECTIONS = {
    "hallmark": "h.all.v2025.1.Hs.symbols.gmt",
    "gobp": "c5.go.bp.v2025.1.Hs.symbols.gmt",
    "kegg": "c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt",
}


METHODS = [
    ("Max", r"AbMIL_ilse_max(\d)ts$"),
    ("Mean", r"AbMIL_ilse_mean(\d)ts$"),
    ("AbMIL", r"AbMIL(\d)ts$"),
    ("ILRA", r"ILRA(\d)ts$"),
    ("S4MIL", r"S4Model_stop_sampling(\d)ts$"),
    ("MambaMIL", r"MambaMILvanira_stop_sampling(\d)ts$"),
    ("SRMambaMIL", r"SRMambaMIL_stop_sampling(\d)ts$"),

    ("2DMamba", r"MambaMIL_2D_stop_sampling(\d)Mamba2DTrainerts$"),
    ("AbReg", r"AbRegMIL(\d)ts$"),
    ("HE2RNA", r"HE2RNA(\d)ComparisonTrainerts$"),
    ("MOSBY", r"SumExpModel_MOSBY(\d)ts$"),
    ("SEQUOIA VIS", r"SEQUOIA_VIS(\d)ComparisonTrainerts$"),
    ("tRNAformer", r"tRNAsformer(\d)ComparisonTrainerts$"),
    ("CPNN", r"ProtoSum_1reg_mse_reg_1e3(\d)DeconvExptsfine$"),


]
METHODS = [(n, p[:-1] + re.escape(SUFFIX) + "$") for n, p in METHODS]


def log_cpm(X):
    X = np.asarray(X, dtype=np.float64)
    s = X.sum(axis=1, keepdims=True)
    return np.log1p(np.divide(X, s, out=np.zeros_like(X), where=s > 0) * 1e4)


def to_common(X, is_log):
    return X if is_log else log_cpm(X)


def membership(members, n_genes):
    rows, cols, vals = [], [], []
    for j, idx in enumerate(members):
        rows.extend(idx.tolist()); cols.extend([j] * len(idx))
        vals.extend([1.0 / len(idx)] * len(idx))
    return csr_matrix((vals, (rows, cols)), shape=(n_genes, len(members)))


def pathway_scores(X, M):
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    return ((X - mu) / np.where(sd > 0, sd, 1.0)) @ M


def corr_cols(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    num = (a * b).sum(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def load_baseline(cohort, fold, pat, slides):
    tag = f"{cohort}-paper"
    d = CPNN_OUT / str(fold) / "prediction"
    hits = [f for f in d.glob(f"{tag}-*.npy") if re.match(pat, f.stem.replace(f"{tag}-", ""))]
    if not hits:
        return None
    o = np.load(hits[0], allow_pickle=True).item()
    names = [s if isinstance(s, str) else s[0] for s in o["slide_name"]]
    order = {s: i for i, s in enumerate(names)}
    if not all(s in order for s in slides):
        return None
    return np.stack(o["preds"])[[order[s] for s in slides]]


def main(a):
    cfg = dict(cohorts.get(a.cohort))
    cfg["gene_root"] = DS_ROOT / f"{a.cohort}-paper-digital_slide"
    genes = pd.read_csv(cfg["processed"] / f"eval_genes_paper_{a.cohort}.txt",
                        header=None)[0].values
    pair = cfg["gene_root"] / "sample_pair_feature_conch"

    sets = {}
    for coll, fn in COLLECTIONS.items():
        idx = pathways.member_index(pathways.parse_gmt(GENESETS / fn), genes,
                                    min_members=a.min_members)
        idx = {n: i for n, i in idx.items() if len(i) <= a.max_members}
        sets[coll] = (list(idx), membership(list(idx.values()), len(genes)))
        print(f"{a.cohort}/{coll}: gene sets {len(idx)}", flush=True)

    rows = []
    for fold in a.folds:
        f = RES / f"pred_cohort_{a.cohort}_{a.mospr_tag}_gene_fold{fold}.npz"
        if not f.exists():
            print(f"  fold {fold}: MoSPR no prediction - skipped"); continue
        z = np.load(f, allow_pickle=True)
        slides = [str(s) for s in z["slides"]]
        obs = log_cpm(np.stack([h5py.File(pair / f"{s}.h5", "r")["tpm"][:] for s in slides]))

        preds = {"MoSPR (ours)": to_common(z["pred"], is_log=True)}
        for name, pat in METHODS:
            x = load_baseline(a.cohort, fold, pat, slides)
            if x is None:
                print(f"  fold {fold}: {name} no prediction"); continue
            preds[name] = to_common(x, is_log=(name != "CPNN"))

        for coll, (names, M) in sets.items():
            S_obs_raw = pathway_scores(obs, M)
            S_obs_raw_rank = rankdata(S_obs_raw, axis=0)
            for name, P in preds.items():
                s = pathway_scores(P, M)
                scc = corr_cols(S_obs_raw_rank, rankdata(s, axis=0))
                pcc = corr_cols(S_obs_raw, s)


                rows.append({"cohort": a.cohort, "collection": coll, "fold": fold,
                             "model": name, "n_sets": len(names),
                             "SCC_median": float(np.nanmedian(scc)),
                             "SCC_mean": float(np.nanmean(scc)),
                             "PCC_median": float(np.nanmedian(pcc)),
                             "PCC_mean": float(np.nanmean(pcc))})
        print(f"[{a.cohort} f{fold}] models {len(preds)} scored", flush=True)

    df = pd.DataFrame(rows)
    out = RES / f"pathway_all_{a.cohort}_{a.mospr_tag}.csv"
    df.to_csv(out, index=False)
    print(f"\nwritten: {out}")
    print(df.pivot_table(index="model", columns="collection",
                         values="SCC_median", aggfunc="mean").round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--min_members", type=int, default=10)
    ap.add_argument("--max_members", type=int, default=500)
    ap.add_argument("--mospr_tag", default="paper",
                    help='MoSPR output tag: "paper" (authors split) or "psplit" (patient split).'
                         ' Baselines are selected with MOSPR_RUN_SUFFIX.')
    main(ap.parse_args())
