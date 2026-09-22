import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import paths as _p
import argparse
import pathlib
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = _p.ROOT
sys.path.insert(0, str(_p.CODE))
import cohorts
from spatial_proteome import pathways

CPOUT = _p.CPNN_REPO / "outputs"
RES = _p.RESULTS
DS = _p.DATA
GENESETS = _p.GENESETS
COLLECTIONS = {"hallmark": "h.all.v2025.1.Hs.symbols.gmt",
               "kegg": "c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt"}

import os as _o
OUT = pathlib.Path(_o.environ.get("MOSPR_WORK_OUT", _p.TABLES)) / "tables/robustness"

MODELS = [("Max", "AbMIL_ilse_max%ts", False),
          ("Mean", "AbMIL_ilse_mean%ts", False),
          ("AbMIL", "AbMIL%ts", False),
          ("HE2RNA", "HE2RNA%ComparisonTrainerts", False),
          ("AbReg", "AbRegMIL%ts", False),
          ("tRNAformer", "tRNAsformer%ComparisonTrainerts", False),
          ("ILRA", "ILRA%ts", False),
          ("S4MIL", "S4Model_stop_sampling%ts", False),
          ("MambaMIL", "MambaMILvanira_stop_sampling%ts", False),
          ("SRMambaMIL", "SRMambaMIL_stop_sampling%ts", False),
          ("MOSBY", "SumExpModel_MOSBY%ts", False),
          ("SEQUOIA VIS", "SEQUOIA_VIS%ComparisonTrainerts", False),
          ("2DMamba", "MambaMIL_2D_stop_sampling%Mamba2DTrainerts", False),
          ("CPNN", "ProtoSum_1reg_mse_reg_1e3%DeconvExptsfine", True)]

SFX = "_pstvf"


def logcpm(X):
    X = np.clip(np.asarray(X, dtype=np.float64), 0, None)
    s = X.sum(axis=1, keepdims=True)
    return np.log1p(np.divide(X, s, out=np.zeros_like(X), where=s > 0) * 1e4)


def corr_cols(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    n = (a * b).sum(0)
    d = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    return np.divide(n, d, out=np.full(n.shape, np.nan), where=d > 0)


def score_matrix(X, genes, slides, gmt, threads, method):
    import gseapy
    df = pd.DataFrame(np.asarray(X).T, index=genes, columns=slides)
    if method == "ssgsea":
        r = gseapy.ssgsea(data=df, gene_sets=gmt, outdir=None,
                          sample_norm_method="rank", min_size=10, max_size=500,
                          threads=threads, no_plot=True, verbose=False)
        val = "NES"
    elif method == "gsva":
        r = gseapy.gsva(data=df, gene_sets=gmt, outdir=None,
                        min_size=10, max_size=500, threads=threads, verbose=False)
        val = "ES"
    else:
        raise ValueError(method)
    piv = r.res2d.pivot(index="Name", columns="Term", values=val).astype(float)
    return piv.reindex(slides)


def main(a):
    cfg = cohorts.get(a.cohort)
    genes = pd.read_csv(pathlib.Path(cfg["processed"]) /
                        f"eval_genes_paper_{a.cohort}.txt", header=None)[0].values
    gmt = pathways.parse_gmt(GENESETS / COLLECTIONS[a.collection])
    pair = DS / f"{a.cohort}-paper-digital_slide/sample_pair_feature_conch"
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for method in a.methods:
        for fold in range(4):
            q = RES / f"pred_cohort_{a.cohort}_fpsplit_tv_gene_fold{fold}.npz"
            z = np.load(q, allow_pickle=True)
            slides = [str(s) for s in z["slides"]]
            Y = logcpm(np.stack([h5py.File(pair / f"{s}.h5", "r")["tpm"][:]
                                 for s in slides]))
            SY = score_matrix(Y, genes, slides, gmt, a.threads, method)
            print(f"[{a.cohort} {method} f{fold}] observed {SY.shape}", flush=True)

            def score(P, name):
                SP = score_matrix(P, genes, slides, gmt, a.threads, method)
                cols = SY.columns.intersection(SP.columns)
                v = corr_cols(rankdata(SY[cols].values, axis=0),
                              rankdata(SP[cols].values, axis=0))
                rows.append({"cohort": a.cohort, "collection": a.collection,
                             "method": method, "model": name,
                             "fold": fold, "n_sets": len(cols),
                             "scc": float(np.nanmedian(v))})

            for name, tmpl, is_rate in MODELS:
                if _o.environ.get("MOSPR_ONLY") and name not in _o.environ["MOSPR_ONLY"].split():
                    continue
                p = (CPOUT / str(fold) / "prediction" /
                     f"{a.cohort}-paper-{tmpl.replace('%', str(fold))}{SFX}.npy")
                if not p.exists():
                    continue
                d = np.load(p, allow_pickle=True).item()
                sl = [str(x) if isinstance(x, str) else str(x[0]) for x in d["slide_name"]]
                P = np.stack(d["preds"]).astype(np.float64)
                if sl != slides or P.shape != Y.shape:
                    print(f"    {name}: slide/shape mismatch - skipped", flush=True)
                    continue
                score(logcpm(P) if is_rate else P, name)

            if not _o.environ.get("MOSPR_ONLY"):
                score(np.asarray(z["pred"], dtype=np.float64), "MoSPR (ours)")

    df = pd.DataFrame(rows)
    tag = "" if a.collection == "hallmark" else f"_{a.collection}"
    out = OUT / f"pathway_robustness_{a.cohort}{tag}.csv"
    df.to_csv(out, index=False)
    print(f"\nwritten: {out} ({len(df)} rows)")
    print(df.pivot_table(index="model", columns="method", values="scc")
          .round(4).sort_values(a.methods[0], ascending=False).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="BRCA")
    ap.add_argument("--methods", nargs="+", default=["ssgsea", "gsva"])
    ap.add_argument("--collection", default="hallmark", choices=list(COLLECTIONS))
    ap.add_argument("--threads", type=int, default=6)
    main(ap.parse_args())
