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
               "gobp": "c5.go.bp.v2025.1.Hs.symbols.gmt",
               "kegg": "c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt"}


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

STAGES = {"stage1_trainval_holdout": ("_psf", "fpsplit"),
          "stage2_refit_trainval": ("_pstvf", "fpsplit_tv")}


def logcpm(X):
    X = np.clip(np.asarray(X, dtype=np.float64), 0, None)
    s = X.sum(axis=1, keepdims=True)
    return np.log1p(np.divide(X, s, out=np.zeros_like(X), where=s > 0) * 1e4)


def corr_cols(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    n = (a * b).sum(0)
    d = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    return np.divide(n, d, out=np.full(n.shape, np.nan), where=d > 0)


def set_scores(X, members):
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    Z = (X - mu) / np.where(sd > 0, sd, 1.0)
    return np.stack([Z[:, i].mean(axis=1) for i in members], axis=1)


def score(Y, P, members):
    so, sp = set_scores(Y, members), set_scores(P, members)
    return float(np.nanmedian(corr_cols(rankdata(so, axis=0), rankdata(sp, axis=0))))


def main(a):
    cfg = cohorts.get(a.cohort)
    genes = pd.read_csv(pathlib.Path(cfg["processed"]) /
                        f"eval_genes_paper_{a.cohort}.txt", header=None)[0].values
    sets = {}
    for coll, fn in COLLECTIONS.items():
        idx = pathways.member_index(pathways.parse_gmt(GENESETS / fn), genes,
                                    min_members=10)
        sets[coll] = [v for v in idx.values() if len(v) <= 500]
        print(f"{coll}: {len(sets[coll])} sets", flush=True)
    pair = DS / f"{a.cohort}-paper-digital_slide/sample_pair_feature_conch"

    obs, rows = {}, []

    def observed(slides, fold):
        key = (fold, len(slides))
        if key not in obs:
            obs[key] = logcpm(np.stack(
                [h5py.File(pair / f"{s}.h5", "r")["tpm"][:] for s in slides]))
        return obs[key]

    for stage, (sfx, mtag) in STAGES.items():
        for name, tmpl, is_rate in MODELS:
            for fold in range(4):
                p = (CPOUT / str(fold) / "prediction" /
                     f"{a.cohort}-paper-{tmpl.replace('%', str(fold))}{sfx}.npy")
                if not p.exists():
                    continue
                d = np.load(p, allow_pickle=True).item()
                sl = [str(x) if isinstance(x, str) else str(x[0]) for x in d["slide_name"]]
                Y = observed(sl, fold)
                P = np.stack(d["preds"]).astype(np.float64)
                if P.shape != Y.shape:
                    print(f"  skipped {name} {stage} f{fold}: shape mismatch", flush=True)
                    continue
                if is_rate:
                    P = logcpm(P)
                rows.append({"cohort": a.cohort, "model": name, "stage": stage,
                             "fold": fold,
                             **{f"{c}_scc": score(Y, P, sets[c]) for c in COLLECTIONS}})

        for fold in range(4):
            q = RES / f"pred_cohort_{a.cohort}_{mtag}_gene_fold{fold}.npz"
            if not q.exists():
                print(f"  MoSPR {stage} f{fold} no prediction: {q.name}", flush=True)
                continue
            z = np.load(q, allow_pickle=True)
            sl = [str(s) for s in z["slides"]]
            Y = observed(sl, fold)
            P = np.asarray(z["pred"], dtype=np.float64)
            if P.shape != Y.shape:
                print(f"  skipped MoSPR {stage} f{fold}: shape mismatch", flush=True)
                continue
            rows.append({"cohort": a.cohort, "model": "MoSPR (ours)", "stage": stage,
                         "fold": fold,
                         **{f"{c}_scc": score(Y, P, sets[c]) for c in COLLECTIONS}})
        print(f"[{a.cohort}] {stage} done ({len(rows)} rows so far)", flush=True)

    df = pd.DataFrame(rows)
    out = _p.TABLES / f"tables/refit/refit_pathway_{a.cohort}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwritten: {out} ({len(df)} rows, models {df.model.nunique()})")
    print(df.pivot_table(index="model", columns="stage", values="hallmark_scc")
          .round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="BRCA")
    main(ap.parse_args())
