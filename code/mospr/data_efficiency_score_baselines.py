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

sys.path.insert(0, str(_p.CODE))
from spatial_proteome import pathways

ROOT = _p.ROOT
CPOUT = _p.CPNN_REPO / "outputs"
DS = _p.DATA
GMT = _p.GENESETS / "h.all.v2025.1.Hs.symbols.gmt"


def _proc(cohort):
    import sys as _sys
    _sys.path.insert(0, str(_p.CODE))
    import cohorts as _c
    return pathlib.Path(_c.get(cohort)["processed"])


MODELS = [("tRNAformer", "tRNAsformer%ComparisonTrainerts", False),
          ("SEQUOIA VIS", "SEQUOIA_VIS%ComparisonTrainerts", False),
          ("MOSBY", "SumExpModel_MOSBY%ts", False),
          ("AbMIL", "AbMIL%ts", False),
          ("MambaMIL", "MambaMILvanira_stop_sampling%ts", False),
          ("CPNN", "ProtoSum_1reg_mse_reg_1e3%DeconvExptsfine", True),
          ("2DMamba", "MambaMIL_2D_stop_sampling%Mamba2DTrainerts", False)]
FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.0]


def logcpm(X):
    X = np.clip(np.asarray(X, dtype=np.float64), 0, None)
    s = X.sum(axis=1, keepdims=True)
    return np.log1p(np.divide(X, s, out=np.zeros_like(X), where=s > 0) * 1e4)


def corr_cols(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    n = (a * b).sum(0); d = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    return np.divide(n, d, out=np.full(n.shape, np.nan), where=d > 0)


def set_scores(X, members):
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    Z = (X - mu) / np.where(sd > 0, sd, 1.0)
    return np.stack([Z[:, i].mean(axis=1) for i in members], axis=1)


def main(a):
    genes = pd.read_csv(_proc(a.cohort) / f"eval_genes_paper_{a.cohort}.txt",
                        header=None)[0].values
    idx = pathways.member_index(pathways.parse_gmt(GMT), genes, min_members=10)
    members = [v for v in idx.values() if len(v) <= 500]
    print(f"{a.cohort}: Hallmark sets {len(members)}", flush=True)
    pair = DS / f"{a.cohort}-paper-digital_slide/sample_pair_feature_conch"

    obs_cache, rows = {}, []
    for label, pat, is_rate in MODELS:
        for frac in FRACTIONS:
            sfx = "_pstvf" if frac >= 1.0 else f"_pstvf_f{int(round(frac*100)):02d}"
            for fold in range(4):
                run = pat.replace("%", str(fold))
                p = CPOUT / str(fold) / "prediction" / f"{a.cohort}-paper-{run}{sfx}.npy"
                if not p.exists():
                    continue
                d = np.load(p, allow_pickle=True).item()
                sl = [str(x) if isinstance(x, str) else str(x[0]) for x in d["slide_name"]]
                key = (fold, len(sl))
                if key not in obs_cache:
                    obs_cache[key] = logcpm(np.stack(
                        [h5py.File(pair / f"{s}.h5", "r")["tpm"][:] for s in sl]))
                Y = obs_cache[key]
                P = np.stack(d["preds"])
                if is_rate: P = logcpm(P)
                so, sp = set_scores(Y, members), set_scores(P, members)
                rows.append({
                    "model": label, "frac": frac, "fold": fold, "n_test": len(sl),
                    "scc": float(np.nanmean(corr_cols(rankdata(Y, axis=0), rankdata(P, axis=0)))),
                    "pcc": float(np.nanmean(corr_cols(Y, P))),
                    "hall_scc": float(np.nanmedian(corr_cols(rankdata(so, axis=0),
                                                             rankdata(sp, axis=0)))),
                    "hall_pcc": float(np.nanmedian(corr_cols(so, sp)))})
                print(f"[{label} frac={frac:<4} f{fold}] gene {rows[-1]['scc']:.4f}  "
                      f"hallmark {rows[-1]['hall_scc']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    out = _p.RESULTS / f"deff_baselines_{a.cohort}_fpsplit_tv.csv"
    df.to_csv(out, index=False)
    print(f"\nwritten: {out} ({len(df)} rows)\n")
    if len(df):
        print(df.pivot_table(index="frac", columns="model",
                             values=["scc", "hall_scc"]).round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="BRCA")
    main(ap.parse_args())
