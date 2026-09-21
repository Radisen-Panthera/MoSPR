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
from scipy.stats import rankdata

ROOT = _p.ROOT
CPOUT = _p.CPNN_REPO / "outputs"
DS = _p.DATA
SRC = _p.TABLES / "tables/final/alpha0/source"
import os as _o
OUT = Path(_o.environ["MOSPR_WORK_OUT"]) / "tables" if _o.environ.get("MOSPR_WORK_OUT") else _p.TABLES / "tables/refit"


METHODS = [("CPNN", r"ProtoSum_1reg_mse_reg_1e3(\d)DeconvExptsfine"),
           ("AbMIL", r"AbMIL(\d)ts"),
           ("AbMIL mean-pool", r"AbMIL_mean(\d)ts"),
           ("AbMIL max-pool", r"AbMIL_max(\d)ts"),
           ("ILRA", r"ILRA(\d)ts"),
           ("S4Model", r"S4Model_stop_sampling(\d)ts"),
           ("MambaMIL", r"MambaMILvanira_stop_sampling(\d)ts"),
           ("SRMambaMIL", r"SRMambaMIL_stop_sampling(\d)ts"),
           ("2DMamba", r"MambaMIL_2D_stop_sampling(\d)Mamba2DTrainerts"),
           ("AbReg", r"AbRegMIL(\d)ts"),
           ("HE2RNA", r"HE2RNA(\d)ComparisonTrainerts"),
           ("MOSBY", r"SumExpModel_MOSBY(\d)ts"),
           ("SEQUOIA VIS", r"SEQUOIA_VIS(\d)ComparisonTrainerts"),
           ("tRNAformer", r"tRNAsformer(\d)ComparisonTrainerts")]


import os as _os_il
if _os_il.environ.get("MOSPR_ILSE") == "1":
    METHODS = [(n, t.replace("AbMIL_max", "AbMIL_ilse_max").replace("AbMIL_mean", "AbMIL_ilse_mean"), *r)
             for n, t, *r in METHODS]
STAGES = {"stage1_trainval_holdout": "_psf", "stage2_refit_trainval": "_pstvf"}


def logcpm(x):
    x = np.asarray(x, dtype=np.float64)
    t = x.sum(axis=-1, keepdims=True)
    return np.log1p(np.divide(x, t, out=np.zeros_like(x), where=t > 0) * 1e4)


def read_txt(p):
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    v = [float(x) for x in lines[-1].split(",")]
    return {"PCC": v[2], "SCC": v[3]}


def cpnn_rescored(cohort, fold, sfx):
    pat = re.compile(METHODS[0][1] + re.escape(sfx) + "$")
    hits = [f for f in (CPOUT / str(fold) / "prediction").glob(f"{cohort}-paper-ProtoSum*.npy")
            if pat.match(f.stem.replace(f"{cohort}-paper-", ""))]
    if not hits:
        return None
    d = np.load(hits[0], allow_pickle=True).item()
    preds = np.stack(d["preds"]).astype(np.float64)
    names = [str(x) for x in d["slide_name"]]
    ds = DS / f"{cohort}-paper-digital_slide/sample_pair_feature_conch"
    tpm = np.stack([h5py.File(ds / f"{s}.h5", "r")["tpm"][:] for s in names])
    if tpm.shape != preds.shape:
        return None
    y, p = logcpm(tpm), logcpm(preds)


    return {"PCC": float(np.nanmean(_corr_cols(p, y))),
            "SCC": float(np.nanmean(_corr_cols(rankdata(p, axis=0), rankdata(y, axis=0))))}


def _corr_cols(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    n = (a * b).sum(0); d = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    return np.divide(n, d, out=np.full(n.shape, np.nan), where=d > 0)


def mospr_rows(cohort):
    out = []
    for stage, f in [("stage1_trainval_holdout", f"cohort_{cohort}_fpsplit_ours.csv"),
                     ("stage2_refit_trainval", f"cohort_{cohort}_fpsplit_tv_ours.csv")]:
        d = pd.read_csv(SRC / f)
        d = d[d.space == "gene"]
        for _, r in d.iterrows():
            out.append({"cohort": cohort, "model": "MoSPR (ours)", "stage": stage,
                        "fold": int(r.fold), "PCC": r.test_PCC, "SCC": r.test_SCC})
    return out


def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cohort in a.cohorts:
        for name, pat in METHODS:
            for stage, sfx in STAGES.items():
                rx = re.compile(pat + re.escape(sfx) + r"\.txt$")
                for fold in range(4):
                    if name == "CPNN":
                        m = cpnn_rescored(cohort, fold, sfx)
                    else:
                        hits = [p for p in (CPOUT / str(fold)).glob(f"{cohort}-paper-*{sfx}.txt")
                                if rx.match(p.name.replace(f"{cohort}-paper-", ""))]
                        m = read_txt(hits[0]) if hits else None
                    if m:
                        rows.append({"cohort": cohort, "model": name, "stage": stage,
                                     "fold": fold, **m})
        rows += mospr_rows(cohort)

    long = pd.DataFrame(rows)
    long.to_csv(OUT / "refit_per_fold_gene.csv", index=False)


    piv = long.pivot_table(index=["cohort", "model", "fold"], columns="stage",
                           values=["SCC", "PCC"])
    res = []
    for (c, mdl), g in piv.groupby(level=[0, 1]):
        row = {"cohort": c, "model": mdl, "n_fold": len(g)}
        for met in ("SCC", "PCC"):
            s1 = g[(met, "stage1_trainval_holdout")].values
            s2 = g[(met, "stage2_refit_trainval")].values
            if np.isnan(s1).all() or np.isnan(s2).all():
                continue
            d = s2 - s1
            row |= {f"{met}_stage1": np.nanmean(s1), f"{met}_stage2": np.nanmean(s2),
                    f"{met}_delta": np.nanmean(d),
                    f"{met}_folds_up": int((d > 0).sum()),
                    f"{met}_verdict": ("same" if abs(np.nanmean(d)) < a.tie
                                       else ("up" if np.nanmean(d) > 0 else "down"))}
        res.append(row)
    summ = pd.DataFrame(res).sort_values(["cohort", "SCC_delta"], ascending=[True, False])
    summ.to_csv(OUT / "refit_summary_gene.csv", index=False)

    pd.set_option("display.width", 200)
    for c in a.cohorts:
        s = summ[summ.cohort == c]
        if not len(s):
            continue
        print(f"\n===== {c} · gene SCC (stage 1 -> stage 2 refit) =====")
        print(s[["model", "SCC_stage1", "SCC_stage2", "SCC_delta", "SCC_folds_up",
                 "SCC_verdict", "n_fold"]].round(4).to_string(index=False))
    print(f"\nwritten: {OUT/'refit_summary_gene.csv'} · {OUT/'refit_per_fold_gene.csv'}")
    up = (summ.SCC_delta > a.tie).sum(); dn = (summ.SCC_delta < -a.tie).sum()
    print(f"summary: up {up}, down {dn}, unchanged(|d|<{a.tie}) {len(summ)-up-dn} of {len(summ)} model x cohort")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["BRCA", "KIRC", "LUAD"])
    ap.add_argument("--tie", type=float, default=0.002,
                    help="changes smaller than this count as unchanged (e.g. 0.333 -> 0.331)")
    main(ap.parse_args())
