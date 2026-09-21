import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import paths as _p
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(_p.CODE))
import cohorts

import os as _os
RES = Path(_os.environ.get("MOSPR_RESULTS_ROOT",
                           str(_p.RESULTS)))
_spec = importlib.util.spec_from_file_location(
    "sw", str(_p.CODE / "design_blocks.py"))
sw = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sw)
h19 = sw.h19

QS = (4, 8, 16, 32, 48, 64, 96, 128)
LAMS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)


def right_singular(C, tol=1e-10):
    G = C @ C.T
    w, U = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    w, U = w[order], U[:, order]
    s = np.sqrt(np.maximum(w, 0.0))
    keep = s > (s[0] if len(s) else 0.0) * tol
    s, U = s[keep], U[:, keep]
    return (U.T @ C) / s[:, None]


def refit_fixed(X, Yz, fit, q, lam):
    C = Yz[fit] - Yz[fit].mean(axis=0, keepdims=True)
    U = right_singular(C)[:q]
    m = Ridge(alpha=lam, fit_intercept=True).fit(X[fit], (Yz @ U.T)[fit])
    return m.predict(X) @ U, {"alpha_q": lam, "rank": q}, {"W_q": m.coef_,
                                                          "b_q": m.intercept_, "U": U}


def fit_lowrank(X, Yz, tr, va, qs, lams):
    C = Yz[tr] - Yz[tr].mean(axis=0, keepdims=True)
    Ut = right_singular(C)
    best = (np.inf, None)
    for q in qs:
        U = Ut[:q]
        Z = Yz @ U.T
        for lam in lams:
            m = Ridge(alpha=lam, fit_intercept=True).fit(X[tr], Z[tr])
            mse = float(np.mean((Yz[va] - m.predict(X[va]) @ U) ** 2))
            if mse < best[0]:
                best = (mse, (lam, q, m, U))
    lam, q, m, U = best[1]
    return m.predict(X) @ U, {"alpha_q": lam, "rank": q}, {"W_q": m.coef_,
                                                           "b_q": m.intercept_, "U": U}


def main(a):
    cfg = cohorts.get(a.cohort)
    genes = pd.read_csv(cfg["processed"] / f"eval_genes_paper_{a.cohort}.txt",
                        header=None)[0].values


    out_tag = a.out_tag or f"{a.cohort}_paper"

    HP = {}
    if a.fit_on == "trainval":
        h = pd.read_csv(a.hp_from)
        for _, r in h.iterrows():
            HP[(r["cohort"], r["space"], int(r["fold"]))] = (int(r["rank"]), float(r["alpha_q"]))
        print(f"stage-1 hyperparameters {len(HP)} loaded: {a.hp_from}", flush=True)

    rows = []
    for fold in a.folds:
        cache = dict(np.load(
            RES / f"micro_cache_{a.cohort}_{a.cache_tag}_fold{fold}.npz",
            allow_pickle=True))
        blocks = sw.build_blocks(cache, a.K, a.d, with_Q=False)
        tr, va, te = cache["idx_tr"], cache["idx_va"], cache["idx_te"]
        te_slides = [str(s) for s in cache["slides"][te]]


        fit = np.concatenate([tr, va]) if a.fit_on == "trainval" else tr
        Xm, mmu, msd = sw.zfit(blocks["M"], fit, return_stats=True)
        Xs, smu, ssd = sw.zfit(blocks["S"], fit, return_stats=True)
        X = np.hstack([Xm, Xs])
        assert X.shape[1] == (a.K + 1) * a.d, (
            f"X width {X.shape[1]} != (K+1)*d {(a.K+1)*a.d}")

        for space in a.spaces:
            Y = cache[f"Y_{space}"].astype(np.float64)
            ymu, ysd = Y[fit].mean(axis=0), Y[fit].std(axis=0)
            ysd[ysd == 0] = 1.0
            Yz = (Y - ymu) / ysd
            if a.fit_on == "trainval":
                key = (a.cohort, space, fold)
                assert key in HP, f"missing stage-1 hyperparameters: {key} ({a.hp_from})"
                q, lam = HP[key]
                predz, p, W = refit_fixed(X, Yz, fit, q, lam)
            else:
                predz, p, W = fit_lowrank(X, Yz, tr, va, a.qs, a.lams)
            unz = lambda idx: predz[idx] * ysd + ymu
            r = {"cohort": a.cohort, "space": space, "fold": fold, "K": a.K,
                 "d": a.d, "primary": True, "model": "prototype-free",
                 "fit_on": a.fit_on, "n_x": X.shape[1], **p,


                 "val_SCC": h19.metrics(Y[va], unz(va))[1],
                 "test_SCC": h19.metrics(Y[te], unz(te))[1],
                 "test_PCC": h19.metrics(Y[te], unz(te))[0]}
            rows.append(r)
            print(f"[{a.cohort} f{fold} {space:>7s}] val {r['val_SCC']:.4f} "
                  f"test {r['test_SCC']:.4f} (q={p['rank']}, lam={p['alpha_q']:g})",
                  flush=True)

            np.savez_compressed(
                RES / f"pred_cohort_{out_tag}_{space}_fold{fold}.npz",
                slides=np.array(te_slides), pred=unz(te).astype(np.float32))
            np.savez_compressed(
                RES / f"weights_cohort_{out_tag}_{space}_fold{fold}.npz",
                genes=np.array(genes), K=a.K, d=a.d,
                ymu=ymu, ysd=ysd,
                xmu=np.concatenate([mmu, smu]), xsd=np.concatenate([msd, ssd]),
                n_M=blocks["M"].shape[1], n_S=blocks["S"].shape[1], **W)

    df = pd.DataFrame(rows)
    df.to_csv(RES / f"cohort_{out_tag}_ours.csv", index=False)
    print(f"\nwritten: cohort_{out_tag}_ours.csv")
    print(df.groupby("space")[["val_SCC", "test_SCC", "test_PCC"]].agg(
        ["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--qs", type=int, nargs="+", default=list(QS))
    ap.add_argument("--lams", type=float, nargs="+", default=list(LAMS))
    ap.add_argument("--cache_tag", default="paper_p512")
    ap.add_argument("--fit_on", choices=["train", "trainval"], default="train",
                    help='final fit set; with trainval, (q, lambda) are read from --hp_from')
    ap.add_argument("--hp_from", default="",
                    help="stage-1 result CSV, e.g. cohort_{cohort}_psplit_ours.csv")
    ap.add_argument("--out_tag", default="",
                    help='output tag; default "{cohort}_paper", patient split uses "{cohort}_psplit"')
    ap.add_argument("--spaces", nargs="+", default=["gene", "pathway"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    main(ap.parse_args())
