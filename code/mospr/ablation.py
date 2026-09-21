import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import paths as _p
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge

sys.path.insert(0, str(_p.CODE))
import cohorts
from spatial_proteome import pathways

RES = Path(str(_p.RESULTS))
GMT = _p.GENESETS / "h.all.v2025.1.Hs.symbols.gmt"
_spec = importlib.util.spec_from_file_location(
    "sw", str(_p.CODE / "design_blocks.py"))
sw = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sw)
h19 = sw.h19

QS = (4, 8, 16, 32, 48, 64, 96, 128)
LAMS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)


def right_singular(C, tol=1e-10):
    w, U = np.linalg.eigh(C @ C.T)
    o = np.argsort(w)[::-1]
    s = np.sqrt(np.maximum(w[o], 0.0)); U = U[:, o]
    keep = s > (s[0] if len(s) else 0.0) * tol
    return (U[:, keep].T @ C) / s[keep][:, None]


def ridge_direct(X, Yz, tr, va, te, lams):
    xm, ym = X[tr].mean(0), Yz[tr].mean(0)
    Xc, Yc = X - xm, Yz - ym
    lam_, Q = np.linalg.eigh(Xc[tr] @ Xc[tr].T)
    Gva, Gte = Xc[va] @ Xc[tr].T, Xc[te] @ Xc[tr].T
    QtY = Q.T @ Yc[tr]
    best = (np.inf, None, None)
    for a in lams:
        A = Q @ (QtY / (lam_ + a)[:, None])
        mse = float(np.mean((Yz[va] - (Gva @ A + ym)) ** 2))
        if mse < best[0]:
            best = (mse, a, Gte @ A + ym)
    return best[2], {"lam": best[1], "q": np.nan}


def ridge_lowrank(X, Yz, tr, va, te, qs, lams):
    C = Yz[tr] - Yz[tr].mean(0, keepdims=True)
    Ut = right_singular(C)
    best = (np.inf, None)
    for q in qs:
        if q > len(Ut):
            continue
        U = Ut[:q]; Z = Yz @ U.T
        for lam in lams:
            m = Ridge(alpha=lam, fit_intercept=True).fit(X[tr], Z[tr])
            mse = float(np.mean((Yz[va] - m.predict(X[va]) @ U) ** 2))
            if mse < best[0]:
                best = (mse, (lam, q, m, U))
    lam, q, m, U = best[1]
    return m.predict(X[te]) @ U, {"lam": lam, "q": q}


def ridge_direct_fixed(X, Yz, fit, te, lam):
    xm, ym = X[fit].mean(0), Yz[fit].mean(0)
    Xc = X - xm
    lam_, Q = np.linalg.eigh(Xc[fit] @ Xc[fit].T)
    A = Q @ ((Q.T @ (Yz[fit] - ym)) / (lam_ + lam)[:, None])
    return (Xc[te] @ Xc[fit].T) @ A + ym, {"lam": lam, "q": np.nan}


def ridge_lowrank_fixed(X, Yz, fit, te, q, lam):
    U = right_singular(Yz[fit] - Yz[fit].mean(0, keepdims=True))[:q]
    m = Ridge(alpha=lam, fit_intercept=True).fit(X[fit], (Yz @ U.T)[fit])
    return m.predict(X[te]) @ U, {"lam": lam, "q": q}


def set_scores(X, members):
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    Z = (X - mu) / np.where(sd > 0, sd, 1.0)
    return np.stack([Z[:, i].mean(axis=1) for i in members], axis=1)


def corr_cols(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    n = (a * b).sum(0); d = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    return np.divide(n, d, out=np.full(n.shape, np.nan), where=d > 0)


def hallmark_scc(Y_true, Y_pred, members):
    o = rankdata(set_scores(Y_true, members), axis=0)
    p = rankdata(set_scores(Y_pred, members), axis=0)
    return float(np.nanmedian(corr_cols(o, p)))


def main(a):
    cfg = cohorts.get(a.cohort)
    genes = pd.read_csv(cfg["processed"] / f"eval_genes_paper_{a.cohort}.txt",
                        header=None)[0].values
    idx = pathways.member_index(pathways.parse_gmt(GMT), genes, min_members=10)
    members = [v for v in idx.values() if len(v) <= 500]
    print(f"{a.cohort}: genes {len(genes)}, Hallmark sets {len(members)}, "
          f"cache {a.cache_tag}", flush=True)

    hp = None
    if a.fit_on == "trainval":
        hp = pd.read_csv(a.hp_from)
        print(f"stage-1 hyperparameters: {a.hp_from}", flush=True)

    rng = np.random.default_rng(a.seed)
    rows = []
    for fold in a.folds:
        cache = dict(np.load(
            RES / f"micro_cache_{a.cohort}_{a.cache_tag}_fold{fold}.npz",
            allow_pickle=True))
        blocks = sw.build_blocks(cache, a.K, a.d, with_Q=False)
        tr, va, te = cache["idx_tr"], cache["idx_va"], cache["idx_te"]


        fit = np.concatenate([tr, va]) if a.fit_on == "trainval" else tr
        Y = cache["Y_gene"].astype(np.float64)
        ymu, ysd = Y[fit].mean(0), Y[fit].std(0); ysd[ysd == 0] = 1.0
        Yz = (Y - ymu) / ysd

        Xm = sw.zfit(blocks["M"], fit)
        Xms = np.hstack([Xm, sw.zfit(blocks["S"], fit)])

        Ssh = blocks["S"][rng.permutation(len(blocks["S"]))]
        Xsh = np.hstack([Xm, sw.zfit(Ssh, fit)])

        XOF = {"Global Direct": (Xm, 0), "Global Low-Rank": (Xm, 1),
               "Spatial Direct": (Xms, 0), "Shuffled-State Low-Rank": (Xsh, 1),
               "Spatial Low-Rank": (Xms, 1)}
        VAR = []
        for name, (Xv, low) in XOF.items():
            if a.fit_on == "trainval":
                r = hp[(hp.fold == fold) & (hp.variant == name)].iloc[0]
                VAR.append((name, (lambda X=Xv, l=low, r=r:
                    ridge_lowrank_fixed(X, Yz, fit, te, int(r["q"]), float(r["lam"])) if l
                    else ridge_direct_fixed(X, Yz, fit, te, float(r["lam"])))))
            else:
                VAR.append((name, (lambda X=Xv, l=low:
                    ridge_lowrank(X, Yz, tr, va, te, a.qs, a.lams) if l
                    else ridge_direct(X, Yz, tr, va, te, a.lams))))
        for name, fn in VAR:
            pz, p = fn()
            pred = pz * ysd + ymu
            g = h19.metrics(Y[te], pred)
            rows.append({"cohort": a.cohort, "fold": fold, "variant": name,
                         "gene_PCC": g[0], "gene_SCC": g[1],
                         "hall_SCC": hallmark_scc(Y[te], pred, members), **p})
            print(f"[{a.cohort} f{fold}] {name:<24s} gene {g[1]:.4f}  "
                  f"hallmark {rows[-1]['hall_SCC']:.4f}  (q={p['q']}, lam={p['lam']:g})",
                  flush=True)

    df = pd.DataFrame(rows)
    out = RES / f"ablation_v2_{a.cohort}_{a.out_tag or a.cache_tag}.csv"
    df.to_csv(out, index=False)
    order = list(XOF)
    print(f"\nwritten: {out}\n")
    print(df.groupby("variant")[["gene_PCC", "gene_SCC", "hall_SCC"]].mean()
          .loc[order].round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="BRCA")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--qs", type=int, nargs="+", default=list(QS))
    ap.add_argument("--lams", type=float, nargs="+", default=list(LAMS))
    ap.add_argument("--cache_tag", default="paper_p512")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit_on", choices=["train", "trainval"], default="train")
    ap.add_argument("--hp_from", help="stage-1 csv, required when fit_on=trainval")
    ap.add_argument("--out_tag", help="output tag (default: cache_tag)")
    main(ap.parse_args())
