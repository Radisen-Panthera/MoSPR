import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import paths as _p
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge

sys.path.insert(0, str(_p.CODE))
from spatial_proteome import pathways              

RES = _p.RESULTS
PROC = _p.RESULTS / "processed"
C2L = _p.DATA / "c2l"
GMT = _p.GENESETS / "h.all.v2025.1.Hs.symbols.gmt"
CPNN_OUT = _p.CPNN_REPO / "outputs"
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
RANKS = (0, 8, 16, 32, 64, 128)
BLOCKS = ["M", "S", "P", "Q"]


def to_target_space(mat):
    x = np.asarray(mat, dtype=np.float64)
    return np.log1p(x / x.sum(axis=0, keepdims=True) * 1e4)


def build_basis(label_key, genes, pathway_members=None, c2l_dir=None):
    inf = pd.read_csv((c2l_dir or C2L) / f"inf_aver_{label_key}.csv",
                      index_col=0)
    assert list(inf.index) == list(genes), "prototype gene axis differs from eval genes"
    raw = inf.values                                                         
    if pathway_members is not None:
        raw = np.stack([raw[idx].sum(axis=0) for idx in pathway_members])
    return to_target_space(raw).T, list(inf.columns)              


def corr_cols(a, b):
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    num = (a * b).sum(axis=0)
    den = np.sqrt((a**2).sum(axis=0) * (b**2).sum(axis=0))
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def metrics(Y, P):
    return (float(np.nanmean(corr_cols(Y, P))),
            float(np.nanmean(corr_cols(rankdata(Y, axis=0), rankdata(P, axis=0)))))


def prototype_decompose(Yz, B, tr):
    G = np.linalg.solve(B @ B.T + 1e-6 * np.eye(len(B)), B)                          
    V = Yz @ G.T                                                                      
    R = Yz - V @ B                                                                 
    Rtr = R[tr] - R[tr].mean(axis=0, keepdims=True)                                      
    _, _, Vt = np.linalg.svd(Rtr, full_matrices=False)
    return V, R, Vt


def fit_two_branch(X_v, X_q, Yz, tr, va, B, V, R, Vt, ranks, alphas_v,
                   alphas_q=None, return_w=False):
    coupled = alphas_q is None
    grid_q = alphas_v if coupled else alphas_q
    best = (np.inf, None)
    for av in alphas_v:
        mv = Ridge(alpha=av, fit_intercept=True).fit(X_v[tr], V[tr])
        resid_va = Yz[va] - mv.predict(X_v[va]) @ B
        for r in ranks:
            if r == 0:
                mse = float(np.mean(resid_va ** 2))
                if mse < best[0]:
                    best = (mse, (av, av, 0))
                continue
            U = Vt[:r]
            Z = (R @ U.T)[tr]
            for aq in ([av] if coupled else grid_q):
                mz = Ridge(alpha=aq, fit_intercept=True).fit(X_q[tr], Z)
                mse = float(np.mean((resid_va - mz.predict(X_q[va]) @ U) ** 2))
                if mse < best[0]:
                    best = (mse, (av, aq, r))
    av, aq, r = best[1]
    mv = Ridge(alpha=av, fit_intercept=True).fit(X_v[tr], V[tr])
    pred = mv.predict(X_v) @ B
    mz = U = None
    if r:
        U = Vt[:r]
        mz = Ridge(alpha=aq, fit_intercept=True).fit(X_q[tr], (R @ U.T)[tr])
        pred = pred + mz.predict(X_q) @ U
    p = {"alpha_v": av, "alpha_q": aq, "rank": r}
    if return_w:
                                                                 
        w = {"W_v": mv.coef_, "b_v": mv.intercept_, "B": B}
        if r:
            w.update(W_q=mz.coef_, b_q=mz.intercept_, U=U)
        return pred, p, w
    return pred, p


def fit_hybrid(X, Yz, tr, va, B, ranks, alphas):
    V, R, Vt = prototype_decompose(Yz, B, tr)
    pred, p = fit_two_branch(X, X, Yz, tr, va, B, V, R, Vt, ranks, alphas)
    return pred, p["alpha_v"], p["rank"]


def fit_free(X, Yz, tr, va, alphas):
    best = (np.inf, None)
    for a in alphas:
        m = Ridge(alpha=a, fit_intercept=True).fit(X[tr], Yz[tr])
        mse = float(np.mean((m.predict(X[va]) - Yz[va]) ** 2))
        if mse < best[0]:
            best = (mse, a)
    m = Ridge(alpha=best[1], fit_intercept=True).fit(X[tr], Yz[tr])
    return m.predict(X), best[1]


def load_cpnn(fold, run, slides):
    path = CPNN_OUT / str(fold) / "prediction" / f"{run.replace('FOLD', str(fold))}.npy"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True).item()
    cs = [s if isinstance(s, str) else s[0] for s in d["slide_name"]]
    order = {s: i for i, s in enumerate(cs)}
    x = np.stack(d["preds"])[[order[s] for s in slides]]
    return np.log1p(x / x.sum(axis=1, keepdims=True) * 1e4)


def main(a):
    genes = pd.read_csv(PROC / "eval_genes.txt", header=None)[0].values
    members = None
    if a.space == "pathway":
        names = pd.read_csv(
            str(_p.DATA / "BRCA-pathway-digital_slide/pathway_names.txt"),
            header=None)[0].values
        idx = pathways.member_index(pathways.parse_gmt(GMT), genes, min_members=10)
        members = [idx[n] for n in names]
    B, ct = build_basis(a.label_key, genes, members)
    print(f"basis: {B.shape} ({len(ct)} cell types, {a.label_key})", flush=True)

    rows, preds_out, te_slides_by_fold = [], {}, {}
    for fold in a.folds:
        d = np.load(RES / f"blocks_{a.tag}_fold{fold}.npz", allow_pickle=True)
        X = np.hstack([d[f"blk_{b}"] for b in BLOCKS]).astype(np.float64)
        Y = d["Y"].astype(np.float64)
        tr, va, te = d["idx_tr"], d["idx_va"], d["idx_te"]
        slides = [str(s) for s in d["slides"]]
        te_slides = [slides[i] for i in te]
        te_slides_by_fold[fold] = te_slides

        ymu, ysd = Y[tr].mean(axis=0), Y[tr].std(axis=0)
        ysd[ysd == 0] = 1.0
        Yz = (Y - ymu) / ysd
        Bz = (B - ymu) / ysd                                                             

        cand = {}
        p, alpha, r = fit_hybrid(X, Yz, tr, va, Bz, RANKS, ALPHAS)
        cand["proto+resid"] = (p, alpha, r)
        p0, a0, _ = fit_hybrid(X, Yz, tr, va, Bz, (0,), ALPHAS)
        cand["proto (r=0)"] = (p0, a0, 0)
        pf, af = fit_free(X, Yz, tr, va, ALPHAS)
        cand["free (M|S|P|Q)"] = (pf, af, -1)

        Yte = Y[te]
        for name, (pred, alpha, r) in cand.items():
            pr = pred[te] * ysd + ymu
            pcc, scc = metrics(Yte, pr)
            rows.append({"fold": fold, "config": name, "alpha": alpha, "rank": r,
                         "PCC": pcc, "SCC": scc})
            preds_out.setdefault(name, {})[fold] = pr
            print(f"[fold {fold}] {name:>16s}: SCC {scc:.4f} PCC {pcc:.4f} "
                  f"(a={alpha:g} r={r})", flush=True)

        cp = load_cpnn(fold, a.cpnn_run, te_slides) if a.space == "gene" else None
        if cp is not None:
            ours = preds_out["proto+resid"][fold]
            oz = (ours - ours.mean(0)) / np.where(ours.std(0) > 0, ours.std(0), 1)
            cz = (cp - cp.mean(0)) / np.where(cp.std(0) > 0, cp.std(0), 1)
            for lam in (0.5,):
                pcc, scc = metrics(Yte, lam * cz + (1 - lam) * oz)
                rows.append({"fold": fold, "config": f"ensemble(CPNN,ours) lam={lam}",
                             "alpha": np.nan, "rank": np.nan, "PCC": pcc, "SCC": scc})
                print(f"[fold {fold}] ensemble lam={lam}: SCC {scc:.4f}", flush=True)
            grid = [(metrics(Yte, l * cz + (1 - l) * oz)[1], l)
                    for l in np.linspace(0, 1, 11)]
            best_scc, best_lam = max(grid)
            rows.append({"fold": fold, "config": "ensemble ORACLE (test-tuned)",
                         "alpha": np.nan, "rank": best_lam, "PCC": np.nan,
                         "SCC": best_scc})
            print(f"[fold {fold}] ensemble oracle: SCC {best_scc:.4f} "
                  f"(lam={best_lam:.1f}) [upper bound, not a valid model]", flush=True)

    for fold, slides_te in te_slides_by_fold.items():
        np.savez_compressed(
            RES / f"pred_hybrid_{a.space}_fold{fold}.npz",
            slides=np.array(slides_te),
            **{f"pred_{n}": byfold[fold].astype(np.float32)
               for n, byfold in preds_out.items() if fold in byfold})

    df = pd.DataFrame(rows)
    df.to_csv(RES / f"hybrid_{a.space}_{a.label_key}.csv", index=False)
    print(f"\n=== {a.space} / {a.label_key}: mean over folds ===")
    print(df.groupby("config")[["PCC", "SCC"]].agg(["mean", "std"]).round(4))
    sel = df[df.config == "proto+resid"]
    print(f"\nselected rank per fold: {list(sel['rank'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", choices=["gene", "pathway"], default="gene")
    ap.add_argument("--tag", default="state_gene")
    ap.add_argument("--label_key", default="celltype_minor",
                    choices=["celltype_minor", "celltype_subset"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--cpnn_run", default="BRCA-ProtoSum_1reg_mse_reg_1e3FOLDDeconvExptsfine")
    main(ap.parse_args())
