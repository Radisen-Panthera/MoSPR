import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import paths as _p
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(_p.CODE))
from spatial_proteome import pq, spectral, state_features

import os as _os_fit
RES = Path(str(_p.RESULTS))
SEED, TAU = 42, 20.0

_spec = importlib.util.spec_from_file_location(
    "hybrid19", str(_p.CODE / "prototype_hybrid.py"))
h19 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h19)


def zfit(X, tr, return_stats=False):
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    return (Z, mu, sd) if return_stats else Z


def build_blocks(cache, K, n_pca=None, with_Q=True, tau=None, R=None):
    mu, cnt, C = cache["micro_mean"], cache["micro_count"], cache["C"]
    tr, va = cache["idx_tr"], cache["idx_va"]


    fit = tr if _os_fit.environ.get("MOSPR_FIT_ON") == "train" else np.concatenate([tr, va])


    if R is None:
        _, R = spectral.macrostate_assignment(cache["emb"], n_macrostates=K, seed=SEED,
                                              sample_weight=cache["micro_sizes"])
    elif R.shape[1] != K:
        raise ValueError(f"R has {R.shape[1]} macrostates, expected K={K}")
    Rl = R.argmax(axis=1)
    sel = [np.flatnonzero(Rl == k) for k in range(K)]

    n = len(mu)
    Hs = np.zeros((n, K, mu.shape[2]))
    counts = np.zeros((n, K))
    for k, idx in enumerate(sel):
        if len(idx) == 0:
            continue
        w = cnt[:, idx]
        counts[:, k] = w.sum(axis=1)
        Hs[:, k] = np.einsum("nm,nmd->nd", w, mu[:, idx]) / np.maximum(
            counts[:, k], 1)[:, None]

    totals = counts.sum(axis=1, keepdims=True)
    P = counts / np.maximum(totals, 1)
    Mb = np.einsum("nk,nkd->nd", counts, Hs) / np.maximum(totals, 1)


    cw = counts[fit]
    centroid = np.einsum("nk,nkd->kd", cw, Hs[fit]) / np.maximum(
        cw.sum(axis=0), 1)[:, None]


    tau = (float(_os_fit.environ["MOSPR_TAU"]) if "MOSPR_TAU" in _os_fit.environ
           else TAU) if tau is None else tau
    Hshr = ((counts[..., None] * Hs + tau * centroid[None]) /
            (counts[..., None] + tau)) if tau > 0 else Hs

    W, pm = cache["pca_components"], cache["pca_mean"]
    if n_pca is not None:
        W = W[:n_pca]
    Sb = np.stack([state_features.state_design((Hshr[i] - pm) @ W.T, P[i], center=True)
                   for i in range(n)])
    out = {"M": Mb, "S": Sb, "P": P}
    if with_Q:
        out["Q"] = np.stack([pq.upper_triangle(pq.interface_enrichment(
            pq.coarse_grain(C[i], R), P[i])) for i in range(n)])
    return out


def evaluate(cache, blocks, space, ranks, label_key, B_cache, return_pred=False,
             return_w=False):
    Y = cache[f"Y_{space}"].astype(np.float64)
    tr, va, te = cache["idx_tr"], cache["idx_va"], cache["idx_te"]
    ymu, ysd = Y[tr].mean(axis=0), Y[tr].std(axis=0)
    ysd[ysd == 0] = 1.0
    Yz = (Y - ymu) / ysd
    Bz = (B_cache[(space, label_key)] - ymu) / ysd
    Xm, mmu, msd = zfit(blocks["M"], tr, return_stats=True)
    Xs, smu, ssd = zfit(blocks["S"], tr, return_stats=True)
    X = np.hstack([Xm, Xs])
    V, R, Vt = h19.prototype_decompose(Yz, Bz, tr)
    fit = h19.fit_two_branch(X, X, Yz, tr, va, Bz, V, R, Vt, ranks,
                             h19.ALPHAS, alphas_q=h19.ALPHAS, return_w=return_w)
    pred, p = fit[0], fit[1]
    unz = lambda idx: pred[idx] * ysd + ymu
    out = {**p, "n_x": X.shape[1],
           "val_SCC": h19.metrics(Y[va], unz(va))[1],
           "test_SCC": h19.metrics(Y[te], unz(te))[1],
           "test_PCC": h19.metrics(Y[te], unz(te))[0]}
    if return_w:
        w = dict(fit[2])
        w.update(ymu=ymu, ysd=ysd,
                 xmu=np.concatenate([mmu, smu]), xsd=np.concatenate([msd, ssd]),
                 n_M=blocks["M"].shape[1], n_S=blocks["S"].shape[1])
        return (out, unz(te), w) if return_pred else (out, w)
    if return_pred:
        return out, unz(te)
    return out


def sweep(a, caches, B_cache, name, settings):


    out = RES / f"sweep_{name}{('_' + a.out_tag) if a.out_tag else ''}.csv"
    rows = []
    for space in a.spaces:
        for setting, tag in settings:
            K, ranks, label_key = setting[:3]
            n_pca = setting[3] if len(setting) > 3 else None
            for fold in a.folds:
                blocks = build_blocks(caches[fold], K, n_pca, with_Q=False)
                r = evaluate(caches[fold], blocks, space, ranks, label_key, B_cache)
                rows.append({"experiment": name, "space": space, "fold": fold,
                             "setting": tag, "K": K, "d": n_pca,
                             "label_key": label_key, **r})
                print(f"[{name}|{space} f{fold}] {tag:<22s} val {r['val_SCC']:.4f} "
                      f"test {r['test_SCC']:.4f} (r={r['rank']})", flush=True)


            pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    for space in a.spaces:
        s = df[df.space == space]
        piv = s.groupby("setting")[["val_SCC", "test_SCC"]].agg(["mean", "std"])
        piv = piv.loc[[t for _, t in settings]]
        print(f"\n=== {name} / {space} ===")
        print(piv.round(4).to_string())
        best_v = piv[("val_SCC", "mean")].idxmax()
        best_t = piv[("test_SCC", "mean")].idxmax()
        print(f"  best on val: {best_v}  | best on test: {best_t}"
              f"{'  (match)' if best_v == best_t else '  <- mismatch'}")
    return df


def main(a):
    caches = {f: dict(np.load(RES / f"{a.cache_prefix}_fold{f}.npz", allow_pickle=True))
              for f in a.folds}
    genes = pd.read_csv(RES.parent / "processed/eval_genes.txt", header=None)[0].values
    B_cache = {}
    for space in a.spaces:
        members = None
        if space == "pathway":
            from spatial_proteome import pathways
            names = pd.read_csv(
                str(_p.DATA / "BRCA-pathway-digital_slide/pathway_names.txt"),
                header=None)[0].values
            idx = pathways.member_index(pathways.parse_gmt(h19.GMT), genes,
                                        min_members=10)
            members = [idx[n] for n in names]
        for lk in ["celltype_minor", "celltype_subset"]:
            B_cache[(space, lk)] = h19.build_basis(lk, genes, members)[0]

    base_ranks, base_lk = h19.RANKS, "celltype_minor"
    if a.experiment in ("joint", "all"):


        settings = [((K, (r,), base_lk), f"K={K},r={r}")
                    for K in a.K_grid for r in a.rank_grid]
        sweep(a, caches, B_cache, "joint", settings)
    if a.experiment in ("dim", "all"):
        settings = [((a.best_K, base_ranks, base_lk, d), f"d={d}") for d in a.dim_grid]
        sweep(a, caches, B_cache, "dim", settings)
    if a.experiment in ("dimK", "all"):
        settings = [((K, base_ranks, base_lk, d), f"K={K},d={d}")
                    for K in a.K_grid for d in a.dim_grid]
        sweep(a, caches, B_cache, "dimK", settings)
    if a.experiment in ("K", "all"):
        settings = [((K, base_ranks, base_lk), f"K={K}") for K in a.K_grid]
        dfk = sweep(a, caches, B_cache, "K", settings)
    if a.experiment in ("rank", "all"):
        K = a.best_K
        fine = tuple(a.rank_grid)
        settings = [((K, (r,) if r == 0 else (0, r), base_lk), f"r={r}")
                    for r in fine]
        sweep(a, caches, B_cache, "rank", settings)
    if a.experiment in ("proto", "all"):
        settings = [((a.best_K, tuple(a.best_ranks), lk),
                     f"{lk} ({29 if lk=='celltype_minor' else 49})")
                    for lk in ["celltype_minor", "celltype_subset"]]
        sweep(a, caches, B_cache, "proto", settings)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment",
                    choices=["K", "rank", "proto", "joint", "dim", "dimK", "all"],
                    default="dim")
    ap.add_argument("--spaces", nargs="+", default=["pathway", "gene"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--K_grid", type=int, nargs="+",
                    default=[4, 6, 8, 10, 12, 14, 16, 20, 24, 30])
    ap.add_argument("--rank_grid", type=int, nargs="+",
                    default=[0, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128])
    ap.add_argument("--dim_grid", type=int, nargs="+",
                    default=[4, 8, 16, 24, 32, 48, 64, 96])
    ap.add_argument("--cache_prefix", default="micro_cache")
    ap.add_argument("--out_tag", default="",
                    help="output filename suffix, for running K-slices in parallel processes")
    ap.add_argument("--best_K", type=int, default=10)
    ap.add_argument("--best_ranks", type=int, nargs="+", default=[0, 8, 16, 32, 64, 128])
    main(ap.parse_args())
