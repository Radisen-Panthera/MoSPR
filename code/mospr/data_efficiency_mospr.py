import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import paths as _p
import argparse
import importlib.util
import os
import pathlib
import pickle
import sys


os.environ.setdefault("MOSPR_TAU", "0")

import numpy as np
import pandas as pd

sys.path.insert(0, str(_p.CODE))
import cohorts
from spatial_proteome import pathways, spectral

RES = pathlib.Path(str(_p.RESULTS))
DS = pathlib.Path(str(_p.DATA))
GMT = _p.GENESETS / "h.all.v2025.1.Hs.symbols.gmt"

_s = importlib.util.spec_from_file_location(
    "sw", str(_p.CODE / "design_blocks.py"))
sw = importlib.util.module_from_spec(_s); _s.loader.exec_module(sw)
_s2 = importlib.util.spec_from_file_location(
    "ab", str(_p.CODE / "ablation.py"))
ab = importlib.util.module_from_spec(_s2); _s2.loader.exec_module(ab)

FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.0]


def frac_tag(f):
    return None if f >= 1.0 else f"f{int(round(f * 100)):02d}"


def aggregate_dense(C):
    tot = C.sum(axis=(1, 2))
    ok = tot > 0
    return (C[ok] / tot[ok][:, None, None]).mean(axis=0) if ok.any() else np.zeros_like(C[0])


def macrostates_from(cache, C_fit, K):
    Cg = aggregate_dense(C_fit)
    d = ((Cg + Cg.T) / 2.0).sum(axis=1)
    keep = d > 0
    M = len(d)
    if keep.all():
        _, emb = spectral.spectral_embedding(Cg, n_vectors=20)
        _, R = spectral.macrostate_assignment(emb, n_macrostates=K, seed=sw.SEED,
                                              sample_weight=cache["micro_sizes"])
        return R

    idx = np.flatnonzero(keep)
    print(f"    isolated microstates {M - len(idx)}  dropped before embedding", flush=True)
    _, emb = spectral.spectral_embedding(Cg[np.ix_(idx, idx)], n_vectors=20)
    lbl, _ = spectral.macrostate_assignment(emb, n_macrostates=K, seed=sw.SEED,
                                            sample_weight=cache["micro_sizes"][idx])
    R = np.zeros((M, K))
    R[idx, lbl] = 1.0

    mu, cnt = cache["micro_mean"], cache["micro_count"]
    w = cnt.sum(axis=0)
    cen = np.einsum("nm,nmd->md", cnt, mu) / np.maximum(w, 1)[:, None]
    mc = np.zeros((K, cen.shape[1]))
    for k in range(K):
        sel = idx[lbl == k]
        if len(sel):
            mc[k] = (w[sel, None] * cen[sel]).sum(0) / max(w[sel].sum(), 1)
    for m in np.flatnonzero(~keep):
        R[m, int(np.argmin(((mc - cen[m]) ** 2).sum(1)))] = 1.0
    return R


def subset_idx(cohort, fold, frac, pos):
    tag = frac_tag(frac)
    if tag is None:
        return None
    sp = pickle.load(open(DS / f"{cohort}-paper-digital_slide" / f"split_patient_{tag}.pkl", "rb"))
    get = lambda k: np.array(sorted(
        pos[pathlib.Path(str(p)).stem] for p in sp[fold][k]
        if pathlib.Path(str(p)).stem in pos), dtype=int)
    return get("train"), get("val")


def main(a):
    cfg = cohorts.get(a.cohort)
    genes = pd.read_csv(cfg["processed"] / f"eval_genes_paper_{a.cohort}.txt", header=None)[0].values
    idx = pathways.member_index(pathways.parse_gmt(GMT), genes, min_members=10)
    members = [v for v in idx.values() if len(v) <= 500]
    print(f"{a.cohort}: genes {len(genes)}, Hallmark sets {len(members)}", flush=True)

    rows = []
    for fold in a.folds:
        cache = dict(np.load(
            RES / f"micro_cache_{a.cohort}_{a.cache_tag}_fold{fold}.npz", allow_pickle=True))
        pos = {str(s): i for i, s in enumerate(cache["slides"])}
        te = cache["idx_te"]
        Y = cache["Y_gene"].astype(np.float64)
        C = cache["C"].astype(np.float64)

        for frac in a.fractions:
            sub = subset_idx(a.cohort, fold, frac, pos)
            tr, va = (cache["idx_tr"], cache["idx_va"]) if sub is None else sub
            fit = np.concatenate([tr, va])


            R = macrostates_from(cache, C[fit], a.K)
            blocks = sw.build_blocks(cache, a.K, a.d, with_Q=False, R=R)


            mu1, sd1 = Y[tr].mean(0), Y[tr].std(0); sd1[sd1 == 0] = 1.0
            Yz1 = (Y - mu1) / sd1
            X1 = np.hstack([sw.zfit(blocks["M"], tr), sw.zfit(blocks["S"], tr)])
            _, p = ab.ridge_lowrank(X1, Yz1, tr, va, te, a.qs, a.lams)


            mu2, sd2 = Y[fit].mean(0), Y[fit].std(0); sd2[sd2 == 0] = 1.0
            Yz2 = (Y - mu2) / sd2
            X2 = np.hstack([sw.zfit(blocks["M"], fit), sw.zfit(blocks["S"], fit)])
            pz, _ = ab.ridge_lowrank_fixed(X2, Yz2, fit, te, int(p["q"]), float(p["lam"]))
            pred = pz * sd2 + mu2

            g = sw.h19.metrics(Y[te], pred)
            hall = ab.hallmark_scc(Y[te], pred, members)
            rows.append({"cohort": a.cohort, "fold": fold, "frac": frac,
                         "n_train": len(tr), "n_fit": len(fit), "n_test": len(te),
                         "pcc": g[0], "scc": g[1], "hall_scc": hall,
                         "q": p["q"], "lam": p["lam"]})
            print(f"[{a.cohort} f{fold} frac={frac:<4}] n_fit={len(fit):>4d}  "
                  f"gene SCC {g[1]:.4f}  hallmark {hall:.4f}  (q={p['q']}, lam={p['lam']:g})",
                  flush=True)

    df = pd.DataFrame(rows)
    out = RES / f"deff_mospr_{a.cohort}_{a.out_tag}.csv"
    df.to_csv(out, index=False)
    print(f"\nwritten: {out}\n")
    print(df.groupby("frac")[["n_fit", "scc", "pcc", "hall_scc"]].mean().round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="BRCA")
    ap.add_argument("--cache_tag", default="fpsplit_p512")
    ap.add_argument("--out_tag", default="fpsplit_tv")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--fractions", type=float, nargs="+", default=FRACTIONS)
    ap.add_argument("--qs", type=int, nargs="+", default=list(ab.QS))
    ap.add_argument("--lams", type=float, nargs="+", default=list(ab.LAMS))
    main(ap.parse_args())
