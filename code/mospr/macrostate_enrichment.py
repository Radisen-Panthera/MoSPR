import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import paths as _p
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(_p.CODE))
import cohorts
from spatial_proteome import pathways, spectral

RES = Path(str(_p.RESULTS))
WEI = Path(str(_p.ROOT / "Baseline_Default"))
GENESETS = Path(str(_p.GENESETS))
COLLECTIONS = {"hallmark": "h.all.v2025.1.Hs.symbols.gmt",
               "gobp": "c5.go.bp.v2025.1.Hs.symbols.gmt",
               "kegg": "c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt"}
SEED = 42


def macro_of_fold(cache, K):
    _, R = spectral.macrostate_assignment(cache["emb"], n_macrostates=K, seed=SEED,
                                          sample_weight=cache["micro_sizes"])
    Rl = R.argmax(axis=1)
    mu, cnt = cache["micro_mean"], cache["micro_count"]
    w = cnt.sum(axis=0)
    mcen = np.einsum("nm,nmd->md", cnt, mu) / np.maximum(w, 1)[:, None]
    C = np.zeros((K, mu.shape[2])); mass = np.zeros(K)
    for k in range(K):
        i = np.flatnonzero(Rl == k)
        mass[k] = w[i].sum()
        if len(i):
            C[k] = (w[i, None] * mcen[i]).sum(0) / max(w[i].sum(), 1)
    return Rl, C, mass / max(mass.sum(), 1)


def match_to(ref, cur):
    n = lambda X: X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    S = n(ref) @ n(cur).T
    r, c = linear_sum_assignment(-S)
    return c, S[r, c]


def enrich_z(load, members, n_null=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    G = len(load); tot = load.sum()
    out = np.zeros(len(members))
    cache = {}
    for j, idx in enumerate(members):
        m = len(idx)
        obs = load[idx].mean() - (tot - load[idx].sum()) / (G - m)
        if m not in cache:
            draws = np.array([load[rng.choice(G, m, replace=False)].mean()
                              for _ in range(n_null)])
            cache[m] = (draws.mean(), draws.std() + 1e-12)
        mu, sd = cache[m]

        obs_null_mu = mu - (tot - m * mu) / (G - m)
        obs_null_sd = sd * (1 + m / (G - m))
        out[j] = (obs - obs_null_mu) / obs_null_sd
    return out


def main(a):
    cfg = cohorts.get(a.cohort)
    genes = pd.read_csv(cfg["processed"] / f"eval_genes_paper_{a.cohort}.txt",
                        header=None)[0].values
    sets = {}
    for coll, fn in COLLECTIONS.items():
        idx = pathways.member_index(pathways.parse_gmt(GENESETS / fn), genes,
                                    min_members=10)
        idx = {n: i for n, i in idx.items() if len(i) <= 500}
        sets[coll] = (list(idx), list(idx.values()))
        print(f"{coll}: {len(idx)} sets", flush=True)

    ref_C = None; perm = {}; cos = {}
    proto_ct, gene_load, proto_load, resid_load, masses = {}, {}, {}, {}, {}
    for fold in a.folds:
        cache = dict(np.load(RES / f"micro_cache_{a.cohort}_paper_p512_fold{fold}.npz",
                             allow_pickle=True))
        _, C, mass = macro_of_fold(cache, a.K)
        if ref_C is None:
            ref_C = C; perm[fold] = np.arange(a.K); cos[fold] = np.ones(a.K)
        else:
            perm[fold], cos[fold] = match_to(ref_C, C)
        masses[fold] = mass[perm[fold]]

        w = np.load(WEI / a.cohort / f"weights/weights_gene_fold{fold}.npz",
                    allow_pickle=True)
        W_v, B, ysd = w["W_v"], w["B"], w["ysd"]
        n_M, d, K = int(w["n_M"]), int(w["d"]), int(w["K"])
        assert K == a.K and (int(w["n_S"]) == K * d)
        has_q = "W_q" in w.files
        W_q, U = (w["W_q"], w["U"]) if has_q else (None, None)


        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "sw", str(_p.CODE / "design_blocks.py"))
        sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)
        blocks = sw.build_blocks(cache, a.K, d, with_Q=False)
        X = np.hstack([blocks["M"], blocks["S"]])
        Xz = (X - w["xmu"]) / w["xsd"]

        ct = np.zeros((a.K, len(W_v))); gl = np.zeros((a.K, len(genes)))
        pl = np.zeros((a.K, len(genes))); rl = np.zeros((a.K, len(genes)))
        for k in range(a.K):
            sl = slice(n_M + k * d, n_M + (k + 1) * d)
            v = Xz[:, sl] @ W_v[:, sl].T
            ct[k] = v.mean(0)
            pl[k] = (v.mean(0) @ B) * ysd
            if has_q:
                q = Xz[:, sl] @ W_q[:, sl].T
                rl[k] = (q.mean(0) @ U) * ysd
            gl[k] = pl[k] + rl[k]
        o = perm[fold]
        proto_ct[fold], gene_load[fold] = ct[o], gl[o]
        proto_load[fold], resid_load[fold] = pl[o], rl[o]
        print(f"[fold{fold}] matched {list(o)} cosine {np.round(cos[fold],3)}", flush=True)

    F = len(a.folds)
    stack = lambda D: np.stack([D[f] for f in a.folds])
    ci = lambda A: 1.96 * A.std(0, ddof=1) / np.sqrt(F)
    CT, GL, PL, RL = map(stack, (proto_ct, gene_load, proto_load, resid_load))
    MA = stack(masses)

    out = RES / f"macrobio_{a.cohort}.npz"
    np.savez_compressed(
        out, cell_types=np.load(WEI / a.cohort / "weights/weights_gene_fold0.npz",
                                allow_pickle=True)["cell_types"],
        genes=np.array(genes), K=a.K, folds=np.array(a.folds),
        match_cos=np.stack([cos[f] for f in a.folds]),
        mass_mean=MA.mean(0), mass_ci=ci(MA),
        ct_mean=CT.mean(0), ct_ci=ci(CT),
        gene_mean=GL.mean(0), gene_ci=ci(GL),
        proto_mean=PL.mean(0), proto_ci=ci(PL),
        resid_mean=RL.mean(0), resid_ci=ci(RL))

    rows = []
    rng = np.random.default_rng(0)
    for coll, (names, members) in sets.items():
        for k in range(a.K):
            z = enrich_z(GL.mean(0)[k], members, n_null=a.n_null, rng=rng)
            for nm, zz, mm in zip(names, z, [len(m) for m in members]):
                rows.append({"collection": coll, "macrostate": k, "set": nm,
                             "size": mm, "z": zz})
        print(f"{coll} enrichment done", flush=True)
    pd.DataFrame(rows).to_csv(RES / f"macrobio_enrich_{a.cohort}.csv", index=False)
    print(f"\nwritten: {out}, macrobio_enrich_{a.cohort}.csv")
    print("\nmacrostate patch mass % (mean +- CI):")
    for k in range(a.K):
        print(f"  k{k}: {100*MA.mean(0)[k]:5.1f} ± {100*ci(MA)[k]:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="BRCA")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--n_null", type=int, default=2000)
    main(ap.parse_args())
