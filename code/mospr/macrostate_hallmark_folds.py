import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import paths as _p
import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MOSPR_TAU", "0")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(_p.CODE))
from spatial_proteome import pathways

RES = Path(str(_p.RESULTS))
OUT = _p.TABLES / "figures"
GENESETS = Path(str(_p.GENESETS))
HALLMARK = "h.all.v2025.1.Hs.symbols.gmt"

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
           "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CPNN = Path(str(_p.CODE))
sw = _load(CPNN / "design_blocks.py", "sw")
mb = _load(CPNN / "macrostate_enrichment.py", "mb")


def gene_loadings(cache, w, K, d, mode):
    n_M = int(w["n_M"])
    cache_te = cache["idx_te"]
    W_q, U, ysd = w["W_q"], w["U"], w["ysd"]
    xmu, xsd = w["xmu"], w["xsd"]
    blocks = sw.build_blocks(cache, K, d, with_Q=False)
    X = np.hstack([blocks["M"], blocks["S"]])
    Xz = (X - xmu) / xsd
    P = blocks["P"]
    gl = np.zeros((K, U.shape[1]))
    for k in range(K):
        sl = slice(n_M + k * d, n_M + (k + 1) * d)
        if mode == "structure":
            v = (xmu[sl] / xsd[sl]) @ W_q[:, sl].T
        else:
            q = Xz[:, sl] @ W_q[:, sl].T
            if mode == "mean":
                v = q.mean(0)
            elif mode == "mean_te":


                v = q[cache_te].mean(0)
            elif mode == "abundance":
                p = P[:, k]
                p = (p - p.mean()) / (p.std() + 1e-12)
                v = p @ q / len(p)
            else:
                raise ValueError(mode)
        gl[k] = (v @ U) * ysd
    return gl


def collect(a):
    wdir = Path(a.weights) / a.cohort
    if not (wdir / "weights_gene_fold0.npz").exists() and (wdir / "weights").exists():
        wdir = wdir / "weights"
    w0 = np.load(wdir / "weights_gene_fold0.npz", allow_pickle=True)
    genes = w0["genes"]
    K, d = int(w0["K"]), int(w0["d"])
    assert K == a.K, f"weights K={K} != --K {a.K}"

    idx = pathways.member_index(pathways.parse_gmt(GENESETS / HALLMARK), genes,
                                min_members=10)
    idx = {n: i for n, i in idx.items() if len(i) <= 500}
    names, members = list(idx), list(idx.values())
    print(f"hallmark: {len(names)} sets (measured genes {len(genes)}), "
          f"loading definition {a.loading}", flush=True)

    ref_C, rows, cosines, loads, perms = None, [], {}, {}, {}
    for fold in a.folds:
        cache = dict(np.load(
            RES / f"micro_cache_{a.cohort}_{a.cache_tag}_fold{fold}.npz",
            allow_pickle=True))
        _, C, _ = mb.macro_of_fold(cache, K)
        if ref_C is None:
            ref_C, perm, cos = C, np.arange(K), np.ones(K)
        else:
            perm, cos = mb.match_to(ref_C, C)
        cosines[fold] = cos

        w = np.load(wdir / f"weights_gene_fold{fold}.npz", allow_pickle=True)
        gl = gene_loadings(cache, w, K, d, a.loading)[perm]
        loads[fold], perms[fold] = gl, perm
        del cache
        print(f"[fold{fold}] matched {list(perm)} cosine {np.round(cos, 3)}", flush=True)

        rng = np.random.default_rng(0)
        for k in range(K):
            z = mb.enrich_z(gl[k], members, n_null=a.n_null, rng=rng)
            for nm, zz, m in zip(names, z, (len(x) for x in members)):
                rows.append({"cohort": a.cohort, "fold": fold, "macrostate": k,
                             "set": nm, "size": m, "z": zz,
                             "match_cos": float(cos[k])})
        print("  enrichment done", flush=True)

    np.savez_compressed(
        RES / f"macro_hallmark_perfold_{a.cohort}{a.suffix}_loadings.npz",
        genes=np.array(genes), folds=np.array(a.folds), K=K, loading=a.loading,
        gene_load=np.stack([loads[f] for f in a.folds]),
        perm=np.stack([perms[f] for f in a.folds]),
        match_cos=np.stack([cosines[f] for f in a.folds]))
    return pd.DataFrame(rows), K, cosines, loads


def figure(df, K, cosines, a, path):
    folds = a.folds
    piv = {f: df[df.fold == f].pivot_table(index="set", columns="macrostate",
                                           values="z")
           for f in folds}

    mean = np.mean([piv[f].values for f in folds], axis=0)
    sets = piv[folds[0]].index.to_numpy()
    keep = np.argsort(-np.abs(mean).max(1))[:a.top]
    order = keep[np.lexsort((-np.abs(mean[keep]).max(1), mean[keep].argmax(1)))]
    sets = sets[order]

    v = np.percentile(np.abs(np.stack([piv[f].values[order] for f in folds])), 99)
    norm = TwoSlopeNorm(0, -v, v)

    fig, axes = plt.subplots(1, len(folds), figsize=(2.5 * len(folds) + 3.4,
                                                     0.20 * len(sets) + 2.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, folds):
        M = piv[f].values[order]
        im = ax.imshow(M, cmap=a.cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(K))
        ax.set_xticklabels([f"MS{k}" for k in range(K)], fontsize=7, rotation=90)
        for k in range(K):
            ax.get_xticklabels()[k].set_color(PALETTE[k % len(PALETTE)])
        ax.set_title(f"fold {f}   (min match cos {cosines[f].min():.2f})",
                     fontsize=8)
        ax.set_xticks(np.arange(-.5, K, 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(sets), 1), minor=True)
        ax.grid(which="minor", color="w", lw=0.4)
        ax.tick_params(which="minor", length=0)

    axes[0].set_yticks(range(len(sets)))
    axes[0].set_yticklabels([s.replace("HALLMARK_", "") for s in sets], fontsize=6)


    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.012, shrink=a.cbar_shrink,
                      aspect=18)
    cb.set_label("enrichment $z$", fontsize=6.5, labelpad=2)
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.5)
    cb.outline.set_linewidth(0.5)
    if not a.no_title:
        fig.suptitle(
            f"{a.cohort} — MoSPR macrostate x Hallmark enrichment, per fold "
            f"(K={K}, top {len(sets)} sets by |z|, loading={a.loading}, "
            f"rows shared across panels)", fontsize=10, y=0.995)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    (pdf_dir := path.parent / "pdf").mkdir(exist_ok=True)
    fig.savefig(pdf_dir / path.with_suffix(".pdf").name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def reproducibility(df, loads, K, a):
    if len(a.folds) < 2:
        print("\nskipping the cross-fold correlation: it needs at least two folds")
        return
    print("\nper-macrostate correlation across folds (pairwise mean / min):")
    piv = {f: df[df.fold == f].pivot_table(index="set", columns="macrostate",
                                           values="z").values for f in a.folds}
    pairs = [(i, j) for ii, i in enumerate(a.folds) for j in a.folds[ii + 1:]]
    for k in range(K):
        zc = [np.corrcoef(piv[i][:, k], piv[j][:, k])[0, 1] for i, j in pairs]
        gc = [np.corrcoef(loads[i][k], loads[j][k])[0, 1] for i, j in pairs]
        print(f"  MS{k}: hallmark z {np.mean(zc):+.3f}/{np.min(zc):+.3f}   "
              f"gene loading {np.mean(gc):+.3f}/{np.min(gc):+.3f}")


def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    csv = RES / f"macro_hallmark_perfold_{a.cohort}{a.suffix}.csv"
    if a.from_csv:

        df = pd.read_csv(csv)
        z = np.load(RES / f"macro_hallmark_perfold_{a.cohort}{a.suffix}"
                          "_loadings.npz", allow_pickle=True)
        K = int(z["K"])
        cosines = {f: z["match_cos"][i] for i, f in enumerate(z["folds"])}
        loads = {f: z["gene_load"][i] for i, f in enumerate(z["folds"])}
        a.loading = str(z["loading"]) if "loading" in z.files else a.loading
    else:
        df, K, cosines, loads = collect(a)
        df.to_csv(csv, index=False)

    png = OUT / f"fig_macro_hallmark_perfold_{a.cohort}{a.suffix}{a.fig_suffix}.png"
    figure(df, K, cosines, a, png)
    print(f"\nwritten: {csv}\n      {png} (+ .pdf)")
    reproducibility(df, loads, K, a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="BRCA")
    ap.add_argument("--weights", default=str(_p.ROOT / "checkpoints"),
                    help="{weights}/{cohort}/weights_gene_fold*.npz")
    ap.add_argument("--cache_tag", default="fpsplit_p512")
    ap.add_argument("--loading", default="structure",
                    choices=["structure", "abundance", "mean", "mean_te"])
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--n_null", type=int, default=2000)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--fig_suffix", default="", help="suffix appended to the figure filename only")
    ap.add_argument("--from_csv", action="store_true",
                    help="redraw from the saved CSV without recomputing enrichment")
    ap.add_argument("--cmap", default="RdBu_r",
                    help="diverging colormap: RdBu_r = red(+)/blue(-), PuOr_r = orange(+)/purple(-)")
    ap.add_argument("--cbar_shrink", type=float, default=0.3,
                    help="colorbar length relative to figure height")
    ap.add_argument("--no_title", action="store_true",
                    help="omit the suptitle; the caption carries the description")
    main(ap.parse_args())
