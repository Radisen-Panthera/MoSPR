import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / 'mospr'))
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
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(_p.CODE))
from spatial_proteome import pathways

RES = Path(str(_p.RESULTS))
OUT = Path(str(_p.TABLES / "figures"))
HALLMARK = "h.all.v2025.1.Hs.symbols.gmt"
COLORS = ["#0000FF", "#FFFF00", "#9D00FF"]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


f52 = _load(_p.CODE / "figure2_slide_pathway.py", "f52")
sw = _load(_p.CODE / "design_blocks.py", "sw")
mb = _load(_p.CODE / "macrostate_enrichment.py", "mb")


def weights_file(root, cohort, fold):
    wdir = Path(root) / cohort
    if not (wdir / f"weights_gene_fold{fold}.npz").exists() and (wdir / "weights").exists():
        wdir = wdir / "weights"
    return wdir / f"weights_gene_fold{fold}.npz"


def patient_contributions(cache, w, n):
    K, d, n_M = int(w["K"]), int(w["d"]), int(w["n_M"])
    blocks = sw.build_blocks(cache, K, d, with_Q=False)
    X = np.hstack([blocks["M"], blocks["S"]])
    xmu, xsd, W, b, U = w["xmu"], w["xsd"], w["W_q"], w["b_q"], w["U"]
    ymu, ysd = w["ymu"], w["ysd"]
    Xz = (X[n] - xmu) / xsd
    pred = ((Xz @ W.T + b) @ U) * ysd + ymu
    base = (b @ U) * ysd + ymu
    terms = {"M": ((Xz[:n_M] @ W[:, :n_M].T) @ U) * ysd}
    contrib = {}
    for k in range(K):
        sl = slice(n_M + k * d, n_M + (k + 1) * d)
        terms[k] = ((Xz[sl] @ W[:, sl].T) @ U) * ysd
        contrib[k] = (((X[n, sl] / xsd[sl]) @ W[:, sl].T) @ U) * ysd
    resid = float(np.abs(base + sum(terms.values()) - pred).max())
    return contrib, blocks["P"][n], resid


def main(a):
    out_dir = Path(a.outdir) if a.outdir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = dict(np.load(RES / f"micro_cache_{a.cohort}_{a.cache_tag}_fold{a.fold}.npz",
                         allow_pickle=True))
    w = np.load(weights_file(a.weights, a.cohort, a.fold), allow_pickle=True)
    K = int(w["K"])

    slides = np.array([str(x) for x in cache["slides"]])
    assert a.slide in set(slides), f"{a.slide} is not in the fold {a.fold} cache"
    n = int(np.flatnonzero(slides == a.slide)[0])
    where = ("test" if n in set(cache["idx_te"]) else
             "val" if n in set(cache["idx_va"]) else "train")
    print(f"[fold{a.fold}] {a.slide} is a {where} slide", flush=True)

    contrib, gamma, resid = patient_contributions(cache, w, n)
    print(f"[fold{a.fold}] decomposition residual {resid:.2e}", flush=True)

    idx = pathways.member_index(pathways.parse_gmt(_p.GENESETS / HALLMARK), w["genes"],
                                min_members=10)
    idx = {s: i for s, i in idx.items() if len(i) <= 500}
    names, members = list(idx), list(idx.values())
    rng = np.random.default_rng(0)
    zmap = {k: pd.Series(mb.enrich_z(contrib[k], members, n_null=a.n_null, rng=rng), index=names)
            for k in range(K)}

    labels, cen = f52.macro_map(cache, K)
    name2file = dict(zip(pd.read_csv(f52.PAIRS)["slide_name"],
                         pd.read_csv(f52.PAIRS)["slide_file"].str.replace(".svs", "", regex=False)))
    coords, micro = f52.assign_patches(a.slide, cen)
    thumb, coords, macro, scale = f52.crop_largest_piece(coords, labels[micro], name2file[a.slide])
    thumb = np.asarray(thumb)
    mass = np.bincount(macro, minlength=K) / len(macro)

    states = sorted(a.states)
    tops, rows = {}, []
    for k in states:
        s = zmap[k]
        tops[k] = s.reindex(s.abs().sort_values(ascending=False).index).head(a.top)
        for r, (st, z) in enumerate(tops[k].items(), 1):
            rows.append({"slide": a.slide, "fold": a.fold, "macrostate": k,
                         "gamma": float(gamma[k]), "tissue_share": float(mass[k]),
                         "rank": r, "set": st, "z": float(z)})
        print(f"  MS{k} gamma {gamma[k]:.3f} tissue {100 * mass[k]:.1f}%  " +
              "  ".join(f"{st.replace('HALLMARK_', '')} {z:+.1f}" for st, z in tops[k].items()),
              flush=True)
    v = max(t.abs().max() for t in tops.values())
    xlim = (-1.15 * v, 1.15 * v)

    n_s = len(states)
    ih = a.col_width * thumb.shape[0] / thumb.shape[1]
    bh = 0.34 * a.top + 0.55
    fig = plt.figure(figsize=(a.col_width * n_s, ih + bh))
    outer = gridspec.GridSpec(1, n_s, figure=fig, wspace=0.06, left=0, right=1, bottom=0, top=1)
    for i, k in enumerate(states):
        cell = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i],
                                                height_ratios=[ih, bh], hspace=0.05)
        f52.panel_slide(fig.add_subplot(cell[0]), thumb, coords, macro, k, scale,
                        mass[k], COLORS[i % len(COLORS)], gray=True, badge=False)
        low = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=cell[1],
                                               width_ratios=[a.label_frac, 1 - a.label_frac],
                                               wspace=0.0)
        axb = fig.add_subplot(low[1])
        m = tops[k]
        f52.panel_bars(axb, m, m, m, xlim, False, labels_left=True,
                       color=COLORS[i % len(COLORS)])
        axb.set_xlabel("Enrichment $z$", fontsize=6.5, labelpad=1)

    p = out_dir / f"figure2_macrostate_{a.cohort}.png"
    fig.savefig(p, dpi=220, bbox_inches="tight", pad_inches=0.02)
    (out_dir / "pdf").mkdir(exist_ok=True)
    fig.savefig(out_dir / "pdf" / (p.stem + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_dir / f"figure2_macrostate_{a.cohort}.csv", index=False)
    print("written:", p, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="BRCA")
    ap.add_argument("--cache_tag", default="fpsplit_p512")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--slide", default="TCGA-AC-A62V-01A_1012_TS")
    ap.add_argument("--states", type=int, nargs="+", default=[0, 5, 6])
    ap.add_argument("--weights", default=str(_p.ROOT / "checkpoints"),
                    help="checkpoint root: {weights}/{cohort}/weights_gene_fold*.npz")
    ap.add_argument("--n_null", type=int, default=2000)
    ap.add_argument("--outdir", default=str(_p.TABLES / "figures"))
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--col_width", type=float, default=4.2)
    ap.add_argument("--label_frac", type=float, default=0.4)
    main(ap.parse_args())
