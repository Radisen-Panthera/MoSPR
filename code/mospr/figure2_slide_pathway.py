import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import paths as _p
import argparse
import os
import sys
import textwrap
from pathlib import Path

os.environ.setdefault("MOSPR_TAU", "0")

import h5py
import numpy as np
import pandas as pd
import openslide
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

sys.path.insert(0, str(_p.CODE))
from spatial_proteome import spectral

RES = _p.RESULTS
OUT = _p.TABLES / "figures"
FEAT_DIR = _p.DATA / "BRCA-paper-digital_slide/sample_pair_feature_conch"
WSI_DIR = Path(os.environ.get("MOSPR_WSI_DIR", str(_p.DATA / "wsi")))
PAIRS = Path(os.environ.get("MOSPR_SLIDE_MANIFEST", str(_p.DATA / "manifests/pairs_named.csv")))

PATCH_PX, SEED = 256, 42

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
UP, DOWN = "#c0392b", "#2166ac"
BAR = "#5a5a5a"



OVERLAY = ["#00E5FF", "#76FF03", "#FFD600", "#00BFA5", "#64FFDA"]


def macro_map(cache, K):
    labels, R = spectral.macrostate_assignment(
        cache["emb"], n_macrostates=K, seed=SEED,
        sample_weight=cache["micro_sizes"])
    cen = (np.einsum("nm,nmd->md", cache["micro_count"], cache["micro_mean"]) /
           np.maximum(cache["micro_count"].sum(axis=0), 1)[:, None])
    return labels, cen


def assign_patches(slide, cen):
    with h5py.File(FEAT_DIR / f"{slide}.h5", "r") as f:
        coords, feat = f["coord"][:], f["feat"][:].astype(np.float64)
    d = ((feat ** 2).sum(1)[:, None] - 2 * feat @ cen.T + (cen ** 2).sum(1)[None])
    return coords, d.argmin(axis=1)


def crop_largest_piece(coords, macro, slide_file, target_w=1800, grid=512, pad=30):
    gx = (coords[:, 0] // grid).astype(np.int64)
    gy = (coords[:, 1] // grid).astype(np.int64)
    gx0, gy0 = gx.min(), gy.min()
    occ = np.zeros((gy.max() - gy0 + 1, gx.max() - gx0 + 1), dtype=bool)
    occ[gy - gy0, gx - gx0] = True
    lbl, _ = ndimage.label(occ, structure=np.ones((3, 3)))
    comp = lbl[gy - gy0, gx - gx0]
    piece = comp == np.bincount(comp[comp > 0]).argmax()

    x0 = int(coords[piece, 0].min()) - pad
    x1 = int(coords[piece, 0].max()) + PATCH_PX + pad
    y0 = int(coords[piece, 1].min()) - pad
    y1 = int(coords[piece, 1].max()) + PATCH_PX + pad
    w0, h0 = x1 - x0, y1 - y0
    with openslide.OpenSlide(str(WSI_DIR / f"{slide_file}.svs")) as sl:
        lvl = sl.level_count - 1
        for c in range(sl.level_count):
            if w0 / sl.level_downsamples[c] <= target_w:
                lvl = c
                break
        ds = sl.level_downsamples[lvl]
        thumb = sl.read_region((x0, y0), lvl,
                               (max(1, int(w0 / ds)), max(1, int(h0 / ds)))).convert("RGB")

    keep = ((coords[:, 0] >= x0) & (coords[:, 0] <= x1) &
            (coords[:, 1] >= y0) & (coords[:, 1] <= y1))
    c = coords[keep].copy()
    c[:, 0] -= x0
    c[:, 1] -= y0
    return thumb, c, macro[keep], thumb.size[0] / w0


def top_sets(df, k, n):
    p = df[df.macrostate == k].pivot_table(index="set", columns="fold", values="z")
    m = p.mean(axis=1)
    sel = m.abs().sort_values(ascending=False).head(n).index
    return m[sel], p.loc[sel].min(axis=1), p.loc[sel].max(axis=1)


def panel_slide(ax, thumb, coords, macro, k, scale, mass, color, gray=True,
                badge=False):
    if gray:


        g = (thumb @ np.array([0.299, 0.587, 0.114]))
        ax.imshow(g, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(thumb)
    box = PATCH_PX * scale
    for xy in coords[macro == k]:
        ax.add_patch(Rectangle((xy[0] * scale, xy[1] * scale), box, box,
                               facecolor=color, edgecolor="none", alpha=0.9))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if badge:
        ax.text(0.014, 0.97, f"MS{k}", transform=ax.transAxes, fontsize=13,
                fontweight="bold", color="#111", ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.26", facecolor=color,
                          edgecolor="none", alpha=0.95))
        ax.text(0.986, 0.97, f"{mass*100:.0f}% of tissue", transform=ax.transAxes,
                fontsize=9, color="#222", ha="right", va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor=color, linewidth=1.0, alpha=0.9))


def _is_light(c):
    r, g, b = matplotlib.colors.to_rgb(c)
    return 0.299 * r + 0.587 * g + 0.114 * b > 0.6


def _bar_style(color):
    return dict(color=color, edgecolor="#333333" if color != BAR else "none",
                linewidth=0.5 if color != BAR else 0)


def panel_bars(ax, mean, lo, hi, xlim, whisker=False, labels_left=False, a_font=6.5,
               color=BAR):
    y = np.arange(len(mean))[::-1]
    ax.barh(y, mean.values, height=0.62, zorder=3, **_bar_style(color))
    if whisker:
        ax.hlines(y, lo.values, hi.values, color="#333", lw=1.1, zorder=4)
    ax.axvline(0, color="#444", lw=0.9, zorder=2)
    for yy, v in zip(y, mean.values):
        inside = abs(v) > 0.5 * xlim[1]
        off = 0.02 * xlim[1] * (1 if v > 0 else -1)
        ax.text(v - off if inside else v + off, yy, f"{v:+.1f}", va="center",
                fontsize=a_font,
                color=("#111" if _is_light(color) else "white") if inside else "#222",
                fontweight="bold" if inside else "normal",
                ha=("right" if v > 0 else "left") if inside
                   else ("left" if v > 0 else "right"))
    ax.set_yticks(y)
    ax.set_yticklabels([textwrap.fill(
        s.replace("HALLMARK_", "").replace("_", " ").title(), 22)
        for s in mean.index], fontsize=a_font)
    if not labels_left:
        ax.yaxis.tick_right()
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlim(*xlim)
    ax.tick_params(axis="x", labelsize=a_font)
    ax.grid(axis="x", alpha=0.18, lw=0.5, zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_ylim(-0.7, len(mean) - 0.3)


def panel_bars_v(ax, mean, ylim, a_font=6.5, color=BAR):
    x = np.arange(len(mean))
    ax.bar(x, mean.values, width=0.6, zorder=3, **_bar_style(color))
    ax.axhline(0, color="#444", lw=0.9, zorder=2)
    for xx, v in zip(x, mean.values):
        inside = abs(v) > 0.6 * ylim[1]
        off = 0.03 * ylim[1] * (1 if v > 0 else -1)
        ax.text(xx, v - off if inside else v + off, f"{v:+.1f}", ha="center",
                fontsize=a_font,
                color=("#111" if _is_light(color) else "white") if inside else "#222",
                fontweight="bold" if inside else "normal",
                va=("top" if v > 0 else "bottom") if inside
                   else ("bottom" if v > 0 else "top"))
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(
        s.replace("HALLMARK_", "").replace("_", " ").title(), 16)
        for s in mean.index], fontsize=a_font)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.6, len(mean) - 0.4)
    ax.tick_params(axis="y", labelsize=a_font)
    ax.set_ylabel("Enrichment $z$", fontsize=a_font, labelpad=1)
    ax.grid(axis="y", alpha=0.18, lw=0.5, zorder=0)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)


def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RES / f"macro_hallmark_perfold_{a.cohort}.csv")
    panel_cache = Path(a.panel_cache) if a.panel_cache else None
    if panel_cache and panel_cache.exists():
        z = np.load(panel_cache, allow_pickle=True)
        thumb = z["thumb"]; coords = z["coords"]; macro = z["macro"]
        scale = float(z["scale"]); mass = z["mass"]; K = a.K
        slide = a.slide
        print(f"reusing panel cache: {panel_cache}", flush=True)
        return _draw(a, df, thumb, coords, macro, scale, mass, K, slide)
    cache = dict(np.load(RES / f"micro_cache_{a.cohort}_{a.cache_tag}_fold{a.fold}.npz",
                         allow_pickle=True))
    K = a.K
    labels, cen = macro_map(cache, K)
    slides = [str(s) for s in cache["slides"]]
    name2file = dict(zip(pd.read_csv(PAIRS)["slide_name"],
                         pd.read_csv(PAIRS)["slide_file"].str.replace(".svs", "",
                                                                      regex=False)))
    slide = a.slide
    assert slide in slides, f"{slide} is not in the fold {a.fold} cache"
    coords, micro = assign_patches(slide, cen)
    thumb, coords, macro, scale = crop_largest_piece(coords, labels[micro],
                                                     name2file[slide])
    mass = np.bincount(macro, minlength=K) / len(macro)
    print(f"{slide}: {len(macro)} patches after cropping, composition "
          + "  ".join(f"MS{k} {mass[k]*100:.0f}%" for k in range(K)), flush=True)
    thumb = np.asarray(thumb)
    if panel_cache:
        np.savez_compressed(panel_cache, thumb=thumb, coords=coords, macro=macro,
                            scale=scale, mass=mass)
    return _draw(a, df, thumb, coords, macro, scale, mass, K, slide)


def spatial_profiles(coords, macro, K, nbin=16):
    x = coords[:, 0].astype(float); y = coords[:, 1].astype(float)
    bx = np.clip(((x - x.min()) / (np.ptp(x) + 1e-9) * nbin).astype(int), 0, nbin - 1)
    by = np.clip(((y - y.min()) / (np.ptp(y) + 1e-9) * nbin).astype(int), 0, nbin - 1)
    cell = by * nbin + bx
    P = np.zeros((K, nbin * nbin))
    for k in range(K):
        P[k] = np.bincount(cell[macro == k], minlength=nbin * nbin)
    return P / np.maximum(P.sum(1, keepdims=True), 1)


def pick_states(df, mass, K, n, min_mass, how="spatial", coords=None, macro=None):
    piv = {k: df[df.macrostate == k].pivot_table(index="set", columns="fold",
                                                 values="z") for k in range(K)}
    folds = sorted(df.fold.unique())
    rep = {}
    for k in range(K):
        P = piv[k]
        rep[k] = np.mean([np.corrcoef(P[i], P[j])[0, 1]
                          for ii, i in enumerate(folds) for j in folds[ii + 1:]])
    cand = [k for k in range(K) if mass[k] >= min_mass]
    if how == "spatial":
        import itertools
        P = spatial_profiles(coords, macro, K)
        C = np.corrcoef(P)
        best = min(itertools.combinations(cand, n),
                   key=lambda c: np.mean([C[i, j] for i, j
                                          in itertools.combinations(c, 2)]))
        order = list(best)
        print(f"mean spatial-overlap correlation {np.mean([C[i, j] for i, j in __import__('itertools').combinations(order, 2)]):+.3f} "
              f"(exhaustive over {n} of {len(cand)} candidates)", flush=True)
    else:
        order = sorted(cand, key=lambda k: -rep[k])[:n]
    print("state selection (fold reproducibility / occupancy in this slide):", flush=True)
    for k in range(K):
        print(f"  MS{k}: r={rep[k]:+.2f}  {mass[k]*100:4.1f}%"
              + ("   <- selected" if k in order else ""), flush=True)
    return sorted(order), rep


def _draw_single(a, df, thumb, coords, macro, scale, mass, K, slide):
    states, rep = pick_states(df, mass, K, a.n_states, a.min_mass, a.select,
                              coords, macro)
    col = ({k: PALETTE[k] for k in states} if a.state_palette
           else {k: OVERLAY[i % len(OVERLAY)] for i, k in enumerate(states)})
    tops = {k: top_sets(df, k, a.top) for k in states}
    v = max(abs(m).max() for m, _, _ in tops.values())
    xlim = (-1.18 * v, 1.18 * v)

    n = len(states)
    fig = plt.figure(figsize=(4.4 * n, 9.2))
    gs = gridspec.GridSpec(2, n, figure=fig, height_ratios=[1.35, 1.0],
                           hspace=0.30, wspace=0.75)

    ax = fig.add_subplot(gs[0, :])
    ax.imshow(thumb)
    box = PATCH_PX * scale
    for k in states:
        for xy in coords[macro == k]:
            ax.add_patch(Rectangle((xy[0] * scale, xy[1] * scale), box, box,
                                   facecolor=col[k], edgecolor="none", alpha=0.9))
    ax.set_xticks([]); ax.set_yticks([])
    handles = [Rectangle((0, 0), 1, 1, facecolor=col[k], edgecolor="#333",
                         linewidth=0.6) for k in states]
    ax.legend(handles, [f"MS{k}  ({mass[k]*100:.0f}% of tissue)" for k in states],
              ncol=len(states), fontsize=10, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), frameon=False, handlelength=1.2)

    for j, k in enumerate(states):
        axb = fig.add_subplot(gs[1, j])
        m, lo, hi = tops[k]
        panel_bars(axb, m, lo, hi, xlim, a.whisker)
        axb.set_title(f"  MS{k}  ", fontsize=12, fontweight="bold", color="#111",
                      loc="left", pad=6,
                      bbox=dict(boxstyle="round,pad=0.25", facecolor=col[k],
                                edgecolor="none"))

    note = ("bar = 4-fold mean, whisker = fold range" if a.whisker
            else "bar = 4-fold mean")
    fig.text(0.5, 0.045, f"Hallmark enrichment $z$   (competitive vs. permutation "
             f"null; {note})", ha="center", fontsize=10)
    if not a.no_title:
        fig.suptitle(f"{a.cohort} — macrostate regions on one patient's slide "
                     f"({slide}) and their Hallmark enrichment (top {a.top} per state)",
                     fontsize=13, y=0.955)
    p = OUT / f"fig_macrostate_slide_pathway_{a.cohort}{a.suffix}.png"
    fig.savefig(p.with_suffix(".png"), dpi=200, bbox_inches="tight")
    (pdf_dir := p.parent / "pdf").mkdir(exist_ok=True)
    fig.savefig(pdf_dir / p.with_suffix(".pdf").name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("written:", p, "(+ .pdf)")


def _draw(a, df, thumb, coords, macro, scale, mass, K, slide):
    if a.layout == "single":
        return _draw_single(a, df, thumb, coords, macro, scale, mass, K, slide)

    states, rep = pick_states(df, mass, K, a.n_states, a.min_mass, a.select,
                              coords, macro)



    hl = [c.strip() for c in a.highlight.split(",") if c.strip()]
    col = ({k: PALETTE[k] for k in states} if a.state_palette
           else {k: hl[i % len(hl)] for i, k in enumerate(states)})
    tops = {k: top_sets(df, k, a.top) for k in states}
    v = (max(max(abs(hi).max(), abs(lo).max()) for _, lo, hi in tops.values())
         if a.whisker else max(abs(m).max() for m, _, _ in tops.values()))
    xlim = (-1.15 * v, 1.15 * v)

    n = len(states)
    ncol = min(a.ncols, n)
    nrow = int(np.ceil(n / ncol))
    h_img, w_img = thumb.shape[:2]
    cw = a.col_width
    ih = cw * h_img / w_img
    bh = (0.34 * a.top + 0.35) if a.bar_orient == "h" else 1.9
    fig = plt.figure(figsize=(cw * ncol, (ih + bh) * nrow))
    outer = gridspec.GridSpec(nrow, ncol, figure=fig, wspace=a.wspace, hspace=0.12,
                              left=0, right=1, bottom=0, top=1)
    for i, k in enumerate(states):
        cell = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[i // ncol, i % ncol],
            height_ratios=[ih, bh], hspace=0.04)
        panel_slide(fig.add_subplot(cell[0]), thumb, coords, macro, k, scale,
                    mass[k], col[k], gray=not a.color_tissue, badge=a.badge)
        m, lo, hi = tops[k]
        bcol = col[k] if a.bar_color == "panel" else BAR


        frac = a.label_frac if a.bar_orient == "h" else 0.10
        low = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=cell[1],
                                               width_ratios=[frac, 1 - frac],
                                               wspace=0.0)
        axb = fig.add_subplot(low[1])
        if a.bar_orient == "h":
            panel_bars(axb, m, lo, hi, xlim, a.whisker, labels_left=True, color=bcol)
            axb.set_xlabel("Enrichment $z$", fontsize=6.5, labelpad=1)
        else:
            panel_bars_v(axb, m, xlim, color=bcol)
        if a.label == "ms":
            axb.set_title(f"MS{k}", fontsize=10, fontweight="bold", loc="left")
        elif a.label == "letter":
            axb.set_title(f"({chr(97 + i)})", fontsize=10, fontweight="bold", loc="left")

    if a.title:
        fig.suptitle(f"{a.cohort} — macrostate regions on one patient's slide "
                     f"({slide}) and their Hallmark enrichment (top {a.top} per state)",
                     fontsize=12, y=1.02)
    p = OUT / f"fig_macrostate_slide_pathway_{a.cohort}{a.suffix}.png"
    fig.savefig(p.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
    (pdf_dir := p.parent / "pdf").mkdir(exist_ok=True)
    fig.savefig(pdf_dir / p.with_suffix(".pdf").name, dpi=220, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    print("written:", p, "(+ .pdf)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="BRCA")
    ap.add_argument("--cache_tag", default="fpsplit_p512")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--slide", default="TCGA-A8-A08P-01A_856_TS",
                    help="representative slide chosen in the notebook")
    ap.add_argument("--pairs_per_row", type=int, default=2,
                    help="number of (slide, bar) pairs per row")
    ap.add_argument("--no_title", action="store_true", help="kept for compatibility; titles are off by default")
    ap.add_argument("--title", action="store_true", help="add a title (off by default)")
    ap.add_argument("--highlight", default="#D50000,#0000FF,#FFFF00,#9D00FF",
                    help="highlight colours; a comma-separated list cycles over the panels "
                         "(check them against the grey tissue with a CVD simulation)")
    ap.add_argument("--bar_color", default="gray", choices=["gray", "panel"],
                    help="bar colour; panel = same colour as that panel highlight")
    ap.add_argument("--bar_orient", default="h", choices=["h", "v"],
                    help="h = horizontal bars, v = vertical bars (transposed)")
    ap.add_argument("--ncols", type=int, default=2, help="panels per row (4 = single row, 2 = 2x2)")
    ap.add_argument("--col_width", type=float, default=4.2, help="width of one panel in inches")
    ap.add_argument("--wspace", type=float, default=0.06)
    ap.add_argument("--label_frac", type=float, default=0.40,
                    help="fraction of the bar panel reserved for pathway names")
    ap.add_argument("--layout", default="grid", choices=["single", "grid"],
                    help="single = one slide plus per-state bars, grid = repeat the slide per state")
    ap.add_argument("--n_states", type=int, default=4)
    ap.add_argument("--select", default="spatial", choices=["spatial", "repro"],
                    help="spatial = least overlapping states, repro = most reproducible across folds")
    ap.add_argument("--label", default="none", choices=["none", "ms", "letter"],
                    help="bar panel header: none = colour strip only, ms = MS index, letter = (a)(b)..")
    ap.add_argument("--badge", action="store_true",
                    help="overlay MS name/share badges on the slide (off by default)")
    ap.add_argument("--min_mass", type=float, default=0.05,
                    help="drop states occupying less than this share of the slide")
    ap.add_argument("--state_palette", action="store_true",
                    help="use the fixed state palette (same colours as the other figures) instead of complementary overlays")
    ap.add_argument("--whisker", action="store_true",
                    help="draw min-max whiskers across folds (off by default)")
    ap.add_argument("--color_tissue", action="store_true",
                    help="show tissue in original H&E colour instead of greyscale")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--panel_cache", default="",
                    help="npz cache of thumbnail/coordinates/assignments; reused when present")
    main(ap.parse_args())
