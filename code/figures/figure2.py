import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / 'mospr'))
import paths as _p
RESULTS = _p.RESULTS
TABLES = _p.TABLES
import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MOSPR_TAU", "0")

import h5py
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CPNN = Path(str(_p.CODE))
_s = importlib.util.spec_from_file_location("f52", CPNN / "figure2_slide_pathway.py")
f52 = importlib.util.module_from_spec(_s); _s.loader.exec_module(f52)

RES = Path(str(_p.RESULTS))
OUT = Path(str(_p.TABLES / "figures"))
                                                
COLORS = ["#0000FF", "#FFFF00", "#9D00FF"]


def slide_candidates(cache, labels, cen, n_probe, min_piece=0.6):
    slides = np.array([str(s) for s in cache["slides"]])[cache["idx_te"]]
    rows = []
    for s in slides:
        with h5py.File(f52.FEAT_DIR / f"{s}.h5", "r") as f:
            co = f["coord"][:]
        gx, gy = (co[:, 0] // 512).astype(int), (co[:, 1] // 512).astype(int)
        occ = np.zeros((gy.max() - gy.min() + 1, gx.max() - gx.min() + 1), bool)
        occ[gy - gy.min(), gx - gx.min()] = True
        from scipy import ndimage
        lbl, _ = ndimage.label(occ, structure=np.ones((3, 3)))
        comp = lbl[gy - gy.min(), gx - gx.min()]
        big = np.bincount(comp[comp > 0]).max() / len(co)
        rows.append({"slide": s, "n": len(co), "piece": big})
    d = pd.DataFrame(rows)
    d = d[d.piece >= min_piece].sort_values("n", ascending=False).head(n_probe)

    ent = []
    for s in d.slide:
        _, micro = f52.assign_patches(s, cen)
        p = np.bincount(labels[micro], minlength=labels.max() + 1).astype(float)
        p /= p.sum()
        ent.append(-(p * np.log(p + 1e-12)).sum())
    d = d.assign(entropy=ent).sort_values("entropy", ascending=False)
    return d


def fold_z(df, npz, fold, K):
    i = list(npz["folds"]).index(fold)
    perm = npz["perm"][i]                                        
    sub = df[df.fold == fold]
    out = {}
    for j in range(K):
        s = sub[sub.macrostate == j].set_index("set")["z"]
        out[int(perm[j])] = s
    return out


def main(a):
    out_dir = Path(a.outdir) if a.outdir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RES / f"macro_hallmark_perfold_{a.cohort}.csv")
    npz = np.load(RES / f"macro_hallmark_perfold_{a.cohort}_loadings.npz", allow_pickle=True)
    K = int(npz["K"])

    cache = dict(np.load(RES / f"micro_cache_{a.cohort}_{a.cache_tag}_fold{a.fold}.npz",
                         allow_pickle=True))
    labels, cen = f52.macro_map(cache, K)
    zmap = fold_z(df, npz, a.fold, K)

    if a.slide:
        slide = a.slide
    else:
        cand = slide_candidates(cache, labels, cen, a.n_probe)
        print(f"[fold{a.fold}] top test candidates\n" + cand.head(8).to_string(index=False), flush=True)
        slide = cand.iloc[0].slide
    sl = np.array([str(x) for x in cache["slides"]])
    where = ("test" if slide in set(sl[cache["idx_te"]])
             else "val" if slide in set(sl[cache["idx_va"]]) else "train")
    if where != "test":
                                                  
                                                    
                                      
        assert a.allow_nontest, (f"{slide}  is {where} of fold {a.fold}. "
                                 f"pass --allow_nontest and state it in the caption")
        print(f"[fold{a.fold}] note: this slide is {where}", flush=True)

    name2file = dict(zip(pd.read_csv(f52.PAIRS)["slide_name"],
                         pd.read_csv(f52.PAIRS)["slide_file"].str.replace(".svs", "", regex=False)))
    coords, micro = f52.assign_patches(slide, cen)
    thumb, coords, macro, scale = f52.crop_largest_piece(coords, labels[micro], name2file[slide])
    thumb = np.asarray(thumb)
    mass = np.bincount(macro, minlength=K) / len(macro)

                                                   
    if a.states:
        states = sorted(a.states)
        print(f"[fold{a.fold}] states given explicitly {states}", flush=True)
    else:
        states, _ = f52.pick_states(df[df.fold == a.fold], mass, K, a.n_states, a.min_mass,
                                    "spatial", coords, macro)
    print(f"[fold{a.fold}] {slide} , states {states}, occupancy "
          + " ".join(f"{100*mass[k]:.0f}%" for k in states), flush=True)

    tops = {}
    for k in states:
        s = zmap[k]
        tops[k] = s.reindex(s.abs().sort_values(ascending=False).index).head(a.top)
    v = max(t.abs().max() for t in tops.values())
    xlim = (-1.15 * v, 1.15 * v)

    n = len(states)
    ih = a.col_width * thumb.shape[0] / thumb.shape[1]
    bh = 0.34 * a.top + 0.55
    fig = plt.figure(figsize=(a.col_width * n, ih + bh))
    outer = gridspec.GridSpec(1, n, figure=fig, wspace=0.06, left=0, right=1, bottom=0, top=1)
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
                       color=COLORS[i % len(COLORS)] if a.bar_color == "panel" else f52.BAR)
        axb.set_xlabel("Enrichment $z$", fontsize=6.5, labelpad=1)

                                                        
                                                        
    tag = f"fold{a.fold}_{slide}"
    p = out_dir / f"fig2_cand_{a.cohort}_{tag}.png"
    fig.savefig(p, dpi=220, bbox_inches="tight", pad_inches=0.02)
    (out_dir / "pdf").mkdir(exist_ok=True)
    fig.savefig(out_dir / "pdf" / (p.stem + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("written:", p, flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="BRCA")
    ap.add_argument("--cache_tag", default="fpsplit_p512")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--slide", default="")
    ap.add_argument("--n_probe", type=int, default=12)
    ap.add_argument("--n_states", type=int, default=3)
    ap.add_argument("--states", type=int, nargs="+", default=None)
    ap.add_argument("--min_mass", type=float, default=0.05)
    ap.add_argument("--outdir", default=str(_p.TABLES / "figures"))
    ap.add_argument("--allow_nontest", action="store_true")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--col_width", type=float, default=4.2)
    ap.add_argument("--label_frac", type=float, default=0.4)
    ap.add_argument("--bar_color", default="panel")
    main(ap.parse_args())
