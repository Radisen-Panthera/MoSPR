import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / 'mospr'))
import paths as _p
RESULTS = _p.RESULTS
TABLES = _p.TABLES
import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              
import numpy as np              
import pandas as pd              
from scipy import stats              

TCRIT = stats.t.ppf(0.975, 3)                

                                                             
LEG = dict(fontsize=7.5, markerscale=0.75, handlelength=1.8, labelspacing=0.28,
           borderaxespad=0.3, handletextpad=0.4)

ROOT = _p.ROOT
RES = RESULTS
FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.0]

COLORS = {"MoSPR": "#2a78d6", "CPNN": "#c2352b", "AbMIL": "#e08a1e",
          "MambaMIL": "#1baf7a", "2DMamba": "#0d7a8c", "SEQUOIA VIS": "#a04ab5",
          "tRNAformer": "#eb6834", "MOSBY": "#5c6b73"}
MARKERS = {"MoSPR": "o", "CPNN": "v", "AbMIL": "P", "MambaMIL": "X",
           "2DMamba": "*", "SEQUOIA VIS": "D", "tRNAformer": "s", "MOSBY": "^"}
                                                      
                                             
              
ORDER = ["MoSPR", "CPNN", "AbMIL", "MambaMIL",
         "2DMamba", "SEQUOIA VIS", "tRNAformer", "MOSBY"]


def _series(d, col):
    mu, n, sd = [], [], []
    for f in FRACTIONS:
        v = d[d.frac == f][col].dropna().values
        n.append(len(v))
        mu.append(v.mean() if len(v) else np.nan)
        sd.append(v.std(ddof=1) if len(v) > 1 else np.nan)
    return mu, n, sd


MOSPR_TAG = "fpsplit_tv"                                      


def curves(cohort, col):
    out = {}
    m = pd.read_csv(RES / f"deff_mospr_{cohort}_{MOSPR_TAG}.csv")
    out["MoSPR"] = _series(m, col)
    b = RES / f"deff_baselines_{cohort}_fpsplit_tv.csv"
    if b.exists():
        d = pd.read_csv(b)
        for lab in d.model.unique():
            out[lab] = _series(d[d.model == lab], col)
    return {k: out[k] for k in ORDER if k in out}


                                                
                                          
YLIM = {"scc": (0.0, 0.50), "hall_scc": (0.0, 0.60)}
YLAB = {"scc": "SCC (Gene)", "hall_scc": "SCC (Hallmark, Median)"}


XTICKS = {"data": [10, 25, 50, 75, 100],                     
          "even": [0, 25, 50, 75, 100],                 
          "fine": [0, 20, 40, 60, 80, 100]}             


def style_axis(ax, col=None, xt="data", x0=0.0):
    ax.set_xlim(x0, 1.03)
    t = [v / 100 for v in XTICKS[xt]]
    ax.set_xticks(t)
    ax.set_xticklabels([f"{int(round(v * 100))}" for v in t])
    ax.set_xlabel("Training Data Fraction (%)")
    if col in YLIM:
        lo, hi = YLIM[col]
        ax.set_ylim(lo, hi)
        ax.set_yticks(np.round(np.arange(lo, hi + 1e-9, 0.1), 2))
        ax.set_ylabel(YLAB[col])
    ax.grid(alpha=0.25)


def band(mu, sd, n, kind):
    if kind == "none":
        return None
    mu, sd = np.asarray(mu, float), np.asarray(sd, float)
    k = TCRIT / np.sqrt(np.maximum(np.asarray(n, float), 1)) if kind == "ci" else 1.0
    h = k * sd
    return mu - h, mu + h


def crossing(cur):
    F = np.asarray(FRACTIONS)
    base = {k: v[0][-1] for k, v in cur.items()
            if k != "MoSPR" and not np.isnan(v[0][-1])}
    if "MoSPR" not in cur or not base:
        return None
    best = max(base, key=base.get)
    y = base[best]
    mo = np.asarray(cur["MoSPR"][0], float)
    hit = np.flatnonzero(mo >= y)
    if not len(hit):
        return None
    i = hit[0]
    x = F[0] if i == 0 else F[i - 1] + (y - mo[i - 1]) / (mo[i] - mo[i - 1]) * (F[i] - F[i - 1])
    return {"best": best, "y": y, "x": float(x), "first_meas": float(F[i]),
            "mospr_at": float(mo[i])}


def annotate_crossing(ax, cur, x0=0.0):
    c = crossing(cur)
    if c is None:
        return None
    ax.axhline(c["y"], color="#555555", lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.plot([c["x"], c["x"]], [ax.get_ylim()[0], c["y"]], color="#555555",
            lw=1.1, ls=(0, (1, 2)), zorder=2)
    ax.plot([c["x"]], [c["y"]], marker="o", ms=5.5, mfc="white", mec="#222222",
            mew=1.3, zorder=6)
    ax.annotate(f"{c['x'] * 100:.0f}%", xy=(c["x"], c["y"]),
                xytext=(-6, 8), textcoords="offset points", ha="right",
                fontsize=10, fontweight="bold", color="#222222", zorder=7)
    ax.text(x0 + 0.01, c["y"], f"{c['best']} @100%", ha="left", va="bottom",
            fontsize=8.5, color="#555555", zorder=7,
            transform=ax.transData)
    return c


def draw(cur, col, bandkind="none", band_only_ours=False, xt="data", x0=0.0,
         cross=True):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    for ax, mode in zip(axes, ("abs", "drop")):
        for lab, (m, nn, sd) in cur.items():
            ours = lab == "MoSPR"
            y = m
            if mode == "drop":
                full = m[-1]
                if not full or np.isnan(full):
                    continue                                    
                y = [(v / full - 1) * 100 if not np.isnan(v) else np.nan for v in m]
            if np.isnan(y).all():
                continue
            if mode == "abs" and (not band_only_ours or ours):
                bd = band(m, sd, nn, bandkind)
                if bd is not None and not np.isnan(bd[0]).all():
                    ax.fill_between(FRACTIONS, bd[0], bd[1], color=COLORS.get(lab),
                                    alpha=0.16 if ours else 0.09, lw=0, zorder=1)
            ax.plot(FRACTIONS, y, marker=MARKERS.get(lab, "o"), color=COLORS.get(lab),
                    lw=2.4 if ours else 1.8, ms=7 if ours else 6,
                    linestyle="-" if ours else "--", alpha=1.0 if ours else 0.9,
                    label="MoSPR (ours)" if ours else lab, zorder=5 if ours else 3)
        if mode == "abs":
            style_axis(ax, col, xt, x0)
            if cross:
                annotate_crossing(ax, cur, x0)
            ax.legend(frameon=False, **LEG, loc="lower right")
        else:
            style_axis(ax, None, xt, x0)
            ax.axhline(0, color="#999999", lw=1, zorder=1)
            ax.set_ylabel("Drop vs. Full Data (%)")
    plt.tight_layout()
    return fig


def draw_combined(cur_gene, cur_path, bandkind="none", band_only_ours=False, xt="data", x0=0.0, legend="left",
                  cross=True):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    for ax, cur, col in ((axes[0], cur_gene, "scc"),
                         (axes[1], cur_path, "hall_scc")):
        for lab, (m, nn, sd) in cur.items():
            ours = lab == "MoSPR"
            if np.isnan(m).all():
                continue
            if not band_only_ours or ours:
                bd = band(m, sd, nn, bandkind)
                if bd is not None and not np.isnan(bd[0]).all():
                    ax.fill_between(FRACTIONS, bd[0], bd[1], color=COLORS.get(lab),
                                    alpha=0.16 if ours else 0.09, lw=0, zorder=1)
            ax.plot(FRACTIONS, m, marker=MARKERS.get(lab, "o"), color=COLORS.get(lab),
                    lw=2.4 if ours else 1.8, ms=7 if ours else 6,
                    linestyle="-" if ours else "--", alpha=1.0 if ours else 0.9,
                    label="MoSPR (ours)" if ours else lab, zorder=5 if ours else 3)
        style_axis(ax, col, xt, x0)
        if cross:
            annotate_crossing(ax, cur, x0)
        if legend == "both":
            ax.legend(frameon=False, **LEG, loc="lower right")
    if legend == "left":
        axes[0].legend(frameon=False, **LEG, loc="lower right")
    plt.tight_layout()
                                            
                                           
    if legend in ("below", "right"):
        h, l = axes[0].get_legend_handles_labels()
        if legend == "below":
            fig.legend(h, l, frameon=False, ncol=4, **{**LEG, "labelspacing": 0.4},
                       loc="lower center", bbox_to_anchor=(0.5, 0.0),
                       columnspacing=1.6)
            fig.subplots_adjust(bottom=0.20)                                 
        else:
            fig.legend(h, l, frameon=False, ncol=1, **LEG,
                       loc="center left", bbox_to_anchor=(0.995, 0.5))
            fig.subplots_adjust(right=0.84)
    return fig


def main(a):
    global MOSPR_TAG
    MOSPR_TAG = a.mospr_tag
    LEG["fontsize"] = a.legend_size
    out = ROOT / a.outdir; out.mkdir(parents=True, exist_ok=True)
    keep = {}
    for col, stem in [("scc", "data_efficiency"),
                      ("hall_scc", "data_efficiency_pathway")]:
        cur = curves(a.cohort, col)
        fig = draw(cur, col, a.band, a.band_only_ours, a.xticks, a.xmin,
                   not a.no_crossing)
        c = crossing(cur)
        if c:
            print(f"   crossing: {c['best']} 100% = {c['y']:.4f}, MoSPR interpolated "
                  f"{c['x']*100:.1f}%, first measured point above {c['first_meas']*100:.0f}% "
                  f"({c['mospr_at']:.4f}, diff {c['mospr_at']-c['y']:+.4f})")
        png = out / f"{stem}_{a.cohort}{a.suffix}.png"
        fig.savefig(png, dpi=300); (png.parent / "pdf").mkdir(exist_ok=True); fig.savefig(png.parent / "pdf" / png.with_suffix(".pdf").name)
        plt.close(fig)
        print(f"written: {png}")
        for lab, (m, n, _sd) in cur.items():
            miss = [f for f, k in zip(FRACTIONS, n) if k < 4]
            note = f"  <- incomplete folds {miss}" if miss else ""
            print(f"   {lab:<12s} " + " ".join(f"{v:.4f}" if not np.isnan(v) else "  -   "
                                                for v in m) + note)
        keep[col] = cur

    fig = draw_combined(keep["scc"], keep["hall_scc"], a.band, a.band_only_ours, a.xticks, a.xmin, a.legend,
                        not a.no_crossing)
    png = out / f"data_efficiency_combined_{a.cohort}{a.suffix}.png"
    fig.savefig(png, dpi=300); (png.parent / "pdf").mkdir(exist_ok=True); fig.savefig(png.parent / "pdf" / png.with_suffix(".pdf").name)
    plt.close(fig)
    print(f"written: {png}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="BRCA")
    ap.add_argument("--outdir", default=str(_p.TABLES / "figures"))
    ap.add_argument("--band", choices=["ci", "std", "none"], default="none")
    ap.add_argument("--band_only_ours", action="store_true")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--xticks", choices=["data", "even", "fine"], default="data")
    ap.add_argument("--xmin", type=float, default=0.0)
    ap.add_argument("--mospr_tag", default="fpsplit_tv_leakfree")
    ap.add_argument("--legend_size", type=float, default=7.5)
    ap.add_argument("--no_crossing", action="store_true")
    ap.add_argument("--legend", choices=["left", "both", "below", "right"], default="below")
    main(ap.parse_args())
