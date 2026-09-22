import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / 'mospr'))
import paths as _p
TABLES = _p.TABLES
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _p.ROOT
SRC = _p.TABLES / "tables/refit/refit_per_fold_gene.csv"
OUT = _p.TABLES
import os as _o
if _o.environ.get("MOSPR_WORK_OUT"):
    OUT = Path(_o.environ["MOSPR_WORK_OUT"])
    SRC = OUT / "tables/refit_per_fold_gene.csv"

RENAME = {"AbMIL max-pool": "Max", "AbMIL mean-pool": "Mean", "S4Model": "S4MIL"}
ORDER = ["Max", "Mean", "AbMIL", "HE2RNA", "AbReg", "tRNAformer", "ILRA", "S4MIL",
         "MambaMIL", "SRMambaMIL", "MOSBY", "SEQUOIA VIS", "2DMamba",
         "CPNN", "MoSPR (ours)"]
S1, S2 = "stage1_trainval_holdout", "stage2_refit_trainval"


def summarise(df, metric):
    rows = []
    for (c, m), g in df.groupby(["cohort", "model"]):
        a = g[g.stage == S1].sort_values("fold")[metric].values
        b = g[g.stage == S2].sort_values("fold")[metric].values
        if len(a) != 4 or len(b) != 4:
            continue
        ci = lambda v: stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        d = b - a
        rows.append({"cohort": c, "model": m, "metric": metric,
                     "before": a.mean(), "before_ci": ci(a),
                     "after": b.mean(), "after_ci": ci(b),
                     "delta": d.mean(), "delta_ci": ci(d),
                     "folds_up": int((d > 0).sum()),
                     "p_paired": stats.ttest_rel(b, a).pvalue,
                     **{f"before_f{i}": x for i, x in enumerate(a)},
                     **{f"after_f{i}": x for i, x in enumerate(b)}})
    r = pd.DataFrame(rows)
    r["_o"] = r.model.map({m: i for i, m in enumerate(ORDER)}).fillna(99)
    return r.sort_values(["cohort", "_o"]).drop(columns="_o")


def to_tex(s, metric, cohorts, path):
    models = [m for m in ORDER if m in set(s.model)]
    L = [f"% TV refitting before/after - {metric}. stage 1 (fit on train, early stop on val) vs",
         "% stage 2 (refit on train+val, lr 1e-4 for 8 epochs). The test set is the same in both.",
         r"% Value = 4-fold mean {\tiny$\pm$95\% CI}. Delta is the fold-paired difference and",
         r"% the parentheses give how many of the four folds improved.",
         r"\begin{tabular}{l" + "ccc" * len(cohorts) + "}", r"\toprule",
         "& " + " & ".join(r"\multicolumn{3}{c}{%s}" % c for c in cohorts) + r"\\"]
    st = 2
    L.append("".join(r"\cmidrule(lr){%d-%d}" % (st + 3 * i, st + 3 * i + 2)
                     for i in range(len(cohorts))))
    L.append("Method & " + " & ".join(["Before & After & $\\Delta$"] * len(cohorts)) + r"\\")
    L.append(r"\midrule")
    for m in models:
        cells = []
        for c in cohorts:
            r = s[(s.cohort == c) & (s.model == m)]
            if not len(r):
                cells += ["--", "--", "--"]
                continue
            r = r.iloc[0]
            cells.append(r"%.3f\,{\tiny$\pm$%.3f}" % (r.before, r.before_ci))
            cells.append(r"%.3f\,{\tiny$\pm$%.3f}" % (r.after, r.after_ci))
            cells.append(r"\textbf{%+.3f}\,{\tiny(%d/4)}" % (r.delta, r.folds_up))
        nm = r"\textbf{MoSPR (ours)}" if m == "MoSPR (ours)" else m
        L.append(f"{nm} & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(L))


def draw(s, metric, cohorts, path):
    models = [m for m in ORDER if m in set(s.model)][::-1]
    fig, axes = plt.subplots(1, len(cohorts), figsize=(4.1 * len(cohorts), 5.0),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cohorts):
        sub = s[s.cohort == c].set_index("model")
        for i, m in enumerate(models):
            if m not in sub.index:
                continue
            r = sub.loc[m]
            ours = m == "MoSPR (ours)"
            ax.plot([r.before, r.after], [i, i], color="#b0b0b0", lw=2, zorder=1)
            ax.scatter(r.before, i, s=34, color="#9a9a9a", zorder=3,
                       label="Before (train only)" if i == 0 else None)
            ax.scatter(r.after, i, s=44, color="#c0392b" if ours else "#2166ac",
                       zorder=4, label="After (train+val refit)" if i == 0 else None)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([m.replace(" (ours)", "") for m in models], fontsize=8)
        for t, m in zip(ax.get_yticklabels(), models):
            if m == "MoSPR (ours)":
                t.set_fontweight("bold")
        ax.set_title(c, fontsize=11)
        ax.set_xlabel(metric, fontsize=9)
        ax.grid(axis="x", alpha=0.25, lw=0.5)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    (path.parent / "pdf").mkdir(exist_ok=True)
    fig.savefig(path.parent / "pdf" / (path.stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(a):
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC).replace({"model": RENAME})
    df = df[df.cohort.isin(a.cohorts)]

    allr = []
    for metric in ("SCC", "PCC"):
        s = summarise(df, metric)
        allr.append(s)
        to_tex(s, metric, a.cohorts, OUT / f"tables/refit/refit_before_after_{metric}.tex")
        draw(s, f"Gene {metric}", a.cohorts, OUT / f"figures/refit_before_after_{metric}.png")
        up = (s.delta > 0).sum()
        print(f"[{metric}] {len(s)}  of them up {up}, down {(s.delta < 0).sum()} · "
              f"mean delta {s.delta.mean():+.4f}, min delta {s.delta.min():+.4f} "
              f"({s.loc[s.delta.idxmin(), 'model']}/{s.loc[s.delta.idxmin(), 'cohort']})")
    pd.concat(allr).round(5).to_csv(OUT / "tables/refit/refit_before_after.csv", index=False)
    print(f"\nwritten: {OUT}/tables/ · {OUT}/figures/")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["BRCA", "KIRC", "LUAD"])
    main(ap.parse_args())
