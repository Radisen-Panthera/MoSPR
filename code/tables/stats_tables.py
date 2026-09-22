import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / 'mospr'))
import paths as _p
TABLES = _p.TABLES
EXCLUDE = set()
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = _p.ROOT

import os
SRC = Path(os.environ.get("MOSPR_STATS_SRC", _p.TABLES / "tables/final/alpha0/source"))
GENE = Path(os.environ.get("MOSPR_STATS_GENE", _p.TABLES / "tables/refit/refit_per_fold_gene.csv"))
OUT = Path(os.environ.get("MOSPR_STATS_OUT", _p.TABLES / "tables/statistics"))


RENAME = {"AbMIL max-pool": "Max", "AbMIL mean-pool": "Mean", "S4Model": "S4MIL"}
ORDER = ["Max", "Mean", "AbMIL", "HE2RNA", "AbReg", "tRNAformer", "ILRA", "S4MIL",
         "MambaMIL", "SRMambaMIL", "MOSBY", "SEQUOIA VIS", "2DMamba",
         "CPNN", "MoSPR (ours)"]
COLLECTIONS = ["hallmark", "gobp", "kegg"]

PRETTY = {"gene PCC": "PCC", "gene SCC": "SCC",
          "hallmark SCC (median)": "Hallmark", "gobp SCC (median)": "GO-BP",
          "kegg SCC (median)": "KEGG"}


def load_long(cohorts):
    rows = []
    g = pd.read_csv(GENE)
    g = g[g.stage == "stage2_refit_trainval"].replace({"model": RENAME})
    for _, r in g.iterrows():
        if r.cohort not in cohorts:
            continue
        rows += [{"cohort": r.cohort, "model": r.model, "metric": f"gene {m}",
                  "fold": int(r.fold), "value": r[m]} for m in ("PCC", "SCC")]
    for c in cohorts:
        p = pd.read_csv(SRC / f"pathway_all_{c}_fpsplit_tv.csv").replace({"model": RENAME})
        for _, r in p[p.collection.isin(COLLECTIONS)].iterrows():
            rows.append({"cohort": c, "model": r.model,
                         "metric": f"{r.collection} SCC (median)",
                         "fold": int(r.fold), "value": r.SCC_median})
    d = pd.DataFrame(rows)
    return d[~d.model.isin(EXCLUDE)].reset_index(drop=True)


def summarise(long):
    out = []
    for (c, m, met), g in long.groupby(["cohort", "model", "metric"]):
        v = g.sort_values("fold").value.values.astype(float)
        n = len(v)
        sd = v.std(ddof=1) if n > 1 else np.nan
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
        out.append({"cohort": c, "model": m, "metric": met, "n_fold": n,
                    "mean": v.mean(), "sd": sd, "ci_half": half,
                    "ci_lo": v.mean() - half, "ci_hi": v.mean() + half,
                    **{f"fold{i}": x for i, x in enumerate(v)}})
    d = pd.DataFrame(out)
    d["_o"] = d.model.map({m: i for i, m in enumerate(ORDER)}).fillna(99)
    return d.sort_values(["metric", "cohort", "_o"]).drop(columns="_o")


def to_tex(summ, metrics, cohorts, path, title):
    head = ("% " + title + "\n"
            "% Value = 4-fold mean +- 95% CI (standard deviation in parentheses).\n"
            "% Intervals use the t distribution across folds, df=3, t=3.182; the unit is the fold.\n"
            "% Intervals are per model; use a paired test to compare models.\n")
    cols = [(c, m) for c in cohorts for m in metrics]
    lines = [head, r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule"]
    lines.append("& " + " & ".join(r"\multicolumn{%d}{c}{%s}" % (len(metrics), c)
                                   for c in cohorts) + r"\\")
    st = 2
    cm = []
    for _ in cohorts:
        cm.append(r"\cmidrule(lr){%d-%d}" % (st, st + len(metrics) - 1))
        st += len(metrics)
    lines.append("".join(cm))
    lines.append("Method & " + " & ".join(PRETTY.get(m, m) for _, m in cols) + r"\\")
    lines.append(r"\midrule")
    models = [m for m in ORDER if m in set(summ.model)]
    for m in models:
        cells = []
        for c, met in cols:
            r = summ[(summ.cohort == c) & (summ.model == m) & (summ.metric == met)]
            cells.append("--" if not len(r) else
                         r"%.3f\,{\tiny$\pm$}%.3f\,{\tiny(%.3f)}"
                         % (r["mean"].iloc[0], r.ci_half.iloc[0], r.sd.iloc[0]))
        nm = r"\textbf{MoSPR (ours)}" if m == "MoSPR (ours)" else m
        lines.append(f"{nm} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))


def _num(x, digits=3):
    if x != x:
        return "--"
    t = f"{x:.{digits}f}"
    if t.startswith("0."):
        return t[1:]
    if t.startswith("-0."):
        return "-" + t[2:]
    return t


def to_tex_full(summ, metrics, cohorts, path, title):
    models = [m for m in ORDER if m in set(summ.model)]
    L = [
        "% " + title,
        r"% Cell = 4-fold mean {\tiny$\pm$95\% CI} {\tiny(SD)}. Leading zeros omitted.",
        r"% Intervals use the t distribution across folds (df=3, t=3.182); the unit is the fold.",
        r"% Intervals are per model; use a paired test to compare models.",
        r"% Requires \usepackage{rotating,booktabs}.",
        r"\begin{sidewaystable}[p]",
        r"\centering",
        r"\tiny",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\renewcommand{\arraystretch}{1.25}",
        r"\begin{tabular}{l" + "c" * len(models) + "}",
        r"\toprule",
        "Metric & " + " & ".join(m.replace(" (ours)", "") for m in models) + r"\\",
    ]
    for c in cohorts:
        L.append(r"\midrule")
        L.append(r"\multicolumn{%d}{l}{\textbf{%s}}\\" % (len(models) + 1, c))
        for met in metrics:
            sub = summ[(summ.cohort == c) & (summ.metric == met)].set_index("model")
            best = sub["mean"].max() if len(sub) else np.nan
            cells = []
            for m in models:
                if m not in sub.index:
                    cells.append("--")
                    continue
                r = sub.loc[m]
                v = _num(r["mean"])
                if abs(r["mean"] - best) < 1e-9:
                    v = r"\textbf{%s}" % v
                cells.append(r"%s\,{\tiny$\pm$%s}\,{\tiny(%s)}"
                             % (v, _num(r.ci_half), _num(r.sd)))
            L.append(r"\quad %s & " % PRETTY.get(met, met) + " & ".join(cells) + r"\\")
    L += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Per-cohort performance across the four folds. Each cell gives the "
        r"mean, the half-width of the 95\% confidence interval across folds "
        r"(Student $t$, $df=3$), and the standard deviation in parentheses; leading "
        r"zeros are omitted. Gene-level PCC and SCC are correlations across slides "
        r"computed per gene and averaged over genes. Pathway metrics are the median "
        r"over gene sets of the per-set Spearman correlation, aggregated from "
        r"gene-level predictions (pathway labels are never used for training). "
        r"Intervals are computed independently per method, so overlapping intervals "
        r"do not imply the absence of a difference: every method sees the same folds "
        r"and the same held-out slides, and method-to-method comparisons should be "
        r"paired. Best value per row in bold.}",
        r"\label{tab:appendix-full}",
        r"\end{sidewaystable}",
        "",
    ]
    path.write_text("\n".join(L))


def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    long = load_long(a.cohorts)
    long.to_csv(OUT / "stats_foldwise_long.csv", index=False)
    summ = summarise(long)
    summ.round(4).to_csv(OUT / "stats_summary.csv", index=False)

    metrics = ["gene PCC", "gene SCC"] + [f"{c} SCC (median)" for c in COLLECTIONS]
    to_tex(summ, ["gene PCC", "gene SCC"], a.cohorts,
           OUT / "appendix_gene_stats.tex",
           "Appendix - gene level, method x cohort")
    to_tex(summ, [f"{c} SCC (median)" for c in COLLECTIONS], a.cohorts,
           OUT / "appendix_pathway_stats.tex",
           "Appendix - pathway level (median per-set Spearman)")
    to_tex_full(summ, metrics, a.cohorts, OUT / "appendix_full_transposed.tex",
                "Appendix, transposed - rows = cohort x metric, columns = method")

    raw = long.pivot_table(index=["metric", "cohort", "model"], columns="fold",
                           values="value").round(4)
    raw.to_csv(OUT / "stats_foldwise_wide.csv")

    print(f"models {summ.model.nunique()}, cohorts {summ.cohort.nunique()}, "
          f"metrics {summ.metric.nunique()}, cells {len(summ)}")
    miss = summ[summ.n_fold != 4]
    print("cells without 4 folds:", "none" if not len(miss) else
          miss[["cohort", "model", "metric", "n_fold"]].to_string(index=False))
    print(f"\nwritten: {OUT}/stats_summary.csv · stats_foldwise_{{long,wide}}.csv · "
          f"appendix_{{gene,pathway}}_stats.tex · appendix_full_transposed.tex")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["BRCA", "KIRC", "LUAD"])
    main(ap.parse_args())
