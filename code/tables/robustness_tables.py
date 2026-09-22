import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / 'mospr'))
import paths as _p
TABLES = _p.TABLES
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(str(_p.TABLES))
import os as _o
if _o.environ.get("MOSPR_WORK_OUT"):
    OUT = Path(_o.environ["MOSPR_WORK_OUT"])
T = OUT / "tables/robustness"
ORDER = ["Max", "Mean", "AbMIL", "HE2RNA", "AbReg", "tRNAformer", "ILRA", "S4MIL",
         "MambaMIL", "SRMambaMIL", "MOSBY", "SEQUOIA VIS", "2DMamba",
         "CPNN", "MoSPR (ours)"]
SCORERS = [("meanz", "mean-z"), ("ssgsea", "ssGSEA"), ("gsva", "GSVA")]


def load(cohort, coll):
    mz = (pd.read_csv(OUT / f"tables/refit/refit_pathway_{cohort}.csv")
          .query("stage=='stage2_refit_trainval'")[["model", "fold", f"{coll}_scc"]]
          .rename(columns={f"{coll}_scc": "scc"}).assign(method="meanz"))
    tag = "" if coll == "hallmark" else f"_{coll}"
    rb = pd.read_csv(T / f"pathway_robustness_{cohort}{tag}.csv")[["model", "fold", "method", "scc"]]
    d = pd.concat([mz, rb]); d["cohort"] = cohort
    return d


def ci(v):
    v = np.asarray(v, float)
    return stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def main(a):
    df = pd.concat([load(c, a.collection) for c in a.cohorts])
    g = (df.groupby(["cohort", "method", "model"]).scc
         .agg(mean="mean", ci=ci, n="size").reset_index())
    g = g[g.n == 4]
    g.round(5).to_csv(T / f"robustness_summary_{a.collection}.csv", index=False)


    print(f"\n===== {a.collection} — MoSPR rank per scorer / winner =====")
    for c in a.cohorts:
        line = []
        for k, lab in SCORERS:
            s = g[(g.cohort == c) & (g.method == k)].set_index("model")["mean"]
            if not len(s):
                line.append(f"{lab}: --"); continue
            r = int(s.rank(ascending=False)["MoSPR (ours)"])
            line.append(f"{lab}: MoSPR rank {r}, winner {s.idxmax()}")
        print(f"  {c}  " + " | ".join(line))

    L = [f"% Pathway-metric robustness: predictions fixed, only the scorer changes ({a.collection}).",
         r"% mean-z = the metric used in the paper; ssGSEA/GSVA = official gseapy 1.3.1 implementations.",
         r"% Value = 4-fold mean {\tiny$\pm$95\% CI} of the per-set Spearman median.",
         r"\begin{tabular}{l" + "ccc" * len(a.cohorts) + "}", r"\toprule",
         "& " + " & ".join(r"\multicolumn{3}{c}{%s}" % c for c in a.cohorts) + r"\\",
         "".join(r"\cmidrule(lr){%d-%d}" % (2 + 3 * i, 4 + 3 * i)
                 for i in range(len(a.cohorts))),
         "Method & " + " & ".join([" & ".join(l for _, l in SCORERS)] * len(a.cohorts)) + r"\\",
         r"\midrule"]
    best = {(c, k): g[(g.cohort == c) & (g.method == k)]["mean"].max()
            for c in a.cohorts for k, _ in SCORERS}
    for m in [x for x in ORDER if x in set(g.model)]:
        cells = []
        for c in a.cohorts:
            for k, _ in SCORERS:
                r = g[(g.cohort == c) & (g.method == k) & (g.model == m)]
                if not len(r):
                    cells.append("--"); continue
                r = r.iloc[0]
                s = r"%.3f\,{\tiny$\pm$%.3f}" % (r["mean"], r["ci"])
                cells.append(r"\textbf{%s}" % s if np.isclose(r["mean"], best[(c, k)]) else s)
        nm = r"\textbf{MoSPR (ours)}" if m == "MoSPR (ours)" else m
        L.append(f"{nm} & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", ""]
    (T / f"appendix_robustness_{a.collection}.tex").write_text("\n".join(L))


    gf = (pd.read_csv(T / "pathway_global_factor.csv")
          .groupby(["cohort", "model"])[["gfac_scc", "gfac_share_of_setscore_R2"]].mean()
          .reset_index())
    P = gf.pivot(index="model", columns="cohort", values="gfac_share_of_setscore_R2")
    Q = gf.pivot(index="model", columns="cohort", values="gfac_scc")
    L2 = [r"% Share of the mean-z set score explained by a transcriptome-wide slide factor, and",
          r"% that factor's own accuracy. A larger share makes scorers disagree more.",
          r"\begin{tabular}{l" + "cc" * len(a.cohorts) + "}", r"\toprule",
          "& " + " & ".join(r"\multicolumn{2}{c}{%s}" % c for c in a.cohorts) + r"\\",
          "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(len(a.cohorts))),
          "Method & " + " & ".join([r"share & $\rho$"] * len(a.cohorts)) + r"\\", r"\midrule"]
    for m in [x for x in ORDER if x in P.index]:
        cells = []
        for c in a.cohorts:
            cells += ["%.2f" % P.loc[m, c], "%.3f" % Q.loc[m, c]]
        nm = r"\textbf{MoSPR (ours)}" if m == "MoSPR (ours)" else m
        L2.append(f"{nm} & " + " & ".join(cells) + r"\\")
    L2 += [r"\bottomrule", r"\end{tabular}", ""]
    (T / "appendix_global_factor.tex").write_text("\n".join(L2))
    print(f"\nwritten: {T}/appendix_robustness_{a.collection}.tex · "
          f"appendix_global_factor.tex · robustness_summary_{a.collection}.csv")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["BRCA", "KIRC", "LUAD"])
    ap.add_argument("--collection", default="hallmark")
    main(ap.parse_args())
