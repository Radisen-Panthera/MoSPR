import argparse
import pathlib
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
COH = ["BRCA", "KIRC", "LUAD"]
TCRIT = stats.t.ppf(0.975, 3)
RENAME = {"AbMIL max-pool": "Max", "AbMIL mean-pool": "Mean", "S4Model": "S4MIL"}


def gene_table():
    rows = []
    for c in COH:
        d = pd.read_csv(ROOT / f"results/per_cohort/{c}/results/baselines_per_fold.csv")
        d["model"] = d.model.replace(RENAME)
        r = pd.read_csv(ROOT / f"results/per_cohort/{c}/results/cpnn_rescored.csv")
        if len(r):
            d = pd.concat([d[d.model != "CPNN"], r.assign(model="CPNN")], ignore_index=True)
        m = pd.read_csv(ROOT / f"results/per_cohort/{c}/results/mospr.csv")
        m = m[(m.space == "gene") & (m.primary)].rename(columns={"test_PCC": "PCC", "test_SCC": "SCC"})
        d = pd.concat([d, m.assign(model="MoSPR (ours)")], ignore_index=True)
        for model, g in d.groupby("model"):
            if len(g) != 4:
                continue
            for met in ("PCC", "SCC"):
                v = g.sort_values("fold")[met].to_numpy(float)
                rows.append({"cohort": c, "model": model, "metric": f"gene {met}", "n_fold": 4,
                             "mean": v.mean(), "sd": v.std(ddof=1),
                             "ci_half": TCRIT * v.std(ddof=1) / 2,
                             **{f"fold{i}": x for i, x in enumerate(v)}})
    return pd.DataFrame(rows)


def pathway_table():
    rows = []
    for c in COH:
        f = ROOT / f"results/tables/final/alpha0/source/pathway_all_{c}_fpsplit_tv.csv"
        d = pd.read_csv(f)
        for (model, coll), g in d.groupby(["model", "collection"]):
            v = g.sort_values("fold").SCC_median.to_numpy(float)
            if len(v) != 4:
                continue
            rows.append({"cohort": c, "model": model, "metric": f"{coll} SCC (median)", "n_fold": 4,
                         "mean": v.mean(), "sd": v.std(ddof=1),
                         "ci_half": TCRIT * v.std(ddof=1) / 2,
                         **{f"fold{i}": x for i, x in enumerate(v)}})
    return pd.DataFrame(rows)


def paired_vs_ours(df, metric):
    out = []
    for c in COH:
        s = df[(df.cohort == c) & (df.metric == metric)].set_index("model")
        if "MoSPR (ours)" not in s.index:
            continue
        ours = s.loc["MoSPR (ours)", [f"fold{i}" for i in range(4)]].to_numpy(float)
        for m in s.index:
            v = s.loc[m, [f"fold{i}" for i in range(4)]].to_numpy(float)
            p = np.nan if m == "MoSPR (ours)" else stats.ttest_rel(v, ours).pvalue
            out.append({"cohort": c, "metric": metric, "model": m,
                        "mean": v.mean(), "delta_vs_ours": v.mean() - ours.mean(), "p_paired": p})
    return pd.DataFrame(out)


def check_against_tex(df):
    bad = []
    cells = {("gene PCC", i * 2): c for i, c in enumerate(COH)}
    cells |= {("gene SCC", i * 2 + 1): c for i, c in enumerate(COH)}
    tex = (ROOT / "results/tables/final/alpha0/table1_gene.tex").read_text()
    for line in tex.split("\n"):
        if "&" not in line or not line.endswith(r"\\") or line.startswith("&"):
            continue
        name = re.sub(r"~\\citep\{.*?\}|\\textbf\{|\\underline\{|\}|\\textsuperscript\{.*?\}", "",
                      line.split("&")[0]).strip()
        name = {"MoSPR (ours)": "MoSPR (ours)"}.get(name, name)
        vals = []
        for x in line[:-2].split("&")[1:]:
            t = re.sub(r"[^0-9.]", "", x)
            vals.append(float(t) if t else np.nan)
        for (met, j), c in cells.items():
            r = df[(df.cohort == c) & (df.model == name) & (df.metric == met)]
            if not len(r):
                continue
            if abs(round(r["mean"].iloc[0], 3) - vals[j]) > 0.0006:
                bad.append((c, name, met, vals[j], round(r["mean"].iloc[0], 4)))
    return bad


def main(a):
    g, p = gene_table(), pathway_table()
    full = pd.concat([g, p], ignore_index=True)
    out = ROOT / "results/tables/reproduced_summary.csv"
    full.round(5).to_csv(out, index=False)
    pd.set_option("display.width", 200)
    for met in ("gene SCC", "hallmark SCC (median)"):
        t = paired_vs_ours(full, met)
        print(f"\n=== {met} (mean, delta vs MoSPR, paired p) ===")
        print(t.pivot_table(index="model", columns="cohort",
                            values=["mean", "delta_vs_ours", "p_paired"]).round(4).to_string())
    print(f"\nwritten: {out}")
    if a.check:
        bad = check_against_tex(full)
        print(f"\nTable 1 cross-check: {len(bad)} mismatch(es)")
        for b in bad:
            print("  ", b)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="diff the aggregates against table1_gene.tex")
    main(ap.parse_args())
