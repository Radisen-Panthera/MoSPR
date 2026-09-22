import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / 'mospr'))
import paths as _p
RESULTS = _p.RESULTS
TABLES = _p.TABLES
import argparse, pathlib
import numpy as np
import pandas as pd
from scipy import stats

R = _p.ROOT
                                                 
import os
SRC = os.environ.get("MOSPR_TABLE_SRC", str(_p.TABLES / "per_cohort"))
TCRIT = stats.t.ppf(0.975, 3)
OURS = "MoSPR (ours)"
COH = ["BRCA", "KIRC", "LUAD"]

                                                   
                                                     
                                                    
                                        
ROWS = [("AbMIL max-pool", r"Max~\citep{wang2018revisiting}"),
        ("AbMIL mean-pool", r"Mean~\citep{wang2018revisiting}"),
        ("AbMIL", r"AbMIL~\citep{ilse2018abmil}"),
        ("HE2RNA", r"HE2RNA~\citep{schmauch2020he2rna}"),
        ("AbReg", r"AbReg~\citep{graziani2022abreg}"),
        ("tRNAformer", r"tRNAformer~\citep{alsaafin2022trnaformer}"),
        ("ILRA", r"ILRA~\citep{xiang2023ilra}"),
        ("S4Model", r"S4MIL~\citep{fillioux2023s4mil}"),
        ("MambaMIL", r"MambaMIL~\citep{yang2024mambamil}"),
        ("SRMambaMIL", r"SRMambaMIL~\citep{yang2024mambamil}"),
        ("MOSBY", r"MOSBY~\citep{senbabaoglu2024mosby}"),
        ("SEQUOIA VIS", r"SEQUOIA VIS~\citep{pizurica2024sequoia}"),
        ("2DMamba", r"2DMamba~\citep{zhang2025twodmamba}"),
        ("CPNN", r"CPNN~\citep{nishimura2026cpnn}"),
        ("__MID__", None),
        (OURS, r"\textbf{MoSPR (ours)}")]


def perfold(c, tag=""):
    d = R / SRC / c / "results"
    g = pd.read_csv(d / "baselines_per_fold.csv")
    sc = [x for x in g.columns if x.upper() == "SCC"][0]
    pc = [x for x in g.columns if x.upper() == "PCC"][0]
    out = {m: (s.sort_values("fold")[pc].values, s.sort_values("fold")[sc].values)
           for m, s in g.groupby("model")}
    r = d / "cpnn_rescored.csv"
    if r.exists() and r.stat().st_size > 20:
        t = pd.read_csv(r).sort_values("fold")
        out["CPNN"] = (t.PCC.values, t.SCC.values)
    m = pd.read_csv(RESULTS / f"cohort_{c}_fpsplit{tag}_tv_ours.csv")
    m = m[(m.primary) & (m.space == "gene")].sort_values("fold")
    out[OURS] = (m.test_PCC.values, m.test_SCC.values)
    return {k: (np.asarray(a, float), np.asarray(b, float))
            for k, (a, b) in out.items() if len(a) == 4}


def fmt_p(p):
    return r"\textless.001" if p < 1e-3 else f"{p:.3f}"



HEADS = {
    "ci":  ("Table 1 - mean +- 95% CI",
            "% Values are the 4-fold mean +- 95% CI, taken from the t distribution across folds\n"
            "% (df=3, t=3.182). The repeated unit is the fold, not the gene: genes are\n"
            "% strongly correlated, so intervals over genes come out unrealistically narrow.\n"
            "% Note: intervals are computed per model. Overlapping intervals of two models do\n"
            "% not mean there is no difference: every model sees the same folds (same test slides),\n"
            "% so comparisons between models must be paired (see the p-value variant).\n"),
    "std": ("Table 1 - mean (fold SD)",
            "% Parentheses hold the 4-fold standard deviation (ddof=1), i.e. spread, not an interval.\n"
            "% With only four folds the standard deviation itself is an unstable estimate.\n"),
    "p":   ("Table 1 - mean and p against MoSPR",
            "% p is for the fold-paired difference in gene SCC between MoSPR and that model,\n"
            "% (n=4 folds, df=3). The null hypothesis is that the mean fold-wise SCC difference\n"
            "% between the two models is zero. All models see the same four folds, so the runs are\n"
            "% paired: treating them as independent samples would count fold difficulty as\n"
            "% noise and lose power.\n"
            "% No multiple-comparison correction (14 comparisons per cohort).\n"),
    "all": ("Table 1 - mean +- 95% CI (SD), p against MoSPR",
            "% Each cell is mean +- 95% CI (t across folds, df=3), with the standard deviation in parentheses.\n"
            "% p is the two-sided paired t-test on the fold-wise gene SCC difference against MoSPR\n"
            "% (n=4, df=3). No multiple-comparison correction.\n"),
}


def build(kind, data, alpha, exclude=()):
    withp = kind in ("p", "all")
    ncol = 3 * (3 if withp else 2)
    spec = "l" + "c" * ncol
    sub = ("PCC & SCC & $p$" if withp else "PCC & SCC")
    span = 3 if withp else 2
    heads, cmids, i = [], [], 2
    for c in COH:
        heads.append(r"\multicolumn{%d}{c}{%s}" % (span, c))
        cmids.append(r"\cmidrule(lr){%d-%d}" % (i, i + span - 1)); i += span

    best = {(c, j): max(v[j].mean() for v in data[c].values()) for c in COH for j in (0, 1)}
    L = []
    for key, lab in [r for r in ROWS if r[0] not in exclude]:
        if key == "__MID__":
            L.append(r"\midrule"); continue
        cells = []
        for c in COH:
            d = data[c]
            for j in (0, 1):
                if key not in d:
                    cells.append("--"); continue
                v = d[key][j]; mu = v.mean()
                sd = v.std(ddof=1); hw = TCRIT * sd / 2.0
                if kind == "ci":
                    s = f"{mu:.3f}\\,{{\\tiny$\\pm$}}{hw:.3f}"
                elif kind == "std":
                    s = f"{mu:.3f}\\,{{\\tiny({sd:.3f})}}"
                elif kind == "p":
                    s = f"{mu:.3f}"
                else:
                    s = f"{mu:.3f}\\,{{\\tiny$\\pm$}}{hw:.3f}\\,{{\\tiny({sd:.3f})}}"
                cells.append(r"\textbf{" + s + "}" if abs(mu - best[(c, j)]) < 1e-12 else s)
            if withp:
                if key == OURS or key not in d:
                    cells.append("--")
                else:
                    _, p = stats.ttest_rel(d[OURS][1], d[key][1])
                    cells.append(fmt_p(p))
        L.append(f"{lab} & " + " & ".join(cells) + r"\\")

    title, note = HEADS[kind]
    hdr = (f"% {title}\n"
           f"% Setting: filtered patches, patient-level split, train+val fine-tuning, seed 2021\n"
           f"% K=8 · d=512 · tau=0 · ALPHA_SMOOTH={alpha}\n" + note)
    return (hdr + r"\begin{tabular}{" + spec + "}\n" + r"\toprule" + "\n"
            + "& " + " & ".join(heads) + r"\\" + "\n"
            + "".join(cmids) + "\n"
            + "Method & " + " & ".join([sub] * 3) + r"\\" + "\n"
            + r"\midrule" + "\n" + "\n".join(L) + "\n"
            + r"\bottomrule" + "\n" + r"\end{tabular}")


def main(a):
    tag = "" if a.alpha == "0" else "_a1"
    data = {c: perfold(c, tag) for c in COH}
    out = R / a.out; out.mkdir(parents=True, exist_ok=True)
    for kind in ("ci", "std", "p", "all"):
        f = out / f"table1_gene_{kind}.tex"
        f.write_text(build(kind, data, a.alpha, a.exclude))
        print(f"  {f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", choices=["0", "1"], default="0")
    ap.add_argument("--out", default=str(_p.TABLES / "tables/final/alpha0"))
    ap.add_argument("--exclude", nargs="*", default=[])
    main(ap.parse_args())
