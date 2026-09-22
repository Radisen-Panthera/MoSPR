import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / 'mospr'))
import paths as _p
RESULTS = _p.RESULTS
TABLES = _p.TABLES
import argparse
import os, pathlib
import pandas as pd

R = _p.ROOT
RES = RESULTS
SRC = os.environ.get("MOSPR_TABLE_SRC", str(_p.TABLES / "per_cohort"))
COH = ["BRCA", "KIRC", "LUAD"]






ROWS1 = [("AbMIL max-pool", r"Max~\citep{wang2018revisiting}"),
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
 ("MoSPR", r"\textbf{MoSPR (ours)}")]

PNAMES = {"MoSPR (ours)": "MoSPR", "Max": "AbMIL max-pool", "Mean": "AbMIL mean-pool",
          "S4MIL": "S4Model"}

HDR = """% Setting: filtered patches (BRCA 15,483), patient-level split, train+val fine-tuning.
% K=8, d=512, tau=0, seed 2021, ALPHA_SMOOTH={alpha}
%
% alpha smooths the adjacency matrix: mean(C)*alpha is added to every cell before the
% spectral embedding, which is what fixes the MoSPR macrostates.
% Baselines do not use it, so their numbers are identical across alpha variants.
% At alpha=1 half of the edge mass becomes an artificial uniform constant; accuracy is
% flat (sweep over 0-10 spans 0.004) but macrostate stability degrades.
%
% Bold = best in column, underline = runner-up.
%
%    (10x Visium) co-training could not be applied as no public spot data exists, and the
%    released checkpoints were trained on TCGA and overlap our test folds.
%
%    Sentence for the caption:
%    the authors' spot-level (10x Visium) co-training could not be applied as no
%    public spot data accompanies these cohorts, and the released checkpoints were
%    trained on TCGA and therefore overlap our test folds. These numbers do not
"""


def load1(alpha_tag):
    tab = {}
    for c in COH:



        f = R / f"{SRC}/{c}/results/baselines_summary.csv"
        if not f.exists():
            f = R / f"Baseline_Filtered_PatientSplit/{c}/results/baselines_summary.csv"
        b = pd.read_csv(f)
        tab[c] = b[b.folds == 4].set_index("model")
        m = RES / f"cohort_{c}_fpsplit{alpha_tag}_tv_ours.csv"
        if m.exists():
            g = pd.read_csv(m)
            g = g[(g.primary) & (g.space == "gene")]
            tab[c].loc["MoSPR", ["SCC", "PCC"]] = [g.test_SCC.mean(), g.test_PCC.mean()]
    return tab


def emit(rows, cols, get, out, caption_extra=""):
    rank = {}
    for j in range(len(cols)):
        v = [(m, get(m, j)) for m, _ in rows if m and m != "__MID__" and get(m, j) is not None]
        v.sort(key=lambda x: -x[1])
        rank[j] = (v[0][0], v[1][0]) if len(v) > 1 else (v[0][0], None)
    def cell(m, j):
        x = get(m, j)
        if x is None: return r"\res"
        s = f"{x:.3f}"
        if m == rank[j][0]: return r"\textbf{" + s + "}"
        if m == rank[j][1]: return r"\underline{" + s + "}"
        return s
    L = []
    for m, label in rows:
        if m == "__MID__": L.append(r"\midrule"); continue
        L.append(label + " & " + " & ".join(cell(m, j) for j in range(len(cols))) + r"\\")
    return "\n".join(L)


def main(a):
    tag = "" if a.alpha == "0" else "_a1"
    out = R / a.out
    out.mkdir(parents=True, exist_ok=True)
    hdr = HDR.format(alpha=a.alpha)


    tab = load1(tag)
    cols = [(c, k) for c in COH for k in ["PCC", "SCC"]]
    def g1(m, j):
        c, k = cols[j]
        t = tab[c]
        return t.loc[m, k] if m in t.index and pd.notna(t.loc[m, k]) else None
    rows1 = [r for r in ROWS1 if r[0] not in a.exclude]
    body = emit(rows1, cols, g1, out)
    (out / "table1_gene.tex").write_text(hdr + r"""\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{BRCA} & \multicolumn{2}{c}{KIRC} & \multicolumn{2}{c}{LUAD}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Method & PCC & SCC & PCC & SCC & PCC & SCC\\
\midrule
""" + body + "\n" + r"""\bottomrule
\end{tabular}""")
    print(f"  table1_gene.tex")


    COLL = ["hallmark", "gobp", "kegg"]
    cols2 = [(c, k) for c in COH for k in COLL]
    p2 = {}
    for c in COH:


        f = RES / f"pathway_all_{c}_fpsplit{tag}_tv.csv"
        if f.exists():
            d = pd.read_csv(f)
            ok = d.groupby("model").fold.nunique(); ok = set(ok[ok == 4].index)
            p2[c] = d[d.model.isin(ok)].pivot_table(index="model", columns="collection",
                                                    values="SCC_median", aggfunc="mean")
    if p2:
        rows2 = [(PNAMES.get(m, m) if m != "__MID__" else m, lb) for m, lb in rows1]
        inv = {v: k for k, v in PNAMES.items()}
        def g2(m, j):
            c, k = cols2[j]
            if c not in p2: return None
            key = inv.get(m, m)
            t = p2[c]
            return t.loc[key, k] if key in t.index else None
        body2 = emit(rows2, cols2, g2, out)
        (out / "table2_pathway.tex").write_text(hdr + r"""\begin{tabular}{lccccccccc}
\toprule
& \multicolumn{3}{c}{BRCA} & \multicolumn{3}{c}{KIRC} & \multicolumn{3}{c}{LUAD}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Method & Hallmark & GO-BP & KEGG & Hallmark & GO-BP & KEGG & Hallmark & GO-BP & KEGG\\
\midrule
""" + body2 + "\n" + r"""\bottomrule
\end{tabular}""")
        print(f"  table2_pathway.tex")


    f3 = RES / f"ablation_v2_BRCA_fpsplit{tag}_tv_p512.csv"
    if f3.exists():
        d = pd.read_csv(f3).groupby("variant")[["gene_SCC", "hall_SCC"]].mean()
        AB = [("Global Direct", "$M$", "Direct"), ("Global Low-Rank", "$M$", "$U_q$"),
              ("Spatial Direct", r"$[M\mid S]$", "Direct"), ("__MID__", None, None),
              ("Shuffled-State Low-Rank", r"$[M\mid S_{\mathrm{shuffle}}]$", "$U_q$"),
              ("Spatial Low-Rank", r"$[M\mid S]$", "$U_q$")]
        gb, hb = d.gene_SCC.idxmax(), d.hall_SCC.idxmax()
        L = []
        for k, hi, mo in AB:
            if k == "__MID__": L.append(r"\midrule"); continue
            lab = r"\textbf{Spatial Low-Rank (ours)}" if k == "Spatial Low-Rank" else k
            f = lambda col, best: (r"\textbf{" + f"{d.loc[k,col]:.3f}" + "}") if k == best else f"{d.loc[k,col]:.3f}"
            L.append(f"{lab} & {hi} & {mo} & {f('gene_SCC',gb)} & {f('hall_SCC',hb)}" + r"\\")
        (out / "table3_ablation.tex").write_text(hdr + r"""\begin{tabular}{lcccc}
\toprule
Variant & Histology input & Molecular output & Gene SCC & Hallmark SCC\\
\midrule
""" + "\n".join(L) + "\n" + r"""\bottomrule
\end{tabular}""")
        print(f"  table3_ablation.tex")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", choices=["0", "1"], default="0")
    ap.add_argument("--out", default=str(_p.TABLES / "tables/final/alpha0"))
    ap.add_argument("--exclude", nargs="*", default=[])
    main(ap.parse_args())
