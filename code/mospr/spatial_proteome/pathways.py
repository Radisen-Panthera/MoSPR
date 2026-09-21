from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HALLMARK_URL = (
    "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/"
    "2025.1.Hs/h.all.v2025.1.Hs.symbols.gmt"
)


def parse_gmt(path: Path) -> dict[str, list[str]]:
    gene_sets = {}
    for line in Path(path).read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        gene_sets[parts[0]] = [g for g in parts[2:] if g]
    return gene_sets


def pathway_coverage(
    gene_sets: dict[str, list[str]], protein_cols, min_members: int = 5
) -> pd.DataFrame:
    measured = set(protein_cols)
    rows = []
    for name, genes in gene_sets.items():
        hit = sorted(measured.intersection(genes))
        rows.append({
            "pathway": name,
            "set_size": len(genes),
            "n_measured": len(hit),
            "coverage": len(hit) / len(genes),
            "kept": len(hit) >= min_members,
        })
    return pd.DataFrame(rows).set_index("pathway").sort_values("n_measured", ascending=False)


def member_index(
    gene_sets: dict[str, list[str]], protein_cols, min_members: int = 5
) -> dict[str, np.ndarray]:
    col_pos = {g: i for i, g in enumerate(protein_cols)}
    out = {}
    for name, genes in gene_sets.items():
        idx = np.array(sorted(col_pos[g] for g in genes if g in col_pos), dtype=int)
        if len(idx) >= min_members:
            out[name] = idx
    return out


def score_mean_z(Y_z: np.ndarray, members: dict[str, np.ndarray], index) -> pd.DataFrame:
    cols = {name: Y_z[:, idx].mean(axis=1) for name, idx in members.items()}
    return pd.DataFrame(cols, index=index)


def score_ssgsea(
    Y_log2: pd.DataFrame, gene_sets: dict[str, list[str]], min_members: int = 5,
    threads: int = 16, seed: int = 42,
) -> pd.DataFrame:
    import gseapy

    expr = Y_log2.T
    res = gseapy.ssgsea(
        data=expr,
        gene_sets={k: list(v) for k, v in gene_sets.items()},
        min_size=min_members,
        max_size=5000,
        threads=threads,
        seed=seed,
        outdir=None,
        verbose=False,
    )
    nes = res.res2d.pivot(index="Name", columns="Term", values="NES")
    nes = nes.astype(float).loc[Y_log2.index]
    return nes
