from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import config


def proportions(micro_labels: np.ndarray, R_labels: np.ndarray, K: int) -> np.ndarray:
    macro = R_labels[micro_labels]
    counts = np.bincount(macro, minlength=K).astype(np.float64)
    return counts / counts.sum()


def coarse_grain(C: sp.csr_matrix, R: np.ndarray) -> np.ndarray:
    return np.asarray(R.T @ (C @ R))


def expected_adjacency(p: np.ndarray, total_edges: float) -> np.ndarray:
    E = 2.0 * total_edges * np.outer(p, p)
    return E


def interface_enrichment(
    Q_raw: np.ndarray,
    p: np.ndarray,
    eps: float = config.EPS_ENRICH,
) -> np.ndarray:
    total_edges = Q_raw.sum() / 2.0
    E = expected_adjacency(p, total_edges)
    return np.log((Q_raw + eps) / (E + eps))


def upper_triangle(Qmat: np.ndarray, include_diag: bool = True) -> np.ndarray:
    K = Qmat.shape[0]
    iu = np.triu_indices(K, k=0 if include_diag else 1)
    return Qmat[iu]


def upper_triangle_names(K: int, include_diag: bool = True) -> list[str]:
    iu = np.triu_indices(K, k=0 if include_diag else 1)
    return [f"MS{a}-MS{b}" for a, b in zip(*iu)]
