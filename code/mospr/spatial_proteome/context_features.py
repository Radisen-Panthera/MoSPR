from __future__ import annotations

import numpy as np
from scipy import sparse

from . import adjacency, config


def neighborhood_mean(
    feats: np.ndarray, coords: np.ndarray, rings: int,
    patch_size: int = config.PATCH_SIZE,
) -> np.ndarray:
    n = len(feats)
    pairs = [adjacency.grid_edges(coords, patch_size=patch_size, ring=r)
             for r in range(1, rings + 1)]
    pairs = [e for e in pairs if len(e)]
    if not pairs:
        return feats.astype(np.float64)
    e = np.vstack(pairs)


    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    A = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                          shape=(n, n))
    total = A @ feats + feats
    count = np.asarray(A.sum(axis=1)).ravel() + 1.0
    return total / count[:, None]


def build_variant(variant: str, z: np.ndarray, ctx: np.ndarray | None) -> np.ndarray:
    if variant == "center":
        return z
    if variant == "mean":
        return ctx
    if variant == "cat":
        return np.hstack([z, ctx])
    if variant == "contrast":
        return np.hstack([z, ctx - z])
    raise ValueError(f"unknown variant {variant!r}")
