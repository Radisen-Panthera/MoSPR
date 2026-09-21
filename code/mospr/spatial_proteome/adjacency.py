from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import config


def _half_ring_offsets(ring: int) -> np.ndarray:
    offsets = [
        (dx, dy)
        for dx in range(-ring, ring + 1)
        for dy in range(-ring, ring + 1)
        if max(abs(dx), abs(dy)) == ring and (dy > 0 or (dy == 0 and dx > 0))
    ]
    return np.asarray(offsets, dtype=np.int64)


def grid_edges(
    coords: np.ndarray, patch_size: int = config.PATCH_SIZE, ring: int = 1,
) -> np.ndarray:
    grid = coords // patch_size
    index = {(int(x), int(y)): i for i, (x, y) in enumerate(grid)}
    offsets = _half_ring_offsets(ring)
    edges = []
    for i, (x, y) in enumerate(grid):
        for dx, dy in offsets:
            j = index.get((int(x) + int(dx), int(y) + int(dy)))
            if j is not None:
                edges.append((i, j))
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2)


def microstate_adjacency(
    labels: np.ndarray,
    edges: np.ndarray,
    n_states: int = config.M_MICROSTATES,
) -> sp.csr_matrix:
    if len(edges) == 0:
        return sp.csr_matrix((n_states, n_states), dtype=np.float64)
    a = labels[edges[:, 0]]
    b = labels[edges[:, 1]]
    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])
    vals = np.ones(len(rows), dtype=np.float64)
    C = sp.coo_matrix((vals, (rows, cols)), shape=(n_states, n_states))
    return C.tocsr()
