from __future__ import annotations

import numpy as np


def macrostate_features(
    feats: np.ndarray, macro: np.ndarray, K: int,
    centroid: np.ndarray | None = None, tau: float = 0.0,
):
    D = feats.shape[1]
    H = np.zeros((K, D), dtype=np.float64)
    counts = np.bincount(macro, minlength=K).astype(np.float64)


    for k in range(K):
        if counts[k]:
            H[k] = feats[macro == k].mean(axis=0)
    if tau > 0 and centroid is not None:
        w = counts[:, None]
        H = (w * H + tau * centroid) / (w + tau)
    return H, counts


def slide_mean(H: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ H


def state_design(
    H: np.ndarray, p: np.ndarray, center: bool = True,
) -> np.ndarray:
    Hc = H - slide_mean(H, p)[None, :] if center else H
    return (p[:, None] * Hc).ravel()


def shuffled_macro(macro: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = macro.copy()
    rng.shuffle(out)
    return out
