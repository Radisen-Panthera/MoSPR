from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans

from . import config


def aggregate_global(C_list: list[sp.csr_matrix]) -> np.ndarray:
    M = C_list[0].shape[0]
    acc = np.zeros((M, M), dtype=np.float64)
    n_used = 0
    for C in C_list:
        total = C.sum()
        if total == 0:
            continue
        acc += (C / total).toarray()
        n_used += 1
    return acc / max(n_used, 1)


def transition_matrix(C_global: np.ndarray, alpha: float = config.ALPHA_SMOOTH) -> np.ndarray:
    a = alpha * C_global.mean()
    Cs = C_global + a
    return Cs / Cs.sum(axis=1, keepdims=True)


def spectral_embedding(
    C_global: np.ndarray,
    n_vectors: int = config.N_EIGENVECTORS,
    alpha: float = config.ALPHA_SMOOTH,
    mode: str = "lambda_psi",
):
    if mode not in ("lambda_psi", "psi", "u"):
        raise ValueError(f"unknown embedding mode: {mode!r}")
    C_sym = (C_global + C_global.T) / 2.0
    C_sym = C_sym + alpha * C_sym.mean()
    d = C_sym.sum(axis=1)
    if not np.all(d > 0):


        dead = np.flatnonzero(d <= 0)
        raise ValueError(
            f"{len(dead)} microstate(s) have zero degree ({dead[:10].tolist()}"
            f"{'...' if len(dead) > 10 else ''}) -- the graph is disconnected at "
            f"alpha={alpha}. Either drop these microstates or pass a small "
            f"positive alpha (config.ALPHA_SMOOTH_LEGACY reproduces the old "
            f"behaviour).")
    d_inv_sqrt = 1.0 / np.sqrt(d)
    S = C_sym * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    eigvals, eigvecs = np.linalg.eigh(S)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    u = eigvecs[:, 1 : n_vectors + 1]
    if mode == "u":
        return eigvals, u
    psi = u * d_inv_sqrt[:, None]
    if mode == "psi":
        return eigvals, psi
    return eigvals, psi * eigvals[1 : n_vectors + 1][None, :]


def joint_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    blocks = []
    for emb in embeddings:
        scale = np.linalg.norm(emb)
        blocks.append(emb / max(scale, 1e-12))
    return np.hstack(blocks)


def macrostate_assignment(
    embedding: np.ndarray,
    n_macrostates: int = config.K_MACROSTATES,
    seed: int = config.RANDOM_SEED,
    sample_weight: np.ndarray | None = None,
    row_normalise: bool = True,
    n_init: int = config.MACRO_N_INIT,
) -> tuple[np.ndarray, np.ndarray]:
    if row_normalise:
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        X = embedding / np.maximum(norms, 1e-12)
    else:
        X = embedding
    km = KMeans(n_clusters=n_macrostates, n_init=n_init, random_state=seed)
    labels = km.fit_predict(X, sample_weight=sample_weight)
    M = embedding.shape[0]
    R = np.zeros((M, n_macrostates), dtype=np.float64)
    R[np.arange(M), labels] = 1.0
    return labels, R
