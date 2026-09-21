from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import KFold

from . import adjacency, config, data, evaluation, pq, spectral


def fit_microstates(
    M: int, sids, seed: int = config.RANDOM_SEED,
    samples_per_patient: int = config.SAMPLES_PER_PATIENT,
) -> MiniBatchKMeans:
    rng = np.random.default_rng(seed)
    samples = []
    for sid in sids:
        _, feats = data.load_slide(sid)
        n = feats.shape[0]
        if n > samples_per_patient:
            idx = rng.choice(n, samples_per_patient, replace=False)
            feats = feats[np.sort(idx)]
        samples.append(feats)
    X = np.concatenate(samples).astype(np.float32)
    km = MiniBatchKMeans(n_clusters=M, batch_size=4096, n_init=10,
                         max_iter=300, random_state=seed)
    km.fit(X)
    return km


def assign_microstates(kmeans: MiniBatchKMeans, sids) -> dict[str, np.ndarray]:
    out = {}
    for sid in sids:
        _, feats = data.load_slide(sid)
        out[sid] = kmeans.predict(feats.astype(np.float32)).astype(np.int64)
    return out


def build_C_matrices(
    labels: dict[str, np.ndarray], M: int, ring: int = 1,
) -> dict[str, sp.csr_matrix]:
    out = {}
    for sid, lab in labels.items():
        coords, _ = data.load_slide(sid, with_features=False)
        edges = adjacency.grid_edges(coords, ring=ring)
        out[sid] = adjacency.microstate_adjacency(lab, edges, n_states=M)
    return out


def derive_macrostates(
    C_list: list[sp.csr_matrix], K: int, micro_sizes: np.ndarray,
    n_vectors: int = config.N_EIGENVECTORS, seed: int = config.RANDOM_SEED,
) -> np.ndarray:
    C_global = spectral.aggregate_global(C_list)
    n_vec = min(n_vectors, C_global.shape[0] - 1)
    _, embedding = spectral.spectral_embedding(C_global, n_vectors=n_vec)
    _, R = spectral.macrostate_assignment(embedding, n_macrostates=K, seed=seed,
                                          sample_weight=micro_sizes)
    return R


def derive_macrostates_multi(
    C_lists: dict[int, list[sp.csr_matrix]], K: int, micro_sizes: np.ndarray,
    n_vectors: int = config.N_EIGENVECTORS, seed: int = config.RANDOM_SEED,
) -> np.ndarray:
    embeddings = []
    for ring in sorted(C_lists):
        C_global = spectral.aggregate_global(C_lists[ring])
        n_vec = min(n_vectors, C_global.shape[0] - 1)
        _, emb = spectral.spectral_embedding(C_global, n_vectors=n_vec)
        embeddings.append(emb)
    joint = spectral.joint_embedding(embeddings)
    _, R = spectral.macrostate_assignment(joint, n_macrostates=K, seed=seed,
                                          sample_weight=micro_sizes)
    return R


def compute_PQ(
    labels: dict[str, np.ndarray], C: dict[str, sp.csr_matrix], R: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    K = R.shape[1]
    R_labels = R.argmax(axis=1)
    tri_names = pq.upper_triangle_names(K)
    P_rows, Q_rows = {}, {}
    for sid, lab in labels.items():
        p = pq.proportions(lab, R_labels, K)
        Q_raw = pq.coarse_grain(C[sid], R)
        P_rows[sid] = p
        Q_rows[sid] = pq.upper_triangle(pq.interface_enrichment(Q_raw, p))
    P_df = pd.DataFrame.from_dict(P_rows, orient="index",
                                  columns=[f"MS{k}" for k in range(K)])
    Q_df = pd.DataFrame.from_dict(Q_rows, orient="index", columns=tri_names)
    return P_df, Q_df


def compute_PQ_multi(
    labels: dict[str, np.ndarray],
    C_by_ring: dict[int, dict[str, sp.csr_matrix]],
    R: np.ndarray,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    K = R.shape[1]
    R_labels = R.argmax(axis=1)
    tri_names = pq.upper_triangle_names(K)
    P_rows = {sid: pq.proportions(lab, R_labels, K) for sid, lab in labels.items()}
    P_df = pd.DataFrame.from_dict(P_rows, orient="index",
                                  columns=[f"MS{k}" for k in range(K)])
    Q_dfs = {}
    for ring, C in sorted(C_by_ring.items()):
        Q_rows = {}
        for sid, lab in labels.items():
            Q_raw = pq.coarse_grain(C[sid], R)
            Q_rows[sid] = pq.upper_triangle(
                pq.interface_enrichment(Q_raw, P_rows[sid]))
        Q_dfs[ring] = pd.DataFrame.from_dict(
            Q_rows, orient="index", columns=[f"r{ring}:{n}" for n in tri_names])
    return P_df, Q_dfs


def combine_PQ(P_df: pd.DataFrame, Q_df: pd.DataFrame, fit_index) -> pd.DataFrame:
    q_mu = Q_df.loc[fit_index].values.mean(axis=0)
    q_sd = Q_df.loc[fit_index].values.std(axis=0)
    q_sd[q_sd == 0] = 1.0
    Qz = (Q_df.values - q_mu) / q_sd
    X = np.hstack([P_df.values, Qz])
    return pd.DataFrame(X, index=P_df.index)


def outer_cv_score(
    X: np.ndarray, Y: np.ndarray, n_folds: int = 5, seed: int = config.RANDOM_SEED,
    n_repeats: int = 1,
) -> pd.Series:
    fold_metrics = []
    for r in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
        for tr, va in kf.split(X):
            model, _, _ = evaluation.fit_ridge_cv(X[tr], Y[tr], seed=seed)
            pred = model.predict(X[va])
            fold_metrics.append(evaluation.summarize(Y[va], pred, "fold"))
    return pd.DataFrame(fold_metrics).mean()
