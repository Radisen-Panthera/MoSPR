from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import config


def rasterize(coords: np.ndarray, patch_size: int = config.PATCH_SIZE):
    grid = (coords // patch_size).astype(np.int64)
    g = grid - grid.min(axis=0)
    shape = (int(g[:, 0].max()) + 1, int(g[:, 1].max()) + 1)
    return g, shape


def tumor_mask(
    g: np.ndarray, macro: np.ndarray, shape: tuple[int, int],
    non_tumor_ms=config.NON_TUMOR_MS,
) -> np.ndarray:
    tumor = np.zeros(shape, dtype=bool)
    is_tumor = ~np.isin(macro, non_tumor_ms)
    tumor[g[is_tumor, 0], g[is_tumor, 1]] = True
    tumor = ndimage.binary_closing(tumor, structure=np.ones((3, 3)))
    tumor = ndimage.binary_opening(tumor, structure=np.ones((3, 3)))
    return tumor


def signed_distance(tumor: np.ndarray, sigma: float = 2.0) -> np.ndarray | None:
    if tumor.all() or not tumor.any():
        return None
    f = (ndimage.distance_transform_edt(~tumor)
         - ndimage.distance_transform_edt(tumor)).astype(np.float64)
    return ndimage.gaussian_filter(f, sigma=sigma)


def patch_gradient(f: np.ndarray, g: np.ndarray):
    dfx, dfy = np.gradient(f)
    vec = np.stack([dfx[g[:, 0], g[:, 1]], dfy[g[:, 0], g[:, 1]]], axis=1)
    mag = np.linalg.norm(vec, axis=1)
    ghat = vec / np.maximum(mag, 1e-12)[:, None]
    return ghat, mag


def directional_cosines(edges: np.ndarray, g: np.ndarray, ghat: np.ndarray):
    d = (g[edges[:, 1]] - g[edges[:, 0]]).astype(np.float64)
    dhat = d / np.linalg.norm(d, axis=1, keepdims=True)
    gu, gv = ghat[edges[:, 0]], ghat[edges[:, 1]]
    cos_uv = np.einsum("ij,ij->i", gu, dhat)
    cos_vu = -np.einsum("ij,ij->i", gv, dhat)
    sin_uv = gu[:, 0] * dhat[:, 1] - gu[:, 1] * dhat[:, 0]
    sin_vu = -(gv[:, 0] * dhat[:, 1] - gv[:, 1] * dhat[:, 0])
    return cos_uv, cos_vu, sin_uv, sin_vu


def directional_pair_mass(
    macro: np.ndarray, edges: np.ndarray,
    cos_uv: np.ndarray, cos_vu: np.ndarray,
    mag: np.ndarray, K: int, min_grad: float = 0.1,
):
    a, b = macro[edges[:, 0]], macro[edges[:, 1]]
    valid_uv = mag[edges[:, 0]] >= min_grad
    valid_vu = mag[edges[:, 1]] >= min_grad
    D = np.zeros((K, K)); Tn = np.zeros((K, K)); N = np.zeros((K, K))
    for src, dst, cos, ok in [(a, b, cos_uv, valid_uv), (b, a, cos_vu, valid_vu)]:
        np.add.at(D, (src[ok], dst[ok]), np.maximum(cos[ok], 0.0))
        np.add.at(Tn, (src[ok], dst[ok]), 1.0 - np.abs(cos[ok]))
        np.add.at(N, (src[ok], dst[ok]), 1.0)
    return D, Tn, N


def asymmetry_features(D: np.ndarray, eps: float = 1.0) -> np.ndarray:
    iu = np.triu_indices(D.shape[0], k=1)
    return (D[iu] - D.T[iu]) / (D[iu] + D.T[iu] + eps)


def tangent_features(Tn: np.ndarray, N: np.ndarray, eps: float = 1.0) -> np.ndarray:
    iu = np.triu_indices(Tn.shape[0], k=0)
    return (Tn[iu] + Tn.T[iu]) / (N[iu] + N.T[iu] + eps)


def asymmetry_names(K: int) -> list[str]:
    iu = np.triu_indices(K, k=1)
    return [f"dir:MS{a}>MS{b}" for a, b in zip(*iu)]


def tangent_names(K: int) -> list[str]:
    iu = np.triu_indices(K, k=0)
    return [f"tan:MS{a}-MS{b}" for a, b in zip(*iu)]
