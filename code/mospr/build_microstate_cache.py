import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import paths as _p
import argparse
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

sys.path.insert(0, str(_p.CODE))
from spatial_proteome import adjacency, config, spectral

import sys
import cohorts
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--cohort", default="BRCA")
_ap.add_argument("--variant", choices=["", "paper"], default="")
_k = _ap.parse_known_args()[0]
CFG = cohorts.get(_k.cohort)
if _k.variant == "paper":
    GENE = CFG["gene_root"].with_name(f"{_k.cohort}-paper-digital_slide")
    PATH = CFG["path_root"].with_name(f"{_k.cohort}-paper-pathway-digital_slide")
else:
    GENE, PATH = CFG["gene_root"], CFG["path_root"]
import os as _os_res
RES = Path(_os_res.environ.get("MOSPR_RESULTS_ROOT",
                               str(_p.RESULTS)))
PATCH, SEED, SAMPLES_PER_SLIDE, M = 256, 42, 2000, 200


def target(tpm):
    tpm = np.asarray(tpm, dtype=np.float64)
    return np.log1p(tpm / tpm.sum() * 1e4).astype(np.float32)


def load_gene(name):
    with h5py.File(GENE / "sample_pair_feature_conch" / f"{name}.h5", "r") as f:
        return (f["coord"][:].astype(np.int64), f["feat"][:].astype(np.float32),
                f["tpm"][:])


def load_pathway_target(name):
    with h5py.File(PATH / "sample_pair_feature_conch" / f"{name}.h5", "r") as f:
        return f["tpm"][:]


def micro_summary(feats, lab):
    n = len(lab)
    onehot = sparse.csr_matrix((np.ones(n, dtype=np.float32), (lab, np.arange(n))),
                               shape=(M, n))
    counts = np.asarray(onehot.sum(axis=1)).ravel()
    sums = onehot @ feats
    return sums / np.maximum(counts, 1)[:, None], counts


def run_fold(fold, split, n_pca, suffix="", fit_on="trainval"):
    t0 = time.time()
    tr, va, te = ([p.stem for p in split[fold][s]] for s in ("train", "val", "test"))
    slides = tr + va + te
    fit_list = tr if fit_on == "train" else tr + va
    fit_slides = set(fit_list)
    rng = np.random.default_rng(SEED)

    samples = []
    for s in fit_list:
        _, feats, _ = load_gene(s)
        if feats.shape[0] > SAMPLES_PER_SLIDE:
            feats = feats[np.sort(rng.choice(feats.shape[0], SAMPLES_PER_SLIDE, False))]
        samples.append(feats)
    sample = np.concatenate(samples)
    del samples
    km = MiniBatchKMeans(n_clusters=M, batch_size=4096, n_init=10,
                         max_iter=300, random_state=SEED).fit(sample)
    sub = sample if len(sample) <= 300_000 else sample[
        np.random.default_rng(SEED).choice(len(sample), 300_000, replace=False)]
    pca = PCA(n_components=n_pca, random_state=SEED).fit(sub)
    del sample, sub
    print(f"[fold {fold}] k-means + PCA ({time.time()-t0:.0f}s)", flush=True)

    mu_all, cnt_all, C_all, Yg, Yp = [], [], [], [], []
    C_fit = []
    for s in slides:
        coords, feats, tpm = load_gene(s)
        lab = km.predict(feats).astype(np.int64)
        mu, cnt = micro_summary(feats, lab)
        C = adjacency.microstate_adjacency(
            lab, adjacency.grid_edges(coords, patch_size=PATCH, ring=1), n_states=M)
        mu_all.append(mu.astype(np.float32))
        cnt_all.append(cnt.astype(np.float32))
        C_all.append(np.asarray(C.todense(), dtype=np.float32))
        Yg.append(target(tpm))
        Yp.append(target(load_pathway_target(s)))
        if s in fit_slides:
            C_fit.append(C)
    print(f"[fold {fold}] per-slide summaries ({time.time()-t0:.0f}s)", flush=True)

    micro_sizes = np.zeros(M)
    for s, c in zip(slides, cnt_all):
        if s in fit_slides:
            micro_sizes += c


    _al = _os_res.environ.get("MOSPR_ALPHA")
    _kw = {} if _al is None else {"alpha": float(_al)}
    if _al is not None:
        print(f"[cache] alpha={_al} (default {config.ALPHA_SMOOTH})", flush=True)
    _, emb = spectral.spectral_embedding(spectral.aggregate_global(C_fit),
                                         n_vectors=20, **_kw)

    pos = {n: i for i, n in enumerate(slides)}
    np.savez(RES / f"micro_cache{suffix}_fold{fold}.npz",
             slides=np.array(slides),
             idx_tr=np.array([pos[s] for s in tr]),
             idx_va=np.array([pos[s] for s in va]),
             idx_te=np.array([pos[s] for s in te]),
             micro_mean=np.stack(mu_all), micro_count=np.stack(cnt_all),
             C=np.stack(C_all), micro_sizes=micro_sizes, emb=emb,
             pca_mean=pca.mean_.astype(np.float32),
             pca_components=pca.components_.astype(np.float32),
             Y_gene=np.stack(Yg), Y_pathway=np.stack(Yp))
    print(f"[fold {fold}] cached ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--cohort", default="BRCA")
    ap.add_argument("--variant", choices=["", "paper"], default="")
    ap.add_argument("--pca", type=int, default=32)
    ap.add_argument("--suffix", default="",
                    help="output suffix so a re-fit does not clobber the cache")
    ap.add_argument("--fit_on", choices=["trainval", "train"], default="trainval",
                    help="slides the representation is fit on: train or trainval")
    a = ap.parse_args()


    _sf = _os_res.environ.get("MOSPR_SPLIT_FILE", "split.pkl")
    print(f"[cache] split={_sf}", flush=True)
    with open(GENE / _sf, "rb") as f:
        split = pickle.load(f)
    for fold in a.folds:
        run_fold(fold, split, a.pca, a.suffix, a.fit_on)
