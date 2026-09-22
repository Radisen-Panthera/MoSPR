import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import paths as _p
import os as _os_sp
"""cell2location prototypes + deconvolution, mirroring CPNN's 4-cell2location_proc.py.

Two stages, because the expensive one does not actually depend on the fold:

  --stage sc         build the QC'd single-cell AnnData once (cached h5ad)
  --stage prototype  NB regression over cell types -> inf_aver (per label_key).
                     The gene axis is bulk ∩ single-cell, which is fold
                     independent, so CPNN's per-fold rerun of this step is
                     redundant; we run it once and share it across folds.
  --stage deconv     Cell2location on the fold's train+val bulk samples ->
                     {fold}/{resolution}_parameter_dict.pkl, the file
                     ProtoSum/PropDataset load. This one *is* fold specific
                     (it must not see test slides).

Label keys follow their BATCH_KEY_DICT for BRCA: batch "Patient";
celltype_minor = 29 types (their "fine"), celltype_subset = 49 types (their
"medium", and the count the paper quotes for its main result).
"""
import argparse
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.io import mmread

import cell2location
from cell2location.models import RegressionModel

sys.path.insert(0, str(_p.CODE))
import cohorts
import sc_loaders

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--cohort", default="BRCA")



_ap.add_argument("--variant", choices=["", "paper"], default="")
_known = _ap.parse_known_args()[0]
CFG = cohorts.get(_known.cohort)
if _known.variant == "paper":
    DST = CFG["gene_root"].with_name(f"{_known.cohort}-paper-digital_slide")
    WORK = CFG["c2l"].with_name(f"{CFG['c2l'].name}_paper")
else:
    DST = CFG["gene_root"]
    WORK = CFG["c2l"]
WORK.mkdir(parents=True, exist_ok=True)
BATCH_KEY = CFG["batch_key"]


def build_sc():
    out = WORK / "sc_adata.h5ad"
    if not out.exists():
        sc_loaders.load_sc(CFG, cache=out)
    return out


def common_gene_view(adata_sc, adata_bulk):
    common = np.intersect1d(adata_bulk.var_names, adata_sc.var_names)
    return adata_sc[:, common].copy(), adata_bulk[:, common].copy()


def run_prototype(label_key):
    out_csv = WORK / f"inf_aver_{label_key}.csv"
    if out_csv.exists():
        print(f"prototypes already at {out_csv}")
        return
    adata_sc = sc.read_h5ad(build_sc())
    adata_bulk = sc.read_h5ad(DST / "adata_bulk_processed.h5ad")
    adata_sc, _ = common_gene_view(adata_sc, adata_bulk)
    adata_sc = adata_sc[~adata_sc.obs[label_key].isna()].copy()
    adata_sc.X = adata_sc.layers["counts"]
    print(f"prototype fit: {adata_sc.shape}, {adata_sc.obs[label_key].nunique()} cell types")

    RegressionModel.setup_anndata(adata=adata_sc, batch_key=BATCH_KEY, labels_key=label_key)
    mod = RegressionModel(adata_sc)
    mod.train(max_epochs=250)
    adata_ref = mod.export_posterior(
        adata_sc, sample_kwargs={"num_samples": 1000, "batch_size": 2500})
    mod.save(str(WORK / f"sc_rec_{label_key}"), overwrite=True)

    names = adata_ref.uns["mod"]["factor_names"]
    inf_aver = adata_ref.varm["means_per_cluster_mu_fg"][
        [f"means_per_cluster_mu_fg_{i}" for i in names]].copy()
    inf_aver.columns = names
    inf_aver.to_csv(out_csv)


    mask = {}
    Xc = adata_sc.layers["counts"]
    for ct in adata_sc.obs[label_key].unique():
        sub = Xc[(adata_sc.obs[label_key] == ct).values]
        m = np.asarray(sub.mean(axis=0)).ravel() if sp.issparse(sub) else sub.mean(axis=0)
        mask[ct] = m == 0
    pd.DataFrame(mask, index=inf_aver.index).to_csv(WORK / f"mask_{label_key}.csv")
    print(f"saved {out_csv}")


def run_deconv(label_key, resolution, fold):
    inf_aver = pd.read_csv(WORK / f"inf_aver_{label_key}.csv", index_col=0)
    mask_df = pd.read_csv(WORK / f"mask_{label_key}.csv", index_col=0)
    adata_bulk = sc.read_h5ad(DST / "adata_bulk_processed.h5ad")
    adata_bulk = adata_bulk[:, inf_aver.index].copy()





    _sf = _os_sp.environ.get("MOSPR_SPLIT_FILE", "split.pkl")
    with open(DST / _sf, "rb") as f:
        split = pickle.load(f)
    print(f"  split={_sf}", flush=True)
    fit_slides = [p.stem for p in split[fold]["train"]] + [p.stem for p in split[fold]["val"]]
    adata_bulk = adata_bulk[fit_slides].copy()
    print(f"fold {fold}: deconvolving {adata_bulk.shape} bulk samples "
          f"over {inf_aver.shape[1]} cell types")

    cell2location.models.Cell2location.setup_anndata(adata_bulk, layer="raw_count")
    mod = cell2location.models.Cell2location(
        adata_bulk, cell_state_df=inf_aver, N_cells_per_location=30, detection_alpha=20)
    mod.train(max_epochs=30000, batch_size=None, train_size=1)
    adata_vis = mod.export_posterior(
        adata_bulk, sample_kwargs={"num_samples": 1000, "batch_size": mod.adata.n_obs})

    parameter_dict = {
        "W": adata_vis.obsm["means_cell_abundance_w_sf"],
        "theta": inf_aver,
        "m_g": adata_vis.uns["mod"]["post_sample_means"]["m_g"],
        "s_eg": adata_vis.uns["mod"]["post_sample_means"]["s_g_gene_add"],
        "y_s": adata_vis.uns["mod"]["post_sample_means"]["detection_y_s"],
        "mask": mask_df.reindex(index=inf_aver.index, columns=inf_aver.columns),
    }

    save_dir = DST / (_os_sp.environ.get("MOSPR_C2L_OUTDIR", "") or str(fold))
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"{resolution}_parameter_dict.pkl", "wb") as f:
        pickle.dump(parameter_dict, f)
    print(f"saved {save_dir}/{resolution}_parameter_dict.pkl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sc", "prototype", "deconv", "all"],
                    required=True)
    ap.add_argument("--cohort", default="BRCA")
    ap.add_argument("--variant", choices=["", "paper"], default="")
    ap.add_argument("--label_key", default=None,
                    help="obs column; defaults to the cohort's fine level")
    ap.add_argument("--resolution", default=None,
                    help="name used in the parameter_dict filename (CPNN's --resolution)")
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()
    inv = {v: k for k, v in CFG["label_keys"].items()}
    label_key = a.label_key or CFG["label_keys"]["fine"]
    res = a.resolution or inv[label_key]
    if a.stage == "sc":
        build_sc()
    elif a.stage == "prototype":
        run_prototype(label_key)
    elif a.stage == "deconv":
        run_deconv(label_key, res, a.fold)
    else:
        run_prototype(label_key)
        for fold in range(4):
            run_deconv(label_key, res, fold)
