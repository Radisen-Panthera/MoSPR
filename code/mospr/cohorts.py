import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import paths as _p
import os
from pathlib import Path

TCGA_ROOT = Path(os.environ.get("MOSPR_TCGA_ROOT", str(_p.DATA / "TCGA")))


import os as _os
DATASET_ROOT = Path(_os.environ.get("MOSPR_DATASET_ROOT",
                                    str(_p.DATA)))
FEATURE_ROOT = Path(_os.environ.get("MOSPR_FEATURE_ROOT",
                                    str(_p.DATA / "features")))
BENCH = Path(str(_p.CODE))
WORK  = _p.RESULTS
REFD  = _p.ROOT / "refdata"

COHORTS = {
    "BRCA": {
        "project": "TCGA-BRCA",
        "sc_kind": "10x_mtx",
        "sc_dir": TCGA_ROOT / "sc_BRCA",
        "batch_key": "Patient",
        "label_keys": {"coarse": "celltype_major", "medium": "celltype_subset",
                       "fine": "celltype_minor"},
        "paper_pairs": 1474, "paper_genes": 14047, "paper_scc": 0.338,
    },
    "KIRC": {
        "project": "TCGA-KIRC",
        "sc_kind": "h5ad",
        "sc_dir": TCGA_ROOT / "sc_KIRC",
        "sc_file": "RCC_upload_final_raw_counts.h5ad",
        "batch_key": "patient",
        "label_keys": {"coarse": "summaryDescription", "medium": "broad_type",
                       "fine": "annotation"},
        "paper_pairs": 681, "paper_genes": 14300, "paper_scc": 0.318,
    },
    "LUAD": {
        "project": "TCGA-LUAD",
        "sc_kind": "umi_txt",
        "sc_dir": TCGA_ROOT / "sc_LUAD",
        "sc_file": "GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz",
        "sc_annot": "GSE131907_Lung_Cancer_cell_annotation.txt.gz",
        "batch_key": "Sample",
        "label_keys": {"coarse": "Cell_type", "medium": "Cell_type.refined",
                       "fine": "Cell_subtype"},


        "origin_key": "Sample_Origin",
        "paper_pairs": 756, "paper_genes": 14520, "paper_scc": 0.304,
    },
}


def get(name: str) -> dict:
    c = dict(COHORTS[name])
    p = c["project"]
    c.update(
        name=name,
        wsi_raw=TCGA_ROOT / p / "WSI",
        wsi_flat=TCGA_ROOT / p / "WSI_flat",
        rna_dir=TCGA_ROOT / p / "RNA",
        pairs_csv=REFD / ("manifests/pairs.csv" if name == "BRCA"
                           else f"manifests/pairs_{p}.csv"),
        pairs_named=REFD / ("manifests/pairs_named.csv" if name == "BRCA"
                             else f"manifests/pairs_named_{name}.csv"),
        manifest_dir=REFD / "manifests",
        processed=_p.PROCESSED / f"processed_{name}" if name != "BRCA" else _p.PROCESSED,
        features=FEATURE_ROOT / f"{p}_features",
        gene_root=DATASET_ROOT / f"{name}-digital_slide",
        path_root=DATASET_ROOT / f"{name}-pathway-digital_slide",
        c2l=DATASET_ROOT / f"c2l_{name}" if name != "BRCA" else DATASET_ROOT / "c2l",
    )
    return c
