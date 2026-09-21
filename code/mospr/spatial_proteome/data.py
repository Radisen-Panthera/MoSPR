from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from . import config


def list_slide_ids() -> list[str]:
    return sorted(p.stem for p in config.FEATURES_DIR.glob("*.h5"))


def patient_to_slide(patient_id: str, slide_ids: set[str] | None = None) -> str | None:
    if slide_ids is None:
        slide_ids = set(list_slide_ids())
    if patient_id in slide_ids:
        return patient_id
    dashed = patient_id.replace("_", "-", 1)
    if dashed in slide_ids:
        return dashed
    return None


def load_proteomics() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(config.PROTEOMICS_CSV, low_memory=False)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df = df.dropna(subset=["Patient_ID"])
    df = df.drop_duplicates(subset="Patient_ID", keep="first")
    df = df.set_index("Patient_ID")
    meta_cols = ["NAME", "Recur", "RFS"]
    protein_cols = [c for c in df.columns if c not in meta_cols]
    Y = df[protein_cols].astype(np.float32)
    meta = df[meta_cols]
    return Y, meta


def load_slide(slide_id: str, with_features: bool = True):
    path = config.FEATURES_DIR / f"{slide_id}.h5"
    with h5py.File(path, "r") as f:
        coords = f["coords"][:].astype(np.int64)
        feats = f["features"][:] if with_features else None
    return coords, feats


def load_cohort() -> pd.DataFrame:
    return pd.read_csv(config.OUTPUT_DIR / "cohort.csv", index_col=0)
