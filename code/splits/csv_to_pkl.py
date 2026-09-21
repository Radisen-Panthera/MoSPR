#!/usr/bin/env python
"""Rebuild the split pickle the pipeline reads from the slide-id CSV shipped in results/.

    python code/splits/csv_to_pkl.py --cohort BRCA --data /path/to/datasets

The CSV (results/per_cohort/{cohort}/split/split_patient_4fold.csv) holds fold, split and slide id.
This writes <data>/{cohort}-paper-digital_slide/split_patient.pkl, a dict
{fold: {"train"|"val"|"test": [Path, ...]}} of absolute paths under the given data root, which is
what dataloaders and build_microstate_cache.py expect. Paths are rebuilt locally so the CSV itself
stays free of machine-specific paths.
"""
import argparse
import pathlib
import pickle
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def main(a):
    csv = ROOT / f"results/per_cohort/{a.cohort}/split/split_patient_4fold.csv"
    d = pd.read_csv(csv)
    base = pathlib.Path(a.data) / f"{a.cohort}-paper-digital_slide" / a.feature_dir
    out = {}
    for fold, g in d.groupby("fold"):
        out[int(fold)] = {k: [base / f"{s}.h5" for s in g[g.split == k].slide]
                          for k in ("train", "val", "test")}
    def present(path):
        if path.exists():
            return True
        stem = path.parent.name
        return any(path.parent.with_name(d.name).joinpath(path.name).exists()
                   for d in path.parent.parent.glob(stem + "_*"))
    missing = [p for f in out.values() for v in f.values() for p in v if not present(p)]
    dst = pathlib.Path(a.data) / f"{a.cohort}-paper-digital_slide" / a.name
    if a.dry_run:
        print(f"would write {dst}")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            pickle.dump(out, f)
        print(f"written: {dst}")
    for fold, s in out.items():
        print(f"  fold {fold}: train {len(s['train'])}, val {len(s['val'])}, test {len(s['test'])}")
    if missing:
        print(f"warning: {len(missing)} slide files do not exist yet, e.g. {missing[0]}")
        print("         extract features first (code/features/extract_features.sh)")
    return 1 if missing and a.strict else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="BRCA")
    ap.add_argument("--data", required=True, help="dataset root that holds {cohort}-paper-digital_slide")
    ap.add_argument("--feature_dir", default="sample_pair_feature",
                    help="subfolder the dataloader rewrites per encoder (kept for compatibility)")
    ap.add_argument("--name", default="split_patient.pkl")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit 1 when slide files are missing")
    sys.exit(main(ap.parse_args()))
