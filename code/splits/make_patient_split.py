import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / 'mospr'))
import paths as _p
import argparse
import collections
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(_p.CODE))
import cohorts

PATIENT = lambda stem: "-".join(stem.split("-")[:3])


def build(paths, n_blocks=4):
    by_pat = collections.defaultdict(list)
    for p in paths:
        by_pat[PATIENT(Path(str(p)).stem)].append(p)
    blocks = [[] for _ in range(n_blocks)]
    counts = [0] * n_blocks
    for pat in sorted(by_pat, key=lambda k: (-len(by_pat[k]), k)):
        i = min(range(n_blocks), key=lambda j: (counts[j], j))
        blocks[i].extend(by_pat[pat]); counts[i] += len(by_pat[pat])
    return blocks


def main(a):
    rows = []
    for c in a.cohorts:
        cfg = cohorts.get(c)
        root = cfg["gene_root"].with_name(f"{c}-paper-digital_slide")
        src = root / "split.pkl"
        old = pickle.load(open(src, "rb"))

        allp = list(old[0]["train"]) + list(old[0]["val"]) + list(old[0]["test"])
        assert len({str(x) for x in allp}) == len(allp), "duplicate slides in the source fold 0"

        blocks = build(allp)
        new = {}
        for f in range(4):
            te = blocks[f]
            va = blocks[(f + 3) % 4]
            tr = [p for j in range(4) if j not in (f, (f + 3) % 4) for p in blocks[j]]
            new[f] = {"train": tr, "val": va, "test": te}
            P = lambda v: {PATIENT(Path(str(x)).stem) for x in v}
            assert not (P(tr) & P(te)), f"{c} f{f}: train/test patient overlap"
            assert not (P(va) & P(te)), f"{c} f{f}: val/test patient overlap"
            assert not (P(tr) & P(va)), f"{c} f{f}: train/val patient overlap"
            o = old[f]
            rows.append({"cohort": c, "fold": f,
                         "train_pat": len(P(tr)), "train_slide": len(tr),
                         "val_pat": len(P(va)), "val_slide": len(va),
                         "test_pat": len(P(te)), "test_slide": len(te),
                         "author_train_slide": len(o["train"]),
                         "author_val_slide": len(o["val"]),
                         "author_test_slide": len(o["test"])})

        seen = set()
        for f in range(4):
            s = {str(x) for x in new[f]["test"]}
            assert not (seen & s), f"{c}: slide overlap between test folds"
            seen |= s
        assert len(seen) == len(allp), f"{c}: union of test folds does not cover all slides"

        dst = root / "split_patient.pkl"
        if not a.dry_run:
            pickle.dump(new, open(dst, "wb"))
        print(f"{c}: patients {len({PATIENT(Path(str(x)).stem) for x in allp})} · "
              f"slides {len(allp)} → {dst.name}{' (dry-run)' if a.dry_run else ''}")

    df = pd.DataFrame(rows)
    out = Path(str(_p.RESULTS / "split_patient_manifest.csv"))
    if not a.dry_run:
        df.to_csv(out, index=False)
    print(f"\n{df.to_string(index=False)}")
    if not a.dry_run:
        print(f"\nwritten: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["BRCA", "KIRC", "LUAD"])
    ap.add_argument("--dry_run", action="store_true")
    main(ap.parse_args())
