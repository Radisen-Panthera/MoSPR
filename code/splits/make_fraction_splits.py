import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / 'mospr'))
import paths as _p
import argparse
import pathlib
import pickle

import numpy as np

FRACTIONS = [0.10, 0.25, 0.50, 0.75]
ROOT = pathlib.Path(str(_p.DATA))


def pid(p):
    return "-".join(pathlib.Path(str(p)).stem.split("-")[:3])


def main(a):
    for cohort in a.cohorts:
        ds = ROOT / f"{cohort}-paper-digital_slide"
        base = pickle.load(open(ds / a.base, "rb"))
        for frac in a.fractions:
            tag = f"f{int(round(frac * 100)):02d}"
            out, rows = {}, []
            for fold in range(4):
                tr, va, te = base[fold]["train"], base[fold]["val"], base[fold]["test"]


                pats = sorted({pid(p) for p in list(tr) + list(va)})
                rng = np.random.default_rng(a.seed)
                keep = set(np.array(pats)[rng.permutation(len(pats))][: max(1, int(round(len(pats) * frac)))])
                tr2 = [p for p in tr if pid(p) in keep]
                va2 = [p for p in va if pid(p) in keep]
                assert va2, f"{cohort} {tag} fold{fold}: val is empty - fraction too small"
                out[fold] = {"train": tr2, "val": va2, "test": te}
                rows.append((fold, len(tr2), len(tr), len(va2), len(va), len(te), len(keep), len(pats)))
            dst = ds / f"{pathlib.Path(a.base).stem}_{tag}.pkl"
            pickle.dump(out, open(dst, "wb"))
            print(f"{cohort} {tag} → {dst.name}")
            for f, t2, t, v2, v, te_, k, p in rows:
                print(f"   fold{f}: train {t2:>4d}/{t:<4d} val {v2:>3d}/{v:<3d} "
                      f"test {te_:>3d}  patients {k}/{p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["BRCA"])
    ap.add_argument("--base", default="split_patient.pkl")
    ap.add_argument("--fractions", type=float, nargs="+", default=FRACTIONS)
    ap.add_argument("--seed", type=int, default=2021)
    main(ap.parse_args())
