import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / 'mospr'))
import paths as _p
import os
import argparse, hashlib, pathlib, pickle, sys

import numpy as np

BASE = pathlib.Path(os.environ.get("MOSPR_REFERENCE_RESULTS", ""))


def fp_split(p):
    sp = pickle.load(open(p, "rb"))
    slides = sorted({pathlib.Path(x).stem for f in sp for k in ("train", "val", "test") for x in sp[f][k]})
    sizes = {f: {k: len(sp[f][k]) for k in ("train", "val", "test")} for f in sp}
    return hashlib.sha1("|".join(slides).encode()).hexdigest()[:16], len(slides), sizes, sp


def fp_genes(p):
    g = np.load(p, allow_pickle=True)
    return hashlib.sha1("|".join(map(str, g)).encode()).hexdigest()[:16], len(g), g


def compare(cohort, new_split, new_genes):
    print(f"\n{'='*60}\n{cohort}\n{'='*60}")
    ok = True

    old_p = BASE / cohort / f"{cohort}_split.pkl"
    if new_split and old_p.exists():
        oh, on, osz, osp = fp_split(old_p)
        nh, nn, nsz, nsp = fp_split(new_split)
        same = oh == nh
        ok &= same
        print(f"split  current {oh} ({on})  |  new {nh} ({nn})  → {'same' if same else 'differs'}")
        if not same:
            oset = {pathlib.Path(x).stem for f in osp for k in ("train","val","test") for x in osp[f][k]}
            nset = {pathlib.Path(x).stem for f in nsp for k in ("train","val","test") for x in nsp[f][k]}
            print(f"   new only: {len(nset-oset)}  current only: {len(oset-nset)}")
            for f in sorted(set(osz) & set(nsz)):
                if osz[f] != nsz[f]:
                    print(f"   fold{f} size change: {osz[f]} → {nsz[f]}")

    old_g = BASE / cohort / f"{cohort}_common_genes.npy"
    if new_genes and old_g.exists():
        oh, on, og = fp_genes(old_g)
        nh, nn, ng = fp_genes(new_genes)
        same = oh == nh
        ok &= same
        print(f"gene axis current {oh} ({on})  |  new {nh} ({nn})  → {'same' if same else 'differs'}")
        if not same:
            print(f"   intersection {len(set(og)&set(ng))}, same order? {list(og)==list(ng)}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort")
    ap.add_argument("--split")
    ap.add_argument("--genes")
    ap.add_argument("--dir", help="folder holding new BRCA/KIRC/LUAD subfolders")
    a = ap.parse_args()

    results = {}
    if a.dir:
        d = pathlib.Path(a.dir)
        for c in ("BRCA", "KIRC", "LUAD"):
            sp = next(iter(d.glob(f"**/{c}_split.pkl")), None)
            gn = next(iter(d.glob(f"**/{c}_common_genes.npy")), None)
            if sp or gn:
                results[c] = compare(c, sp, gn)
    elif a.cohort:
        results[a.cohort] = compare(a.cohort, a.split, a.genes)
    else:
        ap.error("--cohort or --dir is required")

    print(f"\n{'='*60}")
    if all(results.values()):
        print("verdict: identical - no rework needed for BRCA/KIRC; proceed with LUAD only.")
    else:
        bad = [c for c, v in results.items() if not v]
        print(f"verdict: the reference changed for {', '.join(bad)} - regenerate the paper variant for those cohorts.")
        print("      regenerate the paper variant for those cohorts before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
