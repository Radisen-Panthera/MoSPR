#!/usr/bin/env python
"""End-to-end check: run the pipeline in this repository and compare with the released numbers.

    python code/tests/test_pipeline.py --data /path/to/datasets --cohort BRCA --fold 0
    python code/tests/test_pipeline.py --data ... --with-cache     # also rebuild the microstate cache

What it checks, in order:
  1  the microstate cache loads and its split matches results/per_cohort/*/split/split_patient_4fold.csv
  2  design_blocks.build_blocks returns [M | S] of width (K+1)*d with finite entries
  3  train_mospr stage 1 reproduces the released (q, lambda) and stage-1 test SCC
  4  train_mospr stage 2 reproduces the released test SCC and PCC (gene and pathway)
  5  ablation reproduces the released Table 3 rows for this fold
  6  macrostate enrichment reproduces the released Hallmark z for this fold
  7  reproduce_results.py --check passes

Reference values are read from results/, so the test fails if code and shipped results drift apart.
Everything runs on CPU. Without --with-cache an existing cache is required
(MOSPR_RESULTS_ROOT/micro_cache_{cohort}_{tag}_fold{fold}.npz).
"""
import argparse
import importlib.util
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
MOSPR = ROOT / "code/mospr"
TOL = 5e-4
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""), flush=True)
    return ok


def close(a, b, tol=TOL):
    return abs(float(a) - float(b)) <= tol


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run(script, *args):
    cmd = [sys.executable, str(MOSPR / script), *map(str, args)]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-2000:]); print(p.stderr[-2000:])
    return p.returncode == 0, time.time() - t0


def main(a):
    os.environ.setdefault("MOSPR_TAU", "0")
    os.environ["MOSPR_DATASET_ROOT"] = a.data
    os.environ["MOSPR_RESULTS_ROOT"] = a.results
    if a.genesets:
        os.environ["MOSPR_GENESETS"] = a.genesets
    if a.processed:
        os.environ["MOSPR_PROCESSED"] = a.processed
    if a.tables:
        os.environ["MOSPR_TABLES_ROOT"] = a.tables
    res = pathlib.Path(a.results)
    res.mkdir(parents=True, exist_ok=True)
    tag, c, f = a.cache_tag, a.cohort, a.fold

    # reference numbers shipped with the repository
    ref = pd.read_csv(ROOT / f"results/per_cohort/{c}/results/mospr.csv")
    ref = ref[(ref.fold == f) & (ref.primary)]
    ref_gene = ref[ref.space == "gene"].iloc[0]
    ref_path = ref[ref.space == "pathway"].iloc[0]
    abl = pd.read_csv(ROOT / f"results/tables/final/alpha0/source/ablation_v2_{c}_fpsplit_tv_p512.csv")
    abl = abl[abl.fold == f].set_index("variant")

    if a.with_cache:
        ok, dt = run("build_microstate_cache.py", "--cohort", c, "--variant", "paper",
                     "--pca", 512, "--suffix", f"_{c}_{tag}", "--folds", f)
        check("1a. build_microstate_cache runs", ok, f"{dt:.0f}s")

    cache_file = res / f"micro_cache_{c}_{tag}_fold{f}.npz"
    if not check("1b. microstate cache present", cache_file.exists(), str(cache_file)):
        return summarise()
    cache = dict(np.load(cache_file, allow_pickle=True))
    split = pd.read_csv(ROOT / f"results/per_cohort/{c}/split/split_patient_4fold.csv")
    split = split[split.fold == f]
    slides = np.array([str(s) for s in cache["slides"]])
    same = {k: set(slides[cache[f"idx_{v}"]]) == set(split[split.split == k].slide)
            for k, v in (("train", "tr"), ("val", "va"), ("test", "te"))}
    check("1c. cache split matches the shipped split CSV", all(same.values()), str(same))

    sw = load(MOSPR / "design_blocks.py", "sw")
    blocks = sw.build_blocks(cache, a.K, a.d, with_Q=False)
    X = np.hstack([blocks["M"], blocks["S"]])
    check("2. design matrix [M | S] shape and finiteness",
          X.shape[1] == (a.K + 1) * a.d and np.isfinite(X).all(), f"X {X.shape}")

    ok, dt = run("train_mospr.py", "--cohort", c, "--cache_tag", tag, "--folds", f,
                 "--out_tag", f"{c}_test_s1")
    check("3a. train_mospr stage 1 runs", ok, f"{dt:.0f}s")
    s1 = pd.read_csv(res / f"cohort_{c}_test_s1_ours.csv")
    s1g = s1[s1.space == "gene"].iloc[0]
    check("3b. stage 1 selects the released (q, lambda)",
          int(s1g["rank"]) == int(ref_gene["rank"]) and close(s1g.alpha_q, ref_gene.alpha_q, 0),
          f"q={int(s1g['rank'])} lam={s1g.alpha_q:g} vs q={int(ref_gene['rank'])} lam={ref_gene.alpha_q:g}")

    ok, dt = run("train_mospr.py", "--cohort", c, "--cache_tag", tag, "--folds", f,
                 "--fit_on", "trainval", "--hp_from", res / f"cohort_{c}_test_s1_ours.csv",
                 "--out_tag", f"{c}_test_s2")
    check("4a. train_mospr stage 2 runs", ok, f"{dt:.0f}s")
    s2 = pd.read_csv(res / f"cohort_{c}_test_s2_ours.csv")
    g = s2[s2.space == "gene"].iloc[0]
    p = s2[s2.space == "pathway"].iloc[0]
    check("4b. gene test SCC matches the released value",
          close(g.test_SCC, ref_gene.test_SCC), f"{g.test_SCC:.6f} vs {ref_gene.test_SCC:.6f}")
    check("4c. gene test PCC matches the released value",
          close(g.test_PCC, ref_gene.test_PCC), f"{g.test_PCC:.6f} vs {ref_gene.test_PCC:.6f}")
    check("4d. pathway test SCC matches the released value",
          close(p.test_SCC, ref_path.test_SCC), f"{p.test_SCC:.6f} vs {ref_path.test_SCC:.6f}")

    ok1, dt1 = run("ablation.py", "--cohort", c, "--cache_tag", tag, "--folds", f,
                   "--out_tag", f"{c}_test_abl_s1")
    ok, dt = run("ablation.py", "--cohort", c, "--cache_tag", tag, "--folds", f,
                 "--fit_on", "trainval", "--hp_from", res / f"ablation_v2_{c}_{c}_test_abl_s1.csv",
                 "--out_tag", f"{c}_test_abl")
    check("5a. ablation runs (stage 1 then stage 2)", ok1 and ok, f"{dt1 + dt:.0f}s")
    if ok:
        got = pd.read_csv(res / f"ablation_v2_{c}_{c}_test_abl.csv").set_index("variant")
        rows = [v for v in abl.index if v in got.index]
        diffs = {v: (got.loc[v, "gene_SCC"], abl.loc[v, "gene_SCC"]) for v in rows}
        worst = max(abs(x - y) for x, y in diffs.values())
        check("5b. ablation gene SCC matches the released rows", worst <= TOL,
              f"{len(rows)} variants, max |diff| {worst:.2e}")

    ok, dt = run("macrostate_hallmark_folds.py", "--cohort", c, "--cache_tag", tag,
                 "--folds", f, "--weights", a.weights or ROOT / "checkpoints", "--suffix", "_test")
    check("6a. macrostate enrichment runs", ok, f"{dt:.0f}s")
    got_csv = res / f"macro_hallmark_perfold_{c}_test.csv"
    if ok and got_csv.exists():
        g2 = pd.read_csv(got_csv)
        r2 = pd.read_csv(ROOT / f"results/tables/macrostate/macro_hallmark_perfold_{c}.csv")
        m = g2.merge(r2[r2.fold == f], on=["fold", "macrostate", "set"], suffixes=("_new", "_ref"))
        worst = float((m.z_new - m.z_ref).abs().max()) if len(m) else float("inf")
        check("6b. macrostate Hallmark z matches the released values", worst <= 1e-6,
              f"{len(m)} cells, max |diff| {worst:.2e}")

    p = subprocess.run([sys.executable, str(ROOT / "code/reproduce_results.py"), "--check"],
                       capture_output=True, text=True)
    check("7. reproduce_results.py --check", p.returncode == 0, p.stdout.strip().split("\n")[-1])
    return summarise()


def summarise():
    n = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n}/{len(results)} checks passed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="MOSPR_DATASET_ROOT for the run")
    ap.add_argument("--results", default=str(ROOT / "results/run"), help="MOSPR_RESULTS_ROOT for the run")
    ap.add_argument("--genesets", default="", help="MOSPR_GENESETS (MSigDB .gmt directory)")
    ap.add_argument("--processed", default="", help="MOSPR_PROCESSED (per-cohort gene lists)")
    ap.add_argument("--tables", default="", help="MOSPR_TABLES_ROOT for the run")
    ap.add_argument("--weights", default="", help="checkpoint root: {weights}/{cohort}/weights_gene_fold*.npz")
    ap.add_argument("--cohort", default="BRCA")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cache_tag", default="fpsplit_p512")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--with-cache", action="store_true", help="rebuild the microstate cache first (slow)")
    sys.exit(main(ap.parse_args()))
