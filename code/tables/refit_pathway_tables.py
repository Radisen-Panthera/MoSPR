import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / 'mospr'))
import paths as _p
TABLES = _p.TABLES
import argparse
from pathlib import Path

import pandas as pd

CPNN = Path(str(_p.CODE))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import refit_tables as f62

OUT = Path(str(_p.TABLES))
import os as _o
if _o.environ.get("MOSPR_WORK_OUT"):
    OUT = Path(_o.environ["MOSPR_WORK_OUT"])
LABEL = {"hallmark": "Hallmark", "gobp": "GO-BP", "kegg": "KEGG"}


def main(a):
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(OUT / f"tables/refit/refit_pathway_{c}.csv")
                    for c in a.cohorts])

    keep = []
    for coll in a.collections:
        col = f"{coll}_scc"
        s = f62.summarise(df, col)
        keep.append(s)
        f62.to_tex(s, f"{LABEL[coll]} SCC (median)", a.cohorts,
                   OUT / f"tables/refit/refit_before_after_{coll}.tex")
        f62.draw(s, f"{LABEL[coll]} SCC (median)", a.cohorts,
                 OUT / f"figures/refit_before_after_{coll}.png")
        up, dn = (s.delta > 0).sum(), (s.delta < 0).sum()
        print(f"[{coll}] {len(s)} of them up {up}, down {dn}, mean delta {s.delta.mean():+.4f}")
        for r in s[s.delta < 0].itertuples():
            print(f"    down {r.model}/{r.cohort} {r.delta:+.4f} "
                  f"(folds improved {r.folds_up}/4, p={r.p_paired:.2f})")
    pd.concat(keep).round(5).to_csv(OUT / "tables/refit/refit_before_after_pathway.csv",
                                    index=False)
    print(f"\nwritten: {OUT}/tables/refit_before_after_{{hallmark,gobp,kegg}}.tex · "
          f"refit_before_after_pathway.csv\n      {OUT}/figures/ (png + pdf/)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["BRCA", "KIRC", "LUAD"])
    ap.add_argument("--collections", nargs="+", default=["hallmark", "gobp", "kegg"])
    main(ap.parse_args())
