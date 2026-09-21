import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / 'mospr'))
import paths as _p
import argparse
import collections
import csv
import pathlib
import re


METHODS = [
    ("cpnn", ["cpnn", "ProtoSum"], r"ProtoSum_1reg_mse_reg_1e3(\d)DeconvExptsfine$"),
    ("abmil", ["abmil"], r"AbMIL(\d)ts$"),
    ("abmil_adapter", ["abmil_adapter"], r"AbMIL(\d)adapterts$"),
    ("abmil_max", ["abmil_max"], r"AbMIL_max(\d)ts$"),
    ("abmil_mean", ["abmil_mean"], r"AbMIL_mean(\d)ts$"),
    ("ilra", ["ilra"], r"ILRA(\d)ts$"),
    ("s4model", ["s4model"], r"S4Model_stop_sampling(\d)ts$"),
    ("mambamil", ["mambamil"], r"MambaMILvanira_stop_sampling(\d)ts$"),
    ("srmamba", ["srmamba"], r"SRMambaMIL_stop_sampling(\d)ts$"),
    ("mamba2d", ["mamba2d"], r"MambaMIL_2D_stop_sampling(\d)Mamba2DTrainerts$"),
    ("abreg", ["abreg"], r"AbRegMIL(\d)ts$"),
    ("he2rna", ["he2rna"], r"HE2RNA(\d)ComparisonTrainerts$"),
    ("mosby", ["mosby"], r"SumExpModel_MOSBY(\d)ts$"),
    ("sequoia_vis", ["sequoia_vis"], r"SEQUOIA_VIS(\d)ComparisonTrainerts$"),
    ("trnasformer", ["trnasformer"], r"tRNAsformer(\d)ComparisonTrainerts$"),
]
COHORTS = ["BRCA", "KIRC", "LUAD"]

PAT = re.compile(rb"Epoch (\d+), global step \d+: '([a-z_]+)' reached")


def completed(outputs, suffix=""):
    done = set()
    for c in COHORTS:
        for tag, _, pat in METHODS:
            for f in pathlib.Path(outputs).glob(f"*/{c}-paper-*.txt"):
                m = re.fullmatch(pat[:-1] + re.escape(suffix) + "$",
                                 f.stem.replace(f"{c}-paper-", ""))
                if m:
                    done.add((c, tag, int(m.group(1))))
    return done


def scan_logs(dirs, skip=("_filt", "_nostop", "_ln")):
    out = {}
    for d in dirs:
        for f in pathlib.Path(d).rglob("*.log"):
            name = f.name
            if any(x in name for x in skip):
                continue
            c = next((x for x in COHORTS if x in name), None)
            fold = re.search(r"fold(\d)|_f(\d)\b", name)
            if not c or not fold:
                continue
            fold = int(fold.group(1) or fold.group(2))
            tag = None
            for t, frags, _ in METHODS:
                if any(fr.lower() in name.lower() for fr in frags):
                    if tag is None or len(t) > len(tag):
                        tag = t
            if tag is None:
                continue
            hits = PAT.findall(f.read_bytes())
            if not hits:
                continue
            key = (c, tag, fold)

            if key not in out or len(hits) > out[key][2]:
                out[key] = (int(hits[-1][0]), hits[-1][1].decode(), len(hits))
    return out


def main(a):
    done = completed(a.outputs, a.suffix)


    logs = scan_logs(a.log_dirs)
    rows, miss = [], []
    for c in COHORTS:
        for tag, _, _ in METHODS:
            for fold in range(4):
                key = (c, tag, fold)
                if key not in done:
                    continue
                if key not in logs:
                    miss.append(key); continue
                e, mon, n = logs[key]
                rows.append({"cohort": c, "tag": tag, "fold": fold, "done": 1,
                             "best_epoch": e, "monitor": mon, "n_improve": n})
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"finished (by artefact) {len(done)}, E* found {len(rows)}, log missing {len(miss)}")
    for k in miss:
        print("   no log:", k)
    print("monitored metric:", dict(collections.Counter(r["monitor"] for r in rows)))
    print(f"written: {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dirs", nargs="+", default=[
        str(_p.LOGS)])
    ap.add_argument("--outputs", default=str(_p.CPNN_REPO / "outputs"))
    ap.add_argument("--suffix", default="",
                    help='run-name suffix: "" authors protocol, "_ps" patient split, "_tv" train+val')
    ap.add_argument("--out", default=str(_p.RESULTS / "best_epochs.csv"))
    main(ap.parse_args())
