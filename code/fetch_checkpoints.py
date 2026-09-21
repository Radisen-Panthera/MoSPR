import argparse, pathlib, shutil, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main(a):
    out = pathlib.Path(a.out or ROOT / a.what)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()) and not a.force:
        sys.exit(f"{out} is not empty; pass --force to download anyway")
    if "drive.google.com" in a.url and shutil.which("gdown") is None:
        sys.exit("gdown not found: pip install gdown, or pass a direct download link")
    cmd = (["gdown", "--folder", "--output", str(out), a.url] if "drive.google.com" in a.url
           else ["curl", "-L", "-o", str(out / "mospr_checkpoints.tar.gz"), a.url])
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    for t in out.glob("*.tar.gz"):
        subprocess.run(["tar", "xzf", str(t), "-C", str(out)], check=True)
        t.unlink()
    n = sum(1 for _ in out.rglob("*") if _.is_file())
    print(f"{n} files under {out}")
    print("Expected: 168 .ckpt (14 baselines x 4 folds x 3 cohorts) + 24 .npz (MoSPR)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="Google Drive folder link or a direct download URL")
    p.add_argument("--what", default="checkpoints", choices=["checkpoints", "predictions"])
    p.add_argument("--out", default="", help="destination (default: <repo>/<what>)")
    p.add_argument("--force", action="store_true")
    main(p.parse_args())
