import argparse
import pathlib

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def build_encoder(name, device):
    if name == "conch":
        from conch.open_clip_custom import create_model_from_pretrained
        model, preprocess = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch")
        model = model.to(device).eval()
        return (lambda x: model.encode_image(x, proj_contrast=False, normalize=False)), preprocess, 512
    if name == "exaone":
        import timm
        from transformers import AutoModel
        model = AutoModel.from_pretrained("LGAI-DILab/EXAONEPath", trust_remote_code=True).to(device).eval()
        cfg = timm.data.resolve_data_config({}, model=getattr(model, "model", model))
        return (lambda x: model(x)), timm.data.create_transform(**cfg), 768
    if name in ("uni", "gigapath"):
        import timm
        repo = {"uni": "hf-hub:MahmoodLab/UNI", "gigapath": "hf-hub:prov-gigapath/prov-gigapath"}[name]
        model = timm.create_model(repo, pretrained=True, init_values=1e-5, dynamic_img_size=True)
        model = model.to(device).eval()
        cfg = timm.data.resolve_data_config({}, model=model)
        return (lambda x: model(x)), timm.data.create_transform(**cfg), model.num_features
    raise SystemExit(f"unknown encoder: {name}")


class Patches(Dataset):
    def __init__(self, wsi, coords, size, level, tf):
        self.wsi, self.coords, self.size, self.level, self.tf = wsi, coords, size, level, tf

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        x, y = self.coords[i]
        img = self.wsi.read_region((int(x), int(y)), self.level, (self.size, self.size)).convert("RGB")
        return self.tf(img)


def main(a):
    import openslide
    device = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    encode, tf, dim = build_encoder(a.encoder, device)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    slides = {p.stem: p for ext in ("*.svs", "*.tif", "*.tiff", "*.ndpi")
              for p in pathlib.Path(a.slides).rglob(ext)}

    for h5 in sorted(pathlib.Path(a.patches).glob("*.h5")):
        dst = out / h5.name
        if dst.exists():
            print(f"skip {h5.stem}: exists", flush=True)
            continue
        if h5.stem not in slides:
            print(f"skip {h5.stem}: slide file not found", flush=True)
            continue
        with h5py.File(h5, "r") as f:
            coords = f["coords"][:]
        wsi = openslide.OpenSlide(str(slides[h5.stem]))
        loader = DataLoader(Patches(wsi, coords, a.patch_size, a.level, tf),
                            batch_size=a.batch, num_workers=a.workers, pin_memory=True)
        feats = []
        with torch.inference_mode():
            for batch in loader:
                z = encode(batch.to(device, non_blocking=True))
                feats.append(z.float().cpu().numpy())
        feat = np.concatenate(feats).astype(np.float32)
        assert feat.shape == (len(coords), dim), (feat.shape, dim)
        with h5py.File(dst, "w") as f:
            f.create_dataset("coord", data=coords.astype(np.int64))
            f.create_dataset("feat", data=feat)
        print(f"{h5.stem}: {feat.shape[0]} patches x {feat.shape[1]}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--patches", required=True, help="CLAM patch .h5 directory")
    p.add_argument("--slides", required=True, help="whole-slide image directory")
    p.add_argument("--out", required=True)
    p.add_argument("--encoder", default="conch", choices=["conch", "exaone", "uni", "gigapath"])
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    main(p.parse_args())
