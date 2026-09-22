import argparse

from TileSampling import WSITileSampler


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, help='Directory containing whole-slide images, or a single WSI file.')
parser.add_argument('--output_dir', type=str, help='Directory where the per-slide HDF5 files are written.')
parser.add_argument('--endswith', '--end', type=str, default="svs", help='File extension of the WSI files.')
parser.add_argument('--max_depth', type=int, default=0, help='Maximum directory depth searched under --root.')
parser.add_argument('--save_thumb', type=str2bool, default=False, help='Save slide thumbnails.')
parser.add_argument('--tile_size', type=int, default=224, help='Tile size in level-0 pixels.')
parser.add_argument('--min_tiles', type=int, default=5, help='Minimum number of tiles in a region kept after post-filtering.')
parser.add_argument('--is_normalized', type=str2bool, default=False, help='Apply stain normalization to the thumbnail before masking.')
parser.add_argument('--thumb_level', type=int, default=-1, help='Thumbnail level.')

args = parser.parse_args()


def main():
    sampler = WSITileSampler(
        root=args.root,
        output_dir=args.output_dir,
        endswith=args.endswith,
        max_depth=args.max_depth,
        save_thumb=args.save_thumb,
        tile_size=args.tile_size,
        min_tiles=args.min_tiles,
        is_normalized=args.is_normalized,
        thumb_level=args.thumb_level,
    )
    sampler.run()


if __name__ == "__main__":
    main()
