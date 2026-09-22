import os
from TileSampling import WSITileSampler
import pandas as pd
import argparse
import natsort

parser = argparse.ArgumentParser()
parser.add_argument('--root', type= str, help = 'Path to the directory containing Whole Slide Images (.svs) or a single WSI file.')
parser.add_argument('--output_dir', type= str, help = 'Path to the directory where output HDF5 files will be saved.')
parser.add_argument('--endswith', '--end', type= str , default = "svs", help = 'File extension of WSI files.')
parser.add_argument('--max_depth', type= int, default = 0, help = 'Maximum directory depth to search under the root directory.')
parser.add_argument('--save_thumb', type= bool , default = False, help = 'Whether to save thumbnails of WSIs.')
parser.add_argument('--tile_size', type= int , default = 224, help = 'Patch size used for tiling WSIs.')
parser.add_argument('--min_tiles', type= int , default = 5, help = 'Minimum number of tiles required in a valid region during post-filtering.')
parser.add_argument('--is_normalized', type= bool , default = False, help = 'Whether to apply stain normalization.')
parser.add_argument('--thumb_level', type= int , default = -1, help = 'Thumbnail Level')

args = parser.parse_args()

def main() : 
    
    """
    Examples
    --------
    Run the tile processing script on a single WSI file:

    ```bash
    python tile_processing.py \
        --root /data/wsi/DB-00000-H.svs \
        --output_dir /home/user/tiles \
        --save_thumb True
    ```

    Output
    ------
    The following files will be generated:

    ```
    /home/user/tiles/Thumbnails/DB-00000-H.png
    /home/user/tiles/DB-00000-H.h5
    ```
    """
    
    #df = pd.read_csv("/data/yscho_real/virtual_codex/OCM_label_csv/file_list_common.csv")
    #listed = sorted([word for word in list(df['path'])])
    
    #for path in listed : 
    #    args.root = path

    sampler = WSITileSampler(
                            root=args.root,
                            output_dir=args.output_dir,
                            endswith=args.endswith,
                            max_depth=args.max_depth,
                            save_thumb=args.save_thumb,
                            tile_size=args.tile_size,
                            min_tiles=args.min_tiles,
                            is_normalized=args.is_normalized,
                            thumb_level = args.thumb_level
                            )
    sampler.run()
    
if __name__ == "__main__" :
    main()
