
import os
import time
import numpy as np
import h5py
import tqdm
import glob
import natsort
import matplotlib.pyplot as plt
import openslide

from ForegroundMasking import ForegroundMasker
from Tiling import TileSampler
from enhanceStain import get_imagej_enhanced_thumbnail
from criterion import should_apply_color_adjustment

class WSITileSampler:
    def __init__(
        self,
        root,
        output_dir,
        endswith='svs',
        max_depth=0,
        save_thumb=False,
        tile_size=224,
        overlap=0,
        min_tiles=5,
        is_normalized=False,
        thumb_level = -1
    ):
        self.root = root
        self.output_dir = output_dir
        self.tiler = None
        self.tile_size = tile_size
        self.save_thumb = save_thumb
        self.overlap = overlap
        self.min_tiles = min_tiles
        self.masker = ForegroundMasker()
        self.endswith = endswith
        self.is_normalized = is_normalized
        self.max_depth = max_depth
        self.thumb_level = thumb_level

        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir,exist_ok=True)

        self.generate_setting()

    def generate_setting(self) :
        output_dir = self.output_dir[:self.output_dir.rfind('/')]
        settings_path = os.path.join(output_dir, "settings.txt")
        default_settings = {
        "Slide_directory": self.root,
        "tile_size": self.tile_size,
        "overlap": self.overlap,
        "min_tiles": self.min_tiles,
        "is normalized" : self.is_normalized
         }

        if not os.path.exists(settings_path):
            with open(settings_path, "w") as f:
                for key, value in default_settings.items():
                    f.write(f"{key} = {value}\n")
            print(f"Created new settings file: {settings_path}")
        else:
            print(f"Settings file already exists: {settings_path}")

    def load_wsi(self, imgname):
        svs_file = os.path.join(imgname)
        print(f"file {svs_file}")
        slide = openslide.OpenSlide(svs_file)

        lv_dimensions = slide.level_dimensions

        for idx in range(len(lv_dimensions)):
            x,y = lv_dimensions[-(idx+1)]
            if x*y > 1000000:
                self.thumbnail_level = -(idx+1)
                break


        if len(slide.level_dimensions) > 1:
            thumbnail = np.array(slide.get_thumbnail(slide.level_dimensions[self.thumbnail_level]))
            scaler =  int(slide.level_downsamples[self.thumbnail_level])
        else : 
            downscale_factor = 64
            thumbnail = np.array(
                slide.get_thumbnail( 
                    (slide.level_dimensions[0][0]//downscale_factor,
                    slide.level_dimensions[0][1]//downscale_factor) 
                    )
                )
            scaler = downscale_factor

        if self.save_thumb:
            thumbnail_dir = os.path.join(self.output_dir, "Thumbnails")
            os.makedirs(thumbnail_dir, exist_ok= True)
            basename = os.path.splitext(os.path.basename(svs_file))[0]
            save_path = os.path.join(thumbnail_dir, f"{basename}.png")

            plt.imshow(thumbnail)
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

        tiler = TileSampler(
                tile_size=self.tile_size,
                dimensions=slide.dimensions,
                overlap=self.overlap,
                min_tiles=self.min_tiles,
                downscale_factor=scaler,
            )

        self.tiler = tiler
        print(f"Original size: {slide.level_dimensions[0]}")
        print(f"Thumbnail size: {thumbnail.shape}")
        return thumbnail, scaler 

    def compute_foreground_mask(self, image):
        foreground,mask = self.masker.get_foreground(image, self.is_normalized)
        return foreground, mask

    def sample_tiles(self, mask, scaler):
        coords = self.tiler.get_tile(mask)
        return coords


    def save_hdf5(self, fname, coords):
        if fname is None:
            raise ValueError("Missing file name: please set 'fname' before saving HDF5.")

        metadata = {
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "total_tiles": len(coords)
        }
        save_path = os.path.join(self.output_dir,f"{fname}.h5")

        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset('coords', data=np.array(coords), compression='gzip')

            meta_group = hf.create_group('metadata')
            for key, value in metadata.items():
                if isinstance(value, (list, tuple, np.ndarray)):
                    meta_group.create_dataset(key, data=np.array(value))
                else:
                    meta_group.attrs[key] = value

        print(f"Tiling complete. Metadata and tile coordinates saved as {fname}.h5")

        return metadata

    def process_image(self, imgname):
        fname = os.path.basename(imgname)[:-4]
        start = time.time()

        thumbnail, scaler= self.load_wsi(imgname)
        io_time = time.time() - start
        print(f"Image I/O: {io_time:.5f} sec")

        if self.tiler is None:
            raise ValueError("Tiler is Missing!")

        start = time.time()
        _, mask = self.compute_foreground_mask(thumbnail)
        fg_time = time.time() - start
        print(f"Foreground masking: {fg_time:.5f} sec")

        start = time.time()
        coords = self.sample_tiles(mask, scaler)
        metadata = self.save_hdf5(fname, coords)

        ts_time = time.time() - start

        print(f"Tile sampling: {ts_time:.5f} sec")

        return coords, metadata

    def process_images(self, imgList):
        for _, imgname in enumerate(tqdm.tqdm(imgList)):
            name = imgname.split('/')[-1][:-4] + '.h5'
            if name not in os.listdir(self.output_dir) : 
                print(f"Processing {imgname} ...")
                try : 
                    coords,metadata = self.process_image(imgname)
                except Exception as e:
                    print(f"Error processing {imgname}: {type(e).__name__}: {e}")
            else : 
                print(f"Passing {imgname} ...")


    def run(self):
        if os.path.isdir(self.root):
            img_list = []
            ext = self.endswith.lower()

            for root, dirs, files in os.walk(self.root):
                rel = os.path.relpath(root, self.root)
                depth = 0 if rel == '.' else rel.count(os.sep) + 1

                if self.max_depth > 0 and depth > self.max_depth:
                    dirs[:] = []
                    continue

                for fn in files:
                    if fn.lower().endswith(ext):
                        img_list.append(os.path.join(root, fn))

            if not img_list:
                raise ValueError(f"No '{self.endswith}' files found under: {self.root}")

            img_list = natsort.natsorted(img_list)
            self.process_images(img_list)

        elif os.path.isfile(self.root):
            self.process_image(self.root)
        else:
            raise ValueError("Input a valid root directory or file.")
