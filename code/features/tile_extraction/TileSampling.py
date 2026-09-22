'''
modified by YS, date : 2025-04-10

modification :
- adding parameters below : 
    (1) extension name parameter 'end'
    (2) thumbnail level
    (3) downscaling factor

You can check the modified parts with "[YS]"
'''

import os
import time
import math
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
        """
        Initialize the WSI processing class.

        Parameters
        ----------
        root : str
            Path to the directory containing Whole Slide Images (.svs) or a single WSI file.
        output_dir : str
            Path to the directory where output HDF5 files will be saved.
        endswith : str, default='svs'
            File extension of WSI files to search for.
        max_depth : int, default=0
            Maximum directory depth to search under the root directory.
            Set to 0 to search only the top-level directory.
        save_thumb : bool, default=False
            Whether to save thumbnails of WSIs.
            If True, a 'Thumbnail' folder will be created under the output directory.
        tile_size : int, default=224
            Patch size used for tiling WSIs.
        overlap : int, default=0
            Number of overlapping pixels between adjacent tiles.
            Must be smaller than `tile_size`.
        min_tiles : int, default=5
            Minimum number of tiles required in a valid region during post-filtering.
            ### Modify min_tiles (-> Prostate needs to change, default-> BRCA : 150)
        is_normalized : bool, default=False
            Whether to apply stain normalization.
            If True, a default stain normalization template for breast cancer WSIs
            will be used from `./templates/tcga_brca_template.json`.
        """
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
        output_dir = self.output_dir[:self.output_dir.rfind('/')]## removing "/final folder directories"
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
        """
        Loads the whole-slide image using OpenSlide and generates a thumbnail.
        
        Parameters:
            imgname (str): Base name of the image (without extension).
            
        Returns:
            slide: OpenSlide object.
            thumbnail (np.ndarray): Thumbnail image as a NumPy array.
        """
        #svs_file = os.path.join(self.root, f"{imgname}.{self.end}")
        svs_file = os.path.join(imgname) ### for BRCAS dataset
        print("########################## FILE {} #####################".format(svs_file))
        # print(svs_file)
        slide = openslide.OpenSlide(svs_file)
        
        ### [YS] If there's no given thumbnail, we need to extract our own thumbnail with downscaling factor
        ### [SM] If the given thumbnail is too small adjust the thumbnail size most likely to 1000x1000 image
        lv_dimensions = slide.level_dimensions
        
        for idx in range(len(lv_dimensions)):
            x,y = lv_dimensions[-(idx+1)]
            if x*y > 1000000:
                self.thumbnail_level = -(idx+1)
                break
        
       
        if len(slide.level_dimensions) > 1:
            ### [FIX] thumbnail(구 self.thumb_level)과 scaler(self.thumbnail_level)가 서로 다른 레벨을
            ### 참조해 좌표 환산이 어긋나던 버그 — 동일 레벨로 통일.
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
                dimensions = slide.dimensions,
                overlap=0, 
                min_tiles=5,
                downscale_factor =scaler,  
            ) 
        
        self.tiler = tiler
        print(f"Original size: {slide.level_dimensions[0]}")
        print(f"Thumbnail size: {thumbnail.shape}")
        return thumbnail, scaler 

    def compute_foreground_mask(self, image):
        """
        Computes the foreground mask using your get_foreground function.
        
        Parameters:
            image (np.ndarray): Input thumbnail image.
            
        Returns:
            mask (np.ndarray): Foreground mask.
        """
        foreground,mask = self.masker.get_foreground(image, self.is_normalized)
        return foreground, mask
    
    def sample_tiles(self, mask, scaler):
        """
        Calls the tiling function to sample tile coordinates from the slide using the mask.
        
        Parameters:
            slide: The whole-slide image object.
            mask (np.ndarray): Foreground mask.
            
        Returns:
            coords: List of tile starting coordinates (which are also saved to an HDF5 file).
        """
        # get_tile (from Tiling.py) is assumed to save the coordinates to an HDF5 file.
        #coords,metadata = self.tiler.get_tile(slide, mask,imgname,scaler)
        coords = self.tiler.get_tile(mask)
        return coords
    
    
    def save_hdf5(self, fname, coords):
        """
        Save the tile coordinates and metadata to an HDF5 file.
        """
        if fname is None:
            raise ValueError("Missing file name: please set 'fname' before saving HDF5.")
        
        metadata = {
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "total_tiles": len(coords)
        }
        save_path = os.path.join(self.output_dir,f"{fname}.h5")
        
        with h5py.File(save_path, 'w') as hf:
            # coords: numpy 배열로 저장
            hf.create_dataset('coords', data=np.array(coords), compression='gzip')

            # metadata: key-value를 attribute로 저장
            meta_group = hf.create_group('metadata')
            for key, value in metadata.items():
                # 리스트/배열이면 dataset으로, 단일값이면 attribute로 저장
                if isinstance(value, (list, tuple, np.ndarray)):
                    meta_group.create_dataset(key, data=np.array(value))
                else:
                    meta_group.attrs[key] = value
                    
        print(f"Tiling complete. Metadata and tile coordinates saved as {fname}.h5")

        return metadata
    
    def process_image(self, imgname):
        """
        Processes a single image: loads the WSI, computes the foreground mask,
        and samples tiles. It also prints the timing for each step.
        
        Parameters:
            imgname (str): Base name of the image (without extension).
            
        Returns:
            coords: The tile coordinates returned by the tiling function.
        """
        # Simulate a heavy computation to mimic the original code.
        fname = os.path.basename(imgname)[:-4]
        start = time.time()
        math.factorial(100000)
        
        # Load the slide and create a thumbnail.
        thumbnail, scaler= self.load_wsi(imgname)
        io_time = time.time() - start
        print(f"Image I/O: {io_time:.5f} sec")
        
        if self.tiler is None:
            raise ValueError("Tiler is Missing!")
        
        # Compute the foreground mask.
        start = time.time()
        _, mask = self.compute_foreground_mask(thumbnail)
        fg_time = time.time() - start
        print(f"Foreground masking: {fg_time:.5f} sec")
        
        # Sample tiles using the mask.
        start = time.time()
        coords = self.sample_tiles(mask, scaler)
        # Save HDF5 file with metadata.
        metadata = self.save_hdf5(fname, coords)
        
        ts_time = time.time() - start
        
        print(f"Tile sampling: {ts_time:.5f} sec")
        
        return coords, metadata

    def process_images(self, imgList):
        """
        Processes a list of image names.
        
        Parameters:
            imgList (list of str): List of image base names.
            
        Returns:
            results (dict): A dictionary mapping each image name to its tile coordinates.
        """
        for _, imgname in enumerate(tqdm.tqdm(imgList)):
            name = imgname.split('/')[-1][:-4] + '.h5'
            ### [YS] exception code when there's noisy slides (i.e., no meta data etc.)
            if name not in os.listdir(self.output_dir) : 
                print(f"Processing {imgname} ...")
                try : 
                    coords,metadata = self.process_image(imgname)
                except : 
                    print(f"Error generation {imgname} ...")
            else : 
                print(f"Passing {imgname} ...")
                

    def run(self):
        """
        Executes the main workflow for processing Whole Slide Images (WSIs).

        This method determines whether `root` is a directory or a single file.
        It recursively searches for all files with the specified extension (`self.endswith`)
        up to the given `max_depth`, sorts them naturally, and processes them.

        Args:
            None

        Raises:
            ValueError: If `root` is invalid or no files are found.

        Notes:
            - If `root` is a directory, files are processed using `process_images()`.
            - If `root` is a single file, it is processed with `process_image()`.
            - Uses natural sorting (`natsort`) for predictable file ordering.
            - Respects `max_depth` to control recursive search.
        """
        if os.path.isdir(self.root):
            img_list = []
            ext = self.endswith.lower()

            for root, dirs, files in os.walk(self.root):
                # 현재 탐색 중인 root의 깊이 계산 (root이 depth=0)
                rel = os.path.relpath(root, self.root)
                depth = 0 if rel == '.' else rel.count(os.sep) + 1

                # max_depth를 초과하면 하위 디렉토리 탐색 중단
                if self.max_depth > 0 and depth > self.max_depth:
                    dirs[:] = []  # 가지치기
                    continue

                # 지정한 확장자에 맞는 파일만 추가
                for fn in files:
                    if fn.lower().endswith(ext):
                        img_list.append(os.path.join(root, fn))

            # 파일이 하나도 없는 경우 예외 발생
            if not img_list:
                raise ValueError(f"No '{self.endswith}' files found under: {self.root}")

            # 자연 정렬 후 처리
            img_list = natsort.natsorted(img_list)
            self.process_images(img_list)

        elif os.path.isfile(self.root):
            # 단일 파일인 경우 바로 처리
            self.process_image(self.root)
        else:
            raise ValueError("Input a valid root directory or file.")
            
        
            