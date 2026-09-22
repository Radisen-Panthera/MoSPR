'''
modified by YS, date : 2025-04-10

modification :
- adding parameters below : 
    (1) extension name parameter 'end'
    (2) thumbnail level
    (3) downscaling factor

You can check the modified parts with "[YS]"
'''

import cv2
import numpy as np
from multiprocessing import Pool

class TileSampler:
    def __init__(self, tile_size=224, dimensions=(None, None), overlap=0, min_tiles=5, downscale_factor=64):
        """
        Initialize the tile configuration for Tiling processing.

        Parameters
        ----------
        tile_size : int, default=224
            Desired tile size at full resolution (e.g., 224 or 256).
        dimensions : tuple of (int, int), default=(None, None)
            Level-0 (full resolution) dimensions of the WSI in (width, height) format.
        overlap : float, default=0
            Overlap ratio between tiles. 0 means no overlap.
        min_tiles : int, default=5
            Minimum number of tiles required in a valid boundary region.
            (Adjust depending on organ type, e.g., BRCA: 50 → reduced to 50 from 150.)
        downscale_factor : int, default=64
            Downscaling factor applied when no thumbnail is available.
        """
        self.fname = None
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_tiles = min_tiles
        self.dimensions = dimensions
        self.downscale_factor = downscale_factor
        
    def extract_boundaries(self, mask):
        """
        Extract boundaries (contours) from the foreground mask.
        """
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        return contours, hierarchy

    def is_tile_valid(self, tile_mask, tile_coords):
        """
        Check if a tile is valid based on its centroid and border edge points.
        """
        tile_h, tile_w = tile_mask.shape
        # centroid = (tile_coords[0] + tile_w // 2, tile_coords[1] + tile_h // 2)
        if tile_mask[tile_h // 2, tile_w // 2] == 0:
            return False
        top_edge = tile_mask[0, :]
        bottom_edge = tile_mask[-1, :]
        left_edge = tile_mask[:, 0]
        right_edge = tile_mask[:, -1]
        edge_count = (np.count_nonzero(top_edge) + np.count_nonzero(bottom_edge) +
                      np.count_nonzero(left_edge) + np.count_nonzero(right_edge))
        return edge_count >= 2

    def validate_tile(self, args):
        """
        Worker function for processing a single tile.
        """
        x, y, x_idx, y_idx, mask_tile_size, tile = args
    
        if tile.shape[0] != mask_tile_size or tile.shape[1] != mask_tile_size:
            return None  # Skip incomplete tiles at the edges.
        if self.is_tile_valid(tile, (x_idx, y_idx)):
            return (x, y)
        return None

    def sample_tiles(self, mask):
        """
        Tile the mask using multiprocessing.
        """
        unique_vals = np.unique(mask)
        if not np.all(np.isin(unique_vals, [0, 1])):
            mask = (mask / 255).astype(np.uint8)
            
        width, height = self.dimensions
        mask_tile_size = round(self.tile_size/ self.downscale_factor)
        stride = int(self.tile_size - self.tile_size * self.overlap)
        tasks = []
        for h in range(0, height , stride):
            for w in range(0, width, stride):
                h_idx, w_idx = round(h/ self.downscale_factor), round(w/ self.downscale_factor)
                tile = mask[h_idx:h_idx+mask_tile_size, w_idx:w_idx+mask_tile_size]
                valid = np.sum(tile==1) >=((mask_tile_size**2)*0.5) ### [FIX] 0.7은 다공성 조직(폐포/세뇨관)을 과도하게 탈락시켜 0.5로 완화
                if valid :
                    tasks.append((w, h, w_idx, h_idx, mask_tile_size, tile))
        
        with Pool(processes=8) as pool:
            results = pool.map(self.validate_tile, tasks)
        valid_tiles = [coord for coord in results if coord is not None]
        return valid_tiles
    
    def adaptive_minimum_tile_contuors(self,mask, scale_factor):
        """
        For each slide, determin minimum tile numbers must be clustered.
        If the sampled area is less then given threshold, will be dropped from tile sampling
        """
        contours, _ = self.extract_boundaries(mask)
        
        foreground_pixels = np.sum(mask > 0)
        mask_tile_size = self.tile_size // scale_factor
        pixels_per_tile = mask_tile_size * mask_tile_size
        min_tiles = np.ceil(foreground_pixels / pixels_per_tile).astype(int)
        # print(min_tiles,foreground_pixels,pixels_per_tile)
        # print(mask_tile_size,self.tile_size,scale_factor)
        min_tiles = int(min_tiles * 0.0005) #less than 0.1% region will be drop
        
        # Drop small contours: Full-res threshold is (tile_size^2 * min_tiles) pixels.
        # Convert this threshold to mask scale.
        threshold_area_mask = (self.tile_size**2 * min_tiles) / (scale_factor ** 2)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= threshold_area_mask]
        
        return contours

    def filter_tiles_by_boundary(self, valid_tiles, contours):
        """
        For each tile, determine which boundary (contour) it lies in based on its centroid.
        Only retain tiles in boundaries that contain at least self.min_tiles.
        """
        mask_tile_size = round(self.tile_size / self.downscale_factor)
        clusters = {}
        for tile in valid_tiles:
            x_idx, y_idx = tile
            y = round(y_idx/self.downscale_factor)
            x = round(x_idx/self.downscale_factor)
            centroid = (x + mask_tile_size / 2, y + mask_tile_size / 2)
            for idx, contour in enumerate(contours):
                if cv2.pointPolygonTest(contour, centroid, False) >= 0:
                    clusters.setdefault(idx, []).append(tile)
                    break  # Assign to first matching contour.
                
        remaining_tiles = []
        for cluster in clusters.values():
            if len(cluster) >= self.min_tiles:
                remaining_tiles.extend(cluster)
        return remaining_tiles


    def get_tile(self, mask):
        """
        Main function that extracts tile coordinates from a WSI given a foreground mask.
        
        Parameters:
            slide: Whole-slide image object (e.g., from OpenSlide).
            mask (np.ndarray): Foreground mask (assumed to be derived from the thumbnail).
        
        Returns:
            valid_tiles_full (list): List of upscaled tile coordinates in full resolution.
        """
        
        if mask is None:
            raise ValueError("Failed to load the mask image.")
        
        contours = self.adaptive_minimum_tile_contuors(mask, self.downscale_factor)
        
        # Sample valid tile starting coordinates at thumbnail (mask) scale.
        valid_tiles_thumb = self.sample_tiles(mask)

        # Filter the valid tiles by their boundaries.
        filtered_tiles_thumb = self.filter_tiles_by_boundary(valid_tiles_thumb, contours)

        
        return filtered_tiles_thumb