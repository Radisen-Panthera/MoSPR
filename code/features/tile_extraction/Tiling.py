
import cv2
import numpy as np
from multiprocessing import Pool

class TileSampler:
    def __init__(self, tile_size=224, dimensions=(None, None), overlap=0, min_tiles=5, downscale_factor=64):
        self.fname = None
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_tiles = min_tiles
        self.dimensions = dimensions
        self.downscale_factor = downscale_factor

    def extract_boundaries(self, mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        return contours, hierarchy

    def is_tile_valid(self, tile_mask, tile_coords):
        tile_h, tile_w = tile_mask.shape
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
        x, y, x_idx, y_idx, mask_tile_size, tile = args

        if tile.shape[0] != mask_tile_size or tile.shape[1] != mask_tile_size:
            return None
        if self.is_tile_valid(tile, (x_idx, y_idx)):
            return (x, y)
        return None

    def sample_tiles(self, mask):
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
                valid = np.sum(tile==1) >=((mask_tile_size**2)*0.5)
                if valid :
                    tasks.append((w, h, w_idx, h_idx, mask_tile_size, tile))

        with Pool(processes=8) as pool:
            results = pool.map(self.validate_tile, tasks)
        valid_tiles = [coord for coord in results if coord is not None]
        return valid_tiles

    def adaptive_minimum_tile_contuors(self,mask, scale_factor):
        contours, _ = self.extract_boundaries(mask)

        foreground_pixels = np.sum(mask > 0)
        mask_tile_size = self.tile_size // scale_factor
        pixels_per_tile = mask_tile_size * mask_tile_size
        min_tiles = np.ceil(foreground_pixels / pixels_per_tile).astype(int)
        min_tiles = int(min_tiles * 0.0005)

        threshold_area_mask = (self.tile_size**2 * min_tiles) / (scale_factor ** 2)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= threshold_area_mask]

        return contours

    def filter_tiles_by_boundary(self, valid_tiles, contours):
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
                    break

        remaining_tiles = []
        for cluster in clusters.values():
            if len(cluster) >= self.min_tiles:
                remaining_tiles.extend(cluster)
        return remaining_tiles


    def get_tile(self, mask):

        if mask is None:
            raise ValueError("Failed to load the mask image.")

        contours = self.adaptive_minimum_tile_contuors(mask, self.downscale_factor)

        valid_tiles_thumb = self.sample_tiles(mask)

        filtered_tiles_thumb = self.filter_tiles_by_boundary(valid_tiles_thumb, contours)


        return filtered_tiles_thumb
