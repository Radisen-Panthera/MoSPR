import openslide
import numpy as np
from PIL import Image

class AutoEnhancer:
    """
    Minimal class to replicate ImageJ's Auto button brightness/contrast enhancement
    for OpenSlide whole slide images
    """
    
    def __init__(self, thumbnail):
        self.thumbnail = thumbnail
        self.auto_threshold = 5000
        
    def get_enhanced_thumbnail(self, size=None):
        """
        Get ImageJ Auto enhanced thumbnail
        
        Args:
            size (tuple, optional): Thumbnail size. Defaults to lowest resolution level.
            
        Returns:
            PIL.Image: Enhanced thumbnail image
        """
        img_array = np.array(self.thumbnail)
        
        # Handle RGBA by dropping alpha channel
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        # Calculate min/max for each channel
        min_vals = []
        max_vals = []
        
        for channel in range(img_array.shape[2]):
            channel_data = img_array[:, :, channel]
            hist, bins = np.histogram(channel_data, bins=256, range=(0, 255))
            
            min_val, max_val = self._find_thresholds(hist)
            min_vals.append(min_val)
            max_vals.append(max_val)
        
        # Use conservative min/max to prevent color shifts
        global_min = max(min_vals)
        global_max = min(max_vals)
        
        # Apply enhancement
        enhanced = np.zeros_like(img_array, dtype=np.uint8)
        
        for channel in range(img_array.shape[2]):
            channel_data = img_array[:, :, channel].astype(np.float32)
            
            if global_max > global_min:
                stretched = (channel_data - global_min) * 255.0 / (global_max - global_min)
                enhanced[:, :, channel] = np.clip(stretched, 0, 255).astype(np.uint8)
            else:
                enhanced[:, :, channel] = channel_data.astype(np.uint8)
        
        return Image.fromarray(enhanced)
    
    def _find_thresholds(self, hist):
        """Find min/max thresholds using ImageJ's algorithm"""
        total_pixels = np.sum(hist)
        
        if total_pixels < self.auto_threshold:
            # For small images
            min_val = np.argmax(hist > 0)
            max_val = 255 - np.argmax(hist[::-1] > 0)
            min_val = max(0, min_val - 1)
            max_val = min(255, max_val + 1)
        else:
            # For larger images - use 0.01% saturation
            threshold_count = int(total_pixels * 0.0001)
            
            # Find min threshold
            cumsum = 0
            min_val = 0
            for i in range(256):
                cumsum += hist[i]
                if cumsum > threshold_count:
                    min_val = i
                    break
            
            # Find max threshold
            cumsum = 0
            max_val = 255
            for i in range(255, -1, -1):
                cumsum += hist[i]
                if cumsum > threshold_count:
                    max_val = i
                    break
        
        return min_val, max_val

# Simple usage function
def get_imagej_enhanced_thumbnail(thumbnail, size=None):
    """
    Get ImageJ Auto enhanced thumbnail
    
    Args:
        slide_path (str): Path to slide file
        size (tuple, optional): Thumbnail size
        
    Returns:
        PIL.Image: Enhanced thumbnail
    """
    enhancer = AutoEnhancer(thumbnail)
    return enhancer.get_enhanced_thumbnail(size)

