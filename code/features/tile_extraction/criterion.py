import cv2
import numpy as np
from PIL import Image

def needs_color_adjustment(image, saturation_threshold=0.15):
    """
    Determine if image needs color adjustment based on saturation
    """
    # Convert PIL to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Convert to HSV
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Calculate mean saturation (excluding white/background areas)
    saturation = hsv[:, :, 1] / 255.0
    
    # Mask out very bright areas (likely background)
    brightness = hsv[:, :, 2] / 255.0
    mask = brightness < 0.9
    
    if np.sum(mask) > 0:
        mean_saturation = np.mean(saturation[mask])
    else:
        mean_saturation = np.mean(saturation)
    
    return mean_saturation < saturation_threshold

def needs_adjustment_contrast(image, contrast_threshold=30):
    """
    Detect low contrast images that need adjustment
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image.convert('L'))  # Convert to grayscale
    else:
        img_array = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Calculate standard deviation (measure of contrast)
    contrast = np.std(img_array)
    
    return contrast < contrast_threshold

def needs_adjustment_histogram(image, low_intensity_threshold=0.7):
    """
    Check if too much of the image is in low intensity range
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Convert to grayscale for intensity analysis
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Check what percentage of pixels are in lower intensity ranges
    total_pixels = gray.shape[0] * gray.shape[1]
    low_intensity_pixels = np.sum(hist[0:128])  # First half of intensity range
    
    low_intensity_ratio = low_intensity_pixels / total_pixels
    
    return low_intensity_ratio > low_intensity_threshold

def should_apply_color_adjustment(thumbnail):
    """
    Combined approach using multiple criteria
    """
    # Check saturation
    low_saturation = needs_color_adjustment(thumbnail, saturation_threshold=0.15)
    
    # Check contrast
    low_contrast = needs_adjustment_contrast(thumbnail, contrast_threshold=25)
    
    # Check intensity distribution
    low_intensity = needs_adjustment_histogram(thumbnail, low_intensity_threshold=0.65)
    
    # Apply adjustment if at least 2 out of 3 criteria are met
    criteria_met = sum([low_saturation,low_contrast, low_intensity])
    
    return criteria_met >= 2