import cv2
import numpy as np
from PIL import Image

def needs_color_adjustment(image, saturation_threshold=0.15):
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image

    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    saturation = hsv[:, :, 1] / 255.0

    brightness = hsv[:, :, 2] / 255.0
    mask = brightness < 0.9

    if np.sum(mask) > 0:
        mean_saturation = np.mean(saturation[mask])
    else:
        mean_saturation = np.mean(saturation)

    return mean_saturation < saturation_threshold

def needs_adjustment_contrast(image, contrast_threshold=30):
    if isinstance(image, Image.Image):
        img_array = np.array(image.convert('L'))
    else:
        img_array = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    contrast = np.std(img_array)

    return contrast < contrast_threshold

def needs_adjustment_histogram(image, low_intensity_threshold=0.7):
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    total_pixels = gray.shape[0] * gray.shape[1]
    low_intensity_pixels = np.sum(hist[0:128])

    low_intensity_ratio = low_intensity_pixels / total_pixels

    return low_intensity_ratio > low_intensity_threshold

def should_apply_color_adjustment(thumbnail):
    low_saturation = needs_color_adjustment(thumbnail, saturation_threshold=0.15)

    low_contrast = needs_adjustment_contrast(thumbnail, contrast_threshold=25)

    low_intensity = needs_adjustment_histogram(thumbnail, low_intensity_threshold=0.65)

    criteria_met = sum([low_saturation,low_contrast, low_intensity])

    return criteria_met >= 2
