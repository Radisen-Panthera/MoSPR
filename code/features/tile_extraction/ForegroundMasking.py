import os
import cv2
import numpy as np
import math
from skimage.morphology import remove_small_objects, remove_small_holes
from enhanceStain import get_imagej_enhanced_thumbnail
from criterion import should_apply_color_adjustment
from stainNorm import TCGAStainNormalizer

class ForegroundMasker:
    def __init__(self, kernel_size=(7, 7), min_hole_area=10, min_area=10, hole_area_threshold=20): 
        self.kernel_size = kernel_size
        self.min_hole_area = min_hole_area
        self.min_area = min_area
        self.hole_area_threshold = hole_area_threshold
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "templates", "tcga_brca_template.json")
        self.normalizer = TCGAStainNormalizer(output_path="inference_output")
        self.normalizer.load_template(template_path)

    def morphological_cleanup(self, mask):
        kernel = np.ones(self.kernel_size, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        return mask

    def enhance_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2RGB)

    def yuv_red_filter(self, image):
        yuv = cv2.cvtColor(self.enhance_contrast(image), cv2.COLOR_RGB2YUV)
        mask = cv2.inRange(yuv, (0, 0, 225), (255, 120, 255))
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def yuv_green_filter(self, image):
        yuv = cv2.cvtColor(self.enhance_contrast(image), cv2.COLOR_RGB2YUV)
        mask = cv2.inRange(yuv, (0, 90, 0), (255, 255, 120))
        return self.morphological_cleanup(mask)

    def yuv_blue_filter(self, image):
        yuv = cv2.cvtColor(self.enhance_contrast(image), cv2.COLOR_RGB2YUV)
        mask = cv2.inRange(yuv, (0, 130, 0), (255, 255, 120))
        return self.morphological_cleanup(mask)

    def remove_small_holes_(self, mask):
        inverted_mask = cv2.bitwise_not(mask)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_mask, connectivity=8)
        cleaned_mask = mask.copy()
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area <= self.min_hole_area:
                cleaned_mask[labels == i] = 255
        return cleaned_mask

    def get_mask(self, image, is_norm):
        red_mask = self.yuv_red_filter(image)
        green_mask = self.yuv_green_filter(image)
        blue_mask = self.yuv_blue_filter(image)
        yuv_mask = red_mask | green_mask | blue_mask
        cleaned_mask = self.remove_small_holes_(yuv_mask)
        mask = cv2.bitwise_not(cleaned_mask)
        if should_apply_color_adjustment(image):
            image = np.array(get_imagej_enhanced_thumbnail(image))

        sample = cv2.bitwise_and(image, image, mask=mask)

        if is_norm : 
            sample = self.normalizer.normalize_image(
                    image=sample,
                    return_image=True,
                )
        return sample, mask

    def filter_grays(self, image, tolerance=15, output_type="bool"): 
        rgb = image.astype(np.int16)
        rg_diff = np.abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
        rb_diff = np.abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
        gb_diff = np.abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
        mask = ~(rg_diff & rb_diff & gb_diff)
        if output_type == "float":
            return mask.astype(float)
        elif output_type == "uint8":
            return (mask.astype(np.uint8) * 255)
        return mask

    def filter_green_channel(self, image, green_thresh=200, avoid_overmask=False, overmask_thresh=90, output_type="bool"): 
        g_channel = image[:, :, 1]
        mask = (g_channel < green_thresh) & (g_channel > 0)
        mask_percentage = (np.sum(mask) / mask.size) * 100
        if mask_percentage >= overmask_thresh and green_thresh < 255 and not avoid_overmask:
            new_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
            return self.filter_green_channel(image, new_thresh, avoid_overmask, overmask_thresh, output_type)
        if output_type == "float":
            return mask.astype(float)
        elif output_type == "uint8":
            return (mask.astype(np.uint8) * 255)
        return mask

    def paraffin_removal(self, image):
        mask_not_green = self.filter_green_channel(image, green_thresh=255)
        mask_not_gray = self.filter_grays(image, tolerance=15)
        combined_mask = mask_not_gray & mask_not_green
        combined_mask_uint8 = (combined_mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(combined_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned_mask = np.zeros_like(combined_mask_uint8)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_area:
                cv2.drawContours(cleaned_mask, [contour], -1, 255, thickness=cv2.FILLED)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return closed_mask

    def remove_background(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_med = cv2.medianBlur(img_hsv[:, :, 1], 7)
        _, img_otsu = cv2.threshold(img_med, 10, 255, cv2.THRESH_BINARY)
        kernel = np.ones((4, 4), np.uint8)
        img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)
        return img_otsu

    def get_foreground(self, image, is_norm):
        foreground, mask = self.get_mask(image, is_norm)

        gray_filtered = self.paraffin_removal(foreground)
        back_filtered = self.remove_background(foreground)
        final_mask = cv2.bitwise_and(gray_filtered, back_filtered)
        return foreground, final_mask
