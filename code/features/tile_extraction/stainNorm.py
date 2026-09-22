
import os
import cv2
import numpy as np
import openslide
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass, asdict
from sklearn.cluster import KMeans
from scipy import stats
import pickle
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class StainStats:
    hue_mean: float
    hue_std: float
    sat_mean: float
    sat_std: float
    val_mean: float
    val_std: float
    tissue_ratio: float

@dataclass
class NormalizationParams:
    target_hue: float = 160.0
    target_sat: float = 120.0
    target_val: float = 180.0
    hue_tolerance: float = 15.0
    sat_tolerance: float = 40.0
    brightness_threshold: float = 240
    saturation_threshold: float = 30

@dataclass
class NormalizationTemplate:
    reference_stats: StainStats
    normalization_params: NormalizationParams
    creation_date: str
    dataset_info: Dict
    version: str = "1.0"

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

class TCGAStainNormalizer:

    def __init__(self, dataset_path: str = None, output_path: str = "output", 
                 thumbnail_size: Tuple[int, int] = (1024, 1024)):
        if dataset_path:
            self.dataset_path = Path(dataset_path)
        else:
            self.dataset_path = None
        self.output_path = Path(output_path)
        self.thumbnail_size = thumbnail_size
        self.params = NormalizationParams()

        if dataset_path is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)
            (self.output_path / "normalized").mkdir(exist_ok=True)
            (self.output_path / "analysis").mkdir(exist_ok=True)
            (self.output_path / "thumbnails").mkdir(exist_ok=True)
            (self.output_path / "templates").mkdir(exist_ok=True)

        self.slide_stats: Dict[str, StainStats] = {}
        self.reference_stats: Optional[StainStats] = None
        self.normalization_template: Optional[NormalizationTemplate] = None

    def get_slide_list(self) -> List[Path]:
        if not self.dataset_path:
            raise ValueError("Dataset path not set")
        svs_files = list(self.dataset_path.glob("**/*.svs"))
        logger.info(f"Found {len(svs_files)} SVS files in dataset")
        return svs_files

    def extract_thumbnail(self, slide_path: Union[str, Path]) -> Optional[np.ndarray]:
        try:
            slide = openslide.OpenSlide(str(slide_path))
            thumbnail = slide.get_thumbnail(self.thumbnail_size)
            slide.close()
            return np.array(thumbnail)
        except Exception as e:
            logger.error(f"Error extracting thumbnail from {slide_path}: {e}")
            return None

    def analyze_tissue_mask(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        _, tissue_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)

        return tissue_mask.astype(bool)

    def extract_stain_stats(self, image: np.ndarray) -> StainStats:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        tissue_mask = self.analyze_tissue_mask(image)

        brightness_mask = hsv[:, :, 2] < self.params.brightness_threshold
        saturation_mask = hsv[:, :, 1] > self.params.saturation_threshold
        combined_mask = tissue_mask & brightness_mask & saturation_mask

        if np.sum(combined_mask) == 0:
            logger.warning("No tissue detected in image")
            return StainStats(0, 0, 0, 0, 0, 0, 0)

        tissue_hue = hsv[combined_mask, 0].astype(np.float32)
        tissue_sat = hsv[combined_mask, 1].astype(np.float32)
        tissue_val = hsv[combined_mask, 2].astype(np.float32)

        tissue_ratio = np.sum(combined_mask) / (image.shape[0] * image.shape[1])

        return StainStats(
            hue_mean=float(np.mean(tissue_hue)),
            hue_std=float(np.std(tissue_hue)),
            sat_mean=float(np.mean(tissue_sat)),
            sat_std=float(np.std(tissue_sat)),
            val_mean=float(np.mean(tissue_val)),
            val_std=float(np.std(tissue_val)),
            tissue_ratio=float(tissue_ratio)
        )

    def needs_normalization(self, stats: StainStats) -> bool:
        hue_diff = abs(stats.hue_mean - self.params.target_hue)
        if hue_diff > 90:
            hue_diff = 180 - hue_diff

        needs_hue_adjustment = hue_diff > self.params.hue_tolerance
        needs_sat_adjustment = abs(stats.sat_mean - self.params.target_sat) > self.params.sat_tolerance
        low_saturation = stats.sat_mean < 60

        return needs_hue_adjustment or needs_sat_adjustment or low_saturation

    def normalize_slide(self, image: np.ndarray, target_stats: Optional[StainStats] = None) -> np.ndarray:
        if target_stats is None:
            target_stats = self.reference_stats or StainStats(
                self.params.target_hue, self.params.hue_tolerance,
                self.params.target_sat, self.params.sat_tolerance,
                self.params.target_val, 30, 0.3
            )

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

        tissue_mask = self.analyze_tissue_mask(image)
        brightness_mask = hsv[:, :, 2] < self.params.brightness_threshold
        saturation_mask = hsv[:, :, 1] > self.params.saturation_threshold
        combined_mask = tissue_mask & brightness_mask & saturation_mask

        if np.sum(combined_mask) == 0:
            logger.warning("No tissue detected for normalization")
            return image

        current_stats = self.extract_stain_stats(image)

        tissue_pixels = hsv[combined_mask]

        hue_diff = target_stats.hue_mean - current_stats.hue_mean
        if abs(hue_diff) > 90:
            if hue_diff > 0:
                hue_diff -= 180
            else:
                hue_diff += 180

        tissue_pixels[:, 0] = (tissue_pixels[:, 0] + hue_diff) % 180

        if current_stats.sat_std > 0:
            tissue_pixels[:, 1] = (tissue_pixels[:, 1] - current_stats.sat_mean) / current_stats.sat_std
            tissue_pixels[:, 1] = tissue_pixels[:, 1] * target_stats.sat_std + target_stats.sat_mean
            tissue_pixels[:, 1] = np.clip(tissue_pixels[:, 1], 0, 255)

        if current_stats.val_std > 0:
            tissue_pixels[:, 2] = (tissue_pixels[:, 2] - current_stats.val_mean) / current_stats.val_std
            tissue_pixels[:, 2] = tissue_pixels[:, 2] * target_stats.val_std + target_stats.val_mean
            tissue_pixels[:, 2] = np.clip(tissue_pixels[:, 2], 0, 255)

        hsv[combined_mask] = tissue_pixels

        normalized = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return normalized

    def establish_reference(self, sample_size: int = 50) -> StainStats:
        if not self.dataset_path:
            raise ValueError("Dataset path required for establishing reference")

        slide_files = self.get_slide_list()

        if len(slide_files) < sample_size:
            sample_size = len(slide_files)

        np.random.seed(42)
        sample_slides = np.random.choice(slide_files, sample_size, replace=False)

        all_stats = []
        logger.info(f"Analyzing {sample_size} slides to establish reference...")

        for slide_path in tqdm(sample_slides):
            thumbnail = self.extract_thumbnail(slide_path)
            if thumbnail is not None:
                stats = self.extract_stain_stats(thumbnail)
                if stats.tissue_ratio > 0.1:
                    all_stats.append(stats)

        if not all_stats:
            logger.error("No suitable reference slides found")
            return StainStats(160, 15, 120, 40, 180, 30, 0.3)

        hue_values = [s.hue_mean for s in all_stats]
        sat_values = [s.sat_mean for s in all_stats]
        val_values = [s.val_mean for s in all_stats]

        reference_stats = StainStats(
            hue_mean=float(np.median(hue_values)),
            hue_std=float(np.std(hue_values)),
            sat_mean=float(np.median(sat_values)),
            sat_std=float(np.std(sat_values)),
            val_mean=float(np.median(val_values)),
            val_std=float(np.std(val_values)),
            tissue_ratio=float(np.median([s.tissue_ratio for s in all_stats]))
        )

        self.reference_stats = reference_stats
        logger.info(f"Reference established: Hue={reference_stats.hue_mean:.1f}, "
                   f"Sat={reference_stats.sat_mean:.1f}, Val={reference_stats.val_mean:.1f}")

        return reference_stats

    def export_template(self, template_name: str = None) -> str:
        if not self.reference_stats:
            raise ValueError("No reference statistics available. Run establish_reference() first.")

        if template_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            template_name = f"normalization_template_{timestamp}"

        dataset_info = {
            "total_slides": len(self.slide_stats) if self.slide_stats else 0,
            "dataset_path": str(self.dataset_path) if self.dataset_path else None,
            "creation_timestamp": datetime.now().isoformat()
        }

        template = NormalizationTemplate(
            reference_stats=self.reference_stats,
            normalization_params=self.params,
            creation_date=datetime.now().isoformat(),
            dataset_info=dataset_info
        )

        self.normalization_template = template

        json_path = self.output_path / "templates" / f"{template_name}.json"
        template_dict = asdict(template)
        template_dict = convert_to_serializable(template_dict)

        with open(json_path, 'w') as f:
            json.dump(template_dict, f, indent=2)

        pkl_path = self.output_path / "templates" / f"{template_name}.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(template, f)

        logger.info(f"Template exported to: {json_path} and {pkl_path}")
        return str(json_path)

    def load_template(self, template_path: str) -> NormalizationTemplate:
        template_path = Path(template_path)

        if template_path.suffix == '.json':
            with open(template_path, 'r') as f:
                template_dict = json.load(f)

            reference_stats = StainStats(**template_dict['reference_stats'])
            normalization_params = NormalizationParams(**template_dict['normalization_params'])

            template = NormalizationTemplate(
                reference_stats=reference_stats,
                normalization_params=normalization_params,
                creation_date=template_dict['creation_date'],
                dataset_info=template_dict['dataset_info'],
                version=template_dict.get('version', '1.0')
            )

        elif template_path.suffix == '.pkl':
            with open(template_path, 'rb') as f:
                template = pickle.load(f)

        else:
            raise ValueError("Template file must be .json or .pkl")

        self.normalization_template = template
        self.reference_stats = template.reference_stats
        self.params = template.normalization_params

        logger.info(f"Template loaded from: {template_path}")
        logger.info(f"Template info: Created {template.creation_date}, "
                   f"Reference Hue={template.reference_stats.hue_mean:.1f}")

        return template

    def normalize_image(self, image: Union[np.ndarray, Image.Image, str, Path],
                       output_path: str = None,
                       return_image: bool = True,
                       save_comparison: bool = False,
                       comparison_title: str = "Image Normalization") -> Optional[np.ndarray]:
        if not self.reference_stats:
            raise ValueError("No normalization template loaded. Use load_template() first.")

        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if image_path.suffix.lower() in ['.svs', '.tif', '.tiff']:
                original_array = self.extract_thumbnail(image_path)
                if original_array is None:
                    logger.error(f"Failed to load slide from {image_path}")
                    return None
            else:
                pil_image = Image.open(image_path)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                original_array = np.array(pil_image)
        elif isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            original_array = np.array(image)
        elif isinstance(image, np.ndarray):
            original_array = image.copy()
            if len(original_array.shape) != 3 or original_array.shape[2] != 3:
                raise ValueError("Image must be RGB format with shape (H, W, 3)")
        else:
            raise TypeError("Image must be numpy array, PIL Image, or file path")

        if original_array.dtype != np.uint8:
            if original_array.max() <= 1.0:
                original_array = (original_array * 255).astype(np.uint8)
            else:
                original_array = original_array.astype(np.uint8)

        logger.info(f"Normalizing image with shape: {original_array.shape}")

        original_stats = self.extract_stain_stats(original_array)

        needs_norm = self.needs_normalization(original_stats)
        logger.info(f"Normalization needed: {needs_norm}")

        normalized = self.normalize_slide(original_array, self.reference_stats)

        normalized_stats = self.extract_stain_stats(normalized)

        logger.info(f"Original  - Hue: {original_stats.hue_mean:.1f}, "
                   f"Sat: {original_stats.sat_mean:.1f}, Val: {original_stats.val_mean:.1f}")
        logger.info(f"Normalized - Hue: {normalized_stats.hue_mean:.1f}, "
                   f"Sat: {normalized_stats.sat_mean:.1f}, Val: {normalized_stats.val_mean:.1f}")
        logger.info(f"Target    - Hue: {self.reference_stats.hue_mean:.1f}, "
                   f"Sat: {self.reference_stats.sat_mean:.1f}, Val: {self.reference_stats.val_mean:.1f}")

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(normalized).save(output_path)
            logger.info(f"Normalized image saved to: {output_path}")

        if save_comparison:
            if output_path:
                comparison_path = output_path.parent / f"{output_path.stem}_comparison.png"
            else:
                comparison_path = self.output_path / "comparisons" / f"image_comparison.png"
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_comparison_image(original_array, normalized, comparison_path, 
                                      original_stats, normalized_stats, comparison_title)

        if return_image:
            return normalized

        return None

    def normalize_image_batch(self, images: List[Union[np.ndarray, Image.Image, str, Path]],
                             output_dir: str = None,
                             return_images: bool = False,
                             save_comparisons: bool = False) -> List[Optional[np.ndarray]]:
        if not self.reference_stats:
            raise ValueError("No normalization template loaded. Use load_template() first.")

        results = []

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Batch normalizing {len(images)} images...")

        for i, image in enumerate(tqdm(images)):
            try:
                output_path = None
                if output_dir:
                    if isinstance(image, (str, Path)):
                        filename = Path(image).stem
                    else:
                        filename = f"image_{i:04d}"
                    output_path = output_dir / f"{filename}_normalized.png"

                normalized = self.normalize_image(
                    image=image,
                    output_path=output_path,
                    return_image=return_images,
                    save_comparison=save_comparisons,
                    comparison_title=f"Image {i+1} Normalization"
                )

                results.append(normalized)

            except Exception as e:
                logger.error(f"Error normalizing image {i}: {e}")
                results.append(None)

        success_count = sum(1 for r in results if r is not None)
        logger.info(f"Batch normalization complete: {success_count}/{len(images)} successful")

        return results

    def normalize_single_slide(self, slide_path: Union[str, Path], 
                             output_path: str = None, 
                             return_image: bool = False,
                             save_comparison: bool = False) -> Optional[np.ndarray]:
        if not self.reference_stats:
            raise ValueError("No normalization template loaded. Use load_template() first.")

        slide_path = Path(slide_path)
        slide_id = slide_path.stem

        logger.info(f"Normalizing single slide: {slide_id}")

        thumbnail = self.extract_thumbnail(slide_path)
        if thumbnail is None:
            logger.error(f"Failed to extract thumbnail from {slide_path}")
            return None

        original_stats = self.extract_stain_stats(thumbnail)

        needs_norm = self.needs_normalization(original_stats)
        logger.info(f"Normalization needed: {needs_norm}")

        normalized = self.normalize_slide(thumbnail, self.reference_stats)

        normalized_stats = self.extract_stain_stats(normalized)

        logger.info(f"Original  - Hue: {original_stats.hue_mean:.1f}, "
                   f"Sat: {original_stats.sat_mean:.1f}, Val: {original_stats.val_mean:.1f}")
        logger.info(f"Normalized - Hue: {normalized_stats.hue_mean:.1f}, "
                   f"Sat: {normalized_stats.sat_mean:.1f}, Val: {normalized_stats.val_mean:.1f}")
        logger.info(f"Target    - Hue: {self.reference_stats.hue_mean:.1f}, "
                   f"Sat: {self.reference_stats.sat_mean:.1f}, Val: {self.reference_stats.val_mean:.1f}")

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(normalized).save(output_path)
            logger.info(f"Normalized image saved to: {output_path}")

        if save_comparison:
            comparison_path = self.output_path / "comparisons" / f"{slide_id}_comparison.png"
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_comparison_image(thumbnail, normalized, comparison_path, 
                                      original_stats, normalized_stats)

        if return_image:
            return normalized

        return None

    def _save_comparison_image(self, original: np.ndarray, normalized: np.ndarray, 
                             save_path: Path, orig_stats: StainStats, norm_stats: StainStats,
                             title: str = "Stain Normalization Comparison"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(original)
        axes[0].set_title(f'Original\nHue: {orig_stats.hue_mean:.1f}, '
                         f'Sat: {orig_stats.sat_mean:.1f}, Val: {orig_stats.val_mean:.1f}')
        axes[0].axis('off')

        axes[1].imshow(normalized)
        axes[1].set_title(f'Normalized\nHue: {norm_stats.hue_mean:.1f}, '
                         f'Sat: {norm_stats.sat_mean:.1f}, Val: {norm_stats.val_mean:.1f}')
        axes[1].axis('off')

        fig.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Comparison image saved to: {save_path}")

    def batch_normalize_slides(self, slide_paths: List[Union[str, Path]], 
                             output_dir: str, 
                             save_comparisons: bool = False) -> Dict[str, bool]:
        if not self.reference_stats:
            raise ValueError("No normalization template loaded. Use load_template() first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        logger.info(f"Batch normalizing {len(slide_paths)} slides...")

        for slide_path in tqdm(slide_paths):
            slide_path = Path(slide_path)
            slide_id = slide_path.stem

            try:
                output_path = output_dir / f"{slide_id}_normalized.png"
                self.normalize_single_slide(
                    slide_path=slide_path,
                    output_path=output_path,
                    save_comparison=save_comparisons
                )
                results[slide_id] = True

            except Exception as e:
                logger.error(f"Error normalizing {slide_id}: {e}")
                results[slide_id] = False

        success_count = sum(results.values())
        logger.info(f"Batch normalization complete: {success_count}/{len(slide_paths)} successful")

        return results

    def analyze_dataset(self, save_results: bool = True) -> Dict:
        slide_files = self.get_slide_list()

        logger.info(f"Analyzing {len(slide_files)} slides...")

        results = {
            'total_slides': len(slide_files),
            'processed_slides': 0,
            'failed_slides': 0,
            'slides_needing_normalization': 0,
            'slide_stats': {},
            'normalization_needed': {}
        }

        for slide_path in tqdm(slide_files):
            slide_id = slide_path.stem

            try:
                thumbnail = self.extract_thumbnail(slide_path)
                if thumbnail is None:
                    results['failed_slides'] += 1
                    continue

                thumbnail_path = self.output_path / "thumbnails" / f"{slide_id}.png"
                Image.fromarray(thumbnail).save(thumbnail_path)

                stats = self.extract_stain_stats(thumbnail)
                self.slide_stats[slide_id] = stats

                needs_norm = self.needs_normalization(stats)

                results['slide_stats'][slide_id] = {
                    'hue_mean': float(stats.hue_mean),
                    'sat_mean': float(stats.sat_mean),
                    'val_mean': float(stats.val_mean),
                    'tissue_ratio': float(stats.tissue_ratio)
                }
                results['normalization_needed'][slide_id] = bool(needs_norm)

                if needs_norm:
                    results['slides_needing_normalization'] += 1

                results['processed_slides'] += 1

            except Exception as e:
                logger.error(f"Error processing {slide_path}: {e}")
                results['failed_slides'] += 1

        if save_results:
            results_path = self.output_path / "analysis" / "dataset_analysis.json"

            serializable_results = convert_to_serializable(results)

            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)

            stats_path = self.output_path / "analysis" / "slide_statistics.pkl"
            with open(stats_path, 'wb') as f:
                pickle.dump(self.slide_stats, f)

        logger.info(f"Analysis complete: {results['processed_slides']} processed, "
                   f"{results['slides_needing_normalization']} need normalization")

        return results

def create_template_example():
    normalizer = TCGAStainNormalizer(
        dataset_path="/path/to/training/dataset",
        output_path="template_output"
    )

    normalizer.analyze_dataset()
    normalizer.establish_reference(sample_size=100)

    template_path = normalizer.export_template("my_tcga_template")
    print(f"Template saved to: {template_path}")

def image_inference_examples():
    normalizer = TCGAStainNormalizer(output_path="inference_output")

    normalizer.load_template("template_output/templates/my_tcga_template.json")

    normalized_img1 = normalizer.normalize_image(
        image="path/to/image.png",
        output_path="output/normalized_image1.png",
        save_comparison=True
    )

    from PIL import Image
    pil_img = Image.open("path/to/image.jpg")
    normalized_img2 = normalizer.normalize_image(
        image=pil_img,
        output_path="output/normalized_image2.png",
        save_comparison=True
    )

    import numpy as np
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    normalized_img3 = normalizer.normalize_image(
        image=img_array,
        output_path="output/normalized_image3.png",
        save_comparison=True
    )

    normalized_slide = normalizer.normalize_image(
        image="path/to/slide.svs",
        output_path="output/normalized_slide.png",
        save_comparison=True,
        comparison_title="Slide Normalization"
    )

    image_list = [
        "path/to/image1.png",
        "path/to/image2.jpg",
        "path/to/slide.svs",
        pil_img,
        img_array
    ]

    normalized_batch = normalizer.normalize_image_batch(
        images=image_list,
        output_dir="output/batch_normalized",
        return_images=True,
        save_comparisons=True
    )

    print(f"Batch normalization completed: {len([x for x in normalized_batch if x is not None])} successful")

def slide_inference_example():
    normalizer = TCGAStainNormalizer(output_path="inference_output")

    normalizer.load_template("template_output/templates/my_tcga_template.json")

    slide_path = "/path/to/single/slide.svs"
    normalized_image = normalizer.normalize_single_slide(
        slide_path=slide_path,
        output_path="output/normalized_slide.png",
        return_image=True,
        save_comparison=True
    )

    slide_paths = ["/path/to/slide1.svs", "/path/to/slide2.svs"]
    results = normalizer.batch_normalize_slides(
        slide_paths=slide_paths,
        output_dir="output/batch_normalized",
        save_comparisons=True
    )

    print(f"Batch results: {results}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="directory of .svs slides used to fit the template")
    ap.add_argument("--output", default="output")
    a = ap.parse_args()

    normalizer = TCGAStainNormalizer(
        dataset_path=a.dataset,
        output_path=a.output,
        thumbnail_size=(1024, 1024)
    )

    logger.info("Step 1: Analyzing dataset...")
    analysis_results = normalizer.analyze_dataset()

    logger.info("Step 2: Establishing reference...")
    reference_stats = normalizer.establish_reference(sample_size=100)

    logger.info("Step 3: Exporting normalization template...")
    template_path = normalizer.export_template("tcga_brca_template")

    logger.info(f"Template creation complete! Template saved to: {template_path}")

    logger.info("\n--- Example Inference Usage ---")
    logger.info("To use this template for inference:")
    logger.info("1. Load template: normalizer.load_template('path/to/template.json')")
    logger.info("2. Normalize single slide: normalizer.normalize_single_slide('path/to/slide.svs')")

if __name__ == "__main__":
    main()
