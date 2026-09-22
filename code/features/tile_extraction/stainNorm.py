"""
TCGA-BRCA Stain Normalization Pipeline with Template Export/Import
================================================================

Enhanced version with template export/import functionality and single-slide inference
"""

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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class StainStats:
    """Container for stain statistics"""
    hue_mean: float
    hue_std: float
    sat_mean: float
    sat_std: float
    val_mean: float
    val_std: float
    tissue_ratio: float

@dataclass
class NormalizationParams:
    """Normalization parameters"""
    target_hue: float = 160.0      # Target hue for normalized slides (pink)
    target_sat: float = 120.0      # Target saturation
    target_val: float = 180.0      # Target value/brightness
    hue_tolerance: float = 15.0    # Allowed hue variation
    sat_tolerance: float = 40.0    # Allowed saturation variation
    brightness_threshold: float = 240  # Background detection threshold
    saturation_threshold: float = 30   # Minimum saturation for tissue

@dataclass
class NormalizationTemplate:
    """Complete normalization template for export/import"""
    reference_stats: StainStats
    normalization_params: NormalizationParams
    creation_date: str
    dataset_info: Dict
    version: str = "1.0"

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
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
    """
    Main class for TCGA-BRCA stain normalization with template export/import
    """
    
    def __init__(self, dataset_path: str = None, output_path: str = "output", 
                 thumbnail_size: Tuple[int, int] = (1024, 1024)):
        if dataset_path:
            self.dataset_path = Path(dataset_path)
        else:
            self.dataset_path = None
        self.output_path = Path(output_path)
        self.thumbnail_size = thumbnail_size
        self.params = NormalizationParams()
        
        # Create output directories
        if dataset_path is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)
            (self.output_path / "normalized").mkdir(exist_ok=True)
            (self.output_path / "analysis").mkdir(exist_ok=True)
            (self.output_path / "thumbnails").mkdir(exist_ok=True)
            (self.output_path / "templates").mkdir(exist_ok=True)
        
        # Storage for statistics
        self.slide_stats: Dict[str, StainStats] = {}
        self.reference_stats: Optional[StainStats] = None
        self.normalization_template: Optional[NormalizationTemplate] = None
        
    def get_slide_list(self) -> List[Path]:
        """Get list of all .svs files in dataset"""
        if not self.dataset_path:
            raise ValueError("Dataset path not set")
        svs_files = list(self.dataset_path.glob("**/*.svs"))
        logger.info(f"Found {len(svs_files)} SVS files in dataset")
        return svs_files
    
    def extract_thumbnail(self, slide_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Extract thumbnail from SVS slide"""
        try:
            slide = openslide.OpenSlide(str(slide_path))
            thumbnail = slide.get_thumbnail(self.thumbnail_size)
            slide.close()
            return np.array(thumbnail)
        except Exception as e:
            logger.error(f"Error extracting thumbnail from {slide_path}: {e}")
            return None
    
    def analyze_tissue_mask(self, image: np.ndarray) -> np.ndarray:
        """Create tissue mask excluding background"""
        # Convert to grayscale for tissue detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Otsu thresholding for tissue detection
        _, tissue_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
        
        return tissue_mask.astype(bool)
    
    def extract_stain_stats(self, image: np.ndarray) -> StainStats:
        """Extract comprehensive stain statistics from image"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Create tissue mask
        tissue_mask = self.analyze_tissue_mask(image)
        
        # Alternative mask based on brightness and saturation
        brightness_mask = hsv[:, :, 2] < self.params.brightness_threshold
        saturation_mask = hsv[:, :, 1] > self.params.saturation_threshold
        combined_mask = tissue_mask & brightness_mask & saturation_mask
        
        if np.sum(combined_mask) == 0:
            logger.warning("No tissue detected in image")
            return StainStats(0, 0, 0, 0, 0, 0, 0)
        
        # Extract HSV values for tissue regions
        tissue_hue = hsv[combined_mask, 0].astype(np.float32)
        tissue_sat = hsv[combined_mask, 1].astype(np.float32)
        tissue_val = hsv[combined_mask, 2].astype(np.float32)
        
        # Calculate statistics
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
        """Determine if slide needs normalization based on statistics"""
        # Check if hue is significantly different from target
        hue_diff = abs(stats.hue_mean - self.params.target_hue)
        if hue_diff > 90:  # Handle circular nature of hue
            hue_diff = 180 - hue_diff
        
        # Check various criteria
        needs_hue_adjustment = hue_diff > self.params.hue_tolerance
        needs_sat_adjustment = abs(stats.sat_mean - self.params.target_sat) > self.params.sat_tolerance
        low_saturation = stats.sat_mean < 60  # Very washed out
        
        return needs_hue_adjustment or needs_sat_adjustment or low_saturation
    
    def normalize_slide(self, image: np.ndarray, target_stats: Optional[StainStats] = None) -> np.ndarray:
        """Normalize single slide to target appearance"""
        if target_stats is None:
            target_stats = self.reference_stats or StainStats(
                self.params.target_hue, self.params.hue_tolerance,
                self.params.target_sat, self.params.sat_tolerance,
                self.params.target_val, 30, 0.3
            )
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Create tissue mask
        tissue_mask = self.analyze_tissue_mask(image)
        brightness_mask = hsv[:, :, 2] < self.params.brightness_threshold
        saturation_mask = hsv[:, :, 1] > self.params.saturation_threshold
        combined_mask = tissue_mask & brightness_mask & saturation_mask
        
        if np.sum(combined_mask) == 0:
            logger.warning("No tissue detected for normalization")
            return image
        
        # Get current statistics
        current_stats = self.extract_stain_stats(image)
        
        # Normalize each channel
        tissue_pixels = hsv[combined_mask]
        
        # Hue normalization (handle circular nature)
        hue_diff = target_stats.hue_mean - current_stats.hue_mean
        if abs(hue_diff) > 90:
            if hue_diff > 0:
                hue_diff -= 180
            else:
                hue_diff += 180
        
        tissue_pixels[:, 0] = (tissue_pixels[:, 0] + hue_diff) % 180
        
        # Saturation normalization
        if current_stats.sat_std > 0:
            tissue_pixels[:, 1] = (tissue_pixels[:, 1] - current_stats.sat_mean) / current_stats.sat_std
            tissue_pixels[:, 1] = tissue_pixels[:, 1] * target_stats.sat_std + target_stats.sat_mean
            tissue_pixels[:, 1] = np.clip(tissue_pixels[:, 1], 0, 255)
        
        # Value normalization
        if current_stats.val_std > 0:
            tissue_pixels[:, 2] = (tissue_pixels[:, 2] - current_stats.val_mean) / current_stats.val_std
            tissue_pixels[:, 2] = tissue_pixels[:, 2] * target_stats.val_std + target_stats.val_mean
            tissue_pixels[:, 2] = np.clip(tissue_pixels[:, 2], 0, 255)
        
        # Apply normalized values back
        hsv[combined_mask] = tissue_pixels
        
        # Convert back to RGB
        normalized = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return normalized
    
    def establish_reference(self, sample_size: int = 50) -> StainStats:
        """Establish reference statistics from high-quality slides"""
        if not self.dataset_path:
            raise ValueError("Dataset path required for establishing reference")
            
        slide_files = self.get_slide_list()
        
        if len(slide_files) < sample_size:
            sample_size = len(slide_files)
        
        # Sample slides for reference establishment
        np.random.seed(42)  # For reproducibility
        sample_slides = np.random.choice(slide_files, sample_size, replace=False)
        
        all_stats = []
        logger.info(f"Analyzing {sample_size} slides to establish reference...")
        
        for slide_path in tqdm(sample_slides):
            thumbnail = self.extract_thumbnail(slide_path)
            if thumbnail is not None:
                stats = self.extract_stain_stats(thumbnail)
                if stats.tissue_ratio > 0.1:  # Only consider slides with sufficient tissue
                    all_stats.append(stats)
        
        if not all_stats:
            logger.error("No suitable reference slides found")
            return StainStats(160, 15, 120, 40, 180, 30, 0.3)
        
        # Calculate median statistics (more robust than mean)
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
        """Export normalization template for reuse"""
        if not self.reference_stats:
            raise ValueError("No reference statistics available. Run establish_reference() first.")
        
        if template_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            template_name = f"normalization_template_{timestamp}"
        
        # Create template
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
        
        # Save as JSON
        json_path = self.output_path / "templates" / f"{template_name}.json"
        template_dict = asdict(template)
        template_dict = convert_to_serializable(template_dict)
        
        with open(json_path, 'w') as f:
            json.dump(template_dict, f, indent=2)
        
        # Save as pickle (preserves exact numpy types)
        pkl_path = self.output_path / "templates" / f"{template_name}.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(template, f)
        
        logger.info(f"Template exported to: {json_path} and {pkl_path}")
        return str(json_path)
    
    def load_template(self, template_path: str) -> NormalizationTemplate:
        """Load normalization template from file"""
        template_path = Path(template_path)
        
        if template_path.suffix == '.json':
            with open(template_path, 'r') as f:
                template_dict = json.load(f)
            
            # Reconstruct template from dictionary
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
        
        # Apply loaded template
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
        """
        Normalize an image directly (supports multiple input formats)
        
        Args:
            image: Input image as numpy array, PIL Image, or file path
            output_path: Where to save normalized image (optional)
            return_image: Whether to return the normalized image array
            save_comparison: Whether to save side-by-side comparison
            comparison_title: Title for comparison image
            
        Returns:
            Normalized image array if return_image=True, otherwise None
        """
        if not self.reference_stats:
            raise ValueError("No normalization template loaded. Use load_template() first.")
        
        # Convert input to numpy array
        if isinstance(image, (str, Path)):
            # Load from file path
            image_path = Path(image)
            if image_path.suffix.lower() in ['.svs', '.tif', '.tiff']:
                # Handle slide files
                original_array = self.extract_thumbnail(image_path)
                if original_array is None:
                    logger.error(f"Failed to load slide from {image_path}")
                    return None
            else:
                # Handle regular image files
                pil_image = Image.open(image_path)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                original_array = np.array(pil_image)
        elif isinstance(image, Image.Image):
            # PIL Image
            if image.mode != 'RGB':
                image = image.convert('RGB')
            original_array = np.array(image)
        elif isinstance(image, np.ndarray):
            # Numpy array
            original_array = image.copy()
            # Ensure RGB format
            if len(original_array.shape) != 3 or original_array.shape[2] != 3:
                raise ValueError("Image must be RGB format with shape (H, W, 3)")
        else:
            raise TypeError("Image must be numpy array, PIL Image, or file path")
        
        # Ensure proper data type
        if original_array.dtype != np.uint8:
            if original_array.max() <= 1.0:
                original_array = (original_array * 255).astype(np.uint8)
            else:
                original_array = original_array.astype(np.uint8)
        
        logger.info(f"Normalizing image with shape: {original_array.shape}")
        
        # Get original statistics
        original_stats = self.extract_stain_stats(original_array)
        
        # Check if normalization is needed
        needs_norm = self.needs_normalization(original_stats)
        logger.info(f"Normalization needed: {needs_norm}")
        
        # Normalize the image
        normalized = self.normalize_slide(original_array, self.reference_stats)
        
        # Get normalized statistics for comparison
        normalized_stats = self.extract_stain_stats(normalized)
        
        # Log statistics comparison
        logger.info(f"Original  - Hue: {original_stats.hue_mean:.1f}, "
                   f"Sat: {original_stats.sat_mean:.1f}, Val: {original_stats.val_mean:.1f}")
        logger.info(f"Normalized - Hue: {normalized_stats.hue_mean:.1f}, "
                   f"Sat: {normalized_stats.sat_mean:.1f}, Val: {normalized_stats.val_mean:.1f}")
        logger.info(f"Target    - Hue: {self.reference_stats.hue_mean:.1f}, "
                   f"Sat: {self.reference_stats.sat_mean:.1f}, Val: {self.reference_stats.val_mean:.1f}")
        
        # Save normalized image
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(normalized).save(output_path)
            logger.info(f"Normalized image saved to: {output_path}")
        
        # Save comparison image
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
        """
        Normalize multiple images in batch
        
        Args:
            images: List of images (various formats supported)
            output_dir: Directory to save normalized images
            return_images: Whether to return normalized image arrays
            save_comparisons: Whether to save comparison images
            
        Returns:
            List of normalized image arrays if return_images=True
        """
        if not self.reference_stats:
            raise ValueError("No normalization template loaded. Use load_template() first.")
        
        results = []
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Batch normalizing {len(images)} images...")
        
        for i, image in enumerate(tqdm(images)):
            try:
                # Generate output path if directory provided
                output_path = None
                if output_dir:
                    if isinstance(image, (str, Path)):
                        filename = Path(image).stem
                    else:
                        filename = f"image_{i:04d}"
                    output_path = output_dir / f"{filename}_normalized.png"
                
                # Normalize image
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
        """
        Normalize a single slide using loaded template
        
        Args:
            slide_path: Path to the slide (.svs file)
            output_path: Where to save normalized image (optional)
            return_image: Whether to return the normalized image array
            save_comparison: Whether to save side-by-side comparison
            
        Returns:
            Normalized image array if return_image=True, otherwise None
        """
        if not self.reference_stats:
            raise ValueError("No normalization template loaded. Use load_template() first.")
        
        slide_path = Path(slide_path)
        slide_id = slide_path.stem
        
        logger.info(f"Normalizing single slide: {slide_id}")
        
        # Extract thumbnail
        thumbnail = self.extract_thumbnail(slide_path)
        if thumbnail is None:
            logger.error(f"Failed to extract thumbnail from {slide_path}")
            return None
        
        # Get original statistics
        original_stats = self.extract_stain_stats(thumbnail)
        
        # Check if normalization is needed
        needs_norm = self.needs_normalization(original_stats)
        logger.info(f"Normalization needed: {needs_norm}")
        
        # Normalize the slide
        normalized = self.normalize_slide(thumbnail, self.reference_stats)
        
        # Get normalized statistics for comparison
        normalized_stats = self.extract_stain_stats(normalized)
        
        # Log statistics comparison
        logger.info(f"Original  - Hue: {original_stats.hue_mean:.1f}, "
                   f"Sat: {original_stats.sat_mean:.1f}, Val: {original_stats.val_mean:.1f}")
        logger.info(f"Normalized - Hue: {normalized_stats.hue_mean:.1f}, "
                   f"Sat: {normalized_stats.sat_mean:.1f}, Val: {normalized_stats.val_mean:.1f}")
        logger.info(f"Target    - Hue: {self.reference_stats.hue_mean:.1f}, "
                   f"Sat: {self.reference_stats.sat_mean:.1f}, Val: {self.reference_stats.val_mean:.1f}")
        
        # Save normalized image
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(normalized).save(output_path)
            logger.info(f"Normalized image saved to: {output_path}")
        
        # Save comparison image
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
        """Save side-by-side comparison of original and normalized images"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axes[0].imshow(original)
        axes[0].set_title(f'Original\nHue: {orig_stats.hue_mean:.1f}, '
                         f'Sat: {orig_stats.sat_mean:.1f}, Val: {orig_stats.val_mean:.1f}')
        axes[0].axis('off')
        
        # Normalized image
        axes[1].imshow(normalized)
        axes[1].set_title(f'Normalized\nHue: {norm_stats.hue_mean:.1f}, '
                         f'Sat: {norm_stats.sat_mean:.1f}, Val: {norm_stats.val_mean:.1f}')
        axes[1].axis('off')
        
        # Add main title
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison image saved to: {save_path}")
    
    def batch_normalize_slides(self, slide_paths: List[Union[str, Path]], 
                             output_dir: str, 
                             save_comparisons: bool = False) -> Dict[str, bool]:
        """
        Normalize multiple slides using loaded template
        
        Args:
            slide_paths: List of paths to slides
            output_dir: Directory to save normalized images
            save_comparisons: Whether to save comparison images
            
        Returns:
            Dictionary with slide_id: success_status
        """
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
    
    # ... (rest of the original methods remain the same)
    def analyze_dataset(self, save_results: bool = True) -> Dict:
        """Analyze entire dataset and save statistics"""
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
                # Extract thumbnail
                thumbnail = self.extract_thumbnail(slide_path)
                if thumbnail is None:
                    results['failed_slides'] += 1
                    continue
                
                # Save thumbnail
                thumbnail_path = self.output_path / "thumbnails" / f"{slide_id}.png"
                Image.fromarray(thumbnail).save(thumbnail_path)
                
                # Extract statistics
                stats = self.extract_stain_stats(thumbnail)
                self.slide_stats[slide_id] = stats
                
                # Check if normalization needed
                needs_norm = self.needs_normalization(stats)
                
                # Convert to JSON-serializable format
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
        
        # Save results with proper serialization
        if save_results:
            results_path = self.output_path / "analysis" / "dataset_analysis.json"
            
            # Convert all numpy types to Python native types
            serializable_results = convert_to_serializable(results)
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            # Save slide statistics as pickle (preserves numpy types)
            stats_path = self.output_path / "analysis" / "slide_statistics.pkl"
            with open(stats_path, 'wb') as f:
                pickle.dump(self.slide_stats, f)
        
        logger.info(f"Analysis complete: {results['processed_slides']} processed, "
                   f"{results['slides_needing_normalization']} need normalization")
        
        return results

# Usage Examples
def create_template_example():
    """Example: Create and export normalization template"""
    # Initialize normalizer with training dataset
    normalizer = TCGAStainNormalizer(
        dataset_path="/path/to/training/dataset",
        output_path="template_output"
    )
    
    # Analyze dataset and establish reference
    normalizer.analyze_dataset()
    normalizer.establish_reference(sample_size=100)
    
    # Export template
    template_path = normalizer.export_template("my_tcga_template")
    print(f"Template saved to: {template_path}")

def image_inference_examples():
    """Examples: Different ways to normalize images"""
    # Initialize normalizer for inference
    normalizer = TCGAStainNormalizer(output_path="inference_output")
    
    # Load pre-trained template
    normalizer.load_template("template_output/templates/my_tcga_template.json")
    
    # Example 1: Normalize from file path (regular image)
    normalized_img1 = normalizer.normalize_image(
        image="path/to/image.png",
        output_path="output/normalized_image1.png",
        save_comparison=True
    )
    
    # Example 2: Normalize from PIL Image
    from PIL import Image
    pil_img = Image.open("path/to/image.jpg")
    normalized_img2 = normalizer.normalize_image(
        image=pil_img,
        output_path="output/normalized_image2.png",
        save_comparison=True
    )
    
    # Example 3: Normalize from numpy array
    import numpy as np
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    normalized_img3 = normalizer.normalize_image(
        image=img_array,
        output_path="output/normalized_image3.png",
        save_comparison=True
    )
    
    # Example 4: Normalize slide file (.svs)
    normalized_slide = normalizer.normalize_image(
        image="path/to/slide.svs",
        output_path="output/normalized_slide.png",
        save_comparison=True,
        comparison_title="Slide Normalization"
    )
    
    # Example 5: Batch normalize multiple images
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
    """Example: Load template and normalize slides (original function)"""
    # Initialize normalizer for inference (no dataset path needed)
    normalizer = TCGAStainNormalizer(output_path="inference_output")
    
    # Load pre-trained template
    normalizer.load_template("template_output/templates/my_tcga_template.json")
    
    # Normalize single slide (dedicated slide function)
    slide_path = "/path/to/single/slide.svs"
    normalized_image = normalizer.normalize_single_slide(
        slide_path=slide_path,
        output_path="output/normalized_slide.png",
        return_image=True,
        save_comparison=True
    )
    
    # Batch normalize multiple slides
    slide_paths = ["/path/to/slide1.svs", "/path/to/slide2.svs"]
    results = normalizer.batch_normalize_slides(
        slide_paths=slide_paths,
        output_dir="output/batch_normalized",
        save_comparisons=True
    )
    
    print(f"Batch results: {results}")

def main():
    """Main execution function - Template Creation Mode"""
    # Configuration for template creation
    DATASET_PATH = "/home/nas2_fast/Data/Pathology_project/TCGA-BRCA/wsi"
    OUTPUT_PATH = "output"
    
    # Initialize normalizer
    normalizer = TCGAStainNormalizer(
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_PATH,
        thumbnail_size=(1024, 1024)
    )
    
    # Step 1: Analyze dataset
    logger.info("Step 1: Analyzing dataset...")
    analysis_results = normalizer.analyze_dataset()
    
    # Step 2: Establish reference
    logger.info("Step 2: Establishing reference...")
    reference_stats = normalizer.establish_reference(sample_size=100)
    
    # Step 3: Export template
    logger.info("Step 3: Exporting normalization template...")
    template_path = normalizer.export_template("tcga_brca_template")
    
    logger.info(f"Template creation complete! Template saved to: {template_path}")
    
    # Example inference usage
    logger.info("\n--- Example Inference Usage ---")
    logger.info("To use this template for inference:")
    logger.info("1. Load template: normalizer.load_template('path/to/template.json')")
    logger.info("2. Normalize single slide: normalizer.normalize_single_slide('path/to/slide.svs')")

if __name__ == "__main__":
    main()