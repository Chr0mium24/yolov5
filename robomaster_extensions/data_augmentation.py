"""
Unified Data Augmentation Entry Point for RoboMaster

This module provides a unified interface for all three data augmentation strategies:
1. Sticker Swapping - Swap armor stickers between contexts to break background bias
2. Background Mixup - Mix contexts and insert COCO backgrounds for diversity
3. Context Augmentation - Apply adaptive brightness and color adjustments

This entry file manages all augmentation modules and provides a unified API.
"""

import cv2
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json
import os
from tqdm import tqdm
from .config import get_robomaster_config

# Import the specialized augmenters
from .data_augment.sticker_swap import StickerSwapAugmenter
from .data_augment.background_mixup import BackgroundMixupAugmenter
from .data_augment.context_augment import ContextAugmenter


class UnifiedDataAugmenter:
    """
    Unified data augmenter that combines all three augmentation strategies:
    1. Sticker swapping to prevent background bias
    2. Background mixup for context diversity
    3. Context augmentation for robustness
    """

    def __init__(self,
                 sticker_swap_prob: float = 0.3,
                 context_mixup_prob: float = 0.2,
                 brightness_adjust_prob: float = 0.4,
                 coco_insert_prob: float = 0.2,
                 preserve_geometry: bool = True,
                 coco_img_path: Optional[str] = 'data/robomaster/cocoimg'):
        """
        Initialize the unified data augmenter.

        Args:
            sticker_swap_prob: Probability of swapping stickers between contexts
            context_mixup_prob: Probability of mixing different contexts
            brightness_adjust_prob: Probability of applying brightness adjustment
            coco_insert_prob: Probability of inserting COCO background images
            preserve_geometry: Whether to preserve bounding box geometry
            coco_img_path: Path to COCO background image library
        """
        # Initialize specialized augmenters
        self.sticker_swap_augmenter = StickerSwapAugmenter(
            sticker_swap_prob=sticker_swap_prob,
            preserve_geometry=preserve_geometry
        )

        self.background_mixup_augmenter = BackgroundMixupAugmenter(
            context_mixup_prob=context_mixup_prob,
            coco_insert_prob=coco_insert_prob,
            coco_img_path=coco_img_path
        )

        self.context_augmenter = ContextAugmenter(
            brightness_adjust_prob=brightness_adjust_prob
        )

        # Store parameters for compatibility
        self.sticker_swap_prob = sticker_swap_prob
        self.context_mixup_prob = context_mixup_prob
        self.brightness_adjust_prob = brightness_adjust_prob
        self.coco_insert_prob = coco_insert_prob
        self.preserve_geometry = preserve_geometry
        self.coco_img_path = coco_img_path

        # Load RoboMaster configuration
        self.config = get_robomaster_config()
        self.class_map = self.config.class_map
        self.sentry_classes = self.config.sentry_classes
        self.vehicle_classes = self.config.vehicle_classes

    def detect_context(self, labels: np.ndarray, image_size: Tuple[int, int]) -> str:
        """
        Detect if image contains sentry post context based on labels.
        Delegates to the sticker swap augmenter for consistency.

        Args:
            labels: YOLO format labels [class, x_center, y_center, width, height]
            image_size: (height, width) of the image

        Returns:
            Context type: 'sentry', 'vehicle', or 'mixed'
        """
        return self.sticker_swap_augmenter.detect_context(labels, image_size)

    def extract_armor_patch(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Extract armor plate patch from image using bounding box.
        Delegates to the sticker swap augmenter.

        Args:
            image: Input image (H, W, C)
            bbox: Normalized bounding box [x_center, y_center, width, height]

        Returns:
            Extracted armor patch
        """
        return self.sticker_swap_augmenter.extract_armor_patch(image, bbox)

    def place_armor_patch(self, image: np.ndarray, patch: np.ndarray,
                         bbox: np.ndarray) -> np.ndarray:
        """
        Place armor patch back into image at specified location.
        Delegates to the sticker swap augmenter.

        Args:
            image: Target image
            patch: Armor patch to place
            bbox: Target bounding box [x_center, y_center, width, height]

        Returns:
            Image with placed patch
        """
        return self.sticker_swap_augmenter.place_armor_patch(image, patch, bbox)

    def swap_stickers_between_contexts(self, image: np.ndarray,
                                     labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Swap armor stickers between different contexts to break bias.
        Delegates to the sticker swap augmenter.

        Args:
            image: Input image
            labels: YOLO format labels [class, x_center, y_center, width, height]

        Returns:
            Augmented image and modified labels
        """
        return self.sticker_swap_augmenter.swap_stickers_between_contexts(image, labels)

    def context_mixup(self, image1: np.ndarray, labels1: np.ndarray,
                     image2: np.ndarray, labels2: np.ndarray,
                     alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mix two images from different contexts to create synthetic training data.
        Delegates to the background mixup augmenter.

        Args:
            image1, labels1: First image and labels
            image2, labels2: Second image and labels
            alpha: Mixing ratio

        Returns:
            Mixed image and combined labels
        """
        return self.background_mixup_augmenter.context_mixup(image1, labels1, image2, labels2, alpha)

    def brightness_adjust_augmentation(self, image: np.ndarray,
                                     labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply enhanced adaptive brightness adjustment with multiple methods.
        Delegates to the context augmenter.

        Args:
            image: Input image
            labels: Input labels (unchanged)

        Returns:
            Brightness-adjusted image and original labels
        """
        return self.context_augmenter.brightness_adjust_augmentation(image, labels)

    def contrast_adjustment(self, image: np.ndarray,
                           labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply enhanced adaptive contrast adjustment with multiple methods.
        Delegates to the context augmenter.

        Args:
            image: Input image
            labels: Input labels (unchanged)

        Returns:
            Contrast-adjusted image and original labels
        """
        return self.context_augmenter.contrast_adjustment(image, labels)

    def gamma_correction(self, image: np.ndarray,
                        labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply enhanced gamma correction with adaptive range selection.
        Delegates to the context augmenter.

        Args:
            image: Input image
            labels: Input labels (unchanged)

        Returns:
            Gamma-corrected image and original labels
        """
        return self.context_augmenter.gamma_correction(image, labels)

    def histogram_equalization(self, image: np.ndarray,
                              labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply histogram equalization for contrast enhancement.
        Delegates to the context augmenter.

        Args:
            image: Input image
            labels: Input labels (unchanged)

        Returns:
            Histogram equalized image and original labels
        """
        return self.context_augmenter.histogram_equalization(image, labels)

    def clahe_enhancement(self, image: np.ndarray,
                         labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        Delegates to the context augmenter.

        Args:
            image: Input image
            labels: Input labels (unchanged)

        Returns:
            CLAHE enhanced image and original labels
        """
        return self.context_augmenter.clahe_enhancement(image, labels)

    def adaptive_brightness_contrast(self, image: np.ndarray,
                                   labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply combined adaptive brightness and contrast adjustment.
        Delegates to the context augmenter.

        Args:
            image: Input image
            labels: Input labels (unchanged)

        Returns:
            Enhanced image and original labels
        """
        return self.context_augmenter.adaptive_brightness_contrast(image, labels)

    def load_coco_images(self) -> List[np.ndarray]:
        """
        Load COCO background images from the library with caching.
        Delegates to the background mixup augmenter.

        Returns:
            List of COCO background images
        """
        return self.background_mixup_augmenter.load_coco_images()

    def coco_insert_augmentation(self, image: np.ndarray,
                                labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Insert COCO background images to increase background diversity.
        Delegates to the background mixup augmenter.

        Args:
            image: Input image with armor plates
            labels: Input labels

        Returns:
            Image with COCO background insertions and original labels
        """
        return self.background_mixup_augmenter.coco_insert_augmentation(image, labels)

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) of two bounding boxes.
        Delegates to the background mixup augmenter for consistency.

        Args:
            box1, box2: Normalized bounding boxes [x_center, y_center, width, height]

        Returns:
            IoU value
        """
        return self.background_mixup_augmenter._calculate_iou(box1, box2)

    def augment(self, image: np.ndarray, labels: np.ndarray,
                context_pool: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
                augmentation_type: str = 'mixed'
                ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Apply unified data augmentation to image and labels.

        Args:
            image: Input image
            labels: Input labels
            context_pool: Pool of images from different contexts for mixup
            augmentation_type: Type of augmentation ('sticker_swap', 'brightness_adjust',
                             'contrast_adjust', 'gamma_correct', 'histogram_equalize',
                             'clahe_enhance', 'adaptive_enhance', 'coco_insert',
                             'context_mixup', 'mixed')

        Returns:
            Augmented image, labels, and applied augmentation type name
        """
        aug_image, aug_labels = image.copy(), labels.copy()
        applied_augmentation = 'original'

        if augmentation_type == 'sticker_swap':
            # Only apply sticker swapping
            if random.random() < self.sticker_swap_prob:
                aug_image, aug_labels = self.sticker_swap_augmenter.augment(aug_image, aug_labels)
                applied_augmentation = 'sticker_swap'
            else:
                applied_augmentation = 'original'
        elif augmentation_type == 'brightness_adjust':
            # Only apply enhanced brightness adjustment
            if random.random() < self.brightness_adjust_prob:
                aug_image, aug_labels = self.context_augmenter.augment(
                    aug_image, aug_labels, 'brightness')
                applied_augmentation = 'brightness_adjust'
            else:
                applied_augmentation = 'original'
        elif augmentation_type == 'contrast_adjust':
            # Only apply enhanced contrast adjustment
            aug_image, aug_labels = self.context_augmenter.augment(
                aug_image, aug_labels, 'contrast')
            applied_augmentation = 'contrast_adjust'
        elif augmentation_type == 'gamma_correct':
            # Only apply enhanced gamma correction
            aug_image, aug_labels = self.context_augmenter.augment(
                aug_image, aug_labels, 'gamma')
            applied_augmentation = 'gamma_correct'
        elif augmentation_type == 'histogram_equalize':
            # Only apply histogram equalization
            aug_image, aug_labels = self.context_augmenter.augment(
                aug_image, aug_labels, 'histogram')
            applied_augmentation = 'histogram_equalize'
        elif augmentation_type == 'clahe_enhance':
            # Only apply CLAHE enhancement
            aug_image, aug_labels = self.context_augmenter.augment(
                aug_image, aug_labels, 'clahe')
            applied_augmentation = 'clahe_enhance'
        elif augmentation_type == 'adaptive_enhance':
            # Only apply adaptive brightness/contrast enhancement
            aug_image, aug_labels = self.context_augmenter.augment(
                aug_image, aug_labels, 'adaptive')
            applied_augmentation = 'adaptive_enhance'
        elif augmentation_type == 'coco_insert':
            # Only apply COCO insertion
            if random.random() < self.coco_insert_prob:
                aug_image, aug_labels = self.background_mixup_augmenter.augment(
                    aug_image, aug_labels, context_pool=None)
                applied_augmentation = 'coco_insert'
            else:
                applied_augmentation = 'original'
        elif augmentation_type == 'context_mixup':
            # Only apply context mixup
            if random.random() < self.context_mixup_prob and context_pool:
                aug_image, aug_labels = self.background_mixup_augmenter.augment(
                    aug_image, aug_labels, context_pool)
                applied_augmentation = 'context_mixup'
            else:
                applied_augmentation = 'original'
        else:  # 'mixed' - apply multiple augmentations
            # Apply sticker swapping
            aug_image, aug_labels = self.sticker_swap_augmenter.augment(aug_image, aug_labels)

            # Apply context augmentation (brightness, color, contrast, etc.)
            aug_image, aug_labels = self.context_augmenter.augment(aug_image, aug_labels, 'mixed')

            # Apply background mixup (COCO insertion + context mixup)
            aug_image, aug_labels = self.background_mixup_augmenter.augment(
                aug_image, aug_labels, context_pool)

            applied_augmentation = 'mixed'

        return aug_image, aug_labels, applied_augmentation

    def create_balanced_dataset(self, dataset_path: str, output_path: str,
                               generate_all_types: bool = True,
                               augmentation_factor: int = 1):
        """
        Create a balanced dataset with all six augmentation strategies.

        This method creates a unified output structure compatible with the new
        data format where all augmented images are placed in a single directory
        with descriptive prefixes.

        Args:
            dataset_path: Path to original dataset (expects train/ and val/ subdirs)
            output_path: Path to save augmented dataset
            generate_all_types: Whether to generate all six augmentation types
            augmentation_factor: Number of augmented versions per type per original image
        """
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Define six augmentation strategies matching the documentation
        aug_strategies = ['sticker_swap', 'brightness_adjust', 'contrast_adjust',
                         'clahe_enhance', 'adaptive_enhance', 'coco_insert'] if generate_all_types else ['mixed']

        # Create unified output directory structure matching robomaster.yaml expectations
        # Structure: output/images/train, output/images/train_augmented, output/images/val
        for split in ['train', 'val']:
            # Create directories for original data (copy structure)
            (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

            # Create unified augmented data directories
            (output_path / 'images' / f'{split}_augmented').mkdir(parents=True, exist_ok=True)
            (output_path / 'labels' / f'{split}_augmented').mkdir(parents=True, exist_ok=True)

        # Process train and val splits - handle both input formats
        for split in ['train', 'val']:
            # Try two possible input structures
            split_img_path = None
            split_lbl_path = None

            # Format 1: dataset_path/train/images (standard YOLOv5 format)
            if (dataset_path / split / 'images').exists():
                split_img_path = dataset_path / split / 'images'
                split_lbl_path = dataset_path / split / 'labels'
            # Format 2: dataset_path/images/train (alternative format)
            elif (dataset_path / 'images' / split).exists():
                split_img_path = dataset_path / 'images' / split
                split_lbl_path = dataset_path / 'labels' / split

            if split_img_path is None or not split_img_path.exists():
                print(f"Split {split} not found in either format, skipping...")
                continue

            # Process all images in split
            image_files = list(split_img_path.glob('*.jpg')) + list(split_img_path.glob('*.png'))

            context_pools = {'sentry': [], 'vehicle': [], 'mixed': []}

            # First pass: categorize images by context
            print(f"Categorizing {len(image_files)} images in {split} split...")
            for img_file in tqdm(image_files, desc=f"Categorizing {split} images", unit="img"):
                label_file = split_lbl_path / f"{img_file.stem}.txt"

                if not label_file.exists():
                    continue

                # Load image and labels
                image = cv2.imread(str(img_file))
                if image is None:
                    continue

                try:
                    labels = np.loadtxt(str(label_file)).reshape(-1, 5)
                except:
                    continue

                context = self.detect_context(labels, image.shape[:2])
                if context in context_pools:
                    context_pools[context].append((image, labels))

            print(f"Context distribution for {split}: {[(k, len(v)) for k, v in context_pools.items()]}")

            # Second pass: generate augmented data
            for img_file in tqdm(image_files, desc=f"Processing {split} images", unit="img"):
                label_file = split_lbl_path / f"{img_file.stem}.txt"

                if not label_file.exists():
                    continue

                # Load original data
                image = cv2.imread(str(img_file))
                if image is None:
                    continue

                try:
                    labels = np.loadtxt(str(label_file)).reshape(-1, 5)
                except:
                    continue

                context = self.detect_context(labels, image.shape[:2])

                # Save original to unified images directory structure
                original_img_dir = output_path / 'images' / split
                original_lbl_dir = output_path / 'labels' / split

                original_img_dir.mkdir(parents=True, exist_ok=True)
                original_lbl_dir.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(str(original_img_dir / img_file.name), image)
                np.savetxt(str(original_lbl_dir / f"{img_file.stem}.txt"),
                        labels, fmt='%d %.6f %.6f %.6f %.6f')

                # Generate augmented versions
                progress_desc = f"Augmenting {img_file.name}"
                for aug_type in aug_strategies:
                    for aug_idx in range(augmentation_factor):
                        # Select context pool for mixup (different from current context)
                        available_contexts = [k for k, v in context_pools.items()
                                            if k != context and len(v) > 0]
                        context_pool = (context_pools[random.choice(available_contexts)]
                                    if available_contexts else None)

                        # Apply specific augmentation type
                        aug_image, aug_labels, actual_aug_type = self.augment(image, labels, context_pool, aug_type)

                        # Check if augmentation was actually applied
                        aug_applied = actual_aug_type != 'original'

                        # Determine output paths using unified structure
                        aug_img_dir = output_path / 'images' / f'{split}_augmented'
                        aug_lbl_dir = output_path / 'labels' / f'{split}_augmented'

                        # Create filename with augmentation type prefix and index
                        if aug_applied:
                            # Use actual augmentation type in prefix: clahe_enhance_0001_filename.jpg
                            aug_prefix = f"{actual_aug_type}_{aug_idx:04d}"
                        else:
                            # For cases where augmentation was not applied
                            aug_prefix = f"original_{aug_idx:04d}"

                        # Save augmented data with descriptive prefix filename
                        aug_name = f"{aug_prefix}_{img_file.stem}{img_file.suffix}"
                        cv2.imwrite(str(aug_img_dir / aug_name), aug_image)
                        np.savetxt(str(aug_lbl_dir / f"{aug_prefix}_{img_file.stem}.txt"),
                                    aug_labels, fmt='%d %.6f %.6f %.6f %.6f')

                        # Log augmentation results for debugging
                        # if (i + 1) % 100 == 0:
                        #     print(f"  Applied {actual_aug_type} to {img_file.name} -> {aug_name}")
                        #     print(f"  Augmentation {'successful' if aug_applied else 'skipped'}")

                # Progress is now shown by tqdm, remove periodic print

        # Save configuration for tracking
        config = {
            'dataset_structure': 'unified',  # Mark as new unified structure
            'augmentation_strategies': aug_strategies,
            'augmentation_factor': augmentation_factor,
            'sticker_swap_prob': self.sticker_swap_prob,
            'brightness_adjust_prob': self.brightness_adjust_prob,
            'coco_insert_prob': self.coco_insert_prob,
            'context_mixup_prob': self.context_mixup_prob,
            'coco_img_path': self.coco_img_path,
            'output_structure': {
                'original_data': 'images/train, images/val',
                'augmented_data': 'images/train_augmented, images/val_augmented',
                'naming_convention': 'prefix_index_filename.ext'
            }
        }

        config_file = output_path / 'augmentation_config.yaml'
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False)

        # Save brightness statistics in root output directory
        brightness_stats = {
            'mean_brightness_threshold_low': 100,
            'mean_brightness_threshold_high': 180,
            'dark_image_factor_range': [1.2, 1.8],
            'bright_image_factor_range': [0.6, 0.9],
            'normal_image_factor_range': [0.8, 1.3]
        }

        stats_file = output_path / 'brightness_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(brightness_stats, f, indent=2)

        print(f"Dataset augmentation complete.")
        print(f"Generated augmentation strategies: {aug_strategies}")
        print(f"Output structure: unified format compatible with robomaster.yaml")
        print(f"Configuration saved to: {config_file}")
        print(f"Brightness stats saved to: {stats_file}")
        print(f"")
        print(f"Directory structure created:")
        print(f"  {output_path}/images/train/          # Original training images")
        print(f"  {output_path}/images/train_augmented/ # All augmented training images")
        print(f"  {output_path}/images/val/            # Original validation images")
        print(f"  {output_path}/labels/train/          # Original training labels")
        print(f"  {output_path}/labels/train_augmented/ # All augmented training labels")
        print(f"  {output_path}/labels/val/            # Original validation labels")
        print(f"")
        print(f"Update your robomaster.yaml to point to this output directory:")
        print(f"  path: {output_path}")
        print(f"  train: images/train_augmented")
        print(f"  val: images/val")


def test_unified_data_augmenter():
    """Test function for the UnifiedDataAugmenter with all augmentation types."""

    # Create test data
    image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    labels = np.array([
        [0, 0.3, 0.3, 0.1, 0.1],  # sentry
        [3, 0.7, 0.7, 0.1, 0.1],  # standard_1
    ])

    # Initialize augmenter
    augmenter = UnifiedDataAugmenter(
        sticker_swap_prob=1.0,
        brightness_adjust_prob=1.0,
        coco_insert_prob=1.0
    )

    print("Testing different augmentation types:")
    print("Original labels:", labels[:, 0])
    print("Context detection:", augmenter.detect_context(labels, image.shape[:2]))

    # Test sticker swap
    print("\n--- Testing sticker_swap ---")
    aug_image_1, aug_labels_1, aug_type_1 = augmenter.augment(image, labels,
                                                  augmentation_type='sticker_swap')
    print("Sticker swap result labels:", aug_labels_1[:, 0])
    print("Applied augmentation type:", aug_type_1)

    # Test enhanced brightness adjustment
    print("\n--- Testing brightness_adjust ---")
    aug_image_2, aug_labels_2 = augmenter.brightness_adjust_augmentation(image, labels)
    print("Brightness adjust result labels:", aug_labels_2[:, 0])

    # Test enhanced contrast adjustment
    print("\n--- Testing contrast_adjust ---")
    aug_image_3, aug_labels_3 = augmenter.contrast_adjustment(image, labels)
    print("Contrast adjust result labels:", aug_labels_3[:, 0])

    # Test CLAHE enhancement
    print("\n--- Testing clahe_enhance ---")
    aug_image_4, aug_labels_4 = augmenter.clahe_enhancement(image, labels)
    print("CLAHE enhance result labels:", aug_labels_4[:, 0])

    # Test adaptive brightness/contrast
    print("\n--- Testing adaptive_enhance ---")
    aug_image_5, aug_labels_5 = augmenter.adaptive_brightness_contrast(image, labels)
    print("Adaptive enhance result labels:", aug_labels_5[:, 0])

    # Test mixed augmentation
    print("\n--- Testing mixed augmentation ---")
    aug_image_6, aug_labels_6, aug_type_6 = augmenter.augment(image, labels,
                                                  augmentation_type='mixed')
    print("Mixed augmentation result labels:", aug_labels_6[:, 0])
    print("Applied augmentation type:", aug_type_6)

    return aug_image_1, aug_labels_1


# Backward compatibility alias
BackgroundBiasAugmenter = UnifiedDataAugmenter

def test_background_bias_augmenter():
    """Legacy test function for backward compatibility."""
    return test_unified_data_augmenter()


if __name__ == "__main__":
    test_unified_data_augmenter()