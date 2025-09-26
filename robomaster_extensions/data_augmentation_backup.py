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

# Import the specialized augmenters
from .sticker_swap import StickerSwapAugmenter
from .background_mixup import BackgroundMixupAugmenter
from .context_augment import ContextAugmenter


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
                 coco_img_path: Optional[str] = '/Users/cr/yolov5/data/robomaster/cocoimg'):
        """
        Initialize the background bias augmenter.

        Args:
            sticker_swap_prob: Probability of swapping stickers between contexts
            context_mixup_prob: Probability of mixing different contexts
            brightness_adjust_prob: Probability of applying brightness adjustment
            coco_insert_prob: Probability of inserting COCO background images
            preserve_geometry: Whether to preserve bounding box geometry
            coco_img_path: Path to COCO background image library
        """
        self.sticker_swap_prob = sticker_swap_prob
        self.context_mixup_prob = context_mixup_prob
        self.brightness_adjust_prob = brightness_adjust_prob
        self.coco_insert_prob = coco_insert_prob
        self.preserve_geometry = preserve_geometry
        self.coco_img_path = coco_img_path

        # Define class mappings for RoboMaster
        self.class_map = {
            'sentry': 0,
            'hero': 1,
            'engineer': 2,
            'standard_1': 3,
            'standard_2': 4,
            'standard_3': 5,
            'standard_4': 6,
            'standard_5': 7
        }

        # Context definitions
        self.sentry_classes = [0]  # sentry
        self.vehicle_classes = [1, 2, 3, 4, 5, 6, 7]  # all vehicles

        # COCO image cache
        self._coco_image_cache = None
        self._coco_cache_loaded = False

    def detect_context(self, labels: np.ndarray, image_size: Tuple[int, int]) -> str:
        """
        Detect if image contains sentry post context based on labels.

        Args:
            labels: YOLO format labels [class, x_center, y_center, width, height]
            image_size: (height, width) of the image

        Returns:
            Context type: 'sentry', 'vehicle', or 'mixed'
        """
        if len(labels) == 0:
            return 'unknown'

        classes = labels[:, 0].astype(int)
        has_sentry = any(cls in self.sentry_classes for cls in classes)
        has_vehicle = any(cls in self.vehicle_classes for cls in classes)

        if has_sentry and has_vehicle:
            return 'mixed'
        elif has_sentry:
            return 'sentry'
        elif has_vehicle:
            return 'vehicle'
        else:
            return 'unknown'

    def extract_armor_patch(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Extract armor plate patch from image using bounding box.

        Args:
            image: Input image (H, W, C)
            bbox: Normalized bounding box [x_center, y_center, width, height]

        Returns:
            Extracted armor patch
        """
        h, w = image.shape[:2]

        # Convert normalized coordinates to pixel coordinates
        x_center, y_center, width, height = bbox
        x_center *= w
        y_center *= h
        width *= w
        height *= h

        # Calculate bounding box corners
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        return image[y1:y2, x1:x2].copy()

    def place_armor_patch(self, image: np.ndarray, patch: np.ndarray,
                         bbox: np.ndarray) -> np.ndarray:
        """
        Place armor patch back into image at specified location.

        Args:
            image: Target image
            patch: Armor patch to place
            bbox: Target bounding box [x_center, y_center, width, height]

        Returns:
            Image with placed patch
        """
        h, w = image.shape[:2]

        # Convert normalized coordinates to pixel coordinates
        x_center, y_center, width, height = bbox
        x_center *= w
        y_center *= h
        width *= w
        height *= h

        # Calculate target region
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Ensure coordinates are within bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        target_h = y2 - y1
        target_w = x2 - x1

        if target_h > 0 and target_w > 0 and patch.size > 0:
            # Resize patch to fit target region
            resized_patch = cv2.resize(patch, (target_w, target_h))
            image[y1:y2, x1:x2] = resized_patch

        return image

    def swap_stickers_between_contexts(self, image: np.ndarray,
                                     labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Swap armor stickers between different contexts to break bias.

        Args:
            image: Input image
            labels: YOLO format labels [class, x_center, y_center, width, height]

        Returns:
            Augmented image and modified labels
        """
        if len(labels) == 0:
            return image, labels

        context = self.detect_context(labels, image.shape[:2])

        # Only apply swapping to mixed contexts or with certain probability
        if context not in ['sentry', 'vehicle'] and random.random() > self.sticker_swap_prob:
            return image, labels

        # Create copies for modification
        aug_image = image.copy()
        aug_labels = labels.copy()

        # Find sentry and vehicle armor plates
        sentry_indices = []
        vehicle_indices = []

        for i, label in enumerate(labels):
            cls = int(label[0])
            if cls in self.sentry_classes:
                sentry_indices.append(i)
            elif cls in self.vehicle_classes:
                vehicle_indices.append(i)

        # Perform sticker swapping
        if len(sentry_indices) > 0 and len(vehicle_indices) > 0:
            # Randomly select armor plates to swap
            sentry_idx = random.choice(sentry_indices)
            vehicle_idx = random.choice(vehicle_indices)

            # Extract patches
            sentry_patch = self.extract_armor_patch(image, labels[sentry_idx][1:5])
            vehicle_patch = self.extract_armor_patch(image, labels[vehicle_idx][1:5])

            # Place swapped patches
            aug_image = self.place_armor_patch(aug_image, vehicle_patch,
            labels[sentry_idx][1:5])
            aug_image = self.place_armor_patch(aug_image, sentry_patch,
            labels[vehicle_idx][1:5])

            # Swap class labels
            aug_labels[sentry_idx][0] = labels[vehicle_idx][0]
            aug_labels[vehicle_idx][0] = labels[sentry_idx][0]

        return aug_image, aug_labels

    def context_mixup(self, image1: np.ndarray, labels1: np.ndarray,
                     image2: np.ndarray, labels2: np.ndarray,
                     alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mix two images from different contexts to create synthetic training data.

        Args:
            image1, labels1: First image and labels
            image2, labels2: Second image and labels
            alpha: Mixing ratio

        Returns:
            Mixed image and combined labels
        """
        # Ensure images have same size
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        if h1 != h2 or w1 != w2:
            image2 = cv2.resize(image2, (w1, h1))

        # Mix images
        mixed_image = (alpha * image1 + (1 - alpha) * image2).astype(np.uint8)

        # Combine labels (keep all from both images)
        if len(labels1) > 0 and len(labels2) > 0:
            mixed_labels = np.vstack([labels1, labels2])
        elif len(labels1) > 0:
            mixed_labels = labels1
        elif len(labels2) > 0:
            mixed_labels = labels2
        else:
            mixed_labels = np.array([])

        return mixed_image, mixed_labels

    def brightness_adjust_augmentation(self, image: np.ndarray,
                                     labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply adaptive brightness adjustment based on HSV color space.

        Args:
            image: Input image
            labels: Input labels (unchanged)

        Returns:
            Brightness-adjusted image and original labels
        """
        # Convert to HSV for better brightness control
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Calculate current brightness statistics
        v_channel = hsv[:, :, 2]
        mean_brightness = np.mean(v_channel)
        std_brightness = np.std(v_channel)

        # Adaptive brightness adjustment based on current brightness
        if mean_brightness < 100:  # Dark image
            brightness_factor = random.uniform(1.2, 1.8)
        elif mean_brightness > 180:  # Bright image
            brightness_factor = random.uniform(0.6, 0.9)
        else:  # Normal brightness
            brightness_factor = random.uniform(0.8, 1.3)

        # Apply brightness adjustment
        v_channel = v_channel.astype(np.float32)
        v_channel = v_channel * brightness_factor
        v_channel = np.clip(v_channel, 0, 255).astype(np.uint8)

        # Reconstruct HSV and convert back to BGR
        hsv[:, :, 2] = v_channel
        adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return adjusted_image, labels

    def load_coco_images(self) -> List[np.ndarray]:
        """
        Load COCO background images from the library with caching.

        Returns:
            List of COCO background images
        """
        # Return cached images if available
        if self._coco_cache_loaded and self._coco_image_cache is not None:
            return self._coco_image_cache

        coco_images = []

        if not self.coco_img_path or not os.path.exists(self.coco_img_path):
            self._coco_image_cache = coco_images
            self._coco_cache_loaded = True
            return coco_images

        coco_path = Path(self.coco_img_path)

        # Try to load from cache file
        cache_file = coco_path / 'coco_embeddings.pkl'
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self._coco_image_cache = cached_data
                    self._coco_cache_loaded = True
                    print(f"Loaded COCO images from cache: {cache_file}")
                    return cached_data
            except Exception as e:
                print(f"Failed to load COCO cache, loading fresh: {e}")

        # Load images directly from the folder
        print("Loading COCO background images...")
        if coco_path.exists():
            image_files = list(coco_path.glob('*.jpg')) + list(coco_path.glob('*.png'))
            for img_file in image_files:
                img = cv2.imread(str(img_file))
                if img is not None:
                    # Resize to manageable size for faster processing
                    img_resized = cv2.resize(img, (256, 256))
                    coco_images.append(img_resized)
            print(f"Loaded {len(coco_images)} images from {coco_path}")

        # Save to cache for future use
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(coco_images, f)
            print(f"Saved COCO images to cache: {cache_file}")
        except Exception as e:
            print(f"Failed to save COCO cache: {e}")

        self._coco_image_cache = coco_images
        self._coco_cache_loaded = True
        return coco_images

    def coco_insert_augmentation(self, image: np.ndarray,
                                labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Insert COCO background images to increase background diversity.

        Args:
            image: Input image with armor plates
            labels: Input labels

        Returns:
            Image with COCO background insertions and original labels
        """
        if not self.coco_img_path:
            return image, labels

        # Load COCO images (could be cached)
        coco_images = self.load_coco_images()

        if not coco_images:
            return image, labels

        h, w = image.shape[:2]
        result_image = image.copy()

        # Number of COCO insertions (1-3 random insertions)
        num_insertions = random.randint(1, 3)

        for _ in range(num_insertions):
            # Select random COCO image
            coco_img = random.choice(coco_images)
            coco_h, coco_w = coco_img.shape[:2]

            # Random insertion size (10-30% of image size)
            insert_scale = random.uniform(0.1, 0.3)
            insert_w = int(w * insert_scale)
            insert_h = int(h * insert_scale)

            # Resize COCO image
            resized_coco = cv2.resize(coco_img, (insert_w, insert_h))

            # Random position ensuring no overlap with armor plates
            max_attempts = 10
            for attempt in range(max_attempts):
                x = random.randint(0, max(1, w - insert_w))
                y = random.randint(0, max(1, h - insert_h))

                # Check if insertion area overlaps with any armor plate
                insert_box = np.array([
                    (x + insert_w/2) / w,  # x_center normalized
                    (y + insert_h/2) / h,  # y_center normalized
                    insert_w / w,          # width normalized
                    insert_h / h           # height normalized
                ])

                # Check overlap with existing armor plates
                overlap = False
                for label in labels:
                    armor_box = label[1:5]  # [x_center, y_center, width, height]

                    # Calculate IoU
                    iou = self._calculate_iou(insert_box, armor_box)
                    if iou > 0.1:  # Avoid significant overlap
                        overlap = True
                        break

                if not overlap:
                    # Insert COCO image with blending
                    alpha = random.uniform(0.3, 0.7)
                    roi = result_image[y:y+insert_h, x:x+insert_w]
                    blended = cv2.addWeighted(roi, 1-alpha, resized_coco, alpha, 0)
                    result_image[y:y+insert_h, x:x+insert_w] = blended
                    break

        return result_image, labels

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) of two bounding boxes.

        Args:
            box1, box2: Normalized bounding boxes [x_center, y_center, width, height]

        Returns:
            IoU value
        """
        # Convert center format to corner format
        def center_to_corner(box):
            x_center, y_center, width, height = box
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            return x1, y1, x2, y2

        x1_1, y1_1, x2_1, y2_1 = center_to_corner(box1)
        x1_2, y1_2, x2_2, y2_2 = center_to_corner(box2)

        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def augment(self, image: np.ndarray, labels: np.ndarray,
                context_pool: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
                augmentation_type: str = 'mixed'
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply background bias augmentation to image and labels.

        Args:
            image: Input image
            labels: Input labels
            context_pool: Pool of images from different contexts for mixup
            augmentation_type: Type of augmentation ('sticker_swap', 'brightness_adjust',
                             'coco_insert', 'mixed')

        Returns:
            Augmented image and labels
        """
        aug_image, aug_labels = image.copy(), labels.copy()

        if augmentation_type == 'sticker_swap':
            # Only apply sticker swapping
            if random.random() < self.sticker_swap_prob:
                aug_image, aug_labels = self.swap_stickers_between_contexts(
                    aug_image, aug_labels)
        elif augmentation_type == 'brightness_adjust':
            # Only apply brightness adjustment
            if random.random() < self.brightness_adjust_prob:
                aug_image, aug_labels = self.brightness_adjust_augmentation(
                    aug_image, aug_labels)
        elif augmentation_type == 'coco_insert':
            # Only apply COCO insertion
            if random.random() < self.coco_insert_prob:
                aug_image, aug_labels = self.coco_insert_augmentation(
                    aug_image, aug_labels)
        else:  # 'mixed' - apply multiple augmentations
            # Apply sticker swapping
            if random.random() < self.sticker_swap_prob:
                aug_image, aug_labels = self.swap_stickers_between_contexts(
                    aug_image, aug_labels)

            # Apply brightness adjustment
            if random.random() < self.brightness_adjust_prob:
                aug_image, aug_labels = self.brightness_adjust_augmentation(
                    aug_image, aug_labels)

            # Apply COCO insertion
            if random.random() < self.coco_insert_prob:
                aug_image, aug_labels = self.coco_insert_augmentation(
                    aug_image, aug_labels)

            # Apply context mixup if pool is available
            if (context_pool and len(context_pool) > 0 and
                random.random() < self.context_mixup_prob):

                other_image, other_labels = random.choice(context_pool)
                aug_image, aug_labels = self.context_mixup(
                    aug_image, aug_labels, other_image, other_labels)

        return aug_image, aug_labels

    def create_balanced_dataset(self, dataset_path: str, output_path: str,
                               generate_all_types: bool = True,
                               augmentation_factor: int = 1):
        """
        Create a balanced dataset with all three augmentation strategies.

        Args:
            dataset_path: Path to original dataset
            output_path: Path to save augmented dataset
            generate_all_types: Whether to generate all three augmentation types
            augmentation_factor: Number of augmented versions per type per original image
        """
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Define augmentation strategies
        aug_strategies = ['sticker_swap', 'brightness_adjust', 'coco_insert'] if generate_all_types else ['mixed']

        # Create organized output directory structure
        if generate_all_types:
            for split in ['train', 'val']:
                for aug_type in aug_strategies:
                    (output_path / 'images' / f'{split}_augmented' / aug_type).mkdir(parents=True, exist_ok=True)
                    (output_path / 'labels' / f'{split}_augmented' / aug_type).mkdir(parents=True, exist_ok=True)
        else:
            # Standard structure for mixed augmentation
            (output_path / 'images').mkdir(exist_ok=True)
            (output_path / 'labels').mkdir(exist_ok=True)

        # Process train and val splits
        for split in ['train', 'val']:
            split_path = dataset_path / 'images' / split
            if not split_path.exists():
                print(f"Split {split} not found, skipping...")
                continue

            # Process all images in split
            image_files = list(split_path.glob('*.jpg')) + list(split_path.glob('*.png'))

            context_pools = {'sentry': [], 'vehicle': [], 'mixed': []}

            # First pass: categorize images by context
            print(f"Categorizing {len(image_files)} images in {split} split...")
            for img_file in image_files:
                label_file = dataset_path / 'labels' / split / f"{img_file.stem}.txt"

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
            for i, img_file in enumerate(image_files):
                label_file = dataset_path / 'labels' / split / f"{img_file.stem}.txt"

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

                # Save original to train or val directory
                if generate_all_types:
                    original_img_dir = output_path / 'images' / split
                    original_lbl_dir = output_path / 'labels' / split
                else:
                    original_img_dir = output_path / 'images'
                    original_lbl_dir = output_path / 'labels'

                original_img_dir.mkdir(parents=True, exist_ok=True)
                original_lbl_dir.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(str(original_img_dir / img_file.name), image)
                np.savetxt(str(original_lbl_dir / f"{img_file.stem}.txt"),
                          labels, fmt='%d %.6f %.6f %.6f %.6f')

                # Generate augmented versions
                for aug_type in aug_strategies:
                    for aug_idx in range(augmentation_factor):
                        # Select context pool for mixup (different from current context)
                        available_contexts = [k for k, v in context_pools.items()
                                            if k != context and len(v) > 0]
                        context_pool = (context_pools[random.choice(available_contexts)]
                                       if available_contexts else None)

                        # Apply specific augmentation type
                        aug_image, aug_labels = self.augment(image, labels, context_pool, aug_type)

                        # Determine output paths
                        if generate_all_types:
                            aug_img_dir = output_path / 'images' / f'{split}_augmented' / aug_type
                            aug_lbl_dir = output_path / 'labels' / f'{split}_augmented' / aug_type
                            aug_suffix = f"_{aug_idx}"
                        else:
                            aug_img_dir = output_path / 'images'
                            aug_lbl_dir = output_path / 'labels'
                            aug_suffix = f"_aug_{aug_idx}"

                        # Save augmented data
                        aug_name = f"{img_file.stem}{aug_suffix}{img_file.suffix}"
                        cv2.imwrite(str(aug_img_dir / aug_name), aug_image)
                        np.savetxt(str(aug_lbl_dir / f"{img_file.stem}{aug_suffix}.txt"),
                                  aug_labels, fmt='%d %.6f %.6f %.6f %.6f')

                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(image_files)} images in {split}")

        # Save configuration
        config = {
            'augmentation_strategies': aug_strategies,
            'augmentation_factor': augmentation_factor,
            'sticker_swap_prob': self.sticker_swap_prob,
            'brightness_adjust_prob': self.brightness_adjust_prob,
            'coco_insert_prob': self.coco_insert_prob,
            'context_mixup_prob': self.context_mixup_prob,
            'coco_img_path': self.coco_img_path
        }

        config_file = output_path / 'processed' / 'augmentation_config.yaml'
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False)

        # Save brightness statistics
        brightness_stats = {
            'mean_brightness_threshold_low': 100,
            'mean_brightness_threshold_high': 180,
            'dark_image_factor_range': [1.2, 1.8],
            'bright_image_factor_range': [0.6, 0.9],
            'normal_image_factor_range': [0.8, 1.3]
        }

        stats_file = output_path / 'processed' / 'brightness_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(brightness_stats, f, indent=2)

        print(f"Dataset augmentation complete.")
        print(f"Generated augmentation strategies: {aug_strategies}")
        print(f"Configuration saved to: {config_file}")
        print(f"Brightness stats saved to: {stats_file}")


def test_background_bias_augmenter():
    """Test function for the BackgroundBiasAugmenter with all augmentation types."""

    # Create test data
    image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    labels = np.array([
        [0, 0.3, 0.3, 0.1, 0.1],  # sentry
        [3, 0.7, 0.7, 0.1, 0.1],  # standard_1
    ])

    # Initialize augmenter
    augmenter = BackgroundBiasAugmenter(
        sticker_swap_prob=1.0,
        brightness_adjust_prob=1.0,
        coco_insert_prob=1.0
    )

    print("Testing different augmentation types:")
    print("Original labels:", labels[:, 0])
    print("Context detection:", augmenter.detect_context(labels, image.shape[:2]))

    # Test sticker swap
    print("\n--- Testing sticker_swap ---")
    aug_image_1, aug_labels_1 = augmenter.augment(image, labels,
                                                  augmentation_type='sticker_swap')
    print("Sticker swap result labels:", aug_labels_1[:, 0])

    # Test brightness adjustment
    print("\n--- Testing brightness_adjust ---")
    aug_image_2, aug_labels_2 = augmenter.brightness_adjust_augmentation(image, labels)
    print("Brightness adjust result labels:", aug_labels_2[:, 0])

    # Test mixed augmentation
    print("\n--- Testing mixed augmentation ---")
    aug_image_3, aug_labels_3 = augmenter.augment(image, labels,
                                                  augmentation_type='mixed')
    print("Mixed augmentation result labels:", aug_labels_3[:, 0])

    return aug_image_1, aug_labels_1


if __name__ == "__main__":
    test_background_bias_augmenter()