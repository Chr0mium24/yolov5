# background_mixup.py
"""
Background Mixup Data Augmentation for RoboMaster

This module implements background mixup and COCO insertion strategies to increase
background diversity and reduce background bias in RoboMaster armor detection.
"""

import cv2
import numpy as np
import random
import os
import pickle
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from ..config import get_robomaster_config
from .context_detector import create_context_detector


class BackgroundMixupAugmenter:
    """
    Data augmenter that performs context mixup and COCO background insertion
    to increase background diversity and prevent overfitting to specific contexts.
    """

    def __init__(self,
                 context_mixup_prob: float = 0.2,
                 coco_insert_prob: float = 0.2,
                 coco_img_path: Optional[str] = 'data/robomaster/cocoimg'):
        """
        Initialize the background mixup augmenter.

        Args:
            context_mixup_prob: Probability of mixing different contexts
            coco_insert_prob: Probability of inserting COCO background images
            coco_img_path: Path to COCO background image library
        """
        self.context_mixup_prob = context_mixup_prob
        self.coco_insert_prob = coco_insert_prob
        self.coco_img_path = coco_img_path

        # Load RoboMaster configuration
        self.config = get_robomaster_config()
        self.class_map = self.config.class_map
        self.sentry_classes = self.config.sentry_classes
        self.vehicle_classes = self.config.vehicle_classes

        # Initialize unified context detector
        self.context_detector = create_context_detector(detection_strategy='hybrid')

        # COCO image cache
        self._coco_image_cache = None
        self._coco_cache_loaded = False


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

    def load_coco_images(self, target_size: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """
        Load COCO background images from the library with caching.
        All images are standardized to 256x256 for efficiency.

        Args:
            target_size: Ignored - all images are cached at 256x256

        Returns:
            List of COCO background images (256x256)
        """
        # Use fixed 256x256 size for all COCO images
        cache_size = 256
        cache_key = "coco_cache_256"

        # Return cached images if available
        if (self._coco_cache_loaded and
            hasattr(self, cache_key) and
            getattr(self, cache_key) is not None):
            return getattr(self, cache_key)

        coco_images = []

        if not self.coco_img_path or not os.path.exists(self.coco_img_path):
            setattr(self, cache_key, coco_images)
            self._coco_cache_loaded = True
            return coco_images

        coco_path = Path(self.coco_img_path)

        # Try to load from cache file
        cache_file = coco_path / 'coco_cache_256.pkl'
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    setattr(self, cache_key, cached_data)
                    self._coco_cache_loaded = True
                    print(f"Loaded COCO images from cache: {cache_file}")
                    return cached_data
            except Exception as e:
                print(f"Failed to load COCO cache, loading fresh: {e}")

        # Load images directly from the folder
        print(f"Loading COCO background images (256x256)...")
        if coco_path.exists():
            image_files = list(coco_path.glob('*.jpg')) + list(coco_path.glob('*.png'))
            for img_file in image_files:
                img = cv2.imread(str(img_file))
                if img is not None:
                    # Resize to fixed 256x256
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

        setattr(self, cache_key, coco_images)
        self._coco_cache_loaded = True
        return coco_images

    def _find_available_regions(self, image_size: Tuple[int, int],
                               labels: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Find available regions in the image that don't overlap with armor plates.

        Args:
            image_size: (height, width) of the image
            labels: YOLO format labels [class, x_center, y_center, width, height]

        Returns:
            List of available regions as (x, y, width, height) in pixel coordinates
        """
        h, w = image_size

        # Convert labels to pixel coordinates
        occupied_boxes = []
        for label in labels:
            x_center, y_center, width, height = label[1:5]
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            occupied_boxes.append((x1, y1, x2, y2))

        # Grid-based approach to find available regions
        min_size = min(w, h) // 10  # Minimum region size
        max_size = min(w, h) // 3   # Maximum region size

        available_regions = []

        # Try different region sizes
        for size in range(min_size, max_size + 1, min_size // 2):
            # Generate all possible positions for this size
            positions = []
            for x in range(0, w - size + 1, size // 2):
                for y in range(0, h - size + 1, size // 2):
                    positions.append((x, y))

            # Shuffle positions to avoid left-top bias
            random.shuffle(positions)

            # Check each position for overlap
            for x, y in positions:
                region = (x, y, x + size, y + size)

                # Check if this region overlaps with any armor plate
                overlap = False
                for occupied in occupied_boxes:
                    if self._boxes_overlap(region, occupied):
                        overlap = True
                        break

                if not overlap:
                    available_regions.append((x, y, size, size))

        # Shuffle all regions to ensure random distribution
        random.shuffle(available_regions)

        return available_regions

    def _boxes_overlap(self, box1: Tuple[int, int, int, int],
                      box2: Tuple[int, int, int, int]) -> bool:
        """
        Check if two boxes overlap.

        Args:
            box1, box2: Boxes in format (x1, y1, x2, y2)

        Returns:
            True if boxes overlap
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        return not (x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1)

    def coco_insert_augmentation(self, image: np.ndarray,
                                labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Insert COCO background images to increase background diversity.
        Uses dynamic region calculation to ensure successful insertion.

        Args:
            image: Input image with armor plates
            labels: Input labels

        Returns:
            Image with COCO background insertions and original labels
        """
        if not self.coco_img_path:
            return image, labels

        h, w = image.shape[:2]
        result_image = image.copy()

        # Load COCO images (all are 256x256)
        coco_images = self.load_coco_images()

        if not coco_images:
            return image, labels

        # Find all available regions for COCO insertion
        available_regions = self._find_available_regions((h, w), labels)

        if not available_regions:
            return image, labels  # No available space

        # Determine number of insertions based on available regions
        max_insertions = min(len(available_regions), 3)
        num_insertions = random.randint(1, max_insertions)

        # Track inserted regions to avoid overlap between COCO images
        inserted_regions = []

        for i in range(num_insertions):
            # Find available region that doesn't overlap with already inserted ones
            suitable_region = None
            for region in available_regions:
                x, y, region_w, region_h = region
                new_region = (x, y, x + region_w, y + region_h)

                # Check overlap with inserted regions
                overlap_with_inserted = False
                for inserted in inserted_regions:
                    if self._boxes_overlap(new_region, inserted):
                        overlap_with_inserted = True
                        break

                if not overlap_with_inserted:
                    suitable_region = region
                    break

            if suitable_region is None:
                continue  # No suitable region found

            x, y, insert_w, insert_h = suitable_region

            # Select random COCO image and resize to fit the available region
            coco_img = random.choice(coco_images)

            # Resize COCO image to fit the available region
            # Add some randomness to the size (80-100% of available region)
            scale_factor = random.uniform(0.8, 1.0)
            final_w = int(insert_w * scale_factor)
            final_h = int(insert_h * scale_factor)

            # Adjust position to center the smaller COCO image in the region
            offset_x = (insert_w - final_w) // 2
            offset_y = (insert_h - final_h) // 2
            final_x = x + offset_x
            final_y = y + offset_y

            # Resize COCO image
            resized_coco = cv2.resize(coco_img, (final_w, final_h))

            # Insert COCO image with blending
            alpha = random.uniform(0.8, 0.95)
            roi = result_image[final_y:final_y+final_h, final_x:final_x+final_w]
            blended = cv2.addWeighted(roi, 1-alpha, resized_coco, alpha, 0)
            result_image[final_y:final_y+final_h, final_x:final_x+final_w] = blended

            # Record inserted region to prevent overlap
            inserted_regions.append((final_x, final_y, final_x + final_w, final_y + final_h))

            # Remove used region from available regions
            available_regions.remove(suitable_region)

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
                context_pool: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply background mixup augmentation to image and labels.

        Args:
            image: Input image
            labels: Input labels
            context_pool: Pool of images from different contexts for mixup

        Returns:
            Augmented image and labels
        """
        aug_image, aug_labels = image.copy(), labels.copy()

        # Apply COCO insertion
        if random.random() < self.coco_insert_prob:
            aug_image, aug_labels = self.coco_insert_augmentation(aug_image, aug_labels)

        # Apply context mixup if pool is available
        if (context_pool and len(context_pool) > 0 and
            random.random() < self.context_mixup_prob):

            other_image, other_labels = random.choice(context_pool)
            aug_image, aug_labels = self.context_mixup(
                aug_image, aug_labels, other_image, other_labels)

        return aug_image, aug_labels