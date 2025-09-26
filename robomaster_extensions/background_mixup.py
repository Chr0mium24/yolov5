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
from .config import get_robomaster_config


class BackgroundMixupAugmenter:
    """
    Data augmenter that performs context mixup and COCO background insertion
    to increase background diversity and prevent overfitting to specific contexts.
    """

    def __init__(self,
                 context_mixup_prob: float = 0.2,
                 coco_insert_prob: float = 0.2,
                 coco_img_path: Optional[str] = '/Users/cr/yolov5/data/robomaster/cocoimg'):
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