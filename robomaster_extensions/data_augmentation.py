"""
Background Bias Data Augmentation for RoboMaster

This module implements data augmentation strategies to mitigate background bias
where armor plates near sentry posts are misclassified as sentry class regardless
of their actual class.

Solution: Swap stickers between contexts (sentry stickers on vehicles,
vehicle stickers on sentry posts) to break the learned bias.
"""

import cv2
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
import albumentations as A
from pathlib import Path
import json


class BackgroundBiasAugmenter:
    """
    Data augmenter that swaps armor plate stickers between different contexts
    to prevent background bias in sentry detection.
    """

    def __init__(self,
                 sticker_swap_prob: float = 0.3,
                 context_mixup_prob: float = 0.2,
                 preserve_geometry: bool = True):
        """
        Initialize the background bias augmenter.

        Args:
            sticker_swap_prob: Probability of swapping stickers between contexts
            context_mixup_prob: Probability of mixing different contexts
            preserve_geometry: Whether to preserve bounding box geometry
        """
        self.sticker_swap_prob = sticker_swap_prob
        self.context_mixup_prob = context_mixup_prob
        self.preserve_geometry = preserve_geometry

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

    def augment(self, image: np.ndarray, labels: np.ndarray,
                context_pool: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply background bias augmentation to image and labels.

        Args:
            image: Input image
            labels: Input labels
            context_pool: Pool of images from different contexts for mixup

        Returns:
            Augmented image and labels
        """
        aug_image, aug_labels = image.copy(), labels.copy()

        # Apply sticker swapping
        if random.random() < self.sticker_swap_prob:
            aug_image, aug_labels = self.swap_stickers_between_contexts(
                aug_image, aug_labels)

        # Apply context mixup if pool is available
        if (context_pool and len(context_pool) > 0 and
            random.random() < self.context_mixup_prob):

            other_image, other_labels = random.choice(context_pool)
            aug_image, aug_labels = self.context_mixup(
                aug_image, aug_labels, other_image, other_labels)

        return aug_image, aug_labels

    def create_balanced_dataset(self, dataset_path: str, output_path: str,
                               augmentation_factor: int = 2):
        """
        Create a balanced dataset with background bias mitigation.

        Args:
            dataset_path: Path to original dataset
            output_path: Path to save augmented dataset
            augmentation_factor: Number of augmented versions per original image
        """
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create output directories
        (output_path / 'images').mkdir(exist_ok=True)
        (output_path / 'labels').mkdir(exist_ok=True)

        # Process all images in dataset
        image_files = list((dataset_path / 'images').glob('*.jpg')) + \
                      list((dataset_path / 'images').glob('*.png'))

        context_pools = {'sentry': [], 'vehicle': [], 'mixed': []}

        # First pass: categorize images by context
        for img_file in image_files:
            label_file = dataset_path / 'labels' / f"{img_file.stem}.txt"

            if not label_file.exists():
                continue

            # Load image and labels
            image = cv2.imread(str(img_file))
            if image is None:
                continue

            labels = np.loadtxt(str(label_file)).reshape(-1, 5)
            context = self.detect_context(labels, image.shape[:2])

            if context in context_pools:
                context_pools[context].append((image, labels))

        # Second pass: generate augmented data
        for i, img_file in enumerate(image_files):
            label_file = dataset_path / 'labels' / f"{img_file.stem}.txt"

            if not label_file.exists():
                continue

            # Load original data
            image = cv2.imread(str(img_file))
            if image is None:
                continue

            labels = np.loadtxt(str(label_file)).reshape(-1, 5)
            context = self.detect_context(labels, image.shape[:2])

            # Save original
            cv2.imwrite(str(output_path / 'images' / img_file.name), image)
            np.savetxt(str(output_path / 'labels' / f"{img_file.stem}.txt"),
                      labels, fmt='%d %.6f %.6f %.6f %.6f')

            # Generate augmented versions
            for aug_idx in range(augmentation_factor):
                # Select context pool for mixup (different from current context)
                available_contexts = [k for k, v in context_pools.items()
                                    if k != context and len(v) > 0]
                context_pool = (context_pools[random.choice(available_contexts)]
                               if available_contexts else None)

                # Apply augmentation
                aug_image, aug_labels = self.augment(image, labels, context_pool)

                # Save augmented data
                aug_name = f"{img_file.stem}_aug_{aug_idx}{img_file.suffix}"
                cv2.imwrite(str(output_path / 'images' / aug_name), aug_image)
                np.savetxt(str(output_path / 'labels' / f"{img_file.stem}_aug_{aug_idx}.txt"),
                          aug_labels, fmt='%d %.6f %.6f %.6f %.6f')

        print(f"Dataset augmentation complete. Processed {len(image_files)} images.")
        print(f"Context distribution: {[(k, len(v)) for k, v in context_pools.items()]}")


def test_background_bias_augmenter():
    """Test function for the BackgroundBiasAugmenter."""

    # Create test data
    image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    labels = np.array([
        [0, 0.3, 0.3, 0.1, 0.1],  # sentry
        [3, 0.7, 0.7, 0.1, 0.1],  # standard_1
    ])

    # Initialize augmenter
    augmenter = BackgroundBiasAugmenter(sticker_swap_prob=1.0)

    # Test augmentation
    aug_image, aug_labels = augmenter.augment(image, labels)

    print("Original labels:", labels[:, 0])
    print("Augmented labels:", aug_labels[:, 0])
    print("Context detection:", augmenter.detect_context(labels, image.shape[:2]))

    return aug_image, aug_labels


if __name__ == "__main__":
    test_background_bias_augmenter()