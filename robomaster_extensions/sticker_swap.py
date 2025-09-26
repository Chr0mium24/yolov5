"""
Sticker Swapping Data Augmentation for RoboMaster

This module implements sticker swapping strategies to mitigate background bias
by swapping armor plate stickers between different contexts (sentry stickers on vehicles,
vehicle stickers on sentry posts) to break the learned bias.
"""

import cv2
import numpy as np
import random
from typing import List, Tuple, Dict, Optional


class StickerSwapAugmenter:
    """
    Data augmenter that swaps armor plate stickers between different contexts
    to prevent background bias in sentry detection.
    """

    def __init__(self,
                 sticker_swap_prob: float = 0.3,
                 preserve_geometry: bool = True):
        """
        Initialize the sticker swap augmenter.

        Args:
            sticker_swap_prob: Probability of swapping stickers between contexts
            preserve_geometry: Whether to preserve bounding box geometry
        """
        self.sticker_swap_prob = sticker_swap_prob
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

    def augment(self, image: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply sticker swap augmentation to image and labels.

        Args:
            image: Input image
            labels: Input labels

        Returns:
            Augmented image and labels
        """
        if random.random() < self.sticker_swap_prob:
            return self.swap_stickers_between_contexts(image, labels)
        else:
            return image, labels