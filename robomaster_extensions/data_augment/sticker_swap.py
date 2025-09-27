"""
Sticker Swapping Data Augmentation for RoboMaster

This module implements intelligent sticker swapping strategies to mitigate background bias
by swapping armor plate stickers between different contexts (sentry stickers on vehicles,
vehicle stickers on sentry posts) to break the learned bias.

Key Features:
- Identifies sentry stations (RQS=12, BQS=13) vs vehicle armor plates (0-11)
- Aspect ratio compatibility check to prevent excessive deformation
- Smart swapping algorithm with fallback mechanisms
- Compatible with 14-class RoboMaster system (B1-B5, BHero, R1-R5, RHero, RQS, BQS)
"""

import cv2
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from ..config import get_robomaster_config


class StickerSwapAugmenter:
    """
    Data augmenter that swaps armor plate stickers between different contexts
    to prevent background bias in sentry detection.
    """

    def __init__(self,
                 sticker_swap_prob: float = 0.3,
                 preserve_geometry: bool = True,
                 max_aspect_ratio_diff: float = 0.3):
        """
        Initialize the sticker swap augmenter.

        Args:
            sticker_swap_prob: Probability of swapping stickers between contexts
            preserve_geometry: Whether to preserve bounding box geometry
            max_aspect_ratio_diff: Maximum allowed aspect ratio difference for swapping
        """
        self.sticker_swap_prob = sticker_swap_prob
        self.preserve_geometry = preserve_geometry
        self.max_aspect_ratio_diff = max_aspect_ratio_diff

        # Load RoboMaster configuration
        self.config = get_robomaster_config()
        self.class_map = self.config.class_map
        self.sentry_classes = self.config.sentry_classes  # [12, 13] for RQS, BQS
        self.vehicle_classes = self.config.vehicle_classes  # All others

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

    def calculate_aspect_ratio(self, bbox: np.ndarray) -> float:
        """
        Calculate aspect ratio (width/height) of a bounding box.

        Args:
            bbox: Normalized bounding box [x_center, y_center, width, height]

        Returns:
            Aspect ratio (width/height)
        """
        _, _, width, height = bbox
        return width / height if height > 0 else 1.0

    def can_swap_based_on_aspect_ratio(self, bbox1: np.ndarray, bbox2: np.ndarray) -> bool:
        """
        Check if two bounding boxes can be swapped based on aspect ratio similarity.

        Args:
            bbox1: First bounding box
            bbox2: Second bounding box

        Returns:
            True if aspect ratios are similar enough for swapping
        """
        ratio1 = self.calculate_aspect_ratio(bbox1)
        ratio2 = self.calculate_aspect_ratio(bbox2)

        # Calculate relative difference
        diff = abs(ratio1 - ratio2) / max(ratio1, ratio2)
        return diff <= self.max_aspect_ratio_diff

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

        # Perform sticker swapping with aspect ratio check
        if len(sentry_indices) > 0 and len(vehicle_indices) > 0:
            # Try to find compatible pairs based on aspect ratio
            swap_performed = False
            max_attempts = min(5, len(sentry_indices) * len(vehicle_indices))

            for _ in range(max_attempts):
                sentry_idx = random.choice(sentry_indices)
                vehicle_idx = random.choice(vehicle_indices)

                # Check if aspect ratios are compatible for swapping
                sentry_bbox = labels[sentry_idx][1:5]
                vehicle_bbox = labels[vehicle_idx][1:5]

                if self.can_swap_based_on_aspect_ratio(sentry_bbox, vehicle_bbox):
                    # Extract patches
                    sentry_patch = self.extract_armor_patch(image, sentry_bbox)
                    vehicle_patch = self.extract_armor_patch(image, vehicle_bbox)

                    # Place swapped patches
                    aug_image = self.place_armor_patch(aug_image, vehicle_patch, sentry_bbox)
                    aug_image = self.place_armor_patch(aug_image, sentry_patch, vehicle_bbox)

                    # Swap class labels
                    aug_labels[sentry_idx][0] = labels[vehicle_idx][0]
                    aug_labels[vehicle_idx][0] = labels[sentry_idx][0]

                    swap_performed = True
                    break

            if not swap_performed:
                # Log or handle case where no compatible swap was found
                pass

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


def create_sticker_swapper(max_aspect_ratio_diff: float = 0.3,
                          sticker_swap_prob: float = 0.3) -> StickerSwapAugmenter:
    """
    Factory function to create a StickerSwapAugmenter with optimal settings.

    Args:
        max_aspect_ratio_diff: Maximum allowed aspect ratio difference (default: 0.3)
        sticker_swap_prob: Probability of applying sticker swap (default: 0.3)

    Returns:
        Configured StickerSwapAugmenter instance

    Example:
        ```python
        # Create augmenter with default settings
        swapper = create_sticker_swapper()

        # Create augmenter with stricter aspect ratio matching
        strict_swapper = create_sticker_swapper(max_aspect_ratio_diff=0.2)

        # Apply augmentation
        aug_image, aug_labels = swapper.augment(image, labels)
        ```
    """
    return StickerSwapAugmenter(
        sticker_swap_prob=sticker_swap_prob,
        preserve_geometry=True,
        max_aspect_ratio_diff=max_aspect_ratio_diff
    )