"""
Context Augmentation for RoboMaster

This module implements context-based augmentation strategies including brightness
adjustment and adaptive color space transformations to improve model robustness
across different lighting and environmental conditions.
"""

import cv2
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from .config import get_robomaster_config


class ContextAugmenter:
    """
    Data augmenter that applies context-based augmentations including
    adaptive brightness adjustment and color space transformations.
    """

    def __init__(self,
                 brightness_adjust_prob: float = 0.4):
        """
        Initialize the context augmenter.

        Args:
            brightness_adjust_prob: Probability of applying brightness adjustment
        """
        self.brightness_adjust_prob = brightness_adjust_prob

        # Load RoboMaster configuration
        self.config = get_robomaster_config()
        self.class_map = self.config.class_map
        self.sentry_classes = self.config.sentry_classes
        self.vehicle_classes = self.config.vehicle_classes

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

    def color_temperature_adjustment(self, image: np.ndarray,
                                   labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply color temperature adjustment to simulate different lighting conditions.

        Args:
            image: Input image
            labels: Input labels (unchanged)

        Returns:
            Color temperature adjusted image and original labels
        """
        # Random temperature shift
        temp_shift = random.uniform(-50, 50)

        # Convert to LAB color space for better color manipulation
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab = lab.astype(np.float32)

        # Adjust A and B channels for color temperature
        lab[:, :, 1] += temp_shift * 0.3  # A channel (green-red)
        lab[:, :, 2] += temp_shift * 0.2  # B channel (blue-yellow)

        # Clip values and convert back
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        adjusted_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return adjusted_image, labels

    def contrast_adjustment(self, image: np.ndarray,
                           labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply adaptive contrast adjustment.

        Args:
            image: Input image
            labels: Input labels (unchanged)

        Returns:
            Contrast-adjusted image and original labels
        """
        # Calculate current contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)

        # Adaptive contrast factor based on current contrast
        if std_intensity < 30:  # Low contrast
            contrast_factor = random.uniform(1.3, 1.8)
        elif std_intensity > 80:  # High contrast
            contrast_factor = random.uniform(0.7, 0.9)
        else:  # Normal contrast
            contrast_factor = random.uniform(0.9, 1.2)

        # Apply contrast adjustment
        adjusted_image = image.astype(np.float32)
        adjusted_image = (adjusted_image - mean_intensity) * contrast_factor + mean_intensity
        adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

        return adjusted_image, labels

    def gamma_correction(self, image: np.ndarray,
                        labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply gamma correction for exposure adjustment.

        Args:
            image: Input image
            labels: Input labels (unchanged)

        Returns:
            Gamma-corrected image and original labels
        """
        # Random gamma value
        gamma = random.uniform(0.6, 1.4)

        # Build lookup table for gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")

        # Apply gamma correction
        adjusted_image = cv2.LUT(image, table)

        return adjusted_image, labels

    def augment(self, image: np.ndarray, labels: np.ndarray,
                augmentation_type: str = 'mixed') -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply context augmentation to image and labels.

        Args:
            image: Input image
            labels: Input labels
            augmentation_type: Type of augmentation ('brightness', 'color_temp',
                             'contrast', 'gamma', 'mixed')

        Returns:
            Augmented image and labels
        """
        aug_image, aug_labels = image.copy(), labels.copy()

        if augmentation_type == 'brightness':
            # Only apply brightness adjustment
            if random.random() < self.brightness_adjust_prob:
                aug_image, aug_labels = self.brightness_adjust_augmentation(
                    aug_image, aug_labels)
        elif augmentation_type == 'color_temp':
            # Only apply color temperature adjustment
            aug_image, aug_labels = self.color_temperature_adjustment(
                aug_image, aug_labels)
        elif augmentation_type == 'contrast':
            # Only apply contrast adjustment
            aug_image, aug_labels = self.contrast_adjustment(
                aug_image, aug_labels)
        elif augmentation_type == 'gamma':
            # Only apply gamma correction
            aug_image, aug_labels = self.gamma_correction(
                aug_image, aug_labels)
        else:  # 'mixed' - apply multiple augmentations
            # Apply brightness adjustment
            if random.random() < self.brightness_adjust_prob:
                aug_image, aug_labels = self.brightness_adjust_augmentation(
                    aug_image, aug_labels)

            # Apply color temperature with 30% probability
            if random.random() < 0.3:
                aug_image, aug_labels = self.color_temperature_adjustment(
                    aug_image, aug_labels)

            # Apply contrast adjustment with 25% probability
            if random.random() < 0.25:
                aug_image, aug_labels = self.contrast_adjustment(
                    aug_image, aug_labels)

            # Apply gamma correction with 20% probability
            if random.random() < 0.2:
                aug_image, aug_labels = self.gamma_correction(
                    aug_image, aug_labels)

        return aug_image, aug_labels