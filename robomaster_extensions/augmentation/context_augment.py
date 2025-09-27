#context_augment.py
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
from ..config import get_robomaster_config
from .context_detector import create_context_detector


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

        # Initialize unified context detector
        self.context_detector = create_context_detector(detection_strategy='hybrid')


    def brightness_adjust_augmentation(self, image: np.ndarray,
                                     labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply enhanced adaptive brightness adjustment with multiple methods.

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
        brightness_percentiles = np.percentile(v_channel, [10, 90])

        # Enhanced adaptive brightness adjustment with wider ranges
        if mean_brightness < 80:  # Very dark image - stronger enhancement
            brightness_factor = random.uniform(1.5, 2.5)
        elif mean_brightness < 120:  # Dark image
            brightness_factor = random.uniform(1.3, 2.0)
        elif mean_brightness > 200:  # Very bright image - stronger reduction
            brightness_factor = random.uniform(0.4, 0.7)
        elif mean_brightness > 160:  # Bright image
            brightness_factor = random.uniform(0.6, 0.9)
        else:  # Normal brightness - wider variation
            brightness_factor = random.uniform(0.7, 1.5)

        # Apply non-linear brightness adjustment for better contrast preservation
        v_channel_float = v_channel.astype(np.float32)

        # Method 1: Simple multiplication (existing method)
        if random.random() < 0.6:
            adjusted_v = v_channel_float * brightness_factor
        else:
            # Method 2: Power law adjustment for better dynamic range
            gamma = 1.0 / brightness_factor if brightness_factor > 1 else brightness_factor
            adjusted_v = 255.0 * ((v_channel_float / 255.0) ** gamma)

        # Enhanced clipping with soft saturation to prevent harsh cutoff
        adjusted_v = np.where(adjusted_v > 255,
                             255 - np.exp(-(adjusted_v - 255) / 20),  # Soft saturation for high values
                             adjusted_v)
        adjusted_v = np.clip(adjusted_v, 0, 255).astype(np.uint8)

        # Reconstruct HSV and convert back to BGR
        hsv[:, :, 2] = adjusted_v
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
        Apply enhanced adaptive contrast adjustment with multiple methods.

        Args:
            image: Input image
            labels: Input labels (unchanged)

        Returns:
            Contrast-adjusted image and original labels
        """
        # Calculate comprehensive contrast statistics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        intensity_range = np.ptp(gray)  # Peak-to-peak (max - min)

        # Enhanced adaptive contrast factor with more aggressive adjustments
        if std_intensity < 25:  # Very low contrast - strong enhancement
            contrast_factor = random.uniform(1.8, 2.8)
        elif std_intensity < 45:  # Low contrast
            contrast_factor = random.uniform(1.4, 2.2)
        elif std_intensity > 90:  # Very high contrast - strong reduction
            contrast_factor = random.uniform(0.5, 0.8)
        elif std_intensity > 65:  # High contrast
            contrast_factor = random.uniform(0.7, 0.95)
        else:  # Normal contrast - wider variation
            contrast_factor = random.uniform(0.8, 1.6)

        # Choose contrast adjustment method
        method = random.choice(['linear', 'sigmoid', 'adaptive_histogram'])

        if method == 'linear':
            # Enhanced linear contrast adjustment
            adjusted_image = image.astype(np.float32)
            adjusted_image = (adjusted_image - mean_intensity) * contrast_factor + mean_intensity
            adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

        elif method == 'sigmoid':
            # Sigmoid contrast adjustment for smoother transitions
            adjusted_image = image.astype(np.float32) / 255.0
            # Apply sigmoid function: f(x) = 1 / (1 + exp(-gain * (x - cutoff)))
            gain = contrast_factor * 8.0  # Scale factor for sigmoid steepness
            cutoff = mean_intensity / 255.0  # Sigmoid center point
            adjusted_image = 1.0 / (1.0 + np.exp(-gain * (adjusted_image - cutoff)))
            # Normalize back to [0, 255]
            adjusted_image = (adjusted_image * 255).astype(np.uint8)

        else:  # adaptive_histogram
            # Adaptive histogram-based contrast adjustment
            adjusted_image = image.copy()
            for channel in range(3):
                channel_data = image[:, :, channel].astype(np.float32)
                # Calculate histogram
                hist, bins = np.histogram(channel_data.flatten(), bins=256, range=(0, 256))
                # Calculate cumulative distribution function
                cdf = hist.cumsum()
                cdf_normalized = cdf / cdf[-1]  # Normalize to [0, 1]

                # Apply contrast enhancement based on CDF
                enhanced_factor = 1.0 + (contrast_factor - 1.0) * (1.0 - cdf_normalized[channel_data.astype(int)])
                channel_data = channel_data * enhanced_factor
                adjusted_image[:, :, channel] = np.clip(channel_data, 0, 255).astype(np.uint8)

        return adjusted_image, labels

    def gamma_correction(self, image: np.ndarray,
                        labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply enhanced gamma correction with adaptive range selection.

        Args:
            image: Input image
            labels: Input labels (unchanged)

        Returns:
            Gamma-corrected image and original labels
        """
        # Analyze image brightness to determine optimal gamma range
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        # Enhanced gamma range selection based on image characteristics
        if mean_brightness < 80:  # Dark image - use higher gamma for brightening
            gamma = random.uniform(0.4, 0.8)
        elif mean_brightness > 180:  # Bright image - use lower gamma for darkening
            gamma = random.uniform(1.2, 2.0)
        else:  # Normal range - wider gamma variation
            gamma = random.uniform(0.5, 1.8)

        # Build lookup table for gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")

        # Apply gamma correction
        adjusted_image = cv2.LUT(image, table)

        return adjusted_image, labels

    def histogram_equalization(self, image: np.ndarray,
                              labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply histogram equalization for contrast enhancement.

        Args:
            image: Input image
            labels: Input labels (unchanged)

        Returns:
            Histogram equalized image and original labels
        """
        # Convert to YUV color space for better preservation of color information
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # Apply histogram equalization only to luminance channel (Y)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])

        # Convert back to BGR
        adjusted_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        return adjusted_image, labels

    def clahe_enhancement(self, image: np.ndarray,
                         labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

        Args:
            image: Input image
            labels: Input labels (unchanged)

        Returns:
            CLAHE enhanced image and original labels
        """
        # Convert to LAB color space for better results
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Create CLAHE object with adaptive parameters
        clip_limit = random.uniform(2.0, 6.0)  # Adaptive clip limit
        tile_grid_size = random.choice([(4, 4), (8, 8), (12, 12)])  # Varied tile sizes

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        # Apply CLAHE to L channel only
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convert back to BGR
        adjusted_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return adjusted_image, labels

    def adaptive_brightness_contrast(self, image: np.ndarray,
                                   labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply combined adaptive brightness and contrast adjustment.

        Args:
            image: Input image
            labels: Input labels (unchanged)

        Returns:
            Enhanced image and original labels
        """
        # Analyze image statistics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_contrast = np.std(gray)

        # Determine enhancement strategy based on image characteristics
        if mean_brightness < 100 and std_contrast < 40:
            # Dark and low contrast - apply strong enhancement
            enhanced_image, _ = self.brightness_adjust_augmentation(image, labels)
            enhanced_image, _ = self.contrast_adjustment(enhanced_image, labels)
        elif mean_brightness > 180 and std_contrast > 80:
            # Bright and high contrast - apply gentle adjustment
            enhanced_image, _ = self.gamma_correction(image, labels)
        elif std_contrast < 30:
            # Low contrast regardless of brightness
            enhanced_image, _ = self.clahe_enhancement(image, labels)
        else:
            # Mixed enhancement
            method = random.choice(['brightness_contrast', 'gamma_clahe', 'histogram'])
            if method == 'brightness_contrast':
                enhanced_image, _ = self.brightness_adjust_augmentation(image, labels)
                enhanced_image, _ = self.contrast_adjustment(enhanced_image, labels)
            elif method == 'gamma_clahe':
                enhanced_image, _ = self.gamma_correction(image, labels)
                enhanced_image, _ = self.clahe_enhancement(enhanced_image, labels)
            else:
                enhanced_image, _ = self.histogram_equalization(image, labels)

        return enhanced_image, labels

    def augment(self, image: np.ndarray, labels: np.ndarray,
                augmentation_type: str = 'mixed') -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply enhanced context augmentation to image and labels.

        Args:
            image: Input image
            labels: Input labels
            augmentation_type: Type of augmentation ('brightness', 'color_temp',
                             'contrast', 'gamma', 'histogram', 'clahe',
                             'adaptive', 'mixed')

        Returns:
            Augmented image and labels
        """
        aug_image, aug_labels = image.copy(), labels.copy()

        if augmentation_type == 'brightness':
            # Enhanced brightness adjustment
            if random.random() < self.brightness_adjust_prob:
                aug_image, aug_labels = self.brightness_adjust_augmentation(
                    aug_image, aug_labels)
        elif augmentation_type == 'color_temp':
            # Color temperature adjustment
            aug_image, aug_labels = self.color_temperature_adjustment(
                aug_image, aug_labels)
        elif augmentation_type == 'contrast':
            # Enhanced contrast adjustment
            aug_image, aug_labels = self.contrast_adjustment(
                aug_image, aug_labels)
        elif augmentation_type == 'gamma':
            # Enhanced gamma correction
            aug_image, aug_labels = self.gamma_correction(
                aug_image, aug_labels)
        elif augmentation_type == 'histogram':
            # Histogram equalization
            aug_image, aug_labels = self.histogram_equalization(
                aug_image, aug_labels)
        elif augmentation_type == 'clahe':
            # CLAHE enhancement
            aug_image, aug_labels = self.clahe_enhancement(
                aug_image, aug_labels)
        elif augmentation_type == 'adaptive':
            # Adaptive brightness and contrast
            aug_image, aug_labels = self.adaptive_brightness_contrast(
                aug_image, aug_labels)
        else:  # 'mixed' - apply multiple enhanced augmentations
            # Apply brightness adjustment with higher probability
            if random.random() < self.brightness_adjust_prob:
                aug_image, aug_labels = self.brightness_adjust_augmentation(
                    aug_image, aug_labels)

            # Apply color temperature with 35% probability
            if random.random() < 0.35:
                aug_image, aug_labels = self.color_temperature_adjustment(
                    aug_image, aug_labels)

            # Apply contrast adjustment with 40% probability
            if random.random() < 0.4:
                aug_image, aug_labels = self.contrast_adjustment(
                    aug_image, aug_labels)

            # Apply gamma correction with 30% probability
            if random.random() < 0.3:
                aug_image, aug_labels = self.gamma_correction(
                    aug_image, aug_labels)

            # Apply CLAHE with 25% probability
            if random.random() < 0.25:
                aug_image, aug_labels = self.clahe_enhancement(
                    aug_image, aug_labels)

            # Apply histogram equalization with 20% probability
            if random.random() < 0.2:
                aug_image, aug_labels = self.histogram_equalization(
                    aug_image, aug_labels)

            # Apply adaptive enhancement with 15% probability
            if random.random() < 0.15:
                aug_image, aug_labels = self.adaptive_brightness_contrast(
                    aug_image, aug_labels)

        return aug_image, aug_labels