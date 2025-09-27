# context_detector.py
"""
Unified Context Detection Module for RoboMaster Augmentation

This module provides a centralized context detection system that analyzes
images and labels to determine the scene context (sentry, vehicle, mixed, etc.).
It replaces the individual detect_context methods in each augmentation module
to ensure consistency and provide enhanced context analysis capabilities.

Key Features:
- Unified context detection across all augmentation modules
- Enhanced context analysis with spatial and statistical information
- Configurable detection strategies and thresholds
- Support for multi-class context detection
- Compatible with 14-class RoboMaster system
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from ..config import get_robomaster_config


class ContextDetector:
    """
    Unified context detector that analyzes images and labels to determine
    the scene context for RoboMaster augmentation strategies.
    """

    def __init__(self,
                 detection_strategy: str = 'class_based',
                 spatial_analysis: bool = True,
                 confidence_threshold: float = 0.5):
        """
        Initialize the context detector.

        Args:
            detection_strategy: Strategy for context detection
                              ('class_based', 'spatial', 'hybrid')
            spatial_analysis: Whether to include spatial analysis
            confidence_threshold: Minimum confidence for context detection
        """
        self.detection_strategy = detection_strategy
        self.spatial_analysis = spatial_analysis
        self.confidence_threshold = confidence_threshold

        # Load RoboMaster configuration
        self.config = get_robomaster_config()
        self.class_map = self.config.class_map
        self.sentry_classes = self.config.sentry_classes  # [12, 13] for RQS, BQS
        self.vehicle_classes = self.config.vehicle_classes  # [0-11] for vehicles

        # Context type definitions
        self.context_types = {
            'unknown': 0,
            'vehicle': 1,
            'sentry': 2,
            'mixed': 3,
            'crowded': 4,
            'sparse': 5
        }

    def detect_context(self, labels: np.ndarray,
                      image_size: Tuple[int, int],
                      image: Optional[np.ndarray] = None) -> str:
        """
        Detect scene context based on labels and optionally image content.

        Args:
            labels: YOLO format labels [class, x_center, y_center, width, height]
            image_size: (height, width) of the image
            image: Optional image for enhanced analysis

        Returns:
            Context type: 'sentry', 'vehicle', 'mixed', 'crowded', 'sparse', 'unknown'
        """
        if len(labels) == 0:
            return 'unknown'

        # Basic class-based detection
        context = self._detect_class_based_context(labels)

        # Enhanced analysis if requested
        if self.detection_strategy in ['spatial', 'hybrid']:
            spatial_context = self._detect_spatial_context(labels, image_size)

            if self.detection_strategy == 'spatial':
                context = spatial_context
            elif self.detection_strategy == 'hybrid':
                context = self._combine_contexts(context, spatial_context)

        # Additional image-based analysis if image is provided
        if image is not None and self.spatial_analysis:
            image_context = self._detect_image_based_context(image, labels)
            context = self._combine_contexts(context, image_context)

        return context

    def _detect_class_based_context(self, labels: np.ndarray) -> str:
        """
        Detect context based on object classes present in the image.

        Args:
            labels: YOLO format labels

        Returns:
            Basic context type based on classes
        """
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

    def _detect_spatial_context(self, labels: np.ndarray,
                               image_size: Tuple[int, int]) -> str:
        """
        Detect context based on spatial distribution of objects.

        Args:
            labels: YOLO format labels
            image_size: (height, width) of the image

        Returns:
            Spatial context type
        """
        if len(labels) == 0:
            return 'unknown'

        h, w = image_size
        num_objects = len(labels)

        # Calculate object density
        total_area = h * w
        object_areas = []
        for label in labels:
            _, _, width, height = label[1:5]
            area = (width * w) * (height * h)
            object_areas.append(area)

        total_object_area = sum(object_areas)
        density = total_object_area / total_area

        # Calculate spatial distribution
        centers = labels[:, 1:3]  # x_center, y_center
        if len(centers) > 1:
            # Calculate pairwise distances
            distances = []
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    distances.append(dist)

            avg_distance = np.mean(distances) if distances else 0
            std_distance = np.std(distances) if distances else 0
        else:
            avg_distance = 0
            std_distance = 0

        # Context classification based on spatial metrics
        if num_objects >= 5 or density > 0.3:
            return 'crowded'
        elif num_objects <= 2 and density < 0.1:
            return 'sparse'
        elif avg_distance < 0.2:  # Objects close together
            return 'crowded'
        elif avg_distance > 0.6:  # Objects far apart
            return 'sparse'
        else:
            # Fall back to class-based detection
            return self._detect_class_based_context(labels)

    def _detect_image_based_context(self, image: np.ndarray,
                                   labels: np.ndarray) -> str:
        """
        Detect context based on image visual characteristics.

        Args:
            image: Input image
            labels: YOLO format labels

        Returns:
            Image-based context type
        """
        # Analyze image characteristics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate image statistics
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)

        # Edge density analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # Color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_variance = np.var(hsv, axis=(0, 1))

        # Determine context based on visual characteristics
        if edge_density > 0.1 and brightness_std > 50:
            # High detail, likely complex scene
            return 'crowded'
        elif edge_density < 0.05 and brightness_std < 30:
            # Low detail, likely simple scene
            return 'sparse'
        elif mean_brightness < 80:
            # Dark scene, might be indoor/sentry context
            return 'sentry'
        elif mean_brightness > 180 and np.mean(color_variance) > 1000:
            # Bright and colorful, likely outdoor/vehicle context
            return 'vehicle'
        else:
            return 'mixed'

    def _combine_contexts(self, context1: str, context2: str) -> str:
        """
        Combine multiple context detections into a final result.

        Args:
            context1: First context detection result
            context2: Second context detection result

        Returns:
            Combined context result
        """
        # Priority mapping for context combination
        priority = {
            'unknown': 0,
            'sparse': 1,
            'vehicle': 2,
            'sentry': 3,
            'crowded': 4,
            'mixed': 5
        }

        # If either is mixed, return mixed
        if context1 == 'mixed' or context2 == 'mixed':
            return 'mixed'

        # Return the higher priority context
        if priority.get(context1, 0) >= priority.get(context2, 0):
            return context1
        else:
            return context2

    def get_context_info(self, labels: np.ndarray,
                        image_size: Tuple[int, int],
                        image: Optional[np.ndarray] = None) -> Dict:
        """
        Get detailed context information including statistics.

        Args:
            labels: YOLO format labels
            image_size: (height, width) of the image
            image: Optional image for enhanced analysis

        Returns:
            Dictionary with detailed context information
        """
        context = self.detect_context(labels, image_size, image)

        info = {
            'context': context,
            'num_objects': len(labels),
            'object_classes': labels[:, 0].astype(int).tolist() if len(labels) > 0 else [],
            'has_sentry': False,
            'has_vehicle': False,
            'sentry_count': 0,
            'vehicle_count': 0
        }

        if len(labels) > 0:
            classes = labels[:, 0].astype(int)
            info['has_sentry'] = any(cls in self.sentry_classes for cls in classes)
            info['has_vehicle'] = any(cls in self.vehicle_classes for cls in classes)
            info['sentry_count'] = sum(1 for cls in classes if cls in self.sentry_classes)
            info['vehicle_count'] = sum(1 for cls in classes if cls in self.vehicle_classes)

            # Spatial statistics
            h, w = image_size
            areas = []
            centers = []
            for label in labels:
                _, x_center, y_center, width, height = label
                areas.append((width * w) * (height * h))
                centers.append((x_center, y_center))

            info['object_areas'] = areas
            info['object_centers'] = centers
            info['total_object_area'] = sum(areas)
            info['object_density'] = sum(areas) / (h * w)

            if len(centers) > 1:
                distances = []
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
                        distances.append(dist)
                info['avg_object_distance'] = np.mean(distances)
                info['object_distance_std'] = np.std(distances)
            else:
                info['avg_object_distance'] = 0
                info['object_distance_std'] = 0

        # Image-based statistics if image is provided
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            info['image_brightness'] = float(np.mean(gray))
            info['image_brightness_std'] = float(np.std(gray))

            edges = cv2.Canny(gray, 50, 150)
            info['edge_density'] = float(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))

        return info

    def is_suitable_for_augmentation(self, context: str,
                                   augmentation_type: str) -> bool:
        """
        Determine if a given context is suitable for specific augmentation types.

        Args:
            context: Detected context type
            augmentation_type: Type of augmentation to apply

        Returns:
            True if context is suitable for the augmentation
        """
        suitability_map = {
            'sticker_swap': ['mixed', 'sentry', 'vehicle'],
            'background_mixup': ['sparse', 'vehicle', 'sentry'],
            'coco_insert': ['sparse', 'vehicle'],
            'context_augment': ['crowded', 'mixed', 'vehicle', 'sentry'],
            'brightness_adjust': ['vehicle', 'sentry', 'mixed'],
            'contrast_adjust': ['crowded', 'mixed'],
            'clahe_enhance': ['sentry', 'mixed'],
        }

        return context in suitability_map.get(augmentation_type, [])


def create_context_detector(detection_strategy: str = 'hybrid',
                           spatial_analysis: bool = True,
                           confidence_threshold: float = 0.5) -> ContextDetector:
    """
    Factory function to create a ContextDetector with optimal settings.

    Args:
        detection_strategy: Strategy for context detection ('class_based', 'spatial', 'hybrid')
        spatial_analysis: Whether to include spatial analysis
        confidence_threshold: Minimum confidence for context detection

    Returns:
        Configured ContextDetector instance

    Example:
        ```python
        # Create detector with default hybrid strategy
        detector = create_context_detector()

        # Create detector with class-based strategy only
        simple_detector = create_context_detector(detection_strategy='class_based')

        # Detect context
        context = detector.detect_context(labels, image_size, image)

        # Get detailed context information
        context_info = detector.get_context_info(labels, image_size, image)
        ```
    """
    return ContextDetector(
        detection_strategy=detection_strategy,
        spatial_analysis=spatial_analysis,
        confidence_threshold=confidence_threshold
    )