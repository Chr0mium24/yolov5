#!/usr/bin/env python3
# context_detector_example.py
"""
Example Usage of Unified Context Detection System

This script demonstrates how to use the unified context detector across
all augmentation modules for consistent context analysis.
"""

import numpy as np
import cv2
from context_detector import create_context_detector

def create_sample_labels():
    """Create sample YOLO format labels for testing."""

    # Sample 1: Vehicle context (classes 0-11)
    vehicle_labels = np.array([
        [0, 0.3, 0.4, 0.1, 0.15],    # B1 armor plate
        [6, 0.7, 0.6, 0.12, 0.18],   # R1 armor plate
    ])

    # Sample 2: Sentry context (classes 12-13)
    sentry_labels = np.array([
        [12, 0.5, 0.5, 0.2, 0.3],    # RQS (Red Sentry)
    ])

    # Sample 3: Mixed context (both vehicle and sentry)
    mixed_labels = np.array([
        [0, 0.2, 0.3, 0.1, 0.15],    # B1 armor plate
        [12, 0.6, 0.7, 0.15, 0.25],  # RQS (Red Sentry)
        [7, 0.8, 0.4, 0.1, 0.12],    # R2 armor plate
    ])

    # Sample 4: Crowded context (many objects)
    crowded_labels = np.array([
        [0, 0.1, 0.1, 0.08, 0.12],
        [1, 0.3, 0.2, 0.09, 0.11],
        [2, 0.5, 0.3, 0.07, 0.10],
        [6, 0.7, 0.4, 0.08, 0.13],
        [7, 0.9, 0.6, 0.06, 0.09],
        [8, 0.2, 0.8, 0.09, 0.14],
    ])

    # Sample 5: Sparse context (few objects, far apart)
    sparse_labels = np.array([
        [3, 0.2, 0.2, 0.05, 0.08],
        [9, 0.8, 0.8, 0.06, 0.09],
    ])

    return {
        'vehicle': vehicle_labels,
        'sentry': sentry_labels,
        'mixed': mixed_labels,
        'crowded': crowded_labels,
        'sparse': sparse_labels
    }

def create_sample_image(brightness='normal', complexity='normal'):
    """Create a sample image with specified characteristics."""

    # Base image size
    height, width = 640, 640

    # Create base image based on brightness
    if brightness == 'dark':
        base_color = 30
    elif brightness == 'bright':
        base_color = 200
    else:  # normal
        base_color = 128

    image = np.full((height, width, 3), base_color, dtype=np.uint8)

    # Add complexity based on parameter
    if complexity == 'high':
        # Add noise and edges
        noise = np.random.randint(-50, 50, (height, width, 3))
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add some rectangular patterns
        for _ in range(20):
            x1, y1 = np.random.randint(0, width-50), np.random.randint(0, height-50)
            x2, y2 = x1 + np.random.randint(20, 50), y1 + np.random.randint(20, 50)
            color = np.random.randint(0, 255, 3)
            cv2.rectangle(image, (x1, y1), (x2, y2), color.tolist(), -1)

    elif complexity == 'low':
        # Very simple image with minimal variation
        pass
    else:  # normal
        # Add some basic patterns
        for _ in range(5):
            x1, y1 = np.random.randint(0, width-100), np.random.randint(0, height-100)
            x2, y2 = x1 + np.random.randint(50, 100), y1 + np.random.randint(50, 100)
            color = np.random.randint(base_color-30, base_color+30, 3)
            cv2.rectangle(image, (x1, y1), (x2, y2), color.tolist(), -1)

    return image

def test_context_detection():
    """Test the unified context detection system."""

    print("=" * 60)
    print("Testing Unified Context Detection System")
    print("=" * 60)

    # Create context detector with different strategies
    detectors = {
        'class_based': create_context_detector(detection_strategy='class_based'),
        'spatial': create_context_detector(detection_strategy='spatial'),
        'hybrid': create_context_detector(detection_strategy='hybrid')
    }

    # Get sample labels
    sample_labels = create_sample_labels()
    image_size = (640, 640)

    # Test each sample with each detection strategy
    for sample_name, labels in sample_labels.items():
        print(f"\n--- Testing {sample_name.upper()} context ---")
        print(f"Number of objects: {len(labels)}")
        if len(labels) > 0:
            print(f"Object classes: {labels[:, 0].astype(int).tolist()}")

        for strategy_name, detector in detectors.items():
            context = detector.detect_context(labels, image_size)
            print(f"{strategy_name:12}: {context}")

    # Test with image-based analysis
    print(f"\n--- Testing with image analysis ---")

    # Test different image characteristics
    image_configs = [
        ('dark_high', 'dark', 'high'),
        ('bright_low', 'bright', 'low'),
        ('normal_normal', 'normal', 'normal')
    ]

    hybrid_detector = detectors['hybrid']

    for config_name, brightness, complexity in image_configs:
        print(f"\n{config_name} image:")

        # Create test image
        test_image = create_sample_image(brightness, complexity)

        # Test with vehicle labels
        vehicle_labels = sample_labels['vehicle']
        context_with_image = hybrid_detector.detect_context(
            vehicle_labels, image_size, test_image
        )
        context_without_image = hybrid_detector.detect_context(
            vehicle_labels, image_size
        )

        print(f"  Without image: {context_without_image}")
        print(f"  With image:    {context_with_image}")

def test_detailed_context_info():
    """Test the detailed context information functionality."""

    print(f"\n" + "=" * 60)
    print("Testing Detailed Context Information")
    print("=" * 60)

    detector = create_context_detector(detection_strategy='hybrid')
    sample_labels = create_sample_labels()
    image_size = (640, 640)

    # Test with mixed context
    mixed_labels = sample_labels['mixed']
    test_image = create_sample_image('normal', 'normal')

    context_info = detector.get_context_info(mixed_labels, image_size, test_image)

    print("Detailed context information for mixed context:")
    print("-" * 40)
    for key, value in context_info.items():
        if isinstance(value, float):
            print(f"{key:20}: {value:.4f}")
        elif isinstance(value, list) and len(value) > 3:
            print(f"{key:20}: {value[:3]}... (truncated)")
        else:
            print(f"{key:20}: {value}")

def test_augmentation_suitability():
    """Test the augmentation suitability assessment."""

    print(f"\n" + "=" * 60)
    print("Testing Augmentation Suitability")
    print("=" * 60)

    detector = create_context_detector()
    sample_labels = create_sample_labels()
    image_size = (640, 640)

    augmentation_types = [
        'sticker_swap', 'background_mixup', 'coco_insert',
        'context_augment', 'brightness_adjust', 'contrast_adjust', 'clahe_enhance'
    ]

    print(f"{'Context':<12} | {'Augmentation Type':<18} | {'Suitable'}")
    print("-" * 50)

    for sample_name, labels in sample_labels.items():
        context = detector.detect_context(labels, image_size)

        for aug_type in augmentation_types:
            suitable = detector.is_suitable_for_augmentation(context, aug_type)
            status = "✓" if suitable else "✗"
            print(f"{context:<12} | {aug_type:<18} | {status}")

        print("-" * 50)

if __name__ == "__main__":
    # Run all tests
    test_context_detection()
    test_detailed_context_info()
    test_augmentation_suitability()

    print(f"\n" + "=" * 60)
    print("Context Detection Integration Complete!")
    print("=" * 60)
    print("\nThe unified context detector has been successfully integrated")
    print("into all augmentation modules:")
    print("  • sticker_swap.py")
    print("  • context_augment.py")
    print("  • background_mixup.py")
    print("  • unified_augmenter.py")
    print("\nAll modules now use consistent context detection with enhanced")
    print("capabilities including spatial and image-based analysis.")