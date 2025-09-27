# augmentation/__init__.py
"""
RoboMaster Data Augmentation Module

This package provides comprehensive data augmentation capabilities for RoboMaster
armor detection, including background bias mitigation, context-aware augmentation,
and unified context detection.

Key Components:
- context_detector: Unified context detection system
- sticker_swap: Sticker swapping for background bias mitigation
- background_mixup: Background mixing and COCO insertion
- context_augment: Context-aware brightness/contrast adjustments
- unified_augmenter: Unified interface for all augmentation strategies
- pipeline: High-level augmentation pipeline
- dataset_manager: Dataset management utilities
"""

from .context_detector import ContextDetector, create_context_detector
from .sticker_swap import StickerSwapAugmenter, create_sticker_swapper
from .background_mixup import BackgroundMixupAugmenter
from .context_augment import ContextAugmenter
from .unified_augmenter import UnifiedDataAugmenter
from .pipeline import AugmentationPipeline
from .dataset_manager import DatasetManager

__all__ = [
    # Context Detection
    'ContextDetector',
    'create_context_detector',

    # Specialized Augmenters
    'StickerSwapAugmenter',
    'create_sticker_swapper',
    'BackgroundMixupAugmenter',
    'ContextAugmenter',

    # Unified Interface
    'UnifiedDataAugmenter',

    # Pipeline and Management
    'AugmentationPipeline',
    'DatasetManager',
]