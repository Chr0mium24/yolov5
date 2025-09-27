# augmentation/unified_augmenter.py

import cv2
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from pathlib import Path

# Import project-specific configurations and specialized augmenters
from ..config import get_robomaster_config
from .sticker_swap import StickerSwapAugmenter
from .background_mixup import BackgroundMixupAugmenter
from .context_augment import ContextAugmenter
from .context_detector import create_context_detector


class UnifiedDataAugmenter:
    """
    Unified data augmenter that combines multiple augmentation strategies.
    
    This class is now PURE and only operates on in-memory data. It has no
    knowledge of file systems or process orchestration. Its sole responsibility
    is to receive an image and its labels, apply a specified augmentation,
    and return the result.
    """

    def __init__(self,
                 sticker_swap_prob: float = 0.3,
                 context_mixup_prob: float = 0.2,
                 brightness_adjust_prob: float = 0.4,
                 coco_insert_prob: float = 0.2,
                 preserve_geometry: bool = True,
                 coco_img_path: Optional[str] = None):
        """
        Initialize the unified data augmenter by creating instances of
        specialized augmenters.

        Args:
            sticker_swap_prob (float): Probability of swapping stickers.
            context_mixup_prob (float): Probability of mixing different contexts.
            brightness_adjust_prob (float): Probability of applying brightness adjustment.
            coco_insert_prob (float): Probability of inserting a COCO background.
            preserve_geometry (bool): Whether to preserve bounding box geometry.
            coco_img_path (Optional[str]): Path to COCO background image library.
        """
        # Initialize the specialized augmenters that contain the actual logic
        self.sticker_swap_augmenter = StickerSwapAugmenter(
            sticker_swap_prob=sticker_swap_prob,
            preserve_geometry=preserve_geometry
        )

        self.background_mixup_augmenter = BackgroundMixupAugmenter(
            context_mixup_prob=context_mixup_prob,
            coco_insert_prob=coco_insert_prob,
            coco_img_path=coco_img_path
        )

        self.context_augmenter = ContextAugmenter(
            brightness_adjust_prob=brightness_adjust_prob
        )

        # Load RoboMaster configuration for class definitions
        self.config = get_robomaster_config()

        # Initialize unified context detector
        self.context_detector = create_context_detector(detection_strategy='hybrid')


    def augment(self, image: np.ndarray, labels: np.ndarray,
                context_pool: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
                augmentation_type: str = 'mixed'
                ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Apply a specified data augmentation to an in-memory image and its labels.

        Args:
            image (np.ndarray): The input image.
            labels (np.ndarray): The corresponding YOLO format labels.
            context_pool (Optional[List[Tuple]]): A pool of (image, labels) tuples
                from different contexts, used for mixup operations.
            augmentation_type (str): The specific type of augmentation to apply.
                Can be one of ['sticker_swap', 'brightness_adjust', 'contrast_adjust',
                'clahe_enhance', 'adaptive_enhance', 'coco_insert', 'context_mixup', 'mixed'].

        Returns:
            Tuple[np.ndarray, np.ndarray, str]: A tuple containing:
                - The augmented image.
                - The corresponding augmented labels.
                - The name of the augmentation that was actually applied, or 'original'
                  if the probabilistic check failed and no augmentation was performed.
        """
        aug_image, aug_labels = image.copy(), labels.copy()
        applied_augmentation = 'original'

        # Use a temporary variable to store the result of a probabilistic check
        was_applied = False

        # --- Single Augmentation Strategies ---
        if augmentation_type == 'sticker_swap':
            aug_image, aug_labels, was_applied = self.sticker_swap_augmenter.augment(aug_image, aug_labels)
            if was_applied: applied_augmentation = 'sticker_swap'

        elif augmentation_type == 'brightness_adjust':
            aug_image, _, was_applied = self.context_augmenter.brightness_adjust_augmentation(aug_image)
            if was_applied: applied_augmentation = 'brightness_adjust'

        elif augmentation_type == 'contrast_adjust':
            aug_image, _, was_applied = self.context_augmenter.contrast_adjustment(aug_image)
            if was_applied: applied_augmentation = 'contrast_adjust'
            
        elif augmentation_type == 'clahe_enhance':
            aug_image, _, was_applied = self.context_augmenter.clahe_enhancement(aug_image)
            if was_applied: applied_augmentation = 'clahe_enhance'

        elif augmentation_type == 'adaptive_enhance':
            aug_image, _, was_applied = self.context_augmenter.adaptive_brightness_contrast(aug_image)
            if was_applied: applied_augmentation = 'adaptive_enhance'

        elif augmentation_type == 'coco_insert':
            aug_image, aug_labels, was_applied = self.background_mixup_augmenter.coco_insert_augmentation(aug_image, aug_labels)
            if was_applied: applied_augmentation = 'coco_insert'

        elif augmentation_type == 'context_mixup':
            if context_pool:
                aug_image, aug_labels, was_applied = self.background_mixup_augmenter.context_mixup_augmentation(aug_image, aug_labels, context_pool)
                if was_applied: applied_augmentation = 'context_mixup'

        # --- Mixed Augmentation Strategy ---
        elif augmentation_type == 'mixed':
            # In 'mixed' mode, we apply a sequence of augmentations.
            # We consider the augmentation applied if at least one step changes the image.
            
            # 1. Sticker Swapping
            img_before_swap = aug_image.copy()
            aug_image, aug_labels, swap_applied = self.sticker_swap_augmenter.augment(aug_image, aug_labels)
            
            # 2. Context Augmentation (brightness, contrast, etc.)
            img_before_context = aug_image.copy()
            aug_image, _ = self.context_augmenter.augment(aug_image, aug_labels) # 'augment' in sub-module applies a random choice
            
            # 3. Background Mixup (COCO or context mixup)
            img_before_mixup = aug_image.copy()
            aug_image, aug_labels, mixup_applied = self.background_mixup_augmenter.augment(aug_image, aug_labels, context_pool)

            # Check if any augmentation actually changed the image
            if (swap_applied or 
                not np.array_equal(img_before_context, aug_image) or
                mixup_applied):
                applied_augmentation = 'mixed'
        
        else:
             print(f"Warning: Unknown augmentation_type '{augmentation_type}'. Returning original image.")


        return aug_image, aug_labels, applied_augmentation