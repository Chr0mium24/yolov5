# augmentation/pipeline.py

import random
import yaml
import json
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple,Any

# 确保可以导入同级目录的模块
from .dataset_manager import DatasetManager
from .unified_augmenter import UnifiedDataAugmenter

class AugmentationPipeline:
    """
    Orchestrates the entire data augmentation workflow.
    Connects the DatasetManager (for file I/O) with the UnifiedDataAugmenter (for image processing).
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dm = DatasetManager(config['input_path'], config['output_path'])
        self.augmenter = UnifiedDataAugmenter(
            sticker_swap_prob=config['sticker_swap_prob'],
            context_mixup_prob=config['context_mixup_prob'],
            brightness_adjust_prob=config['brightness_adjust_prob'],
            coco_insert_prob=config['coco_insert_prob'],
            coco_img_path=config.get('coco_img_path') # Use .get for optional param
        )

    def run(self):
        """Executes the full augmentation pipeline."""
        print("Starting augmentation pipeline...")
        self.dm.setup_directories()

        # --- Process Training Split ---
        print("\nProcessing 'train' split...")
        train_files = self.dm.get_data_files('train')
        context_pools = self._categorize_data(train_files)
        self._process_train_split(train_files, context_pools)

        # --- Process Validation Split (typically just copying) ---
        print("\nProcessing 'val' split...")
        self._process_val_split()

        # --- Final Report ---
        self._generate_report()
        print("\nAugmentation pipeline finished successfully!")

    def _categorize_data(self, data_files: List[Path]) -> Dict[str, List[Tuple[Any, Any]]]:
        """Pre-scans data to categorize it by context for mixup operations."""
        print(f"Categorizing {len(data_files)} images by context...")
        context_pools = {'sentry': [], 'vehicle': [], 'mixed': []}
        for img_path, lbl_path in tqdm(data_files, desc="Categorizing"):
            item = self.dm.load_data_item(img_path, lbl_path)
            if item:
                image, labels = item
                context = self.augmenter.detect_context(labels, image.shape[:2])
                if context in context_pools:
                    context_pools[context].append((image, labels))
        print(f"Context distribution: {[(k, len(v)) for k, v in context_pools.items()]}")
        return context_pools

    def _process_train_split(self, train_files: List[Path], context_pools: Dict):
        """Applies augmentations to the training data."""
        for img_path, lbl_path in tqdm(train_files, desc="Augmenting train set"):
            item = self.dm.load_data_item(img_path, lbl_path)
            if not item:
                continue
            
            image, labels = item
            
            # Save a copy of the original image to the augmented set
            self.dm.save_data_item(image, labels, 'train_augmented', img_path.name)
            
            # Determine current context to select a different pool for mixup
            current_context = self.augmenter.detect_context(labels, image.shape[:2])
            
            # Generate augmented versions
            for aug_type in self.config['strategies']:
                for i in range(self.config['augmentation_factor']):
                    # Select a context pool for mixup that is different from the current image's context
                    available_pools = [p for c, p in context_pools.items() if c != current_context and p]
                    mixup_pool = random.choice(available_pools) if available_pools else None
                    
                    aug_img, aug_lbls, actual_aug_type = self.augmenter.augment(image, labels, mixup_pool, aug_type)
                    
                    # Save only if augmentation was actually applied
                    if actual_aug_type != 'original':
                        self.dm.save_augmented_item(aug_img, aug_lbls, img_path.name, actual_aug_type, i)

    def _process_val_split(self):
        """Copies validation data to the output directory without augmentation."""
        val_files = self.dm.get_data_files('val')
        for img_path, lbl_path in tqdm(val_files, desc="Copying val set"):
            item = self.dm.load_data_item(img_path, lbl_path)
            if item:
                image, labels = item
                self.dm.save_data_item(image, labels, 'val', img_path.name)

    def _generate_report(self):
        """Saves configuration and statistics files and prints a final summary."""
        # Save YAML config
        report_config = {k: v for k, v in self.config.items() if k not in ['input_path', 'output_path']}
        report_config['output_path'] = str(self.dm.output_path)
        config_file = self.dm.output_path / 'augmentation_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(report_config, f, default_flow_style=False)

        # Save brightness stats
        brightness_stats = {
            'mean_brightness_threshold_low': 100,
            # ... other stats ...
        }
        stats_file = self.dm.output_path / 'brightness_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(brightness_stats, f, indent=2)

        print("\n--- Augmentation Summary ---")
        print(f"Configuration saved to: {config_file}")
        print(f"Stats saved to: {stats_file}")
        print("Update your robomaster.yaml to point to this output directory:")
        print(f"  path: {self.dm.output_path}")
        print(f"  train: train_augmented")
        print(f"  val: val")