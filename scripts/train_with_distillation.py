#!/usr/bin/env python3
"""
RoboMaster Training Script with Knowledge Distillation

This script provides a complete training pipeline for RoboMaster armor plate detection
with all optimizations including progressive distillation, CrossKD, label smoothing,
background bias mitigation, and active learning.

Usage:
    python scripts/train_with_distillation.py --student-cfg models/robomaster_yolov5n.yaml \
                                             --teacher-weights yolov5x.pt \
                                             --data data/robomaster.yaml \
                                             --epochs 300
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

# Add project root to path
ROOT = Path(__file__).parents[1]
sys.path.append(str(ROOT))

# Import RoboMaster components
from robomaster_extensions.robomaster_trainer import RoboMasterTrainer, create_default_config
from robomaster_extensions.data_augmentation import UnifiedDataAugmenter

# Import YOLOv5 components
from utils.dataloaders import create_dataloader
from utils.general import LOGGER, check_yaml, increment_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RoboMaster Training with Distillation')

    # Model arguments
    parser.add_argument('--student-cfg', type=str, required=True,
                       help='Student model configuration file')
    parser.add_argument('--teacher-weights', type=str, default=None,
                       help='Teacher model weights file')

    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Dataset configuration file')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Training image size')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr0', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                       help='Weight decay')

    # Optimization arguments
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing factor')
    parser.add_argument('--distillation-weight', type=float, default=0.5,
                       help='Weight for distillation loss')
    parser.add_argument('--distillation-method', type=str, default='crosskd',
                       choices=['crosskd', 'progressive'],
                       help='Distillation method')

    # Background bias mitigation
    parser.add_argument('--background-bias-augment', action='store_true',
                       help='Enable background bias augmentation')
    parser.add_argument('--sticker-swap-prob', type=float, default=0.3,
                       help='Probability of sticker swapping')

    # Active learning
    parser.add_argument('--active-learning', action='store_true',
                       help='Enable active learning')
    parser.add_argument('--active-learning-interval', type=int, default=50,
                       help='Active learning iteration interval')

    # Output arguments
    parser.add_argument('--project', type=str, default='runs/robomaster_train',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='exp',
                       help='Experiment name')
    parser.add_argument('--exist-ok', action='store_true',
                       help='Overwrite existing experiment')

    # Device arguments
    parser.add_argument('--device', type=str, default='',
                       help='CUDA device (empty for auto-select)')

    # Analysis arguments
    parser.add_argument('--analyze-attention', action='store_true',
                       help='Perform Grad-CAM attention analysis')
    parser.add_argument('--analysis-interval', type=int, default=50,
                       help='Analysis interval (epochs)')

    return parser.parse_args()


def load_dataset_config(data_path: str) -> dict:
    """Load dataset configuration."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)

    return data_config


def collect_augmented_paths(base_path: str, data_root: str) -> list:
    """Collect paths from augmented data structure."""
    from pathlib import Path

    full_path = Path(data_root) / base_path
    paths = []

    # First add the original path if it exists
    if full_path.exists():
        paths.append(str(full_path))

    # Check if there's an augmented version
    augmented_path = Path(str(full_path) + '_augmented')
    if augmented_path.exists():
        # Add all augmented subdirectories
        subdirs = [d for d in augmented_path.iterdir() if d.is_dir()]
        for subdir in subdirs:
            paths.append(str(subdir))

    # Return at least the original path if nothing else exists
    return paths if paths else [str(full_path)]


def create_combined_path_file(paths: list, temp_file: str = '/tmp/combined_paths.txt'):
    """Create a temporary file with all image paths for the dataloader."""
    from pathlib import Path
    import glob

    all_images = []
    for path in paths:
        # Find all image files in each path
        path_obj = Path(path)
        if path_obj.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                all_images.extend(glob.glob(str(path_obj / ext)))

    # Write all image paths to temporary file
    with open(temp_file, 'w') as f:
        for img_path in all_images:
            f.write(f"{img_path}\n")

    return temp_file


def create_dataloaders(data_config: dict, args) -> tuple:
    """Create training and validation dataloaders."""
    from pathlib import Path

    # Handle augmented data paths
    train_config = data_config['train']
    if isinstance(train_config, list):
        # If train is a list of paths, collect all augmented paths
        train_paths = []
        for path in train_config:
            train_paths.extend(collect_augmented_paths(path, data_config['path']))
    else:
        # If train is a single path
        train_paths = collect_augmented_paths(train_config, data_config['path'])

    # For now, use the first available path (fallback to standard structure)
    if train_paths:
        train_path = train_paths[0]
    else:
        # Fallback to first train path if it's a list, otherwise use as-is
        fallback_train = train_config[0] if isinstance(train_config, list) else train_config
        train_path = str(Path(data_config['path']) / fallback_train)

    print(f"Using training path: {train_path}")
    print(f"Available training paths: {train_paths}")

    # Training dataloader
    train_loader, _ = create_dataloader(
        path=train_path,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        stride=32,  # YOLOv5 stride
        single_cls=False,
        hyp=None,  # Use default hyperparameters
        augment=True,
        cache=False,
        rect=False,
        rank=-1,
        workers=8,
        image_weights=False,
        quad=False,
        prefix='train: ',
        shuffle=True
    )

    # Handle validation paths
    val_paths = collect_augmented_paths(data_config['val'], data_config['path'])
    val_path = val_paths[0] if val_paths else str(Path(data_config['path']) / data_config['val'])

    print(f"Using validation path: {val_path}")

    # Validation dataloader
    val_loader, _ = create_dataloader(
        path=val_path,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        stride=32,
        single_cls=False,
        hyp=None,
        augment=False,
        cache=False,
        rect=True,  # Rectangular inference for validation
        rank=-1,
        workers=8,
        image_weights=False,
        quad=False,
        prefix='val: ',
        shuffle=False
    )

    return train_loader, val_loader


def setup_save_directory(args) -> Path:
    """Setup save directory for training outputs."""
    save_dir = increment_path(
        Path(args.project) / args.name,
        exist_ok=args.exist_ok
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save training arguments
    with open(save_dir / 'args.yaml', 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    return save_dir


def create_training_config(args, data_config: dict) -> dict:
    """Create training configuration from arguments."""
    config = create_default_config()

    # Update with command line arguments
    config.update({
        'epochs': args.epochs,
        'lr0': args.lr0,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'imgsz': args.imgsz,

        # Dataset info
        'num_classes': data_config.get('nc', 8),

        # Optimization settings
        'label_smoothing': args.label_smoothing,
        'distillation_weight': args.distillation_weight,
        'distillation_method': args.distillation_method,

        # Background bias settings
        'background_bias_augment': args.background_bias_augment,
        'sticker_swap_prob': args.sticker_swap_prob,

        # Active learning settings
        'use_active_learning': args.active_learning,
        'active_learning_interval': args.active_learning_interval,

        # Analysis settings
        'analysis_interval': args.analysis_interval,

        # Device
        'device': args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    })

    return config


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    LOGGER.info(f"RoboMaster Training Script")
    LOGGER.info(f"Arguments: {vars(args)}")

    # Load dataset configuration
    data_config = load_dataset_config(args.data)
    LOGGER.info(f"Dataset: {data_config.get('path', 'Unknown')}")
    LOGGER.info(f"Classes: {data_config.get('nc', 8)}")

    # Setup save directory
    save_dir = setup_save_directory(args)
    LOGGER.info(f"Results will be saved to: {save_dir}")

    # Create training configuration
    config = create_training_config(args, data_config)

    # Create dataloaders
    LOGGER.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(data_config, args)
    LOGGER.info(f"Training samples: {len(train_loader.dataset)}")
    LOGGER.info(f"Validation samples: {len(val_loader.dataset)}")

    # Initialize trainer
    LOGGER.info("Initializing RoboMaster trainer...")
    trainer = RoboMasterTrainer(config)

    # Load models
    trainer.load_models(
        student_cfg=args.student_cfg,
        teacher_weights=args.teacher_weights
    )

    # Setup training
    trainer.setup_training(train_loader, val_loader)

    # Perform initial analysis if requested
    if args.analyze_attention:
        LOGGER.info("Performing initial attention analysis...")
        # Get sample images for analysis
        sample_images = []
        for batch_images, _ in val_loader:
            if len(sample_images) >= 10:
                break
            for img in batch_images[:2]:  # Take first 2 images from batch
                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype('uint8')
                sample_images.append(img_np)

        trainer.analyze_model_attention(
            sample_images,
            str(save_dir / 'initial_attention_analysis')
        )

    # Start training
    LOGGER.info(f"Starting training for {args.epochs} epochs...")
    trainer.train(epochs=args.epochs, save_dir=str(save_dir))

    # Final analysis if requested
    if args.analyze_attention:
        LOGGER.info("Performing final attention analysis...")
        trainer.analyze_model_attention(
            sample_images,
            str(save_dir / 'final_attention_analysis')
        )

    LOGGER.info("Training completed successfully!")
    LOGGER.info(f"Best model saved to: {save_dir / 'best.pt'}")

    # Print summary
    print("\n" + "="*50)
    print("ROBOMASTER TRAINING SUMMARY")
    print("="*50)
    print(f"Student Model: {args.student_cfg}")
    if args.teacher_weights:
        print(f"Teacher Weights: {args.teacher_weights}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Final Results: {save_dir}")
    print(f"Best Fitness: {trainer.best_fitness:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()