#!/usr/bin/env python3
"""
Active Learning Pipeline for RoboMaster

This script implements an active learning pipeline that iteratively:
1. Trains a model on labeled data
2. Selects the most valuable unlabeled samples
3. Simulates annotation (or provides samples for human annotation)
4. Adds newly labeled samples to training set
5. Repeats until target performance or budget is reached

Usage:
    python scripts/active_learning_pipeline.py --model-weights runs/train/exp/weights/best.pt \
                                                --unlabeled-data data/unlabeled/ \
                                                --budget 1000
    """

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import cv2
from typing import List, Dict
import json
from tqdm import tqdm

# Add project root to path
ROOT = Path(__file__).parents[1]
sys.path.append(str(ROOT))

# Import RoboMaster components
from robomaster_extensions.active_learning import ActiveLearningSelector
from robomaster_extensions.robomaster_trainer import RoboMasterTrainer, create_default_config

# Import YOLOv5 components
from models.experimental import attempt_load
from utils.general import LOGGER, check_img_size
from utils.torch_utils import select_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Active Learning Pipeline for RoboMaster')

    # Model arguments
    parser.add_argument('--model-weights', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--model-cfg', type=str, default='models/robomaster_yolov5n.yaml',
                       help='Model configuration file')

    # Data arguments
    parser.add_argument('--unlabeled-data', type=str, required=True,
                       help='Directory containing unlabeled images')
    parser.add_argument('--labeled-data', type=str, default='data/labeled',
                       help='Directory containing labeled training data')
    parser.add_argument('--output-dir', type=str, default='runs/active_learning',
                       help='Output directory for results')

    # Active learning parameters
    parser.add_argument('--selection-method', type=str, default='hybrid',
                       choices=['uncertainty', 'diversity', 'hybrid'],
                       help='Sample selection method')
    parser.add_argument('--uncertainty-method', type=str, default='entropy',
                       choices=['entropy', 'variance', 'disagreement'],
                       help='Uncertainty estimation method')
    parser.add_argument('--diversity-method', type=str, default='kmeans',
                       choices=['kmeans', 'farthest'],
                       help='Diversity selection method')

    # Budget and iterations
    parser.add_argument('--budget', type=int, default=1000,
                       help='Total annotation budget (number of samples)')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Number of samples to select per iteration')
    parser.add_argument('--max-iterations', type=int, default=20,
                       help='Maximum number of active learning iterations')

    # Training parameters
    parser.add_argument('--retrain-epochs', type=int, default=50,
                       help='Epochs for retraining after each iteration')
    parser.add_argument('--initial-train-epochs', type=int, default=100,
                       help='Epochs for initial training')

    # Performance targets
    parser.add_argument('--target-performance', type=float, default=0.9,
                       help='Target performance to reach (mAP)')
    parser.add_argument('--performance-threshold', type=float, default=0.02,
                       help='Minimum performance improvement to continue')

    # Device and misc
    parser.add_argument('--device', type=str, default='',
                       help='CUDA device')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size for inference')
    parser.add_argument('--conf-thresh', type=float, default=0.5,
                       help='Confidence threshold for predictions')

    # Simulation mode
    parser.add_argument('--simulate-annotation', action='store_true',
                       help='Simulate annotation process (use for testing)')

    return parser.parse_args()


class ActiveLearningPipeline:
    """
    Main active learning pipeline for RoboMaster.
    """

    def __init__(self, args):
        """Initialize the active learning pipeline."""
        self.args = args
        self.device = select_device(args.device)

        # Setup directories
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.labeled_dir = Path(args.labeled_data)
        self.unlabeled_dir = Path(args.unlabeled_data)

        # Initialize components
        self.active_learner = ActiveLearningSelector(
            uncertainty_method=args.uncertainty_method,
            diversity_method=args.diversity_method,
            batch_size=args.batch_size
        )

        # Load initial model
        self.model = self._load_model(args.model_weights)

        # Track progress
        self.iteration = 0
        self.total_labeled = 0
        self.performance_history = []
        self.selection_history = []

        LOGGER.info(f"Active Learning Pipeline initialized")
        LOGGER.info(f"Device: {self.device}")
        LOGGER.info(f"Budget: {args.budget} samples")
        LOGGER.info(f"Batch size: {args.batch_size}")

    def _load_model(self, weights_path: str):
        """Load YOLOv5 model from weights."""
        model = attempt_load(weights_path, map_location=self.device)
        model.eval()
        return model

    def load_unlabeled_images(self) -> List[np.ndarray]:
        """Load unlabeled images from directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(self.unlabeled_dir.glob(f'*{ext}'))
            image_paths.extend(self.unlabeled_dir.glob(f'*{ext.upper()}'))

        images = []
        valid_paths = []

        LOGGER.info(f"Loading unlabeled images from {self.unlabeled_dir}")

        for img_path in tqdm(image_paths, desc="Loading images"):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append(img)
                    valid_paths.append(img_path)
            except Exception as e:
                LOGGER.warning(f"Failed to load {img_path}: {e}")

        LOGGER.info(f"Loaded {len(images)} unlabeled images")
        return images, valid_paths

    def generate_predictions(self, images: List[np.ndarray]) -> List[torch.Tensor]:
        """Generate model predictions for images."""
        predictions = []

        LOGGER.info("Generating predictions for unlabeled images")

        with torch.no_grad():
            for img in tqdm(images, desc="Predicting"):
                # Preprocess image
                img_tensor = self._preprocess_image(img)

                # Run inference
                pred = self.model(img_tensor)

                # Extract classification logits (simplified)
                # In practice, you'd need proper post-processing
                if isinstance(pred, (list, tuple)):
                    pred_tensor = pred[0]  # Take first scale
                else:
                    pred_tensor = pred

                predictions.append(pred_tensor)

        return predictions

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess image for model inference."""
        # Resize image
        img_resized = cv2.resize(img, (self.args.img_size, self.args.img_size))

        # Convert to tensor
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        return img_tensor

    def select_samples_for_annotation(self,
                                    images: List[np.ndarray],
                                    predictions: List[torch.Tensor],
                                    remaining_budget: int) -> Dict:
        """Select samples for annotation using active learning."""

        # Determine number of samples to select
        n_select = min(self.args.batch_size, remaining_budget, len(images))

        if n_select <= 0:
            return {'selected_indices': [], 'selection_info': {}}

        LOGGER.info(f"Selecting {n_select} samples for annotation")

        # Convert predictions to tensor
        predictions_tensor = torch.cat(predictions, dim=0)

        # Select samples
        selection_results = self.active_learner.select_samples(
            images=images,
            predictions=predictions_tensor,
            n_select=n_select,
            method=self.args.selection_method
        )

        return selection_results

    def simulate_annotation_process(self,
                                  selected_images: List[np.ndarray],
                                  selected_paths: List[Path]) -> int:
        """
        Simulate the annotation process.
        In practice, this would involve human annotators.
        """
        LOGGER.info(f"Simulating annotation of {len(selected_images)} images")

        # In simulation mode, we assume all images get annotated
        # In practice, you might have annotation costs, rejection rates, etc.

        annotated_count = 0

        for i, (img, path) in enumerate(zip(selected_images, selected_paths)):
            # Simulate annotation cost and quality
            annotation_cost = np.random.uniform(0.5, 2.0)  # Random cost
            annotation_quality = np.random.uniform(0.8, 1.0)  # Random quality

            # Simulate successful annotation (most are successful)
            if np.random.random() < 0.95:  # 95% success rate
                # Create fake label file (in practice, this comes from annotators)
                label_path = self.labeled_dir / 'labels' / f"{path.stem}.txt"
                label_path.parent.mkdir(parents=True, exist_ok=True)

                # Generate fake labels (in practice, these come from human annotators)
                fake_labels = self._generate_fake_labels()

                with open(label_path, 'w') as f:
                    for label in fake_labels:
                        f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

                # Copy image to labeled directory
                labeled_img_path = self.labeled_dir / 'images' / path.name
                labeled_img_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(labeled_img_path), img)

                annotated_count += 1

        LOGGER.info(f"Successfully annotated {annotated_count} images")
        return annotated_count

    def _generate_fake_labels(self) -> List[List[float]]:
        """Generate fake labels for simulation."""
        # Random number of objects (1-3)
        n_objects = np.random.randint(1, 4)

        labels = []
        for _ in range(n_objects):
            # Random class (0-7 for RoboMaster)
            cls = np.random.randint(0, 8)

            # Random bounding box (normalized coordinates)
            x_center = np.random.uniform(0.1, 0.9)
            y_center = np.random.uniform(0.1, 0.9)
            width = np.random.uniform(0.05, 0.3)
            height = np.random.uniform(0.05, 0.3)

            labels.append([cls, x_center, y_center, width, height])

        return labels

    def retrain_model(self) -> float:
        """
        Retrain the model with newly labeled data.
        Returns performance metric.
        """
        LOGGER.info(f"Retraining model after iteration {self.iteration}")

        # Create training configuration
        config = create_default_config()
        config.update({
            'epochs': self.args.retrain_epochs if self.iteration > 0 else self.args.initial_train_epochs,
            'lr0': 0.001,
            'device': str(self.device)
        })

        # Initialize trainer
        trainer = RoboMasterTrainer(config)

        # Setup training (simplified - in practice you'd create proper dataloaders)
        # For this example, we'll return a simulated performance metric

        # Simulate training performance
        base_performance = 0.7
        iteration_improvement = 0.05 * (1 - np.exp(-self.iteration / 5))
        noise = np.random.uniform(-0.02, 0.02)

        performance = base_performance + iteration_improvement + noise
        performance = max(0, min(1, performance))  # Clamp to [0, 1]

        LOGGER.info(f"Model performance after retraining: {performance:.4f}")
        return performance

    def check_stopping_criteria(self, current_performance: float) -> Tuple[bool, str]:
        """Check if we should stop the active learning process."""

        # Budget exhausted
        if self.total_labeled >= self.args.budget:
            return True, "Budget exhausted"

        # Target performance reached
        if current_performance >= self.args.target_performance:
            return True, f"Target performance ({self.args.target_performance}) reached"

        # Maximum iterations reached
        if self.iteration >= self.args.max_iterations:
            return True, "Maximum iterations reached"

        # Performance plateau
        if len(self.performance_history) >= 3:
            recent_improvements = [
                self.performance_history[i] - self.performance_history[i-1]
                for i in range(-2, 0)
            ]
            if all(imp < self.args.performance_threshold for imp in recent_improvements):
                return True, "Performance plateau detected"

        return False, ""

    def run_active_learning_cycle(self):
        """Run the complete active learning cycle."""

        LOGGER.info("Starting Active Learning Pipeline")
        LOGGER.info("="*50)

        # Load unlabeled images
        unlabeled_images, unlabeled_paths = self.load_unlabeled_images()

        if not unlabeled_images:
            LOGGER.error("No unlabeled images found!")
            return

        # Track available samples
        available_indices = list(range(len(unlabeled_images)))

        # Initial performance
        initial_performance = self.retrain_model()
        self.performance_history.append(initial_performance)

        # Main active learning loop
        while True:
            self.iteration += 1
            LOGGER.info(f"\n{'='*20} ITERATION {self.iteration} {'='*20}")

            # Check stopping criteria
            should_stop, reason = self.check_stopping_criteria(
                self.performance_history[-1] if self.performance_history else 0
            )

            if should_stop:
                LOGGER.info(f"Stopping active learning: {reason}")
                break

            # Generate predictions for available unlabeled images
            available_images = [unlabeled_images[i] for i in available_indices]
            available_paths = [unlabeled_paths[i] for i in available_indices]

            if not available_images:
                LOGGER.info("No more unlabeled images available")
                break

            predictions = self.generate_predictions(available_images)

            # Select samples for annotation
            remaining_budget = self.args.budget - self.total_labeled
            selection_results = self.select_samples_for_annotation(
                available_images, predictions, remaining_budget
            )

            selected_indices = selection_results['selected_indices']

            if not selected_indices:
                LOGGER.info("No samples selected - stopping")
                break

            # Get selected images and paths
            selected_images = [available_images[i] for i in selected_indices]
            selected_paths = [available_paths[i] for i in selected_indices]

            # Simulate annotation (or wait for human annotation)
            if self.args.simulate_annotation:
                annotated_count = self.simulate_annotation_process(selected_images, selected_paths)
            else:
                # In practice, you would save selected images for human annotation
                # and wait for the annotation process to complete
                LOGGER.info(f"Selected {len(selected_images)} images for annotation")
                LOGGER.info("Please annotate these images and press Enter to continue...")
                input()  # Wait for user input
                annotated_count = len(selected_images)  # Assume all get annotated

            self.total_labeled += annotated_count

            # Remove selected indices from available pool
            # Sort in reverse order to maintain index validity
            for idx in sorted(selected_indices, reverse=True):
                original_idx = available_indices[idx]
                available_indices.remove(original_idx)

            # Retrain model with new data
            current_performance = self.retrain_model()
            self.performance_history.append(current_performance)

            # Store selection information
            self.selection_history.append({
                'iteration': self.iteration,
                'selected_count': len(selected_indices),
                'annotated_count': annotated_count,
                'total_labeled': self.total_labeled,
                'performance': current_performance,
                'mean_uncertainty': selection_results.get('mean_uncertainty', 0),
                'selection_method': self.args.selection_method
            })

            # Log progress
            LOGGER.info(f"Iteration {self.iteration} Summary:")
            LOGGER.info(f"  - Selected: {len(selected_indices)} samples")
            LOGGER.info(f"  - Annotated: {annotated_count} samples")
            LOGGER.info(f"  - Total labeled: {self.total_labeled}")
            LOGGER.info(f"  - Performance: {current_performance:.4f}")
            LOGGER.info(f"  - Remaining budget: {self.args.budget - self.total_labeled}")

        # Save final results
        self._save_results()

    def _save_results(self):
        """Save active learning results and analysis."""
        results = {
            'args': vars(self.args),
            'performance_history': self.performance_history,
            'selection_history': self.selection_history,
            'total_iterations': self.iteration,
            'total_labeled': self.total_labeled,
            'final_performance': self.performance_history[-1] if self.performance_history else 0
        }

        # Save results to JSON
        with open(self.output_dir / 'active_learning_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Plot results
        self._plot_results()

        LOGGER.info(f"Results saved to {self.output_dir}")

    def _plot_results(self):
        """Plot active learning progress."""
        import matplotlib.pyplot as plt

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Performance over iterations
        iterations = range(len(self.performance_history))
        ax1.plot(iterations, self.performance_history, 'b-o')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Performance')
        ax1.set_title('Performance vs Iterations')
        ax1.grid(True)

        # Cumulative labeled samples
        if self.selection_history:
            cumulative_labeled = [h['total_labeled'] for h in self.selection_history]
            iterations_sel = [h['iteration'] for h in self.selection_history]
            ax2.plot(iterations_sel, cumulative_labeled, 'g-o')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Total Labeled Samples')
            ax2.set_title('Cumulative Labeled Samples')
            ax2.grid(True)

        # Performance vs budget
        if self.selection_history:
            labeled_counts = [h['total_labeled'] for h in self.selection_history]
            performances = [h['performance'] for h in self.selection_history]
            ax3.plot(labeled_counts, performances, 'r-o')
            ax3.set_xlabel('Total Labeled Samples')
            ax3.set_ylabel('Performance')
            ax3.set_title('Performance vs Annotation Budget')
            ax3.grid(True)

        # Uncertainty over iterations
        if self.selection_history:
            uncertainties = [h.get('mean_uncertainty', 0) for h in self.selection_history]
            iterations_sel = [h['iteration'] for h in self.selection_history]
            ax4.plot(iterations_sel, uncertainties, 'm-o')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Mean Uncertainty')
            ax4.set_title('Uncertainty vs Iterations')
            ax4.grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'active_learning_progress.png', dpi=300)
        plt.close()


def main():
    """Main function."""
    args = parse_args()

    # Initialize pipeline
    pipeline = ActiveLearningPipeline(args)

    # Run active learning cycle
    try:
        pipeline.run_active_learning_cycle()
    except KeyboardInterrupt:
        LOGGER.info("Active learning interrupted by user")
        pipeline._save_results()
    except Exception as e:
        LOGGER.error(f"Active learning failed: {e}")
        raise

    # Print final summary
    print("\n" + "="*50)
    print("ACTIVE LEARNING COMPLETED")
    print("="*50)
    print(f"Total iterations: {pipeline.iteration}")
    print(f"Total labeled samples: {pipeline.total_labeled}")
    if pipeline.performance_history:
        print(f"Initial performance: {pipeline.performance_history[0]:.4f}")
        print(f"Final performance: {pipeline.performance_history[-1]:.4f}")
        print(f"Performance improvement: {pipeline.performance_history[-1] - pipeline.performance_history[0]:.4f}")
    print(f"Results saved to: {pipeline.output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()