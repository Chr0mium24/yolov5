"""
Grad-CAM Visualization for RoboMaster Armor Plate Detection

This module implements Grad-CAM (Gradient-weighted Class Activation Mapping)
to analyze and visualize which regions the model focuses on when making
predictions. This helps identify background bias and model behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


class GradCAM:
    """
    Grad-CAM implementation for YOLOv5 models.
    """

    def __init__(self, model, target_layers: List[str]):
        """
        Initialize Grad-CAM.

        Args:
            model: YOLOv5 model
            target_layers: Names of target layers for visualization
        """
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.hooks = []

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on target layers."""

        def save_gradient(name):
            def hook(grad):
                self.gradients[name] = grad
            return hook

        def save_activation(name):
            def hook(module, input, output):
                self.activations[name] = output
            return hook

        # Find and register hooks on target layers
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                # Forward hook for activations
                hook_f = module.register_forward_hook(save_activation(name))
                self.hooks.append(hook_f)

                # Backward hook for gradients
                hook_b = module.register_backward_hook(save_gradient(name))
                self.hooks.append(hook_b)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_cam(self,
                    input_tensor: torch.Tensor,
                    target_class: int = None,
                    target_layer: str = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for visualization
            target_layer: Target layer name

        Returns:
            Grad-CAM heatmap
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)

        # Select target layer
        if target_layer is None:
            target_layer = self.target_layers[0]

        # Select target class
        if target_class is None:
            # Use class with highest confidence
            if isinstance(output, (list, tuple)):
                # YOLOv5 returns list of outputs from different scales
                output_tensor = output[0]  # Use first scale
            else:
                output_tensor = output

            # Find highest confidence prediction
            probs = F.softmax(output_tensor.view(-1, output_tensor.shape[-1]), dim=1)
            target_class = torch.argmax(probs, dim=1)[0].item()

        # Compute gradients
        self.model.zero_grad()

        # Backward pass for target class
        if isinstance(output, (list, tuple)):
            score = output[0][0, :, :, :, 5 + target_class].sum()
        else:
            score = output[0, target_class]

        score.backward()

        # Get gradients and activations
        gradients = self.gradients[target_layer]  # (batch, channels, height, width)
        activations = self.activations[target_layer]  # (batch, channels, height, width)

        # Compute channel weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # Global average pooling

        # Generate CAM
        cam = torch.sum(weights * activations, dim=1)  # (batch, height, width)
        cam = F.relu(cam)  # ReLU to keep positive influence

        # Normalize CAM
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)

        return cam.detach().cpu().numpy()

    def visualize_cam(self,
                     image: np.ndarray,
                     cam: np.ndarray,
                     alpha: float = 0.4) -> np.ndarray:
        """
        Overlay CAM heatmap on original image.

        Args:
            image: Original image (H, W, C)
            cam: CAM heatmap (H, W)
            alpha: Overlay alpha

        Returns:
            Visualization image
        """
        # Resize CAM to match image size
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam.squeeze(), (w, h))

        # Convert to colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

        # Overlay on original image
        result = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

        return result


class GradCAMAnalyzer:
    """
    Analyzer for RoboMaster-specific Grad-CAM analysis.
    """

    def __init__(self, model):
        """
        Initialize Grad-CAM analyzer.

        Args:
            model: YOLOv5 model
        """
        self.model = model

        # Define important layers for analysis
        self.target_layers = [
            'model.24',  # Final detection layer
            'model.17',  # P3 detection layer
            'model.20',  # P4 detection layer
            'model.23',  # P5 detection layer
        ]

        self.gradcam = GradCAM(model, self.target_layers)

    def analyze_background_bias(self,
                               images: List[np.ndarray],
                               predictions: List[torch.Tensor],
                               save_dir: str = 'gradcam_analysis') -> Dict:
        """
        Analyze background bias using Grad-CAM.

        Args:
            images: List of input images
            predictions: List of model predictions
            save_dir: Directory to save analysis results

        Returns:
            Analysis results
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        results = {
            'sentry_bias_detected': [],
            'attention_maps': [],
            'bias_scores': []
        }

        for i, (image, pred) in enumerate(zip(images, predictions)):
            # Convert image to tensor
            input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            # Generate CAM for each class
            class_cams = {}
            for class_id in range(8):  # 8 RoboMaster classes
                try:
                    cam = self.gradcam.generate_cam(input_tensor, target_class=class_id)
                    class_cams[class_id] = cam.squeeze()
                except:
                    class_cams[class_id] = np.zeros((image.shape[0], image.shape[1]))

            # Analyze sentry bias
            sentry_cam = class_cams[0]  # Sentry class
            vehicle_cams = [class_cams[j] for j in range(1, 8)]  # Vehicle classes

            # Check if sentry attention overlaps with vehicle regions
            bias_detected = self._detect_bias_overlap(sentry_cam, vehicle_cams, threshold=0.5)
            results['sentry_bias_detected'].append(bias_detected)

            # Compute bias score
            bias_score = self._compute_bias_score(sentry_cam, vehicle_cams)
            results['bias_scores'].append(bias_score)

            # Save visualizations
            self._save_class_visualizations(image, class_cams, save_path / f'sample_{i}')

            results['attention_maps'].append(class_cams)

        # Summary statistics
        results['bias_detection_rate'] = np.mean(results['sentry_bias_detected'])
        results['mean_bias_score'] = np.mean(results['bias_scores'])

        return results

    def _detect_bias_overlap(self,
                           sentry_cam: np.ndarray,
                           vehicle_cams: List[np.ndarray],
                           threshold: float = 0.5) -> bool:
        """
        Detect if sentry attention overlaps significantly with vehicle regions.

        Args:
            sentry_cam: Sentry class attention map
            vehicle_cams: Vehicle class attention maps
            threshold: Overlap threshold

        Returns:
            True if bias detected
        """
        # Create binary masks
        sentry_mask = sentry_cam > np.percentile(sentry_cam, 75)

        bias_detected = False
        for vehicle_cam in vehicle_cams:
            vehicle_mask = vehicle_cam > np.percentile(vehicle_cam, 75)

            # Compute overlap
            overlap = np.logical_and(sentry_mask, vehicle_mask)
            overlap_ratio = np.sum(overlap) / (np.sum(sentry_mask) + 1e-8)

            if overlap_ratio > threshold:
                bias_detected = True
                break

        return bias_detected

    def _compute_bias_score(self,
                          sentry_cam: np.ndarray,
                          vehicle_cams: List[np.ndarray]) -> float:
        """
        Compute numerical bias score.

        Args:
            sentry_cam: Sentry attention map
            vehicle_cams: Vehicle attention maps

        Returns:
            Bias score (0-1, higher means more bias)
        """
        # Normalize attention maps
        sentry_norm = sentry_cam / (np.max(sentry_cam) + 1e-8)

        bias_scores = []
        for vehicle_cam in vehicle_cams:
            vehicle_norm = vehicle_cam / (np.max(vehicle_cam) + 1e-8)

            # Compute spatial correlation
            correlation = np.corrcoef(sentry_norm.flatten(), vehicle_norm.flatten())[0, 1]
            bias_scores.append(max(0, correlation))  # Only positive correlations

        return np.mean(bias_scores)

    def _save_class_visualizations(self,
                                 image: np.ndarray,
                                 class_cams: Dict[int, np.ndarray],
                                 save_dir: Path):
        """Save visualization for each class."""
        save_dir.mkdir(parents=True, exist_ok=True)

        class_names = {
            0: 'sentry', 1: 'hero', 2: 'engineer',
            3: 'standard_1', 4: 'standard_2', 5: 'standard_3',
            6: 'standard_4', 7: 'standard_5'
        }

        # Create subplot for all classes
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for class_id, cam in class_cams.items():
            # Generate visualization
            vis = self.gradcam.visualize_cam(image, cam)

            # Plot
            axes[class_id].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            axes[class_id].set_title(f'{class_names[class_id]} (Class {class_id})')
            axes[class_id].axis('off')

        plt.tight_layout()
        plt.savefig(save_dir / 'class_attention_maps.png', dpi=300, bbox_inches='tight')
        plt.close()

    def compare_attention_patterns(self,
                                 before_images: List[np.ndarray],
                                 after_images: List[np.ndarray],
                                 save_dir: str = 'attention_comparison') -> Dict:
        """
        Compare attention patterns before and after bias mitigation.

        Args:
            before_images: Images before bias mitigation
            after_images: Images after bias mitigation
            save_dir: Save directory

        Returns:
            Comparison results
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Analyze both sets
        before_results = self.analyze_background_bias(before_images, None, str(save_path / 'before'))
        after_results = self.analyze_background_bias(after_images, None, str(save_path / 'after'))

        # Compare results
        comparison = {
            'bias_reduction': before_results['bias_detection_rate'] - after_results['bias_detection_rate'],
            'score_improvement': before_results['mean_bias_score'] - after_results['mean_bias_score'],
            'before_bias_rate': before_results['bias_detection_rate'],
            'after_bias_rate': after_results['bias_detection_rate']
        }

        # Create comparison visualization
        self._plot_bias_comparison(before_results, after_results, save_path)

        return comparison

    def _plot_bias_comparison(self,
                            before_results: Dict,
                            after_results: Dict,
                            save_path: Path):
        """Plot bias comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Bias detection rates
        categories = ['Before', 'After']
        detection_rates = [before_results['bias_detection_rate'], after_results['bias_detection_rate']]

        ax1.bar(categories, detection_rates, color=['red', 'green'], alpha=0.7)
        ax1.set_ylabel('Bias Detection Rate')
        ax1.set_title('Bias Detection Rate Comparison')
        ax1.set_ylim(0, 1)

        # Bias scores
        bias_scores = [before_results['mean_bias_score'], after_results['mean_bias_score']]

        ax2.bar(categories, bias_scores, color=['red', 'green'], alpha=0.7)
        ax2.set_ylabel('Mean Bias Score')
        ax2.set_title('Mean Bias Score Comparison')

        plt.tight_layout()
        plt.savefig(save_path / 'bias_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        if hasattr(self, 'gradcam'):
            self.gradcam.remove_hooks()


def test_gradcam():
    """Test function for Grad-CAM implementation."""
    print("Testing Grad-CAM Implementation")
    print("-" * 40)

    # Create dummy model (simplified)
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.classifier = nn.Linear(128, 8)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    # Create dummy data
    model = DummyModel()
    model.eval()

    # Dummy image
    image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    print("Created dummy model and data")

    # Test basic Grad-CAM
    try:
        gradcam = GradCAM(model, ['conv2'])
        cam = gradcam.generate_cam(input_tensor, target_class=0)
        print(f"Generated CAM with shape: {cam.shape}")

        # Test visualization
        vis = gradcam.visualize_cam(image, cam)
        print(f"Generated visualization with shape: {vis.shape}")

        gradcam.remove_hooks()
        print("Grad-CAM test completed successfully")

    except Exception as e:
        print(f"Grad-CAM test failed: {e}")

    return True


if __name__ == "__main__":
    test_gradcam()