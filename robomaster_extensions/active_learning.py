"""
Active Learning for RoboMaster Armor Plate Detection

This module implements active learning strategies to efficiently select
the most valuable samples for annotation, reducing labeling costs while
maintaining model performance.

Key strategies:
1. Uncertainty-based sampling using Shannon entropy
2. Diversity-based sampling to avoid redundant samples
3. Hybrid approaches combining uncertainty and diversity
4. Budget-aware selection for optimal annotation efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import cv2
from .config import get_robomaster_config
from pathlib import Path
import json
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


class UncertaintyEstimator:
    """
    Estimate prediction uncertainty using various methods.
    """

    def __init__(self, method: str = 'entropy'):
        """
        Initialize uncertainty estimator.

        Args:
            method: Uncertainty estimation method ('entropy', 'variance', 'disagreement')
        """
        self.method = method

    def shannon_entropy(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Compute Shannon entropy for classification uncertainty.

        Args:
            probabilities: Class probabilities (batch_size, num_classes)

        Returns:
            Entropy values (batch_size,)
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        log_probs = torch.log(probabilities + eps)
        entropy = -torch.sum(probabilities * log_probs, dim=1)
        return entropy

    def prediction_variance(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute prediction variance from multiple forward passes (Monte Carlo Dropout).

        Args:
            predictions: List of prediction tensors from multiple passes

        Returns:
            Variance values
        """
        # Stack predictions
        stacked_preds = torch.stack(predictions, dim=0)  # (num_passes, batch_size, ...)

        # Compute variance across passes
        variance = torch.var(stacked_preds, dim=0)

        # Return mean variance across classes/features
        return torch.mean(variance, dim=1)

    def model_disagreement(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute disagreement between multiple models or predictions.

        Args:
            predictions: List of predictions from different models

        Returns:
            Disagreement scores
        """
        if len(predictions) < 2:
            raise ValueError("Need at least 2 predictions for disagreement")

        disagreements = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                # KL divergence between predictions
                pred_i = F.softmax(predictions[i], dim=1)
                pred_j = F.softmax(predictions[j], dim=1)

                kl_div = F.kl_div(torch.log(pred_i + 1e-8), pred_j, reduction='none')
                disagreements.append(torch.sum(kl_div, dim=1))

        # Average disagreement
        avg_disagreement = torch.stack(disagreements).mean(dim=0)
        return avg_disagreement

    def compute_uncertainty(self, predictions: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Compute uncertainty based on the specified method.

        Args:
            predictions: Model predictions (single tensor or list for ensemble methods)

        Returns:
            Uncertainty scores
        """
        if self.method == 'entropy':
            if isinstance(predictions, list):
                predictions = predictions[0]
            probs = F.softmax(predictions, dim=1)
            return self.shannon_entropy(probs)

        elif self.method == 'variance':
            if not isinstance(predictions, list):
                raise ValueError("Variance method requires list of predictions")
            return self.prediction_variance(predictions)

        elif self.method == 'disagreement':
            if not isinstance(predictions, list):
                raise ValueError("Disagreement method requires list of predictions")
            return self.model_disagreement(predictions)

        else:
            raise ValueError(f"Unknown uncertainty method: {self.method}")


class DiversitySelector:
    """
    Select diverse samples to avoid redundant annotations.
    """

    def __init__(self, method: str = 'kmeans', n_clusters: int = 10):
        """
        Initialize diversity selector.

        Args:
            method: Diversity selection method ('kmeans', 'coreset', 'farthest')
            n_clusters: Number of clusters for clustering-based methods
        """
        self.method = method
        self.n_clusters = n_clusters

    def extract_features(self, images: List[np.ndarray], model=None) -> np.ndarray:
        """
        Extract features for diversity computation.

        Args:
            images: List of input images
            model: Feature extraction model (if None, use simple features)

        Returns:
            Feature matrix (num_samples, feature_dim)
        """
        if model is not None:
            # Use deep features from model
            features = []
            with torch.no_grad():
                for img in images:
                    # Convert to tensor and extract features
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
                    feat = model.extract_features(img_tensor)
                    features.append(feat.cpu().numpy().flatten())
            return np.array(features)
        else:
            # Use simple hand-crafted features
            features = []
            for img in images:
                # Histogram-based features
                hist_r = cv2.calcHist([img], [0], None, [32], [0, 256])
                hist_g = cv2.calcHist([img], [1], None, [32], [0, 256])
                hist_b = cv2.calcHist([img], [2], None, [32], [0, 256])

                # Edge features
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

                # Combine features
                feat = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten(), [edge_density]])
                features.append(feat)

            return np.array(features)

    def kmeans_diversity(self, features: np.ndarray, n_select: int) -> List[int]:
        """
        Select diverse samples using K-means clustering.

        Args:
            features: Feature matrix
            n_select: Number of samples to select

        Returns:
            Indices of selected samples
        """
        # Perform K-means clustering
        n_clusters = min(self.n_clusters, len(features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)

        # Select samples closest to cluster centers
        selected_indices = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if not np.any(cluster_mask):
                continue

            cluster_features = features[cluster_mask]
            cluster_center = kmeans.cluster_centers_[cluster_id]

            # Find closest sample to center
            distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
            closest_idx = np.argmin(distances)

            # Convert back to original index
            original_indices = np.where(cluster_mask)[0]
            selected_indices.append(original_indices[closest_idx])

        # If we need more samples, select based on distance to centers
        if len(selected_indices) < n_select:
            remaining = n_select - len(selected_indices)
            all_distances = []

            for i, feat in enumerate(features):
                if i in selected_indices:
                    all_distances.append(float('-inf'))
                else:
                    # Distance to nearest selected sample
                    if selected_indices:
                        min_dist = min(np.linalg.norm(feat - features[j]) for j in selected_indices)
                    else:
                        min_dist = 0
                    all_distances.append(min_dist)

            # Select samples with largest distances
            additional_indices = np.argsort(all_distances)[-remaining:]
            selected_indices.extend(additional_indices.tolist())

        return selected_indices[:n_select]

    def farthest_point_sampling(self, features: np.ndarray, n_select: int) -> List[int]:
        """
        Select diverse samples using farthest point sampling.

        Args:
            features: Feature matrix
            n_select: Number of samples to select

        Returns:
            Indices of selected samples
        """
        n_samples = len(features)
        selected = []

        # Start with random point
        selected.append(np.random.randint(n_samples))

        for _ in range(1, min(n_select, n_samples)):
            distances = []

            for i in range(n_samples):
                if i in selected:
                    distances.append(0)
                else:
                    # Minimum distance to any selected point
                    min_dist = min(np.linalg.norm(features[i] - features[j]) for j in selected)
                    distances.append(min_dist)

            # Select point with maximum distance
            farthest_idx = np.argmax(distances)
            selected.append(farthest_idx)

        return selected

    def select_diverse_samples(self, images: List[np.ndarray], n_select: int, model=None) -> List[int]:
        """
        Select diverse samples.

        Args:
            images: Input images
            n_select: Number of samples to select
            model: Optional model for feature extraction

        Returns:
            Indices of selected samples
        """
        features = self.extract_features(images, model)

        if self.method == 'kmeans':
            return self.kmeans_diversity(features, n_select)
        elif self.method == 'farthest':
            return self.farthest_point_sampling(features, n_select)
        else:
            raise ValueError(f"Unknown diversity method: {self.method}")


class ActiveLearningSelector:
    """
    Main active learning selector that combines uncertainty and diversity.
    """

    def __init__(self,
                 uncertainty_method: str = 'entropy',
                 diversity_method: str = 'kmeans',
                 uncertainty_weight: float = 0.7,
                 diversity_weight: float = 0.3,
                 batch_size: int = 100):
        """
        Initialize active learning selector.

        Args:
            uncertainty_method: Method for uncertainty estimation
            diversity_method: Method for diversity selection
            uncertainty_weight: Weight for uncertainty in hybrid selection
            diversity_weight: Weight for diversity in hybrid selection
            batch_size: Batch size for selection
        """
        self.uncertainty_estimator = UncertaintyEstimator(uncertainty_method)
        self.diversity_selector = DiversitySelector(diversity_method)
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.batch_size = batch_size

        # Statistics tracking
        self.selection_history = []
        self.performance_history = []

    def compute_sample_scores(self,
                            images: List[np.ndarray],
                            predictions: Union[torch.Tensor, List[torch.Tensor]],
                            model=None) -> np.ndarray:
        """
        Compute scores for sample selection combining uncertainty and diversity.

        Args:
            images: Input images
            predictions: Model predictions
            model: Model for feature extraction

        Returns:
            Sample scores
        """
        # Compute uncertainty scores
        uncertainty_scores = self.uncertainty_estimator.compute_uncertainty(predictions)
        uncertainty_scores = uncertainty_scores.cpu().numpy()

        # Normalize uncertainty scores
        uncertainty_scores = (uncertainty_scores - uncertainty_scores.min()) / \
                           (uncertainty_scores.max() - uncertainty_scores.min() + 1e-8)

        # Compute diversity scores
        features = self.diversity_selector.extract_features(images, model)

        # Simple diversity score: average distance to other samples
        diversity_scores = []
        for i in range(len(features)):
            distances = [np.linalg.norm(features[i] - features[j])
                        for j in range(len(features)) if i != j]
            avg_distance = np.mean(distances) if distances else 0
            diversity_scores.append(avg_distance)

        diversity_scores = np.array(diversity_scores)

        # Normalize diversity scores
        if diversity_scores.max() > diversity_scores.min():
            diversity_scores = (diversity_scores - diversity_scores.min()) / \
                             (diversity_scores.max() - diversity_scores.min())

        # Combine scores
        combined_scores = (self.uncertainty_weight * uncertainty_scores +
                          self.diversity_weight * diversity_scores)

        return combined_scores

    def select_samples(self,
                      images: List[np.ndarray],
                      predictions: Union[torch.Tensor, List[torch.Tensor]],
                      n_select: int,
                      model=None,
                      method: str = 'hybrid') -> Dict:
        """
        Select samples for annotation.

        Args:
            images: Input images
            predictions: Model predictions
            n_select: Number of samples to select
            model: Model for feature extraction
            method: Selection method ('uncertainty', 'diversity', 'hybrid')

        Returns:
            Selection results dictionary
        """
        if method == 'uncertainty':
            # Pure uncertainty-based selection
            uncertainty_scores = self.uncertainty_estimator.compute_uncertainty(predictions)
            uncertainty_scores = uncertainty_scores.cpu().numpy()
            selected_indices = np.argsort(uncertainty_scores)[-n_select:].tolist()

        elif method == 'diversity':
            # Pure diversity-based selection
            selected_indices = self.diversity_selector.select_diverse_samples(images, n_select, model)

        elif method == 'hybrid':
            # Hybrid selection
            sample_scores = self.compute_sample_scores(images, predictions, model)
            selected_indices = np.argsort(sample_scores)[-n_select:].tolist()

        else:
            raise ValueError(f"Unknown selection method: {method}")

        # Compute statistics for selected samples
        uncertainty_scores = self.uncertainty_estimator.compute_uncertainty(predictions)
        uncertainty_scores = uncertainty_scores.cpu().numpy()

        selected_uncertainties = uncertainty_scores[selected_indices]

        results = {
            'selected_indices': selected_indices,
            'selected_uncertainties': selected_uncertainties,
            'mean_uncertainty': np.mean(selected_uncertainties),
            'std_uncertainty': np.std(selected_uncertainties),
            'total_samples': len(images),
            'selection_method': method
        }

        # Update selection history
        self.selection_history.append(results)

        return results

    def budget_aware_selection(self,
                             images: List[np.ndarray],
                             predictions: Union[torch.Tensor, List[torch.Tensor]],
                             annotation_costs: List[float],
                             budget: float,
                             model=None) -> Dict:
        """
        Select samples considering annotation budget and costs.

        Args:
            images: Input images
            predictions: Model predictions
            annotation_costs: Cost of annotating each sample
            budget: Available annotation budget
            model: Model for feature extraction

        Returns:
            Budget-aware selection results
        """
        # Compute sample scores
        sample_scores = self.compute_sample_scores(images, predictions, model)

        # Compute efficiency scores (value / cost)
        efficiency_scores = sample_scores / (np.array(annotation_costs) + 1e-8)

        # Greedy selection based on efficiency
        selected_indices = []
        total_cost = 0
        available_indices = list(range(len(images)))

        # Sort by efficiency (descending)
        sorted_indices = np.argsort(efficiency_scores)[::-1]

        for idx in sorted_indices:
            if total_cost + annotation_costs[idx] <= budget:
                selected_indices.append(idx)
                total_cost += annotation_costs[idx]
                available_indices.remove(idx)

        results = {
            'selected_indices': selected_indices,
            'total_cost': total_cost,
            'budget_utilization': total_cost / budget,
            'avg_efficiency': np.mean(efficiency_scores[selected_indices]) if selected_indices else 0,
            'num_selected': len(selected_indices)
        }

        return results

    def adaptive_selection(self,
                          images: List[np.ndarray],
                          predictions: Union[torch.Tensor, List[torch.Tensor]],
                          model_performance: float,
                          target_performance: float = 0.9,
                          model=None) -> Dict:
        """
        Adaptively select samples based on current model performance.

        Args:
            images: Input images
            predictions: Model predictions
            model_performance: Current model performance (e.g., mAP)
            target_performance: Target performance threshold
            model: Model for feature extraction

        Returns:
            Adaptive selection results
        """
        # Adjust selection strategy based on performance gap
        performance_gap = target_performance - model_performance

        if performance_gap > 0.1:
            # Large gap: focus on uncertainty
            method = 'uncertainty'
            n_select = min(self.batch_size * 2, len(images))
        elif performance_gap > 0.05:
            # Medium gap: balanced approach
            method = 'hybrid'
            n_select = self.batch_size
        else:
            # Small gap: focus on diversity
            method = 'diversity'
            n_select = max(self.batch_size // 2, 10)

        results = self.select_samples(images, predictions, n_select, model, method)
        results['performance_gap'] = performance_gap
        results['adaptive_method'] = method

        return results

    def save_selection_history(self, filepath: str):
        """Save selection history to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'selection_history': self.selection_history,
                'performance_history': self.performance_history
            }, f)

    def load_selection_history(self, filepath: str):
        """Load selection history from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.selection_history = data['selection_history']
            self.performance_history = data['performance_history']

    def plot_selection_statistics(self, save_path: Optional[str] = None):
        """Plot selection statistics over time."""
        if not self.selection_history:
            print("No selection history available")
            return

        iterations = range(len(self.selection_history))
        mean_uncertainties = [h['mean_uncertainty'] for h in self.selection_history]
        num_selected = [h.get('num_selected', len(h['selected_indices'])) for h in self.selection_history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot mean uncertainty
        ax1.plot(iterations, mean_uncertainties, 'b-o')
        ax1.set_xlabel('Selection Iteration')
        ax1.set_ylabel('Mean Uncertainty')
        ax1.set_title('Uncertainty Evolution')
        ax1.grid(True)

        # Plot number of samples selected
        ax2.bar(iterations, num_selected, alpha=0.7)
        ax2.set_xlabel('Selection Iteration')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Samples Selected per Iteration')
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def test_active_learning():
    """Test function for active learning components."""

    # Create dummy data
    batch_size = 100
    config = get_robomaster_config()
    num_classes = config.num_classes
    image_size = (640, 640, 3)

    # Dummy images (random)
    images = [np.random.randint(0, 255, image_size, dtype=np.uint8) for _ in range(batch_size)]

    # Dummy predictions
    predictions = torch.randn(batch_size, num_classes)

    print("Testing Active Learning Components")
    print("-" * 50)

    # Test uncertainty estimation
    uncertainty_estimator = UncertaintyEstimator('entropy')
    uncertainty_scores = uncertainty_estimator.compute_uncertainty(predictions)
    print(f"Uncertainty scores shape: {uncertainty_scores.shape}")
    print(f"Mean uncertainty: {uncertainty_scores.mean().item():.4f}")

    # Test diversity selection
    diversity_selector = DiversitySelector('kmeans', n_clusters=10)
    diverse_indices = diversity_selector.select_diverse_samples(images[:20], 10)
    print(f"Selected diverse samples: {len(diverse_indices)}")

    # Test active learning selector
    al_selector = ActiveLearningSelector()
    selection_results = al_selector.select_samples(images[:30], predictions[:30], 10)
    print(f"Active learning selected: {len(selection_results['selected_indices'])}")
    print(f"Mean uncertainty of selected: {selection_results['mean_uncertainty']:.4f}")

    # Test budget-aware selection
    annotation_costs = [1.0 + 0.5 * np.random.random() for _ in range(30)]  # Random costs
    budget_results = al_selector.budget_aware_selection(
        images[:30], predictions[:30], annotation_costs, budget=15.0)
    print(f"Budget-aware selected: {budget_results['num_selected']}")
    print(f"Budget utilization: {budget_results['budget_utilization']:.2%}")

    print("\nAll active learning tests completed successfully!")

    return {
        'uncertainty_scores': uncertainty_scores,
        'diverse_indices': diverse_indices,
        'selection_results': selection_results,
        'budget_results': budget_results
    }


if __name__ == "__main__":
    test_active_learning()