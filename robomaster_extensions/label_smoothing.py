"""
Label Smoothing for Fine-grained Classification in RoboMaster

This module implements label smoothing techniques specifically designed for
fine-grained armor plate classification in RoboMaster. The goal is to:
1. Reduce intra-class distance
2. Increase inter-class distance
3. Improve model calibration for similar classes
4. Reduce misclassification between similar armor types
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from .data_augment.config import get_robomaster_config


class AdaptiveLabelSmoothingLoss(nn.Module):
    """
    Adaptive label smoothing loss that adjusts smoothing based on class similarity.

    For RoboMaster, armor plates from different robots may look very similar,
    so we apply stronger smoothing between similar classes and weaker smoothing
    between dissimilar classes.
    """

    def __init__(self,
                 num_classes: int = 8,
                 base_smoothing: float = 0.1,
                 similarity_matrix: Optional[torch.Tensor] = None,
                 temperature: float = 1.0,
                 reduction: str = 'mean'):
        """
        Initialize adaptive label smoothing loss.

        Args:
            num_classes: Number of armor classes
            base_smoothing: Base smoothing factor
            similarity_matrix: Pre-computed class similarity matrix
            temperature: Temperature for similarity-based smoothing
            reduction: Loss reduction method
        """
        super().__init__()
        self.num_classes = num_classes
        self.base_smoothing = base_smoothing
        self.temperature = temperature
        self.reduction = reduction

        # Load RoboMaster configuration
        self.config = get_robomaster_config()
        self.class_names = self.config.class_names

        # Initialize similarity matrix
        if similarity_matrix is not None:
            self.register_buffer('similarity_matrix', similarity_matrix)
        else:
            self.register_buffer('similarity_matrix', self._create_default_similarity_matrix())

    def _create_default_similarity_matrix(self) -> torch.Tensor:
        """
        Create default class similarity matrix for RoboMaster armor plates.

        Returns:
            Similarity matrix (num_classes, num_classes)
        """
        # Initialize with base similarity
        similarity = torch.ones(self.num_classes, self.num_classes) * 0.1

        # Set diagonal to 1.0 (self-similarity)
        similarity.fill_diagonal_(1.0)

        # Define higher similarities between similar robot types
        # Standard robots (3-7) are more similar to each other
        standard_classes = [3, 4, 5, 6, 7]
        for i in standard_classes:
            for j in standard_classes:
                if i != j:
                    similarity[i, j] = 0.6  # High similarity between standard robots

        # Hero and Engineer are somewhat similar (both special roles)
        similarity[1, 2] = similarity[2, 1] = 0.4

        # Sentry is unique, lower similarity to others
        for i in range(1, self.num_classes):
            similarity[0, i] = similarity[i, 0] = 0.2

        return similarity

    def create_smooth_labels(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Create smoothed labels based on class similarities.

        Args:
            targets: Ground truth class indices (batch_size,)

        Returns:
            Smoothed label distribution (batch_size, num_classes)
        """
        batch_size = targets.size(0)
        device = targets.device

        # Create one-hot encoding
        one_hot = torch.zeros(batch_size, self.num_classes, device=device)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        # Get similarity rows for target classes
        target_similarities = self.similarity_matrix[targets]  # (batch_size, num_classes)

        # Apply temperature scaling to similarities
        scaled_similarities = target_similarities / self.temperature
        similarity_weights = F.softmax(scaled_similarities, dim=1)

        # Create smoothed labels
        smooth_labels = (1.0 - self.base_smoothing) * one_hot + \
                       self.base_smoothing * similarity_weights

        return smooth_labels

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive label smoothing loss.

        Args:
            logits: Model predictions (batch_size, num_classes)
            targets: Ground truth class indices (batch_size,)

        Returns:
            Smoothed cross-entropy loss
        """
        # Create smoothed labels
        smooth_labels = self.create_smooth_labels(targets)

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=1)

        # Cross-entropy with smoothed labels
        loss = -torch.sum(smooth_labels * log_probs, dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CurriculumLabelSmoothing(nn.Module):
    """
    Curriculum-based label smoothing that adapts the smoothing factor
    during training based on model confidence and training progress.
    """

    def __init__(self,
                 num_classes: int = 8,
                 initial_smoothing: float = 0.2,
                 final_smoothing: float = 0.05,
                 warmup_epochs: int = 10,
                 total_epochs: int = 300):
        """
        Initialize curriculum label smoothing.

        Args:
            num_classes: Number of classes
            initial_smoothing: Starting smoothing factor
            final_smoothing: Ending smoothing factor
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
        """
        super().__init__()
        self.num_classes = num_classes
        self.initial_smoothing = initial_smoothing
        self.final_smoothing = final_smoothing
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

        self.current_epoch = 0
        self.current_smoothing = initial_smoothing

    def update_epoch(self, epoch: int):
        """Update current epoch and smoothing factor."""
        self.current_epoch = epoch

        if epoch < self.warmup_epochs:
            # Warmup phase: linearly increase smoothing
            progress = epoch / self.warmup_epochs
            self.current_smoothing = self.initial_smoothing * (1.0 - progress) + \
                                   self.final_smoothing * progress
        else:
            # Decay phase: exponential decay
            decay_progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            decay_factor = np.exp(-3.0 * decay_progress)  # Exponential decay
            self.current_smoothing = self.final_smoothing + \
                                   (self.initial_smoothing - self.final_smoothing) * decay_factor

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute curriculum label smoothing loss.

        Args:
            logits: Model predictions
            targets: Ground truth targets

        Returns:
            Smoothed loss
        """
        batch_size = targets.size(0)
        device = targets.device

        # Create one-hot encoding
        one_hot = torch.zeros(batch_size, self.num_classes, device=device)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        # Apply current smoothing
        smooth_labels = (1.0 - self.current_smoothing) * one_hot + \
                       self.current_smoothing / self.num_classes

        # Compute loss
        log_probs = F.log_softmax(logits, dim=1)
        loss = -torch.sum(smooth_labels * log_probs, dim=1)

        return loss.mean()


class ConfidenceAwareLabelSmoothing(nn.Module):
    """
    Confidence-aware label smoothing that adapts smoothing based on
    model's prediction confidence for each sample.
    """

    def __init__(self,
                 num_classes: int = 8,
                 base_smoothing: float = 0.1,
                 confidence_threshold: float = 0.8,
                 adaptive_factor: float = 2.0):
        """
        Initialize confidence-aware label smoothing.

        Args:
            num_classes: Number of classes
            base_smoothing: Base smoothing factor
            confidence_threshold: Confidence threshold for adaptation
            adaptive_factor: Factor for confidence-based adaptation
        """
        super().__init__()
        self.num_classes = num_classes
        self.base_smoothing = base_smoothing
        self.confidence_threshold = confidence_threshold
        self.adaptive_factor = adaptive_factor

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence-aware label smoothing loss.

        Args:
            logits: Model predictions
            targets: Ground truth targets

        Returns:
            Adaptive smoothed loss
        """
        batch_size = targets.size(0)
        device = targets.device

        # Compute prediction confidence
        probs = F.softmax(logits, dim=1)
        max_probs, predicted = torch.max(probs, dim=1)

        # Adapt smoothing based on confidence
        # Lower confidence -> higher smoothing
        confidence_factor = torch.clamp(1.0 - max_probs, min=0.1, max=1.0)
        adaptive_smoothing = self.base_smoothing * confidence_factor * self.adaptive_factor

        # Create one-hot encoding
        one_hot = torch.zeros(batch_size, self.num_classes, device=device)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        # Apply adaptive smoothing
        adaptive_smoothing = adaptive_smoothing.unsqueeze(1)  # (batch_size, 1)
        smooth_labels = (1.0 - adaptive_smoothing) * one_hot + \
                       adaptive_smoothing / self.num_classes

        # Compute loss
        log_probs = F.log_softmax(logits, dim=1)
        loss = -torch.sum(smooth_labels * log_probs, dim=1)

        return loss.mean()


class MixupLabelSmoothing(nn.Module):
    """
    Combine label smoothing with mixup augmentation for better
    fine-grained classification performance.
    """

    def __init__(self,
                 num_classes: int = 8,
                 smoothing: float = 0.1,
                 mixup_alpha: float = 0.4):
        """
        Initialize mixup label smoothing.

        Args:
            num_classes: Number of classes
            smoothing: Label smoothing factor
            mixup_alpha: Mixup alpha parameter
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.mixup_alpha = mixup_alpha

    def mixup_data(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup to input data and labels.

        Args:
            x: Input data
            y: Labels

        Returns:
            Mixed data, original labels, mixed labels, mixing coefficient
        """
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
        """
        Compute mixup loss with label smoothing.

        Args:
            logits: Model predictions
            y_a: First set of labels
            y_b: Second set of labels
            lam: Mixing coefficient

        Returns:
            Mixup loss with label smoothing
        """
        # Create smoothed labels for both sets
        smooth_labels_a = self.create_smooth_labels(y_a)
        smooth_labels_b = self.create_smooth_labels(y_b)

        # Mix the labels
        mixed_labels = lam * smooth_labels_a + (1 - lam) * smooth_labels_b

        # Compute loss
        log_probs = F.log_softmax(logits, dim=1)
        loss = -torch.sum(mixed_labels * log_probs, dim=1)

        return loss.mean()

    def create_smooth_labels(self, targets: torch.Tensor) -> torch.Tensor:
        """Create smoothed one-hot labels."""
        batch_size = targets.size(0)
        device = targets.device

        one_hot = torch.zeros(batch_size, self.num_classes, device=device)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        smooth_labels = (1.0 - self.smoothing) * one_hot + \
                       self.smoothing / self.num_classes

        return smooth_labels

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute mixup label smoothing loss.

        Args:
            logits: Model predictions
            targets: Ground truth targets
            inputs: Input data (for mixup, if None, only apply label smoothing)

        Returns:
            Loss value
        """
        if inputs is not None and self.training:
            # Apply mixup during training
            mixed_x, y_a, y_b, lam = self.mixup_data(inputs, targets)
            # Note: You would need to re-run the model with mixed_x
            # This is typically done at the training loop level
            return self.mixup_criterion(logits, y_a, y_b, lam)
        else:
            # Standard label smoothing
            smooth_labels = self.create_smooth_labels(targets)
            log_probs = F.log_softmax(logits, dim=1)
            loss = -torch.sum(smooth_labels * log_probs, dim=1)
            return loss.mean()


class RoboMasterLabelSmoothingManager:
    """
    Manager class that combines different label smoothing strategies
    for optimal RoboMaster armor plate classification.
    """

    def __init__(self,
                 strategy: str = 'adaptive',
                 num_classes: int = 8,
                 **kwargs):
        """
        Initialize label smoothing manager.

        Args:
            strategy: Smoothing strategy ('adaptive', 'curriculum', 'confidence', 'mixup')
            num_classes: Number of armor classes
            **kwargs: Additional parameters for specific strategies
        """
        self.strategy = strategy
        self.num_classes = num_classes

        # Load RoboMaster configuration for class names
        self.config = get_robomaster_config()
        self.class_names = self.config.class_names

        if strategy == 'adaptive':
            self.loss_fn = AdaptiveLabelSmoothingLoss(num_classes, **kwargs)
        elif strategy == 'curriculum':
            self.loss_fn = CurriculumLabelSmoothing(num_classes, **kwargs)
        elif strategy == 'confidence':
            self.loss_fn = ConfidenceAwareLabelSmoothing(num_classes, **kwargs)
        elif strategy == 'mixup':
            self.loss_fn = MixupLabelSmoothing(num_classes, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute label smoothing loss."""
        return self.loss_fn(logits, targets, **kwargs)

    def update_epoch(self, epoch: int):
        """Update epoch for curriculum learning strategies."""
        if hasattr(self.loss_fn, 'update_epoch'):
            self.loss_fn.update_epoch(epoch)


def test_label_smoothing():
    """Test function for label smoothing implementations."""

    # Test parameters
    batch_size = 16
    num_classes = 8

    # Create dummy data
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    print("Testing Label Smoothing Implementations for RoboMaster")
    print("-" * 60)

    # Test adaptive label smoothing
    adaptive_loss = AdaptiveLabelSmoothingLoss(num_classes, base_smoothing=0.1)
    adaptive_result = adaptive_loss(logits, targets)
    print(f"Adaptive Label Smoothing Loss: {adaptive_result.item():.4f}")

    # Test curriculum label smoothing
    curriculum_loss = CurriculumLabelSmoothing(num_classes)
    curriculum_loss.update_epoch(50)  # Simulate mid-training
    curriculum_result = curriculum_loss(logits, targets)
    print(f"Curriculum Label Smoothing Loss: {curriculum_result.item():.4f}")
    print(f"Current smoothing factor: {curriculum_loss.current_smoothing:.4f}")

    # Test confidence-aware label smoothing
    confidence_loss = ConfidenceAwareLabelSmoothing(num_classes)
    confidence_result = confidence_loss(logits, targets)
    print(f"Confidence-Aware Label Smoothing Loss: {confidence_result.item():.4f}")

    # Test mixup label smoothing
    mixup_loss = MixupLabelSmoothing(num_classes)
    mixup_result = mixup_loss(logits, targets)
    print(f"Mixup Label Smoothing Loss: {mixup_result.item():.4f}")

    # Test manager
    manager = RoboMasterLabelSmoothingManager('adaptive', num_classes)
    manager_result = manager.compute_loss(logits, targets)
    print(f"Manager (Adaptive) Loss: {manager_result.item():.4f}")

    print("\nAll label smoothing tests completed successfully!")

    return {
        'adaptive': adaptive_result,
        'curriculum': curriculum_result,
        'confidence': confidence_result,
        'mixup': mixup_result,
        'manager': manager_result
    }


if __name__ == "__main__":
    test_label_smoothing()