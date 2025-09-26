"""
Progressive Knowledge Distillation for YOLOv5

This module implements progressive distillation strategy:
YOLOv5x -> YOLOv5l -> YOLOv5m -> YOLOv5s -> YOLOv5n

Each larger model serves as teacher for the next smaller model,
preventing the large gap between teacher and student models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from pathlib import Path
import yaml
import logging
from copy import deepcopy

# Import YOLOv5 components
import sys
sys.path.append('/Users/cr/yolov5')
from models.yolo import Model
from utils.torch_utils import de_parallel, ModelEMA
from utils.general import LOGGER


class FeatureAdaptationLayer(nn.Module):
    """
    Adaptation layer to match feature dimensions between teacher and student.
    """

    def __init__(self, student_channels: int, teacher_channels: int):
        super().__init__()
        self.adaptation = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, 1, bias=False),
            nn.BatchNorm2d(teacher_channels),
            nn.ReLU(inplace=True)
        ) if student_channels != teacher_channels else nn.Identity()

    def forward(self, x):
        return self.adaptation(x)


class ProgressiveDistillationLoss(nn.Module):
    """
    Loss function for progressive knowledge distillation.
    Combines original task loss with distillation loss.
    """

    def __init__(self,
                 alpha: float = 0.7,
                 temperature: float = 4.0,
                 feature_loss_weight: float = 0.5):
        """
        Initialize distillation loss.

        Args:
            alpha: Weight for distillation loss vs original loss
            temperature: Temperature for softmax in distillation
            feature_loss_weight: Weight for feature-level distillation
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.feature_loss_weight = feature_loss_weight

    def forward(self,
                student_outputs: List[torch.Tensor],
                teacher_outputs: List[torch.Tensor],
                student_features: Optional[List[torch.Tensor]] = None,
                teacher_features: Optional[List[torch.Tensor]] = None,
                targets: Optional[torch.Tensor] = None,
                original_loss: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute progressive distillation loss.

        Args:
            student_outputs: Student model predictions
            teacher_outputs: Teacher model predictions
            student_features: Student intermediate features
            teacher_features: Teacher intermediate features
            targets: Ground truth targets
            original_loss: Original task loss

        Returns:
            Dictionary of loss components
        """
        losses = {}
        total_loss = 0

        # Logits distillation loss
        logits_loss = 0
        for s_out, t_out in zip(student_outputs, teacher_outputs):
            # Reshape outputs for distillation
            s_logits = s_out[..., 5:].contiguous().view(-1, s_out.shape[-1] - 5)
            t_logits = t_out[..., 5:].contiguous().view(-1, t_out.shape[-1] - 5)

            # Apply temperature scaling
            s_probs = F.log_softmax(s_logits / self.temperature, dim=1)
            t_probs = F.softmax(t_logits / self.temperature, dim=1)

            # KL divergence loss
            logits_loss += F.kl_div(s_probs, t_probs, reduction='batchmean')

        logits_loss *= (self.temperature ** 2)
        losses['logits'] = logits_loss
        total_loss += logits_loss

        # Feature distillation loss
        if student_features is not None and teacher_features is not None:
            feature_loss = 0
            min_layers = min(len(student_features), len(teacher_features))

            for i in range(min_layers):
                s_feat = student_features[i]
                t_feat = teacher_features[i]

                # Spatial alignment if needed
                if s_feat.shape[2:] != t_feat.shape[2:]:
                    s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear')

                # Channel alignment if needed
                if s_feat.shape[1] != t_feat.shape[1]:
                    s_feat = F.adaptive_avg_pool2d(s_feat, (1, 1)).flatten(1)
                    t_feat = F.adaptive_avg_pool2d(t_feat, (1, 1)).flatten(1)
                    s_feat = s_feat.view(s_feat.size(0), -1)
                    t_feat = t_feat.view(t_feat.size(0), -1)

                # L2 distance loss
                feature_loss += F.mse_loss(s_feat, t_feat)

            feature_loss *= self.feature_loss_weight
            losses['feature'] = feature_loss
            total_loss += feature_loss

        # Combine with original loss
        if original_loss is not None:
            losses['original'] = original_loss
            total_loss = self.alpha * total_loss + (1 - self.alpha) * original_loss

        losses['total'] = total_loss
        return losses


class ProgressiveDistillationTrainer:
    """
    Progressive distillation trainer that sequentially trains smaller models
    using larger models as teachers.
    """

    def __init__(self,
                 model_configs: Dict[str, str],
                 distillation_config: Dict = None):
        """
        Initialize progressive distillation trainer.

        Args:
            model_configs: Dict mapping model names to config paths
            distillation_config: Distillation hyperparameters
        """
        self.model_configs = model_configs
        self.distillation_config = distillation_config or {
            'alpha': 0.7,
            'temperature': 4.0,
            'feature_loss_weight': 0.5
        }

        # Initialize distillation loss
        self.distillation_loss = ProgressiveDistillationLoss(**self.distillation_config)

        # Model order for progressive distillation
        self.model_order = ['yolov5x', 'yolov5l', 'yolov5m', 'yolov5s', 'yolov5n']

        # Storage for trained models
        self.trained_models = {}

    def load_model(self, config_path: str, device: str = 'cpu') -> Model:
        """Load YOLOv5 model from config."""
        try:
            # Load model configuration
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)

            # Create model
            model = Model(cfg=config_path, ch=3, nc=cfg.get('nc', 80))
            model = model.to(device)

            return model
        except Exception as e:
            LOGGER.error(f"Failed to load model from {config_path}: {e}")
            raise

    def extract_features(self, model: Model, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract intermediate features from YOLOv5 model."""
        features = []

        # Hook function to capture features
        def hook_fn(module, input, output):
            features.append(output)

        # Register hooks on key layers
        hooks = []
        for i, layer in enumerate(model.model):
            if hasattr(layer, 'cv1'):  # Conv layers in C3 blocks
                hook = layer.register_forward_hook(hook_fn)
                hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            _ = model(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return features

    def distill_model_pair(self,
                          teacher_model: Model,
                          student_model: Model,
                          train_loader,
                          val_loader,
                          epochs: int = 100,
                          device: str = 'cuda',
                          save_dir: str = 'runs/distill') -> Model:
        """
        Distill knowledge from teacher to student model.

        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            device: Training device
            save_dir: Directory to save checkpoints

        Returns:
            Trained student model
        """
        # Setup
        teacher_model.eval()
        student_model.train()

        # Move models to device
        teacher_model = teacher_model.to(device)
        student_model = student_model.to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=0.001, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Training loop
        best_loss = float('inf')

        for epoch in range(epochs):
            student_model.train()
            epoch_losses = {'total': 0, 'logits': 0, 'feature': 0, 'original': 0}

            for batch_idx, (images, targets) in enumerate(train_loader):
                images = images.to(device)
                targets = targets.to(device)

                # Teacher predictions (no gradients)
                with torch.no_grad():
                    teacher_outputs = teacher_model(images)
                    teacher_features = self.extract_features(teacher_model, images)

                # Student predictions
                student_outputs = student_model(images)
                student_features = self.extract_features(student_model, images)

                # Compute original loss (you would need to implement this based on your loss function)
                # original_loss = compute_yolo_loss(student_outputs, targets)
                original_loss = torch.tensor(0.0, device=device)  # Placeholder

                # Compute distillation loss
                loss_dict = self.distillation_loss(
                    student_outputs, teacher_outputs,
                    student_features, teacher_features,
                    targets, original_loss
                )

                # Backward pass
                optimizer.zero_grad()
                loss_dict['total'].backward()
                optimizer.step()

                # Accumulate losses
                for key, value in loss_dict.items():
                    epoch_losses[key] += value.item()

                # Log progress
                if batch_idx % 100 == 0:
                    LOGGER.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss_dict["total"].item():.4f}')

            # Update learning rate
            scheduler.step()

            # Validation
            val_loss = self.validate_model(student_model, val_loader, device)

            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint = {
                    'model': student_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_loss': best_loss
                }
                torch.save(checkpoint, f'{save_dir}/best_student.pt')

            # Log epoch results
            avg_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
            LOGGER.info(f'Epoch {epoch}: Train Loss = {avg_losses["total"]:.4f}, '
                       f'Val Loss = {val_loss:.4f}')

        return student_model

    def validate_model(self, model: Model, val_loader, device: str) -> float:
        """Validate model and return average loss."""
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                # Compute validation loss (placeholder)
                loss = torch.tensor(1.0)  # You would implement actual validation loss
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def run_progressive_distillation(self,
                                   train_loader,
                                   val_loader,
                                   target_model: str = 'yolov5n',
                                   base_weights: Optional[str] = None,
                                   device: str = 'cuda',
                                   save_dir: str = 'runs/progressive_distill') -> Model:
        """
        Run complete progressive distillation pipeline.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            target_model: Final target model size
            base_weights: Path to base model weights
            device: Training device
            save_dir: Save directory

        Returns:
            Final distilled model
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Find starting point in model order
        if target_model not in self.model_order:
            raise ValueError(f"Target model {target_model} not in supported models")

        target_idx = self.model_order.index(target_model)

        # Load or train the largest teacher model
        if base_weights:
            # Load pre-trained weights
            teacher_model = self.load_model(self.model_configs[self.model_order[0]], device)
            checkpoint = torch.load(base_weights, map_location=device)
            if 'model' in checkpoint:
                state_dict = checkpoint['model'].float().state_dict()
            else:
                state_dict = checkpoint.float().state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
            teacher_model.load_state_dict(state_dict)
            LOGGER.info(f"Loaded pre-trained weights from {base_weights}")
        else:
            # Train the largest model from scratch
            teacher_model = self.load_model(self.model_configs[self.model_order[0]], device)
            # You would implement normal training here
            LOGGER.info("Training largest model from scratch...")

        # Progressive distillation
        current_teacher = teacher_model

        for i in range(1, target_idx + 1):
            student_model_name = self.model_order[i]

            LOGGER.info(f"Distilling {self.model_order[i-1]} -> {student_model_name}")

            # Load student model
            student_model = self.load_model(self.model_configs[student_model_name], device)

            # Distill knowledge
            trained_student = self.distill_model_pair(
                current_teacher, student_model,
                train_loader, val_loader,
                epochs=100, device=device,
                save_dir=f"{save_dir}/{student_model_name}"
            )

            # Save intermediate model
            checkpoint = {
                'model': trained_student.state_dict(),
                'model_name': student_model_name,
                'distillation_config': self.distillation_config
            }
            torch.save(checkpoint, f"{save_dir}/{student_model_name}_distilled.pt")

            # Update teacher for next iteration
            current_teacher = trained_student
            self.trained_models[student_model_name] = trained_student

        LOGGER.info(f"Progressive distillation complete. Final model: {target_model}")
        return current_teacher

    def compare_models(self, test_loader, device: str = 'cuda') -> Dict:
        """Compare performance of all trained models."""
        results = {}

        for model_name, model in self.trained_models.items():
            model.eval()
            test_loss = self.validate_model(model, test_loader, device)

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            results[model_name] = {
                'test_loss': test_loss,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
            }

        return results


def create_model_configs() -> Dict[str, str]:
    """Create default model configuration mapping."""
    base_path = Path('/Users/cr/yolov5/models')

    return {
        'yolov5n': str(base_path / 'yolov5n.yaml'),
        'yolov5s': str(base_path / 'yolov5s.yaml'),
        'yolov5m': str(base_path / 'yolov5m.yaml'),
        'yolov5l': str(base_path / 'yolov5l.yaml'),
        'yolov5x': str(base_path / 'yolov5x.yaml'),
    }


def test_progressive_distillation():
    """Test function for progressive distillation."""

    # Create dummy data loaders (you would use real data in practice)
    batch_size = 16
    dummy_data = [(torch.randn(batch_size, 3, 640, 640),
                   torch.randn(batch_size, 10, 6)) for _ in range(10)]

    # Initialize trainer
    model_configs = create_model_configs()
    trainer = ProgressiveDistillationTrainer(model_configs)

    # Test model loading
    try:
        model = trainer.load_model(model_configs['yolov5n'])
        print(f"Successfully loaded model with {sum(p.numel() for p in model.parameters())} parameters")

        # Test feature extraction
        dummy_input = torch.randn(1, 3, 640, 640)
        features = trainer.extract_features(model, dummy_input)
        print(f"Extracted {len(features)} feature maps")

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_progressive_distillation()