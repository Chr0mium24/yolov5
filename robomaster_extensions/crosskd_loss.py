"""
CrossKD: Cross-Task Knowledge Distillation for Object Detection

This module implements the CrossKD method specifically designed for object detection tasks.
Unlike traditional KL divergence-based distillation, CrossKD uses cross-task learning
to achieve better knowledge transfer and can make students outperform teachers.

Reference: CrossKD: Cross-Task Knowledge Distillation for Object Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
import math


class CrossKDLoss(nn.Module):
    """
    CrossKD loss function for object detection knowledge distillation.

    The key insight is to use different task formulations between teacher and student
    to enable more effective knowledge transfer.
    """

    def __init__(self,
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 temperature: float = 4.0,
                 iou_threshold: float = 0.5,
                 conf_threshold: float = 0.5,
                 feature_adaptation: bool = True):
        """
        Initialize CrossKD loss.

        Args:
            alpha: Weight for classification distillation
            beta: Weight for localization distillation
            temperature: Temperature for knowledge distillation
            iou_threshold: IoU threshold for positive/negative assignment
            conf_threshold: Confidence threshold for teacher predictions
            feature_adaptation: Whether to use feature adaptation layers
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.feature_adaptation = feature_adaptation

        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')

    def compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between two sets of boxes.

        Args:
            box1: (N, 4) boxes in (x1, y1, x2, y2) format
            box2: (M, 4) boxes in (x1, y1, x2, y2) format

        Returns:
            (N, M) IoU matrix
        """
        # Expand dimensions for broadcasting
        box1 = box1.unsqueeze(1)  # (N, 1, 4)
        box2 = box2.unsqueeze(0)  # (1, M, 4)

        # Compute intersection
        inter_min = torch.max(box1[..., :2], box2[..., :2])  # (N, M, 2)
        inter_max = torch.min(box1[..., 2:], box2[..., 2:])  # (N, M, 2)
        inter_wh = torch.clamp(inter_max - inter_min, min=0)  # (N, M, 2)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]  # (N, M)

        # Compute areas
        area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])  # (N, 1)
        area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])  # (1, M)

        # Compute union and IoU
        union_area = area1 + area2 - inter_area
        iou = inter_area / (union_area + 1e-7)

        return iou.squeeze()

    def xywh2xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from (x_center, y_center, width, height) to (x1, y1, x2, y2)."""
        x_c, y_c, w, h = boxes.unbind(-1)
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def filter_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Filter teacher predictions based on confidence threshold.

        Args:
            predictions: (batch_size, num_anchors, 4+1+num_classes)

        Returns:
            Filtered predictions
        """
        batch_size = predictions.shape[0]
        filtered_preds = []

        for b in range(batch_size):
            pred = predictions[b]  # (num_anchors, 4+1+num_classes)

            # Extract confidence scores
            conf = pred[:, 4]  # (num_anchors,)

            # Filter by confidence
            mask = conf > self.conf_threshold
            filtered_pred = pred[mask]

            if len(filtered_pred) == 0:
                # Keep at least one prediction if all filtered out
                max_conf_idx = torch.argmax(conf)
                filtered_pred = pred[max_conf_idx:max_conf_idx+1]

            filtered_preds.append(filtered_pred)

        return filtered_preds

    def cross_task_assignment(self,
                            student_preds: List[torch.Tensor],
                            teacher_preds: List[torch.Tensor]) -> List[Dict]:
        """
        Perform cross-task assignment between student and teacher predictions.

        Args:
            student_preds: List of student predictions per batch
            teacher_preds: List of teacher predictions per batch

        Returns:
            Assignment information for each batch
        """
        assignments = []

        for s_pred, t_pred in zip(student_preds, teacher_preds):
            if len(s_pred) == 0 or len(t_pred) == 0:
                assignments.append({
                    'student_indices': torch.tensor([], dtype=torch.long),
                    'teacher_indices': torch.tensor([], dtype=torch.long),
                    'ious': torch.tensor([])
                })
                continue

            # Extract boxes
            s_boxes = self.xywh2xyxy(s_pred[:, :4])  # Student boxes
            t_boxes = self.xywh2xyxy(t_pred[:, :4])  # Teacher boxes

            # Compute IoU matrix
            iou_matrix = self.compute_iou(s_boxes, t_boxes)  # (num_student, num_teacher)

            # Find best matches (Hungarian-like assignment)
            student_indices = []
            teacher_indices = []
            match_ious = []

            # Greedy assignment based on IoU
            used_teachers = set()
            for s_idx in range(len(s_pred)):
                # Find best teacher match for this student
                available_teachers = [t for t in range(len(t_pred)) if t not in used_teachers]
                if not available_teachers:
                    break

                best_iou = -1
                best_teacher = -1
                for t_idx in available_teachers:
                    if iou_matrix[s_idx, t_idx] > best_iou:
                        best_iou = iou_matrix[s_idx, t_idx].item()
                        best_teacher = t_idx

                if best_iou > self.iou_threshold:
                    student_indices.append(s_idx)
                    teacher_indices.append(best_teacher)
                    match_ious.append(best_iou)
                    used_teachers.add(best_teacher)

            assignments.append({
                'student_indices': torch.tensor(student_indices, dtype=torch.long),
                'teacher_indices': torch.tensor(teacher_indices, dtype=torch.long),
                'ious': torch.tensor(match_ious)
            })

        return assignments

    def classification_distillation_loss(self,
                                       student_cls: torch.Tensor,
                                       teacher_cls: torch.Tensor) -> torch.Tensor:
        """
        Compute classification knowledge distillation loss.

        Args:
            student_cls: Student classification logits
            teacher_cls: Teacher classification logits

        Returns:
            Classification distillation loss
        """
        # Apply temperature scaling
        student_soft = F.log_softmax(student_cls / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_cls / self.temperature, dim=-1)

        # KL divergence loss
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')

        return kl_loss * (self.temperature ** 2)

    def localization_distillation_loss(self,
                                     student_boxes: torch.Tensor,
                                     teacher_boxes: torch.Tensor,
                                     ious: torch.Tensor) -> torch.Tensor:
        """
        Compute localization knowledge distillation loss.

        Args:
            student_boxes: Student box predictions
            teacher_boxes: Teacher box predictions
            ious: IoU values for weighting

        Returns:
            Localization distillation loss
        """
        # IoU-weighted L1 loss for box regression
        box_diff = torch.abs(student_boxes - teacher_boxes)
        iou_weights = ious.unsqueeze(-1).expand_as(box_diff)

        # Apply IoU weighting
        weighted_loss = box_diff * iou_weights

        return weighted_loss.mean()

    def confidence_distillation_loss(self,
                                   student_conf: torch.Tensor,
                                   teacher_conf: torch.Tensor,
                                   ious: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence knowledge distillation loss.

        Args:
            student_conf: Student confidence scores
            teacher_conf: Teacher confidence scores
            ious: IoU values for target confidence

        Returns:
            Confidence distillation loss
        """
        # Use IoU as target for confidence learning
        iou_targets = ious.detach()

        # MSE loss between student confidence and IoU-calibrated teacher confidence
        teacher_calibrated = teacher_conf * iou_targets

        loss = F.mse_loss(student_conf, teacher_calibrated)

        return loss

    def forward(self,
                student_outputs: List[torch.Tensor],
                teacher_outputs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute CrossKD loss.

        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs

        Returns:
            Dictionary of loss components
        """
        device = student_outputs[0].device
        batch_size = student_outputs[0].shape[0]

        # Initialize losses
        total_cls_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        total_conf_loss = torch.tensor(0.0, device=device)
        num_matches = 0

        # Process each detection head
        for s_out, t_out in zip(student_outputs, teacher_outputs):
            # Reshape outputs: (batch, anchors, grid_h, grid_w, 4+1+classes)
            # -> (batch, anchors*grid_h*grid_w, 4+1+classes)
            s_out = s_out.view(batch_size, -1, s_out.shape[-1])
            t_out = t_out.view(batch_size, -1, t_out.shape[-1])

            # Filter teacher predictions by confidence
            filtered_teacher_preds = self.filter_predictions(t_out)

            # Convert student outputs to list format
            student_preds = [s_out[b] for b in range(batch_size)]

            # Perform cross-task assignment
            assignments = self.cross_task_assignment(student_preds, filtered_teacher_preds)

            # Compute losses for matched predictions
            for b, assignment in enumerate(assignments):
                s_indices = assignment['student_indices']
                t_indices = assignment['teacher_indices']
                match_ious = assignment['ious']

                if len(s_indices) == 0:
                    continue

                # Extract matched predictions
                s_matched = student_preds[b][s_indices]  # (num_matches, 4+1+classes)
                t_matched = filtered_teacher_preds[b][t_indices]  # (num_matches, 4+1+classes)

                # Extract components
                s_boxes = s_matched[:, :4]
                s_conf = s_matched[:, 4]
                s_cls = s_matched[:, 5:]

                t_boxes = t_matched[:, :4]
                t_conf = t_matched[:, 4]
                t_cls = t_matched[:, 5:]

                # Classification distillation
                cls_loss = self.classification_distillation_loss(s_cls, t_cls)
                total_cls_loss += cls_loss

                # Localization distillation
                box_loss = self.localization_distillation_loss(s_boxes, t_boxes, match_ious)
                total_box_loss += box_loss

                # Confidence distillation
                conf_loss = self.confidence_distillation_loss(s_conf, t_conf, match_ious)
                total_conf_loss += conf_loss

                num_matches += len(s_indices)

        # Average losses
        if num_matches > 0:
            total_cls_loss /= num_matches
            total_box_loss /= num_matches
            total_conf_loss /= num_matches

        # Combine losses
        total_loss = (self.alpha * total_cls_loss +
                     self.beta * total_box_loss +
                     (1 - self.alpha - self.beta) * total_conf_loss)

        return {
            'total': total_loss,
            'classification': total_cls_loss,
            'localization': total_box_loss,
            'confidence': total_conf_loss,
            'num_matches': num_matches
        }


class CrossKDTrainer:
    """
    Trainer class that integrates CrossKD loss with YOLOv5 training.
    """

    def __init__(self,
                 teacher_model,
                 student_model,
                 crosskd_config: Dict = None):
        """
        Initialize CrossKD trainer.

        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to train
            crosskd_config: CrossKD configuration
        """
        self.teacher_model = teacher_model
        self.student_model = student_model

        # Default CrossKD configuration
        default_config = {
            'alpha': 0.7,
            'beta': 0.3,
            'temperature': 4.0,
            'iou_threshold': 0.5,
            'conf_threshold': 0.5,
            'feature_adaptation': True
        }
        self.crosskd_config = {**default_config, **(crosskd_config or {})}

        # Initialize CrossKD loss
        self.crosskd_loss = CrossKDLoss(**self.crosskd_config)

        # Set teacher to evaluation mode
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def compute_distillation_loss(self,
                                images: torch.Tensor,
                                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute CrossKD distillation loss.

        Args:
            images: Input images
            targets: Ground truth targets

        Returns:
            Dictionary of loss components
        """
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(images)

        # Student forward pass
        student_outputs = self.student_model(images)

        # Compute CrossKD loss
        crosskd_losses = self.crosskd_loss(student_outputs, teacher_outputs)

        return crosskd_losses

    def train_step(self,
                   images: torch.Tensor,
                   targets: torch.Tensor,
                   original_loss_fn,
                   distillation_weight: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Perform one training step with CrossKD.

        Args:
            images: Input images
            targets: Ground truth targets
            original_loss_fn: Original YOLOv5 loss function
            distillation_weight: Weight for distillation loss

        Returns:
            Combined loss dictionary
        """
        # Compute original task loss
        student_outputs = self.student_model(images)
        original_loss = original_loss_fn(student_outputs, targets)

        # Compute distillation loss
        distillation_losses = self.compute_distillation_loss(images, targets)

        # Combine losses
        total_loss = ((1 - distillation_weight) * original_loss +
                     distillation_weight * distillation_losses['total'])

        return {
            'total': total_loss,
            'original': original_loss,
            'distillation': distillation_losses['total'],
            'cls_distill': distillation_losses['classification'],
            'box_distill': distillation_losses['localization'],
            'conf_distill': distillation_losses['confidence'],
            'num_matches': distillation_losses['num_matches']
        }


def test_crosskd_loss():
    """Test function for CrossKD loss."""

    # Create dummy data
    batch_size = 2
    num_classes = 8  # RoboMaster classes

    # Dummy teacher outputs (larger model)
    teacher_outputs = [
        torch.randn(batch_size, 3, 80, 80, 5 + num_classes),  # Large scale
        torch.randn(batch_size, 3, 40, 40, 5 + num_classes),  # Medium scale
        torch.randn(batch_size, 3, 20, 20, 5 + num_classes),  # Small scale
    ]

    # Dummy student outputs (smaller model)
    student_outputs = [
        torch.randn(batch_size, 3, 80, 80, 5 + num_classes),
        torch.randn(batch_size, 3, 40, 40, 5 + num_classes),
        torch.randn(batch_size, 3, 20, 20, 5 + num_classes),
    ]

    # Initialize CrossKD loss
    crosskd_loss = CrossKDLoss(alpha=0.7, beta=0.3, temperature=4.0)

    # Compute loss
    losses = crosskd_loss(student_outputs, teacher_outputs)

    print("CrossKD Loss Test Results:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.item():.4f}")
        else:
            print(f"{key}: {value}")

    return losses


if __name__ == "__main__":
    test_crosskd_loss()