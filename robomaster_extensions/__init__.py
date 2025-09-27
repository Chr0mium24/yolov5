"""
RoboMaster YOLOv5 Extensions

This package contains specialized modules for RoboMaster competition:
- Background bias mitigation through data augmentation
- Progressive knowledge distillation
- CrossKD object detection distillation
- Label smoothing for fine-grained classification
- Active learning for efficient data annotation
"""

__version__ = "1.0.0"
__author__ = "RoboMaster Team"

try:
    from .data_augmentation import UnifiedDataAugmenter
except ImportError:
    UnifiedDataAugmenter = None

try:
    from .distillation import ProgressiveDistillationTrainer
except ImportError:
    ProgressiveDistillationTrainer = None

try:
    from .crosskd_loss import CrossKDLoss
except ImportError:
    CrossKDLoss = None

try:
    from .label_smoothing import RoboMasterLabelSmoothingManager
except ImportError:
    RoboMasterLabelSmoothingManager = None

try:
    from .active_learning import ActiveLearningSelector
except ImportError:
    ActiveLearningSelector = None

try:
    from .grad_cam import GradCAMAnalyzer
except ImportError:
    GradCAMAnalyzer = None

try:
    from .robomaster_trainer import RoboMasterTrainer
except ImportError:
    RoboMasterTrainer = None

__all__ = [
    "UnifiedDataAugmenter",
    "ProgressiveDistillationTrainer",
    "CrossKDLoss",
    "RoboMasterLabelSmoothingManager",
    "ActiveLearningSelector",
    "GradCAMAnalyzer",
    "RoboMasterTrainer"
]