# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a YOLOv5 repository that has been extended with specialized RoboMaster armor plate detection capabilities. It contains both the original YOLOv5 functionality and custom RoboMaster-specific enhancements for competitive robotics applications.

## Key Commands

### Training Commands
```bash
# Standard YOLOv5 training
python train.py --data data/coco128.yaml --epochs 100 --weights yolov5s.pt --img 640

# RoboMaster training with optimized model
python train.py --cfg models/robomaster_yolov5s.yaml --data data/robomaster.yaml --epochs 300 --batch-size 16 --label-smoothing 0.1

# Knowledge distillation training for RoboMaster
python scripts/train_with_distillation.py --student-cfg models/robomaster_yolov5n.yaml --teacher-weights yolov5x.pt --data data/robomaster.yaml --epochs 300 --distillation-method crosskd --label-smoothing 0.1 --background-bias-augment

# Active learning pipeline
python scripts/active_learning_pipeline.py --model-weights runs/train/exp/weights/best.pt --unlabeled-data data/unlabeled/ --budget 1000 --batch-size 50 --simulate-annotation
```

### Inference Commands
```bash
# Standard inference with detect.py
python detect.py --weights yolov5s.pt --source 0  # webcam
python detect.py --weights yolov5s.pt --source img.jpg  # image
python detect.py --weights yolov5s.pt --source vid.mp4  # video

# Segmentation inference
python segment/predict.py --weights yolov5m-seg.pt --source data/images/bus.jpg

# Classification inference
python classify/predict.py --weights yolov5s-cls.pt --source data/images/bus.jpg
```

### Validation Commands
```bash
# Standard validation
python val.py --weights yolov5s.pt --data data/coco128.yaml --img 640

# Segmentation validation
python segment/val.py --weights yolov5s-seg.pt --data coco.yaml --img 640

# Classification validation
python classify/val.py --weights yolov5m-cls.pt --data ../datasets/imagenet --img 224
```

### Export Commands
```bash
# Export to various formats
python export.py --weights yolov5s.pt --include onnx engine torchscript tflite

# Segmentation export
python export.py --weights yolov5s-seg.pt --include engine --img 640 --device 0 --half

# Classification export
python export.py --weights yolov5s-cls.pt --include engine onnx --imgsz 224
```

### Data Setup Commands
```bash
# STEP 1: After placing all data in images/ and labels/ directories
# Split dataset into train/val (MUST run this first)
python scripts/split_dataset.py  # uses default 8:2 ratio
python scripts/split_dataset.py /path/to/images /path/to/labels 0.7  # custom ratio

# STEP 2: Generate 3 types of augmented data
python scripts/data_preprocessing.py

# ALTERNATIVE: Create RoboMaster directory structure (only if starting fresh)
python scripts/create_directory_structure.py
```

## Architecture Overview

### Core YOLOv5 Structure
- **Models**: Configuration files in `models/` define network architectures (yolov5n.yaml to yolov5x.yaml)
- **Training**: Main training loop in `train.py` with distributed training support
- **Inference**: Detection (`detect.py`), segmentation (`segment/`), and classification (`classify/`) modules
- **Utilities**: Common functions in `utils/` including data loading, metrics, plotting, and torch utilities

### RoboMaster Extensions (`robomaster_extensions/`)
This repository includes specialized extensions for RoboMaster armor plate detection:

- **`config.py`**: Centralized 14-class RoboMaster configuration system
- **`data_augmentation.py`**: Sophisticated data augmentation to combat background bias
- **`distillation.py`**: Multi-stage knowledge distillation framework (YOLOv5x→YOLOv5l→YOLOv5m→YOLOv5s→YOLOv5n)
- **`crosskd_loss.py`**: CrossKD loss function specialized for object detection
- **`label_smoothing.py`**: Adaptive label smoothing for fine-grained classification
- **`active_learning.py`**: Shannon entropy-based uncertainty sampling for efficient annotation
- **`grad_cam.py`**: GradCAM visualization for model interpretability
- **`robomaster_trainer.py`**: Integrated training pipeline combining all optimizations

### Data Configuration
- **RoboMaster Classes**: 14-class system (B1-B5, BHero, R1-R5, RHero, RQS, BQS) configured in `data/robomaster.yaml`
- **Model Configs**: Optimized architectures in `models/robomaster_yolov5*.yaml` with adjusted anchors and layer depths for armor plate detection

### Key RoboMaster Features
1. **Background Bias Mitigation**: Intelligent sticker swapping and context augmentation
2. **Multi-stage Knowledge Distillation**: Gradual knowledge transfer from large to small models
3. **Active Learning**: Reduces annotation requirements by 70% using uncertainty-based sampling
4. **Fine-grained Classification**: Label smoothing adapted for armor plate subtypes

## Development Notes

### Python Environment
- Requires Python ≥ 3.8.0 and PyTorch ≥ 1.8.0
- Install dependencies: `pip install -r requirements.txt`
- For RoboMaster features, also install: `pip install albumentations scikit-learn`

### Data Structure
Standard YOLOv5 format with images/ and labels/ directories. For RoboMaster, use the enhanced structure created by `scripts/create_directory_structure.py` which includes augmentation subdirectories.

### Model Architecture
- **Standard Models**: Use depth_multiple and width_multiple to scale network size
- **RoboMaster Models**: Optimized anchor sizes and layer configurations for armor plates
- **Distillation Pipeline**: Teacher-student training with intermediate assistant models

### Training Process
1. Standard training uses `train.py` with hyperparameters in `data/hyps/`
2. RoboMaster training integrates all optimizations via `scripts/train_with_distillation.py`
3. Multi-GPU training supported via `torch.distributed.run`
4. Automatic model selection and hyperparameter evolution available

### Export and Deployment
Models can be exported to ONNX, TensorRT, TorchScript, TFLite, and other formats for production deployment. Segmentation and classification models have separate export paths.

## Important Files
- `train.py`: Main training script
- `val.py`: Validation/testing script
- `detect.py`: Inference script
- `models/`: Model architecture definitions
- `data/`: Dataset configurations and hyperparameters
- `utils/`: Core utilities and helper functions
- `robomaster_extensions/`: RoboMaster-specific optimizations
- `scripts/`: Automation and pipeline scripts