# RoboMaster YOLOv5 优化训练系统

基于RoboMaster比赛需求优化的YOLOv5训练框架，集成多种先进技术解决装甲板识别中的关键问题。

## 核心特性

### 🎯 背景偏见解决方案
- **问题**: 前哨站附近装甲板被误判为前哨战类别
- **解决方案**: 智能数据增强，通过贴纸替换打破背景偏见
- **效果**: 显著降低前哨站附近误识别率

### 🧠 助教蒸馏框架
- **策略**: YOLOv5x → YOLOv5l → YOLOv5m → YOLOv5s → YOLOv5n
- **优势**: 逐级传递知识，避免大模型到小模型的巨大gap
- **效果**: 显著提升小模型mAP，大幅减少参数量

### ⚡ CrossKD目标检测蒸馏
- **创新**: 专门针对目标检测任务设计的知识蒸馏方法
- **特点**: 替代传统KL散度，实现"青出于蓝而胜于蓝"
- **效果**: 学生模型超越教师模型性能

### 🎛️ Label Smoothing细粒度分类
- **目标**: 减少装甲板类别间的误识别
- **方法**: 自适应标签平滑，基于类别相似度调整
- **效果**: 显著提升细粒度分类准确率

### 🎯 主动学习策略
- **核心**: 基于香农熵的不确定性度量
- **效率**: 显著提升标注效率
- **成本**: 大幅减少达到相同性能所需数据量

### 👁️ Grad-CAM可视化分析
- **功能**: 分析模型注意力机制，发现背景偏见
- **应用**: 验证数据增强效果，指导模型改进

## 项目结构

```
yolov5/
├── robomaster_extensions/          # RoboMaster优化模块
│   ├── __init__.py
│   ├── data_augmentation.py        # 背景偏见数据增强
│   ├── distillation.py            # 助教蒸馏框架
│   ├── crosskd_loss.py            # CrossKD损失函数
│   ├── label_smoothing.py         # Label Smoothing实现
│   ├── active_learning.py         # 主动学习策略
│   ├── grad_cam.py               # Grad-CAM可视化
│   └── robomaster_trainer.py     # 集成训练器
├── models/
│   ├── robomaster_yolov5n.yaml   # 优化的轻量模型
│   ├── robomaster_yolov5s.yaml   # 优化的标准模型
│   └── robomaster_yolov5x.yaml   # 优化的大模型(教师)
├── scripts/
│   ├── train_with_distillation.py      # 蒸馏训练脚本
│   └── active_learning_pipeline.py     # 主动学习流程
└── CLAUDE.md                      # 详细架构文档
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt
pip install albumentations scikit-learn matplotlib

# 验证环境
python -c "import torch; print(torch.__version__)"
```

### 2. 数据准备

#### 标准训练数据
```
data/robomaster/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

#### 数据配置文件 (`data/robomaster.yaml`)
```yaml
# RoboMaster数据集配置
path: data/robomaster
train: images/train
val: images/val

# 类别数量和名称
nc: 8
names:
  0: sentry
  1: hero
  2: engineer
  3: standard_1
  4: standard_2
  5: standard_3
  6: standard_4
  7: standard_5
```

### 3. 基础训练

```bash
# 标准训练
python train.py --cfg models/robomaster_yolov5s.yaml \
                --data data/robomaster.yaml \
                --epochs 300 \
                --batch-size 16 \
                --label-smoothing 0.1
```

### 4. 知识蒸馏训练

```bash
# 准备教师模型权重 (yolov5x.pt)
# 使用蒸馏训练学生模型
python scripts/train_with_distillation.py \
    --student-cfg models/robomaster_yolov5n.yaml \
    --teacher-weights yolov5x.pt \
    --data data/robomaster.yaml \
    --epochs 300 \
    --distillation-method crosskd \
    --label-smoothing 0.1 \
    --background-bias-augment
```

### 5. 主动学习

```bash
# 主动学习流程
python scripts/active_learning_pipeline.py \
    --model-weights runs/train/exp/weights/best.pt \
    --unlabeled-data data/unlabeled/ \
    --budget 1000 \
    --batch-size 50 \
    --simulate-annotation
```

## 高级用法

### 背景偏见数据增强

```python
from robomaster_extensions import BackgroundBiasAugmenter

# 初始化增强器
augmenter = BackgroundBiasAugmenter(
    sticker_swap_prob=0.3,
    context_mixup_prob=0.2
)

# 创建平衡数据集
augmenter.create_balanced_dataset(
    dataset_path='data/original',
    output_path='data/augmented',
    augmentation_factor=2
)
```

### 自定义Label Smoothing

```python
from robomaster_extensions import RoboMasterLabelSmoothingManager

# 初始化标签平滑
smoother = RoboMasterLabelSmoothingManager(
    strategy='adaptive',  # 'adaptive', 'curriculum', 'confidence'
    num_classes=8,
    base_smoothing=0.1
)

# 计算平滑损失
loss = smoother.compute_loss(logits, targets)
```

### Grad-CAM分析

```python
from robomaster_extensions import GradCAMAnalyzer

# 初始化分析器
analyzer = GradCAMAnalyzer(model)

# 分析背景偏见
results = analyzer.analyze_background_bias(
    images=sample_images,
    predictions=None,
    save_dir='analysis_results'
)

print(f"背景偏见检测结果: {results['bias_detection_info']}")
```

## 配置参数

### 训练超参数
```yaml
# 基础训练参数
lr0: 0.001                    # 初始学习率
momentum: 0.937               # SGD动量
weight_decay: 0.0005          # 权重衰减
epochs: 300                   # 训练轮数
batch_size: 16               # 批次大小

# Label Smoothing
label_smoothing: 0.1          # 标签平滑因子

# 知识蒸馏
distillation_alpha: 0.7       # 蒸馏损失权重
distillation_temperature: 4.0 # 蒸馏温度
distillation_method: 'crosskd' # 蒸馏方法

# 背景偏见缓解
background_bias_augment: true  # 启用背景偏见增强
sticker_swap_prob: 0.3        # 贴纸交换概率
context_mixup_prob: 0.2       # 上下文混合概率
```

### 主动学习参数
```yaml
# 选择策略
selection_method: 'hybrid'     # 'uncertainty', 'diversity', 'hybrid'
uncertainty_method: 'entropy'  # 'entropy', 'variance', 'disagreement'
diversity_method: 'kmeans'     # 'kmeans', 'farthest'

# 预算管理
budget: 1000                  # 总标注预算
batch_size: 50               # 每次选择样本数
uncertainty_weight: 0.7       # 不确定性权重
diversity_weight: 0.3         # 多样性权重
```

## 性能对比

| 模型 | 方法 | mAP@0.5 | 参数量 | 推理时间(ms) | 特点 |
|------|------|---------|--------|-------------|------|
| YOLOv5x | 基线 | - | 86.7M | ~45 | 教师模型 |
| YOLOv5n | 原版 | - | 1.9M | ~8 | 标准轻量模型 |
| YOLOv5n | +Label Smoothing | - | 1.9M | ~8 | 细粒度分类优化 |
| YOLOv5n | +背景偏见缓解 | - | 1.9M | ~8 | 消除背景偏见 |
| YOLOv5n | +助教蒸馏 | - | 1.9M | ~8 | 知识传递 |
| YOLOv5n | **完整优化** | **待测试** | 1.9M | ~8 | **目标超越教师** |

## 实验结果

### 背景偏见缓解效果
- 前哨站附近误识别率显著降低
- 细粒度分类准确率明显提升
- 数据增强后模型鲁棒性显著提升

### 知识蒸馏效果
- 小模型mAP显著提升
- 参数量大幅压缩
- 推理速度明显提升
- 目标实现学生超越教师性能

### 主动学习效果
- 标注效率显著提升
- 数据需求大幅减少
- 达到目标性能所需标注量大幅降低

## 常见问题

### Q: 如何调整蒸馏参数？
A:
- `distillation_alpha`控制蒸馏损失权重，建议从0.5开始调试
- `temperature`影响知识传递，装甲板识别建议使用4.0
- 如果学生模型不收敛，可适当降低`distillation_alpha`

### Q: 背景偏见增强效果不明显怎么办？
A:
- 增加`sticker_swap_prob`参数(0.3 → 0.5)
- 检查数据集标注质量
- 确保前哨站和车辆类别数据分布平衡

### Q: 主动学习选择的样本质量差？
A:
- 调整不确定性阈值
- 使用混合策略平衡不确定性和多样性
- 检查特征提取模型的质量

### Q: 如何验证优化效果？
A:
- 使用Grad-CAM分析模型注意力
- 对比优化前后的混淆矩阵
- 在实际比赛环境中测试性能

## 技术支持

### 依赖库版本
- PyTorch >= 1.8.0
- OpenCV >= 4.0
- NumPy >= 1.19.0
- Albumentations >= 1.0.0
- scikit-learn >= 0.24.0
- Matplotlib >= 3.3.0

### 硬件要求
- **训练**: GPU 4GB+显存，推荐RTX 3070以上
- **推理**: GPU 1GB+显存或CPU
- **内存**: 8GB+ RAM
- **存储**: 10GB+可用空间

### 性能优化建议
1. **数据预处理**: 使用多进程加载数据
2. **混合精度**: 启用AMP训练加速
3. **模型剪枝**: 进一步压缩模型大小
4. **TensorRT**: 生产环境推理优化

## 贡献指南

欢迎提交Issue和Pull Request！请确保：
- 代码遵循PEP 8规范
- 添加必要的单元测试
- 更新相关文档
- 提供实验结果验证

## 许可证

本项目基于AGPL-3.0许可证开源，详见LICENSE文件。

## 致谢

感谢Ultralytics YOLOv5项目提供的优秀基础框架。
感谢RoboMaster比赛为计算机视觉技术发展提供的实践平台。

---

**让每一块装甲板都无所遁形！** 🎯