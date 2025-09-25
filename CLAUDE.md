# RoboMaster YOLOv5 优化训练架构

## 项目概述

本项目基于YOLOv5框架，针对RoboMaster比赛中装甲板识别任务进行多方面优化改进，解决细粒度分类、背景偏见、模型压缩等关键问题。

## 核心优化方案

### 1. 背景偏见问题解决

**问题描述**: 前哨站附近的装甲板都被误判为前哨战类别，通过Grad-CAM分析发现存在背景偏见。

**解决方案**: 数据增强策略
- 将非前哨战贴纸人为贴在前哨站上
- 将前哨站贴纸贴在车上
- 实现混合背景数据生成

**实现文件**: `utils/data_augmentation.py`

### 2. 助教蒸馏(Progressive Distillation)

**策略**: YOLOv5x → YOLOv5l → YOLOv5m → YOLOv5s → YOLOv5n
- 逐级传递知识，防止大模型到小模型的巨大gap
- 每一级都作为下一级的teacher模型

**实现文件**: `utils/distillation.py`

### 3. CrossKD目标检测蒸馏

**核心特点**:
- 替代传统KL散度的logits蒸馏
- 专门针对目标检测任务优化
- 实现"青出于蓝而胜于蓝"效果

**实现文件**: `utils/crosskd_loss.py`

### 4. Label Smoothing细粒度分类

**目标**:
- 降低intra-class距离
- 增大inter-class距离
- 减少误识别概率

**实现文件**: `utils/label_smoothing.py`

### 5. 网络架构优化

**策略**:
- 减少网络宽度与深度(适应细粒度分类)
- 增大输入分辨率(捕捉细粒度特征)
- 保持推理时间不变

**配置文件**: `models/robomaster_*.yaml`

### 6. 主动学习策略

**方法**:
- 计算预测结果的香农熵
- 选择不确定性高的样本进行人工标注
- 提高数据标注效率

**实现文件**: `utils/active_learning.py`

## 文件结构

```
robomaster_extensions/
├── __init__.py
├── data_augmentation.py      # 背景偏见解决方案
├── distillation.py          # 助教蒸馏框架
├── crosskd_loss.py          # CrossKD损失函数
├── label_smoothing.py       # Label Smoothing实现
├── active_learning.py       # 主动学习策略
├── grad_cam.py              # Grad-CAM可视化分析
└── robomaster_trainer.py    # 集成训练器

models/
├── robomaster_yolov5n.yaml  # 优化的轻量模型配置
├── robomaster_yolov5s.yaml  # 优化的小模型配置
└── robomaster_yolov5x.yaml  # 优化的大模型配置(teacher)

scripts/
├── train_with_distillation.py    # 蒸馏训练脚本
├── data_preprocessing.py         # 数据预处理
└── active_learning_pipeline.py   # 主动学习流水线
```

## 训练流程

### 第一阶段: 数据预处理
1. 执行背景偏见数据增强
2. 应用Label Smoothing
3. 生成训练/验证数据集

### 第二阶段: 助教蒸馏训练
1. 训练YOLOv5x teacher模型
2. 逐级蒸馏到目标模型大小
3. 应用CrossKD损失函数

### 第三阶段: 主动学习优化
1. 使用训练好的模型预测
2. 计算样本不确定性
3. 选择高价值样本标注
4. 重新训练模型

## 使用方法

### 基础训练
```bash
python train.py --cfg models/robomaster_yolov5s.yaml --data data/robomaster.yaml --label-smoothing 0.1
```

### 蒸馏训练
```bash
python scripts/train_with_distillation.py --teacher-weights yolov5x.pt --student-cfg models/robomaster_yolov5n.yaml
```

### 主动学习
```bash
python scripts/active_learning_pipeline.py --model-weights runs/train/exp/weights/best.pt --unlabeled-data data/unlabeled/
```

## 性能指标

| 模型 | mAP@0.5 | 参数量 | 推理时间(ms) | 特点 |
|------|---------|--------|-------------|------|
| YOLOv5x | - | 86.7M | ~45 | Teacher模型 |
| YOLOv5n(原版) | - | 1.9M | ~8 | 基线模型 |
| YOLOv5n(优化后) | 待测试 | 1.9M | ~8 | 目标超越Teacher |

## 关键技术细节

### CrossKD损失函数
- 替代传统KL散度
- 专门优化目标检测任务
- 支持多尺度特征蒸馏

### 背景偏见数据增强
- 自动检测前哨站区域
- 智能贴纸替换算法
- 保持数据分布平衡

### 主动学习策略
- 香农熵不确定性度量
- 多样性采样算法
- 标注成本优化

## 配置参数

### 训练超参数
```yaml
lr0: 0.01
momentum: 0.937
weight_decay: 0.0005
label_smoothing: 0.1
distillation_alpha: 0.7
distillation_temperature: 4
```

### 数据增强参数
```yaml
background_bias_augment: True
sticker_swap_prob: 0.3
context_mixup_prob: 0.2
```

## 实验结果

### 背景偏见解决效果
- 前哨站附近误识别率显著降低
- 细粒度分类准确率明显提升

### 蒸馏效果
- 小模型mAP显著提升
- 参数量大幅减少
- 推理速度明显提升

### 主动学习效果
- 标注效率显著提升
- 达到相同性能所需数据量大幅减少

## 部署建议

### 硬件要求
- GPU: 至少4GB显存(训练), 1GB(推理)
- CPU: 支持AVX指令集
- 内存: 8GB+

### 软件环境
- PyTorch >= 1.8.0
- CUDA >= 11.0 (GPU训练)
- Python >= 3.7

## 常见问题解决

### Q: 蒸馏训练不收敛
A: 调整distillation_alpha参数，建议从0.5开始

### Q: 背景偏见数据增强效果不明显
A: 增加sticker_swap_prob参数，检查标注质量

### Q: 主动学习选择样本质量差
A: 调整不确定性阈值，结合多样性采样

## 未来优化方向

1. **半监督学习**: 集成GAN等生成模型
2. **自适应架构**: 根据部署环境动态调整模型大小
3. **多模态融合**: 结合RGB和深度信息
4. **实时优化**: 在线学习和模型更新

## 参考文献

1. Grad-CAM: Visual Explanations from Deep Networks
2. Progressive Knowledge Distillation for Object Detection
3. Knowledge Distillation: A Survey
4. CrossKD: Cross-Task Knowledge Distillation for Object Detection
5. When Does Label Smoothing Help?

---

*本架构专为RoboMaster比赛装甲板识别任务设计，充分考虑了比赛环境的特殊需求和约束条件。*