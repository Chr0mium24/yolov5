#!/usr/bin/env python3
"""
RoboMaster数据预处理脚本
执行背景偏见数据增强，生成训练就绪的数据集
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from robomaster_extensions.data_augmentation import BackgroundBiasAugmenter


def main():
    parser = argparse.ArgumentParser(description='RoboMaster数据预处理与增强')
    parser.add_argument('--input', type=str, required=True,
                       help='原始数据集路径 (包含images/和labels/子目录)')
    parser.add_argument('--output', type=str, required=True,
                       help='输出增强数据集路径')
    parser.add_argument('--aug-factor', type=int, default=2,
                       help='增强倍数 (每张原图生成多少增强版本)')
    parser.add_argument('--sticker-swap-prob', type=float, default=0.3,
                       help='贴纸交换概率')
    parser.add_argument('--context-mixup-prob', type=float, default=0.2,
                       help='上下文混合概率')

    args = parser.parse_args()

    # 验证输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入路径不存在: {input_path}")
        return

    if not (input_path / 'images').exists() or not (input_path / 'labels').exists():
        print(f"错误: 输入路径必须包含 images/ 和 labels/ 子目录")
        print(f"当前路径内容: {list(input_path.iterdir())}")
        return

    # 初始化增强器
    augmenter = BackgroundBiasAugmenter(
        sticker_swap_prob=args.sticker_swap_prob,
        context_mixup_prob=args.context_mixup_prob,
        preserve_geometry=True
    )

    print(f"开始数据增强...")
    print(f"输入路径: {input_path}")
    print(f"输出路径: {args.output}")
    print(f"增强倍数: {args.aug_factor}")
    print(f"贴纸交换概率: {args.sticker_swap_prob}")
    print(f"上下文混合概率: {args.context_mixup_prob}")

    # 执行数据增强
    augmenter.create_balanced_dataset(
        dataset_path=str(input_path),
        output_path=args.output,
        augmentation_factor=args.aug_factor
    )

    print(f"数据增强完成! 结果保存在: {args.output}")


if __name__ == "__main__":
    main()