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

from robomaster_extensions.data_augmentation import UnifiedDataAugmenter


def main():
    parser = argparse.ArgumentParser(description='RoboMaster数据预处理与增强')
    parser.add_argument('--input', type=str, default='.',
                       help='原始数据集路径 (支持新格式train/val或旧格式images/labels)')
    parser.add_argument('--output', type=str, default='./augmented_data',
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

    # 检测数据结构格式
    new_format = (input_path / 'train').exists() and (input_path / 'val').exists()
    old_format = (input_path / 'images').exists() and (input_path / 'labels').exists()

    if new_format:
        print(f"检测到新格式数据结构: train/val")
        # 检查新格式的完整性
        train_valid = (input_path / 'train' / 'images').exists() and (input_path / 'train' / 'labels').exists()
        val_valid = (input_path / 'val' / 'images').exists() and (input_path / 'val' / 'labels').exists()
        if not (train_valid and val_valid):
            print(f"错误: 新格式数据结构不完整")
            print(f"需要: train/images/, train/labels/, val/images/, val/labels/")
            return
    elif old_format:
        print(f"检测到旧格式数据结构: images/labels")
        # 检查旧格式的完整性
        if not (input_path / 'images' / 'train').exists() or not (input_path / 'labels' / 'train').exists():
            print(f"错误: 旧格式数据结构不完整")
            print(f"需要: images/train/, images/val/, labels/train/, labels/val/")
            return
    else:
        print(f"错误: 无法识别数据结构格式")
        print(f"支持的格式:")
        print(f"  新格式: train/images/, train/labels/, val/images/, val/labels/")
        print(f"  旧格式: images/train/, images/val/, labels/train/, labels/val/")
        print(f"当前路径内容: {list(input_path.iterdir())}")
        return

    # 如果是新格式，需要临时转换为旧格式给增强器使用
    temp_path = None
    if new_format:
        # 创建临时旧格式结构
        import tempfile
        import shutil
        temp_path = Path(tempfile.mkdtemp())
        print(f"创建临时旧格式结构: {temp_path}")

        # 创建旧格式目录
        (temp_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (temp_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (temp_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (temp_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

        # 创建符号链接指向原始数据
        for split in ['train', 'val']:
            src_img = input_path / split / 'images'
            src_lbl = input_path / split / 'labels'
            dst_img = temp_path / 'images' / split
            dst_lbl = temp_path / 'labels' / split

            # 复制文件而不是创建链接（兼容性更好）
            if src_img.exists():
                for img_file in src_img.iterdir():
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        shutil.copy2(img_file, dst_img / img_file.name)

            if src_lbl.exists():
                for lbl_file in src_lbl.iterdir():
                    if lbl_file.suffix.lower() == '.txt':
                        shutil.copy2(lbl_file, dst_lbl / lbl_file.name)

        dataset_path_for_augmenter = str(temp_path)
    else:
        dataset_path_for_augmenter = str(input_path)

    # 初始化增强器
    augmenter = UnifiedDataAugmenter(
        sticker_swap_prob=args.sticker_swap_prob,
        context_mixup_prob=args.context_mixup_prob,
        preserve_geometry=True
    )

    print(f"开始数据增强...")
    print(f"输入路径: {input_path} ({'新格式' if new_format else '旧格式'})")
    print(f"输出路径: {args.output}")
    print(f"增强倍数: {args.aug_factor}")
    print(f"贴纸交换概率: {args.sticker_swap_prob}")
    print(f"上下文混合概率: {args.context_mixup_prob}")

    try:
        # 执行数据增强
        augmenter.create_balanced_dataset(
            dataset_path=dataset_path_for_augmenter,
            output_path=args.output,
            augmentation_factor=args.aug_factor
        )
        print(f"数据增强完成! 结果保存在: {args.output}")
    finally:
        # 清理临时目录
        if temp_path and temp_path.exists():
            shutil.rmtree(temp_path)
            print(f"已清理临时目录: {temp_path}")


if __name__ == "__main__":
    main()