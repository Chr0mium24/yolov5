#!/usr/bin/env python3
"""
创建RoboMaster数据集目录结构的脚本
自动生成images和labels文件夹的完整目录树
"""

import os
from pathlib import Path

def create_directory_structure(base_path="./"):
    """
    创建完整的数据集目录结构
    如果images/labels已存在，只创建增强数据的子目录

    Args:
        base_path: 基础路径，默认为当前目录
    """
    base_path = Path(base_path)

    # 检查是否已有基础目录
    has_images = (base_path / "images").exists()
    has_labels = (base_path / "labels").exists()

    if has_images or has_labels:
        print("检测到已存在的数据目录，仅创建增强数据子目录...")

    # 定义目录结构
    directories = []

    # 如果没有基础目录，创建所有目录
    if not has_images:
        directories.extend([
            "images/train",
            "images/val",
        ])

    if not has_labels:
        directories.extend([
            "labels/train",
            "labels/val",
        ])

    # 总是创建增强数据目录
    directories.extend([
        "images/train_augmented/sticker_swap",
        "images/train_augmented/brightness_adjust",
        "images/train_augmented/coco_insert",
        "images/val_augmented/sticker_swap",
        "images/val_augmented/brightness_adjust",
        "images/val_augmented/coco_insert",
        "labels/train_augmented/sticker_swap",
        "labels/train_augmented/brightness_adjust",
        "labels/train_augmented/coco_insert",
        "labels/val_augmented/sticker_swap",
        "labels/val_augmented/brightness_adjust",
        "labels/val_augmented/coco_insert"
    ])

    # 创建所有目录
    created_dirs = []
    for directory in directories:
        dir_path = base_path / directory
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(str(dir_path))
            print(f"✓ 创建目录: {dir_path}")
        else:
            print(f"- 目录已存在: {dir_path}")

    print(f"\n目录结构创建完成! 共创建 {len(created_dirs)} 个新目录")
    return created_dirs

if __name__ == "__main__":
    import sys

    # 获取基础路径参数
    base_path = sys.argv[1] if len(sys.argv) > 1 else "./"

    print("正在创建RoboMaster数据集目录结构...")
    print(f"基础路径: {os.path.abspath(base_path)}\n")

    create_directory_structure(base_path)