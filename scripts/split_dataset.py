#!/usr/bin/env python3
"""
自动分割数据集为train/val的脚本
支持从单个目录自动分割到train/val结构
"""

import shutil
import random
from pathlib import Path
from typing import List, Tuple

def get_image_label_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    """
    获取图片和对应标签文件的配对列表

    Args:
        images_dir: 图片目录
        labels_dir: 标签目录

    Returns:
        配对的(图片文件, 标签文件)列表
    """
    pairs = []

    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # 遍历图片文件
    for img_file in images_dir.iterdir():
        if img_file.suffix.lower() in image_extensions:
            # 查找对应的标签文件
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                pairs.append((img_file, label_file))
            else:
                print(f"警告: 找不到对应标签文件 {label_file}")

    return pairs

def split_dataset(source_images_dir: str, source_labels_dir: str,
                 train_ratio: float = 0.8, random_seed: int = 42):
    """
    自动分割数据集为train/val

    Args:
        source_images_dir: 源图片目录
        source_labels_dir: 源标签目录
        train_ratio: 训练集比例 (默认0.8)
        random_seed: 随机种子
    """
    source_images = Path(source_images_dir)
    source_labels = Path(source_labels_dir)

    if not source_images.exists():
        raise FileNotFoundError(f"图片目录不存在: {source_images}")
    if not source_labels.exists():
        raise FileNotFoundError(f"标签目录不存在: {source_labels}")

    # 获取所有图片-标签配对
    pairs = get_image_label_pairs(source_images, source_labels)

    if not pairs:
        print("错误: 未找到任何有效的图片-标签配对")
        return

    print(f"找到 {len(pairs)} 个图片-标签配对")

    # 随机打乱
    random.seed(random_seed)
    random.shuffle(pairs)

    # 计算分割点
    train_count = int(len(pairs) * train_ratio)
    train_pairs = pairs[:train_count]
    val_pairs = pairs[train_count:]

    print(f"训练集: {len(train_pairs)} 个文件")
    print(f"验证集: {len(val_pairs)} 个文件")

    # 创建目标目录 - YOLOv5标准格式: train/images, train/labels, val/images, val/labels
    base_dir = source_images.parent
    train_img_dir = base_dir / "train" / "images"
    val_img_dir = base_dir / "val" / "images"
    train_label_dir = base_dir / "train" / "labels"
    val_label_dir = base_dir / "val" / "labels"

    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 移动训练集文件
    print("\n正在移动训练集文件...")
    for img_file, label_file in train_pairs:
        shutil.move(str(img_file), str(train_img_dir / img_file.name))
        shutil.move(str(label_file), str(train_label_dir / label_file.name))

    # 移动验证集文件
    print("正在移动验证集文件...")
    for img_file, label_file in val_pairs:
        shutil.move(str(img_file), str(val_img_dir / img_file.name))
        shutil.move(str(label_file), str(val_label_dir / label_file.name))

    print(f"\n数据集分割完成!")
    print(f"训练集: {train_img_dir}")
    print(f"验证集: {val_img_dir}")

    print("\n正在删除原始的 images 和 labels 文件夹...")
    try:
        shutil.rmtree(source_images)
        shutil.rmtree(source_labels)
        print("原始文件夹删除成功。")
    except OSError as e:
        print(f"删除原始文件夹时出错: {e}")


def split_existing_mixed_data(base_path: str = "./", train_ratio: float = 0.8):
    """
    如果images和labels目录下直接放着混合的文件，自动分割为train/val

    Args:
        base_path: 基础路径
        train_ratio: 训练集比例
    """
    base_path = Path(base_path)
    images_dir = base_path / "images"
    labels_dir = base_path / "labels"

    # 如果根目录下没有images/labels，尝试data/robomaster/目录
    if not images_dir.exists() or not labels_dir.exists():
        print("未找到images/和labels/目录，尝试查找data/robomaster/目录...")
        robomaster_images = base_path / "data" / "robomaster" / "images"
        robomaster_labels = base_path / "data" / "robomaster" / "labels"

        if robomaster_images.exists() and robomaster_labels.exists():
            print(f"找到RoboMaster数据目录: {robomaster_images}")
            images_dir = robomaster_images
            labels_dir = robomaster_labels
        else:
            print("错误: 未找到数据目录，请确保以下任一目录结构存在:")
            print("  方案1: ./images/ 和 ./labels/")
            print("  方案2: ./data/robomaster/images/ 和 ./data/robomaster/labels/")
            print("\n请先将您的图片和标签文件放到相应目录中，然后重新运行此脚本。")
            return

    # 检查目录是否为空
    if not any(images_dir.iterdir()) or not any(labels_dir.iterdir()):
        print(f"错误: 数据目录为空")
        print(f"图片目录: {images_dir} ({'空' if not any(images_dir.iterdir()) else '有文件'})")
        print(f"标签目录: {labels_dir} ({'空' if not any(labels_dir.iterdir()) else '有文件'})")
        print("\n请先将您的图片和标签文件放到相应目录中，然后重新运行此脚本。")
        return

    # 检查是否已经有train/val子目录
    base_dir = images_dir.parent
    if (base_dir / "train").exists() or (base_dir / "val").exists():
        print("检测到已有train/val目录，请确认是否要重新分割")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            return

    split_dataset(str(images_dir), str(labels_dir), train_ratio)

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # 默认模式：处理当前目录下的混合数据
        print("自动分割模式：处理images/和labels/目录下的混合数据")
        split_existing_mixed_data()

    elif len(sys.argv) == 3:
        # 指定源目录模式
        images_dir = sys.argv[1]
        labels_dir = sys.argv[2]
        print(f"指定目录模式：{images_dir} -> train/val")
        split_dataset(images_dir, labels_dir)

    elif len(sys.argv) == 4:
        # 指定源目录和比例
        images_dir = sys.argv[1]
        labels_dir = sys.argv[2]
        train_ratio = float(sys.argv[3])
        print(f"指定目录和比例模式：{images_dir} -> train/val (比例: {train_ratio})")
        split_dataset(images_dir, labels_dir, train_ratio)

    else:
        print("用法:")
        print("  python split_dataset.py                           # 自动处理当前目录")
        print("  python split_dataset.py <images_dir> <labels_dir> # 指定源目录")
        print("  python split_dataset.py <images_dir> <labels_dir> <train_ratio> # 指定比例")