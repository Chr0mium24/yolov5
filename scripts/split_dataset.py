import shutil
import random
import argparse
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

def get_image_label_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    """
    获取图片和对应标签文件的配对列表。

    Args:
        images_dir: 图片目录。
        labels_dir: 标签目录。

    Returns:
        配对的 (图片文件, 标签文件) 列表。
    """
    pairs = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    print("正在扫描文件并进行配对...")
    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions]

    for img_file in tqdm(image_files, desc="正在查找配对文件"):
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            pairs.append((img_file, label_file))
        else:
            print(f"\n警告: 找不到图片 '{img_file.name}' 对应的标签文件 '{label_file.name}'")
    
    return pairs

def split_dataset(source_images_dir: str, source_labels_dir: str,
                 train_ratio: float = 0.8, random_seed: int = 42, use_move: bool = False):
    """
    自动分割数据集为 train/val。

    Args:
        source_images_dir: 源图片目录。
        source_labels_dir: 源标签目录。
        train_ratio: 训练集比例。
        random_seed: 随机种子。
        use_move: 如果为 True，则移动文件；否则复制文件。
    """
    source_images = Path(source_images_dir)
    source_labels = Path(source_labels_dir)

    if not source_images.exists():
        raise FileNotFoundError(f"图片目录不存在: {source_images}")
    if not source_labels.exists():
        raise FileNotFoundError(f"标签目录不存在: {source_labels}")

    pairs = get_image_label_pairs(source_images, source_labels)

    if not pairs:
        print("错误: 未找到任何有效的图片-标签配对。")
        return

    print(f"\n共找到 {len(pairs)} 个有效的图片-标签配对。")

    random.seed(random_seed)
    random.shuffle(pairs)

    train_count = int(len(pairs) * train_ratio)
    train_pairs = pairs[:train_count]
    val_pairs = pairs[train_count:]

    print(f"训练集规模: {len(train_pairs)} 个")
    print(f"验证集规模: {len(val_pairs)} 个")

    base_dir = source_images.parent
    train_img_dir = base_dir / "train" / "images"
    val_img_dir = base_dir / "val" / "images"
    train_label_dir = base_dir / "train" / "labels"
    val_label_dir = base_dir / "val" / "labels"

    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    operation = shutil.move if use_move else shutil.copy
    op_name = "移动" if use_move else "复制"

    print(f"\n正在{op_name}训练集文件...")
    for img_file, label_file in tqdm(train_pairs, desc="处理训练集"):
        operation(str(img_file), str(train_img_dir / img_file.name))
        operation(str(label_file), str(train_label_dir / label_file.name))

    print(f"\n正在{op_name}验证集文件...")
    for img_file, label_file in tqdm(val_pairs, desc="处理验证集"):
        operation(str(img_file), str(val_img_dir / img_file.name))
        operation(str(label_file), str(val_label_dir / label_file.name))

    print(f"\n数据集分割完成!")
    print(f"训练集图片: {train_img_dir}")
    print(f"验证集图片: {val_img_dir}")

    if use_move:
        print("\n正在清理空的原始文件夹...")
        try:
            # 仅当文件夹为空时才会被删除
            source_images.rmdir()
            source_labels.rmdir()
            print("原始文件夹清理成功。")
        except OSError as e:
            print(f"清理原始文件夹时出错（可能文件夹非空或无权限）: {e}")
    else:
        print("\n原始数据已保留。")

def find_and_split_data(base_path: str, train_ratio: float, random_seed: int, use_move: bool):
    """
    在指定基础路径下查找数据目录并执行分割。
    """
    base_path = Path(base_path)
    images_dir = base_path / "images"
    labels_dir = base_path / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print("在当前目录未找到 images/ 和 labels/，尝试查找 data/robomaster/ 目录...")
        robomaster_images = base_path / "data" / "robomaster" / "images"
        robomaster_labels = base_path / "data" / "robomaster" / "labels"
        if robomaster_images.exists() and robomaster_labels.exists():
            print(f"找到 RoboMaster 数据目录: {robomaster_images.parent}")
            images_dir, labels_dir = robomaster_images, robomaster_labels
        else:
            print("错误: 未找到有效的数据目录。请确保以下任一结构存在:")
            print("  1. ./images/ 和 ./labels/")
            print("  2. ./data/robomaster/images/ 和 ./data/robomaster/labels/")
            return

    if not any(images_dir.iterdir()) or not any(labels_dir.iterdir()):
        print(f"错误: 数据目录为空。请将文件放入以下目录后重试:\n  图片: {images_dir}\n  标签: {labels_dir}")
        return

    base_dir = images_dir.parent
    if (base_dir / "train").exists() or (base_dir / "val").exists():
        print("警告: 检测到已存在 train/ 或 val/ 目录。")
        response = input("继续执行将可能会覆盖现有文件，是否继续? (y/n): ")
        if response.lower() != 'y':
            print("操作已取消。")
            return

    split_dataset(str(images_dir), str(labels_dir), train_ratio, random_seed, use_move)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="自动将图片和标签数据集分割为训练集和验证集。",
        formatter_class=argparse.RawTextHelpFormatter  # 保持帮助信息格式
    )

    parser.add_argument(
        '--images_dir', type=str, default=None,
        help='源图片目录的路径。\n如果未提供此参数和 --labels_dir，脚本将进入自动检测模式。'
    )
    parser.add_argument(
        '--labels_dir', type=str, default=None,
        help='源标签目录的路径。'
    )
    parser.add_argument(
        '--train_ratio', type=float, default=0.8,
        help='训练集所占的比例 (0.0 到 1.0 之间)。\n默认为 0.8。'
    )
    parser.add_argument(
        '--random_seed', type=int, default=42,
        help='用于复现随机分割结果的种子。\n默认为 42。'
    )
    parser.add_argument(
        '--move', action='store_true',
        help='使用移动代替默认的复制操作。\n这是一个破坏性操作，原始文件将被移动到新目录。'
    )

    args = parser.parse_args()

    if (args.images_dir and not args.labels_dir) or (not args.images_dir and args.labels_dir):
        parser.error("--images_dir 和 --labels_dir 必须同时提供。")

    if args.images_dir and args.labels_dir:
        # 模式1: 用户明确指定目录
        print("--- 指定目录模式 ---")
        split_dataset(
            args.images_dir,
            args.labels_dir,
            args.train_ratio,
            args.random_seed,
            args.move
        )
    else:
        # 模式2: 自动检测当前目录下的数据
        print("--- 自动检测模式 ---")
        find_and_split_data(
            base_path=".",
            train_ratio=args.train_ratio,
            random_seed=args.random_seed,
            use_move=args.move
        )