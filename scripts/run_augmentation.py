#run_augmentation.py
"""
RoboMaster数据预处理与增强主运行脚本

该脚本负责解析命令行参数，配置并启动数据增强流水线。
"""

import argparse
import sys
from pathlib import Path

try:
    from robomaster_extensions.augmentation.pipeline import AugmentationPipeline
except ImportError:
    # 为了在项目根目录直接运行时也能找到模块，添加路径
    sys.path.append(str(Path(__file__).parent.parent))
    from robomaster_extensions.augmentation.pipeline import AugmentationPipeline


def main():
    """主函数：解析参数，验证路径，并运行增强流水线。"""
    parser = argparse.ArgumentParser(
        description='RoboMaster数据预处理与增强 (基于新架构)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # 更好地显示默认值
    )
    
    # --- 路径参数 ---
    parser.add_argument('--input', type=str, default='robomaster',
                        help='原始数据集路径 (e.g., data/robomaster)')
    parser.add_argument('--output', type=str, default='robomaster',
                        help='输出增强数据集的路径 (e.g., data/robomaster_augmented)')
    parser.add_argument('--coco-path', type=str, default=None,
                        help='COCO背景图片库路径。如果未提供，将尝试在输入路径下查找 "cocoimg"')

    # --- 增强控制参数 ---
    parser.add_argument('--strategies', nargs='+', default=[
                            'sticker_swap', 'brightness_adjust', 'clahe_enhance', 'coco_insert'
                        ],
                        help='要执行的增强策略列表 (空格分隔)。')
    parser.add_argument('--aug-factor', type=int, default=1,
                        help='增强倍数 (每种策略为每张原图生成多少个版本)')

    # --- 概率参数 ---
    parser.add_argument('--sticker-swap-prob', type=float, default=0.5,
                        help='贴纸交换概率')
    parser.add_argument('--context-mixup-prob', type=float, default=0.2,
                        help='上下文混合概率 (仅在 "mixed" 策略中使用)')
    parser.add_argument('--brightness-adjust-prob', type=float, default=0.6,
                        help='亮度调整概率')
    parser.add_argument('--coco-insert-prob', type=float, default=0.3,
                        help='COCO背景插入概率')

    args = parser.parse_args()

    # 1. 验证输入路径 (您的原始逻辑，非常好，予以保留)
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入路径不存在: {input_path}")
        sys.exit(1) # 使用 sys.exit 终止脚本

    print("正在验证数据集结构...")
    train_valid = (input_path / 'train' / 'images').exists() and (input_path / 'train' / 'labels').exists()
    val_valid = (input_path / 'val' / 'images').exists() and (input_path / 'val' / 'labels').exists()

    if not (train_valid and val_valid):
        print("错误: 数据集结构不完整或格式不正确。")
        # ... (错误提示信息保持不变) ...
        sys.exit(1)
    
    print("数据集结构验证通过。")

    # 2. 构造配置字典 (这是核心变化)
    # 我们将所有命令行参数打包成一个字典，传递给流水线
    config = {
        # 路径
        "input_path": args.input,
        "output_path": args.output,
        "coco_img_path": args.coco_path if args.coco_path else str(input_path / 'cocoimg'),
        
        # 增强控制
        "strategies": args.strategies,
        "augmentation_factor": args.aug_factor,
        
        # 概率
        "sticker_swap_prob": args.sticker_swap_prob,
        "context_mixup_prob": args.context_mixup_prob,
        "brightness_adjust_prob": args.brightness_adjust_prob,
        "coco_insert_prob": args.coco_insert_prob,
    }

    print("\n--- 增强配置 ---")
    for key, value in config.items():
        print(f"{key:<25}: {value}")
    print("---------------------\n")

    # 3. 初始化并运行流水线
    try:
        pipeline = AugmentationPipeline(config=config)
        pipeline.run()
        print(f"\n数据增强成功! 结果已保存在: {args.output}")
    except Exception as e:
        print(f"\n错误: 数据增强过程中发生意外: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()