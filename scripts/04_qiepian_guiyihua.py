# /home/zhz/Deepl/subway_surfers_ai/scripts/normalize_slices.py

import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# --- [核心] 确保能找到项目根目录 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# --- [核心] 配置区 ---
# 原始切片素材的来源目录
SOURCE_SLICES_DIR = PROJECT_ROOT / "data" / "slices"
# 标准化后切片的输出目录
NORMALIZED_SLICES_DIR = PROJECT_ROOT / "data" / "normalized_slices"
# 对于特别小的物体（如金币），设定一个最小尺寸，防止它们被缩得看不见
MINIMUM_SIZE = (16, 16) 

# --- 辅助函数1: 计算一个类别所有切片的中位数尺寸 ---
def get_median_size(class_dir):
    """
    计算给定类别目录下所有PNG图片尺寸的中位数。

    Args:
        class_dir (Path): 存放同一类别切片的目录。

    Returns:
        tuple or None: (中位数宽度, 中位数高度)，如果目录为空则返回None。
    """
    sizes = []
    # 遍历目录下的所有 .png 文件
    for slice_path in class_dir.glob("*.png"):
        try:
            with Image.open(slice_path) as img:
                sizes.append(img.size) # img.size 返回 (宽度, 高度)
        except Exception as e:
            print(f"警告：无法读取图片 {slice_path}: {e}")
            
    if not sizes:
        return None
    
    # 将尺寸列表转换为numpy数组，方便计算
    sizes_arr = np.array(sizes)
    # 分别计算宽度和高度的中位数，并取整
    median_width = int(np.median(sizes_arr[:, 0]))
    median_height = int(np.median(sizes_arr[:, 1]))
    
    # 确保尺寸不小于我们设定的最小值
    final_width = max(median_width, MINIMUM_SIZE[0])
    final_height = max(median_height, MINIMUM_SIZE[1])
    
    return (final_width, final_height)

# --- 辅助函数2: 对一个类别的所有切片进行归一化处理 ---
def normalize_slices_for_class(class_dir, target_size, output_base_dir):
    """
    将一个类别的所有切片缩放到目标尺寸并保存。

    Args:
        class_dir (Path): 源切片目录。
        target_size (tuple): (目标宽度, 目标高度)。
        output_base_dir (Path): 标准化切片的根输出目录。
    """
    # 在输出目录中创建对应的子目录
    output_class_dir = output_base_dir / class_dir.name
    output_class_dir.mkdir(parents=True, exist_ok=True)
    
    # 遍历所有源切片
    for slice_path in class_dir.glob("*.png"):
        try:
            with Image.open(slice_path) as img:
                # 使用高质量的LANCZOS算法进行缩放
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                # 构建并保存到新目录
                output_path = output_class_dir / slice_path.name
                resized_img.save(output_path)
        except Exception as e:
            print(f"警告：处理文件 {slice_path} 时出错: {e}")

# --- 主执行函数 ---
def main():
    """
    遍历所有原始切片，计算每个类别的标准尺寸，
    并生成一个全新的、所有切片尺寸统一的“标准件库”。
    """
    print("--- 启动切片归一化程序：创建“标准件库” ---")
    
    if not SOURCE_SLICES_DIR.exists():
        print(f"错误：源切片目录 '{SOURCE_SLICES_DIR}' 不存在！")
        return
        
    # 获取所有类别的子目录
    class_dirs = [d for d in SOURCE_SLICES_DIR.iterdir() if d.is_dir()]
    
    print(f"找到 {len(class_dirs)} 个类别，开始处理...")
    
    # 使用tqdm创建进度条
    for class_dir in tqdm(class_dirs, desc="归一化进度"):
        # 1. 计算该类别的“标准尺寸”
        target_size = get_median_size(class_dir)
        
        if target_size:
            # 2. 对该类别的所有图片进行缩放和保存
            normalize_slices_for_class(class_dir, target_size, NORMALIZED_SLICES_DIR)
        else:
            print(f"警告：目录 '{class_dir.name}' 为空或无法读取，已跳过。")
            
    print("--- ✅ 切片归一化全部完成！---")
    print(f"所有“标准件”已保存到: '{NORMALIZED_SLICES_DIR}'")

if __name__ == '__main__':
    main()