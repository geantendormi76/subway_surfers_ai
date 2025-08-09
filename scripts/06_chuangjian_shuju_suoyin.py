# /home/zhz/Deepl/subway_surfers_ai/scripts/create_dataset_index.py

import sys
from pathlib import Path
import random

# --- [核心] 确保能找到项目根目录 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# --- [核心] 配置区 ---
# 定义我们所有的数据源目录
REAL_IMAGES_DIR = PROJECT_ROOT / "data" / "images"
SYNTHETIC_IMAGES_DIR = PROJECT_ROOT / "data" / "yolo_dataset" / "images"
# 定义索引文件的输出目录
OUTPUT_DIR = PROJECT_ROOT / "data"

# 定义验证集在真实数据中占的比例
VALIDATION_SPLIT_RATIO = 0.2

def create_index_files():
    """
    遍历真实数据和合成数据目录，创建YOLOv8所需的 train.txt 和 val.txt 索引文件。
    """
    print("--- 启动数据集索引文件生成程序 ---")

    # --- 1. 收集所有图片路径 ---
    real_image_paths = sorted([p for p in REAL_IMAGES_DIR.glob('*.png')])
    synthetic_image_paths = sorted([p for p in SYNTHETIC_IMAGES_DIR.glob('*.jpg')])

    if not real_image_paths:
        print(f"错误：在 '{REAL_IMAGES_DIR}' 中没有找到任何真实图片 (.png)！")
        return

    print(f"找到 {len(real_image_paths)} 张真实图片。")
    print(f"找到 {len(synthetic_image_paths)} 张合成图片。")

    # --- 2. 划分真实数据为训练集和验证集 ---
    # 打乱真实图片列表以确保随机划分
    random.shuffle(real_image_paths)
    
    split_index = int(len(real_image_paths) * (1 - VALIDATION_SPLIT_RATIO))
    real_train_paths = real_image_paths[:split_index]
    real_val_paths = real_image_paths[split_index:]

    print(f"真实数据已划分为: {len(real_train_paths)} 张训练图片, {len(real_val_paths)} 张验证图片。")

    # --- 3. 构建最终的训练集和验证集路径列表 ---
    # 训练集 = 真实训练集 + 全部合成集
    final_train_paths = real_train_paths + synthetic_image_paths
    # 验证集 = 真实验证集
    final_val_paths = real_val_paths
    
    random.shuffle(final_train_paths) # 再次打乱，让真实数据和合成数据混合

    # --- 4. 写入索引文件 ---
    def write_index_file(filepath, paths):
        # 将绝对路径转换为相对于 data 目录的相对路径
        relative_paths = [p.relative_to(OUTPUT_DIR) for p in paths]
        with open(filepath, 'w') as f:
            for path in relative_paths:
                # YOLO需要 'images/xxx.png' 这样的格式
                f.write(f"./{path.as_posix()}\n") 
        print(f"✅ 成功写入 {len(paths)} 条记录到 {filepath}")

    write_index_file(OUTPUT_DIR / "train.txt", final_train_paths)
    write_index_file(OUTPUT_DIR / "val.txt", final_val_paths)

    print("--- 数据集索引文件生成完毕！ ---")

if __name__ == '__main__':
    create_index_files()