# scripts/14_split_trajectories.py (V3.0 - Idempotent Final Version)
import shutil
from pathlib import Path
import random
import math

# --- 配置区 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 注意：我们将源目录硬编码为原始轨迹的生成位置
SOURCE_DIR = PROJECT_ROOT / "data" / "trajectories"
TRAIN_DIR = PROJECT_ROOT / "data" / "train_trajectories"
VAL_DIR = PROJECT_ROOT / "data" / "val_trajectories"
VAL_SPLIT = 0.25 # 明确设置为0.25，确保4个文件中至少有1个进入验证集

print("--- 启动幂等数据集划分脚本 V3.0 ---")

# --- 步骤 1: 清理目标目录，确保幂等性 ---
# 无论之前是什么状态，都先恢复到干净状态
print("正在清理旧的训练/验证集目录...")
if TRAIN_DIR.exists():
    shutil.rmtree(TRAIN_DIR)
if VAL_DIR.exists():
    shutil.rmtree(VAL_DIR)

TRAIN_DIR.mkdir(exist_ok=True)
VAL_DIR.mkdir(exist_ok=True)
print("清理完成。")

# --- 步骤 2: 从源目录加载文件列表 ---
all_files = list(SOURCE_DIR.glob("*.pkl.xz"))
print(f"在源目录 {SOURCE_DIR} 中找到 {len(all_files)} 个轨迹文件。")

if not all_files:
    print("\033[91m错误：源目录为空！请先运行 '12_generate_final_trajectories.py' 生成轨迹文件。\033[0m")
    exit()

random.seed(42) # 使用固定的随机种子，确保每次划分结果都一样
random.shuffle(all_files)

# --- 步骤 3: 健壮的划分逻辑 ---
num_total_files = len(all_files)
if num_total_files < 2:
    num_val_files = 0
else:
    # 使用向上取整确保验证集至少有1个文件
    num_val_files = math.ceil(num_total_files * VAL_SPLIT)
    # 保护逻辑，确保训练集也至少有1个文件
    if num_val_files >= num_total_files:
        num_val_files = num_total_files - 1

split_idx = num_val_files
val_files = all_files[:split_idx]
train_files = all_files[split_idx:]

# --- 步骤 4: 使用复制（copy）而不是移动（move） ---
print("正在将文件复制到目标目录...")
for f in train_files:
    shutil.copy(f, TRAIN_DIR / f.name)
for f in val_files:
    shutil.copy(f, VAL_DIR / f.name)

print("\n\033[92m数据集划分完毕：\033[0m")
print(f"  - 训练集: {len(train_files)} 个文件 -> {TRAIN_DIR}")
print(f"  - 验证集: {len(val_files)} 个文件 -> {VAL_DIR}")
print("\n原始轨迹文件仍保留在源目录中，可以安全地重复运行此脚本。")