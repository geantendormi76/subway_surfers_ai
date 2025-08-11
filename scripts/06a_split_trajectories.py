# scripts/06a_split_trajectories.py (数据集划分工具)

import os
from pathlib import Path
import random
import shutil

def main():
    project_root = Path(__file__).resolve().parents[1]
    
    # [核心] 源目录应该是 trajectories，因为 v7 版本的 08 脚本直接生成干净数据
    source_dir = project_root / "data" / "trajectories"
    
    train_dir = project_root / "data" / "train_trajectories"
    val_dir = project_root / "data" / "val_trajectories"
    
    # 清理旧目录，确保每次划分都是全新的
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if val_dir.exists():
        shutil.rmtree(val_dir)
        
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    all_files = sorted(list(source_dir.glob("*.pkl.xz")))
    if not all_files:
        print(f"错误：在源目录 {source_dir} 中没有找到任何轨迹文件。请先运行 08_shengcheng_guiji_shuju.py。")
        return

    random.seed(42) # 使用固定种子确保每次划分结果一致
    random.shuffle(all_files)
    
    # 80/20 划分
    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print("--- 正在按回合划分数据集 ---")
    print(f"源目录: {source_dir}")
    print(f"总计 {len(all_files)} 个轨迹文件。")
    print(f"训练集: {len(train_files)} 个文件 -> 将移动到 {train_dir}")
    print(f"验证集: {len(val_files)} 个文件 -> 将移动到 {val_dir}")
    
    # 使用 shutil.move 来移动文件
    for f in train_files:
        shutil.move(str(f), str(train_dir / f.name))
        
    for f in val_files:
        shutil.move(str(f), str(val_dir / f.name))
        
    print("\n--- 数据集划分完成！ ---")
    print(f"源目录 {source_dir} 现在应该是空的。")

if __name__ == "__main__":
    main()