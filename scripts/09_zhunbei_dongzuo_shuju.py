# scripts/zhunbei_dongzuo_shuju.py
import torch
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split

# 确保可以导入项目模块
XIANGMU_GEN_MULU = Path(__file__).resolve().parents[1]
if str(XIANGMU_GEN_MULU) not in sys.path:
    sys.path.insert(0, str(XIANGMU_GEN_MULU))

from subway_surfers_ai.action_recognition.dataloader import ActionRecognitionDataset

def main():
    print("--- 开始准备动作识别数据集 ---")
    
    BIAOZHU_LUJING = XIANGMU_GEN_MULU / "data" / "dongzuo_shibie_shuju" / "annotations" / "dongzuo_biaozhu.json"
    YUAN_ZHEN_MULU = XIANGMU_GEN_MULU / "data" / "dongzuo_shibie_shuju" / "raw_frames"
    SHUCHU_MULU = XIANGMU_GEN_MULU / "data" / "dongzuo_shibie_shuju" / "processed_dataset"
    SHUCHU_MULU.mkdir(parents=True, exist_ok=True)

    # 1. 初始化完整数据集
    dataset = ActionRecognitionDataset(BIAOZHU_LUJING, YUAN_ZHEN_MULU)
    print(f"成功加载完整数据集，共 {len(dataset)} 个样本。")

    # 2. 划分训练集和验证集
    indices = list(range(len(dataset)))
    labels = [item['action'] for item in dataset.biaozhu_liebiao]
    
    # 使用 stratify 参数确保训练集和验证集中各类别的比例相似
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    print(f"数据集已划分为：{len(train_dataset)} 个训练样本，{len(val_dataset)} 个验证样本。")

    # 3. 保存处理好的数据集到磁盘
    # 这是一个好习惯，避免每次训练都重新读取和处理原始图片
    torch.save(train_dataset, SHUCHU_MULU / "xunlianji.pt")
    torch.save(val_dataset, SHUCHU_MULU / "yanzhengji.pt")
    
    print(f"✅ 预处理完成的数据集已保存到: {SHUCHU_MULU}")

if __name__ == "__main__":
    main()