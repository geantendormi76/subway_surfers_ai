# /home/zhz/Deepl/subway_surfers_ai/train_decision_model.py (v3 - 简化加载版)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm

# 确保能导入项目内的模块
import sys
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from subway_surfers_ai.decision.model import StARformer
from subway_surfers_ai.decision.config import ModelConfig, TrainConfig
from subway_surfers_ai.decision.dataloader import create_dataloader

def train(main_config_path):
    # --- 1. 加载主配置文件 ---
    with open(main_config_path, 'r') as f:
        main_config = yaml.safe_load(f)
    
    # --- 2. [核心修改] 直接从主配置文件获取数据目录路径 ---
    data_path = main_config.get('data')
    if not data_path:
        raise ValueError("主配置文件中未找到 'data' 键，用于指向轨迹数据目录。")

    # --- 3. 组合配置并初始化 ---
    train_params = main_config.get('train_params', {})
    model_params = main_config.get('model_params', {})

    train_config = TrainConfig(**train_params)
    model_config = ModelConfig(**model_params)
    
    print(f"设备: {train_config.device}")
    print(f"将从以下目录加载数据: {data_path}")
    
    # --- 4. 初始化模型、数据加载器、优化器 ---
    model = StARformer(model_config).to(train_config.device)
    # [核心修改] 将 data_path (目录路径) 直接传给 create_dataloader
    dataloader = create_dataloader(data_path, train_config)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=train_config.learning_rate, 
        weight_decay=train_config.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()
    
    # --- 5. 训练循环 (保持不变) ---
    for epoch in range(train_config.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{train_config.epochs}")
        
        for batch in pbar:
            states = batch['states'].to(train_config.device)
            actions = batch['actions'].to(train_config.device)
            rtgs = batch['rtgs'].to(train_config.device)
            timesteps = batch['timesteps'].to(train_config.device)
            target_actions = actions

            action_logits = model(states, actions, rtgs, timesteps)
            
            loss = loss_fn(action_logits.view(-1, model_config.act_dim), target_actions.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}")

    print("训练完成！")
    
    # --- 6. 保存模型 (保持不变) ---
    save_path = PROJECT_ROOT / "models" / "decision_model.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'train_config': train_config
        }, save_path)
    print(f"✅ 模型已成功保存到: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="decision_model_config.yaml",
                        help="主训练配置文件的路径")
    args = parser.parse_args()
    train(args.config)