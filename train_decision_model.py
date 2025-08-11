# train_decision_model.py (v7 - 带验证与早停版)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import math
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from subway_surfers_ai.decision.model import StARformer
from subway_surfers_ai.decision.config import ModelConfig, TrainConfig
from subway_surfers_ai.decision.dataloader import create_dataloader

def get_lr_scheduler(optimizer, total_epochs, steps_per_epoch, warmup_epochs=3):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def train(main_config_path):
    # 1. 配置加载
    with open(main_config_path, 'r', encoding='utf-8') as f:
        main_config = yaml.safe_load(f)
    
    # [核心修改] 定义训练集和验证集的目录路径
    train_data_path = PROJECT_ROOT / "data" / "train_trajectories"
    val_data_path = PROJECT_ROOT / "data" / "val_trajectories"

    train_params = main_config.get('train_params', {})
    model_params = main_config.get('model_params', {})
    train_config = TrainConfig(**train_params)
    model_config = ModelConfig(**model_params)
    
    print(f"设备: {train_config.device}")
    print(f"将从以下目录加载训练数据: {train_data_path}")
    print(f"将从以下目录加载验证数据: {val_data_path}")
    
    # 2. 初始化
    model = StARformer(model_config).to(train_config.device)
    train_dataloader = create_dataloader(train_data_path, train_config)
    val_dataloader = create_dataloader(val_data_path, train_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    
    class_weights = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0]).to(train_config.device)
    # 检查配置中是否有label_smoothing, 没有则默认为0
    label_smoothing = train_params.get('label_smoothing', 0.0)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    
    scheduler = get_lr_scheduler(optimizer, train_config.epochs, len(train_dataloader))

    # 3. 早停机制变量
    best_val_loss = float('inf')
    patience = 10 # 如果连续10个epoch验证集loss没有改善，就停止
    patience_counter = 0
    best_model_state = None

    # 4. 训练与验证循环
    for epoch in range(train_config.epochs):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{train_config.epochs} [训练]")
        for batch in pbar:
            states, actions, rtgs, timesteps = batch['states'].to(train_config.device), batch['actions'].to(train_config.device), batch['rtgs'].to(train_config.device), batch['timesteps'].to(train_config.device)
            action_logits = model(states, actions, rtgs, timesteps)
            loss = loss_fn(action_logits.view(-1, model_config.act_dim), actions.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # --- 验证循环 ---
        model.eval()
        total_val_loss = 0
        total_correct_actions = 0
        total_actions = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{train_config.epochs} [验证]"):
                states, actions, rtgs, timesteps = batch['states'].to(train_config.device), batch['actions'].to(train_config.device), batch['rtgs'].to(train_config.device), batch['timesteps'].to(train_config.device)
                action_logits = model(states, actions, rtgs, timesteps)
                loss = loss_fn(action_logits.view(-1, model_config.act_dim), actions.view(-1))
                total_val_loss += loss.item()

                preds = torch.argmax(action_logits, dim=-1)
                mask = actions > 0
                total_correct_actions += (preds[mask] == actions[mask]).sum().item()
                total_actions += mask.sum().item()

        avg_val_loss = total_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
        action_accuracy = total_correct_actions / total_actions if total_actions > 0 else 0
        
        print(f"Epoch {epoch+1} | 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f} | 关键动作准确率: {action_accuracy:.4f}")

        # --- 早停逻辑 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            print(f"  (发现新最佳模型！保存中...)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  (验证损失已连续 {patience} 个Epoch未改善。触发早停。)")
                break
    
    # 5. 保存最佳模型
    print("训练完成！")
    if best_model_state:
        save_dir = PROJECT_ROOT / "models"
        save_dir.mkdir(parents=True, exist_ok=True)
        weights_path = save_dir / "decision_model_weights.pt"
        torch.save(best_model_state, weights_path)
        print(f"✅ 最佳模型权重已成功保存到: {weights_path}")
        model_config_path = save_dir / "decision_model_config.json"
        with open(model_config_path, 'w') as f: json.dump(model_params, f, indent=4)
        train_config_path = save_dir / "decision_train_config.json"
        with open(train_config_path, 'w') as f:
            train_params_dict = {k: v for k, v in vars(train_config).items() if not k.startswith('__')}
            json.dump(train_params_dict, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="decision_model_config.yaml",
                        help="主训练配置文件的路径")
    args = parser.parse_args()
    train(args.config)