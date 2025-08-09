# /home/zhz/Deepl/subway_surfers_ai/subway_surfers_ai/decision/dataloader.py (v3 - 目录加载版)

import torch
import pickle
import lzma
import numpy as np
import random
from torch.utils.data import Dataset, WeightedRandomSampler
from subway_surfers_ai.perception.state_builder import build_state_tensor
from pathlib import Path

class TrajectoryDataset(Dataset):
    def __init__(self, trajectory_dir, config):
        self.config = config
        self.sequence_length = config.sequence_length
        
        # [核心修改] 遍历目录加载所有 .pkl.xz 文件
        trajectory_paths = sorted(list(Path(trajectory_dir).glob("*.pkl.xz")))
        if not trajectory_paths:
            raise FileNotFoundError(f"在目录 {trajectory_dir} 中没有找到任何 .pkl.xz 轨迹文件")

        print(f"正在从目录 {trajectory_dir} 加载轨迹数据...")
        
        trajectories = []
        for path in trajectory_paths:
            print(f"  - 加载: {path.name}")
            with lzma.open(path, 'rb') as f:
                trajectories.append(pickle.load(f))
        
        # [核心修改] 将多个轨迹列表合并为一个长的总轨迹列表
        self.trajectory = [step for traj in trajectories for step in traj]
        
        self._initialize()
        print(f"加载完成！共 {len(trajectories)} 个轨迹, 总计 {len(self.trajectory)} 步, 可生成 {len(self.valid_indices)} 个序列。")

    def _initialize(self):
        """
        [核心对齐] 预计算所有有效索引和采样权重。
        """
        self.valid_indices = []
        self.sample_weights = []
        
        # 找到所有动作帧
        action_frames = {i for i, step in enumerate(self.trajectory) if step['action'] != 0}
        
        # 动作比例，用于计算重采样权重
        # [鲁棒性修改] 避免除以零的错误
        if len(self.trajectory) == 0:
            action_ratio = 0.0
        else:
            action_ratio = len(action_frames) / len(self.trajectory)

        if action_ratio == 0.0:
            weight_action = 1.0
        else:
            weight_action = 1.0 / action_ratio
            
        if action_ratio == 1.0:
            weight_noop = 1.0
        else:
            weight_noop = 1.0 / (1.0 - action_ratio)

        last_action_frame = -float('inf')
        for i in range(len(self.trajectory) - self.sequence_length):
            self.valid_indices.append(i)
            
            # --- [对齐] 动作重采样逻辑 ---
            # 查找当前序列片段的结束帧 (i + sequence_length - 1)
            end_frame_idx = i + self.sequence_length - 1
            is_action_frame = self.trajectory[end_frame_idx]['action'] != 0
            
            if is_action_frame:
                last_action_frame = end_frame_idx

            # 论文中的重采样策略：提高动作帧和其后一定范围内帧的权重
            # Sj = max(1/(1-ra), 1/(ra*(j-ti+1)))
            # 这是一个简化但有效的实现
            if end_frame_idx - last_action_frame < self.config.action_resample_window:
                self.sample_weights.append(weight_action)
            else:
                self.sample_weights.append(weight_noop)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        
        # --- [对齐] 随机间隔采样逻辑 ---
        segment_indices = [start_idx]
        current_idx = start_idx
        for _ in range(self.sequence_length - 1):
            # 在 [1, random_interval] 之间随机选择一个步长
            step = random.randint(1, self.config.random_interval)
            current_idx += step
            # 确保索引不越界
            if current_idx >= len(self.trajectory):
                current_idx = len(self.trajectory) - 1
            segment_indices.append(current_idx)
            
        segment = [self.trajectory[i] for i in segment_indices]
        
        # 后续处理与之前版本相同
        states = [build_state_tensor(step['state']) for step in segment]
        actions = [step['action'] for step in segment]
        rtgs = [step['rtg'] for step in segment]
        timesteps = [step['timestep'] for step in segment]

        states_tensor = torch.from_numpy(np.stack(states, axis=0).transpose(0, 3, 1, 2)).float()
        
        return {
            'states': states_tensor,
            'actions': torch.tensor(actions, dtype=torch.long),
            'rtgs': torch.tensor(rtgs, dtype=torch.float32).unsqueeze(-1),
            'timesteps': torch.tensor(timesteps, dtype=torch.long)
        }

def create_dataloader(trajectory_dir, config): # [核心修改] 参数名改为 trajectory_dir
    """
    创建一个包含 WeightedRandomSampler 的 DataLoader。
    """
    dataset = TrajectoryDataset(trajectory_dir, config)
    
    # [鲁棒性修改] 确保有样本可以采样
    if len(dataset) == 0:
        # 返回一个空的 DataLoader
        return torch.utils.data.DataLoader([])

    sampler = WeightedRandomSampler(
        weights=dataset.sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True # 推荐开启以加速数据到GPU的传输
    )
    return dataloader