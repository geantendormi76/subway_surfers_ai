# /home/zhz/Deepl/subway_surfers_ai/subway_surfers_ai/decision/dataloader.py (v2 - 对齐katacr)

import torch
import pickle
import lzma
import numpy as np
import random
from torch.utils.data import Dataset, WeightedRandomSampler
from subway_surfers_ai.perception.state_builder import build_state_tensor

class TrajectoryDataset(Dataset):
    def __init__(self, trajectory_path, config):
        self.config = config
        self.sequence_length = config.sequence_length
        
        print(f"正在加载轨迹数据: {trajectory_path}...")
        with lzma.open(trajectory_path, 'rb') as f:
            self.trajectory = pickle.load(f)
        
        self._initialize()
        print(f"加载完成！共 {len(self.trajectory)} 步, 可生成 {len(self.valid_indices)} 个序列。")

    def _initialize(self):
        """
        [核心对齐] 预计算所有有效索引和采样权重。
        """
        self.valid_indices = []
        self.sample_weights = []
        
        # 找到所有动作帧
        action_frames = {i for i, step in enumerate(self.trajectory) if step['action'] != 0}
        
        # 动作比例，用于计算重采样权重
        action_ratio = len(action_frames) / len(self.trajectory)
        weight_action = 1.0 / action_ratio
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

def create_dataloader(trajectory_path, config):
    """
    创建一个包含 WeightedRandomSampler 的 DataLoader。
    """
    dataset = TrajectoryDataset(trajectory_path, config)
    sampler = WeightedRandomSampler(
        weights=dataset.sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers
    )
    return dataloader