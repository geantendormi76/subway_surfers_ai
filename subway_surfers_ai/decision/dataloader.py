# /home/zhz/Deepl/subway_surfers_ai/subway_surfers_ai/decision/dataloader.py (v3 - 支持多数据集)

import torch
import pickle
import lzma
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from tqdm import tqdm

# 确保能导入项目模块
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from subway_surfers_ai.perception.state_builder import build_state_tensor

class TrajectoryDataset(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.sequence_length = config.sequence_length
        
        # --- [核心修改] 加载并拼接所有轨迹数据 ---
        self._load_and_merge_trajectories(data_path)
        
        self._initialize()
        print(f"所有轨迹加载完成！共 {len(self.trajectory)} 步, 可生成 {len(self.valid_indices)} 个序列。")

    def _load_and_merge_trajectories(self, data_path):
        """
        [对齐katacr] 支持加载单个文件或整个目录下的所有 .pkl.xz 文件。
        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"数据路径不存在: {path}")

        if path.is_file():
            files_to_load = [path]
        elif path.is_dir():
            files_to_load = sorted(list(path.glob('*.pkl.xz')))
        
        if not files_to_load:
            raise ValueError(f"在路径 {path} 中没有找到任何 .pkl.xz 轨迹文件。")

        print(f"正在从以下文件加载轨迹数据: {files_to_load}")
        
        all_trajectories = []
        for file_path in tqdm(files_to_load, desc="加载并合并轨迹文件"):
            with lzma.open(file_path, 'rb') as f:
                all_trajectories.extend(pickle.load(f))
        
        self.trajectory = all_trajectories

    def _initialize(self):
        self.valid_indices = []
        self.sample_weights = []
        action_frames = {i for i, step in enumerate(self.trajectory) if step['action'] != 0}
        
        # 处理没有动作的罕见情况
        if len(action_frames) == 0:
            action_ratio = 0.0
            weight_action = 1.0
            weight_noop = 1.0
        else:
            action_ratio = len(action_frames) / len(self.trajectory)
            weight_action = 1.0 / action_ratio
            weight_noop = 1.0 / (1.0 - action_ratio)

        last_action_frame = -float('inf')
        for i in range(len(self.trajectory) - self.sequence_length):
            self.valid_indices.append(i)
            end_frame_idx = i + self.sequence_length - 1
            is_action_frame = self.trajectory[end_frame_idx]['action'] != 0
            if is_action_frame:
                last_action_frame = end_frame_idx
            if end_frame_idx - last_action_frame < self.config.action_resample_window:
                self.sample_weights.append(weight_action)
            else:
                self.sample_weights.append(weight_noop)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        segment_indices = [start_idx]
        current_idx = start_idx
        for _ in range(self.sequence_length - 1):
            step = random.randint(1, self.config.random_interval)
            current_idx += step
            if current_idx >= len(self.trajectory):
                current_idx = len(self.trajectory) - 1
            segment_indices.append(current_idx)
        segment = [self.trajectory[i] for i in segment_indices]
        
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

def create_dataloader(data_path, config):
    """
    创建一个包含 WeightedRandomSampler 的 DataLoader。
    """
    dataset = TrajectoryDataset(data_path, config)
    sampler = WeightedRandomSampler(
        weights=dataset.sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers
    )
    return dataloader