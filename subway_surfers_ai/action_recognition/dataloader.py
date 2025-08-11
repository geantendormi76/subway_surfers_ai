# subway_surfers_ai/dongzuo_shibie/shuju_jiazaiqi.py
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import json

class ActionRecognitionDataset(Dataset):
    """
    用于动作识别模型的PyTorch数据集类。
    它负责从标注文件中加载信息，并从磁盘读取、预处理视频帧。
    """
    def __init__(self, biaozhu_lujing, zhen_mulu, zhen_geshu=8, tupian_chicun=(64, 64)):
        """
        Args:
            biaozhu_lujing (Path): dongzuo_biaozhu.json 文件的路径。
            zhen_mulu (Path): 存放原始视频帧的目录。
            zhen_geshu (int): 每个动作样本包含的帧数。
            tupian_chicun (tuple): 将每帧图像缩放到的尺寸 (W, H)。
        """
        self.zhen_mulu = Path(zhen_mulu)
        self.zhen_geshu = zhen_geshu
        self.tupian_chicun = tupian_chicun
        
        with open(biaozhu_lujing, 'r') as f:
            self.biaozhu_liebiao = json.load(f)
        
        self.action_to_id = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}

    def __len__(self):
        return len(self.biaozhu_liebiao)

    def __getitem__(self, idx):
        biaozhu = self.biaozhu_liebiao[idx]
        start_frame = biaozhu['start_frame']
        end_frame = biaozhu['end_frame']
        action = biaozhu['action']
        video_name = biaozhu['video_name']

        # 均匀采样N帧
        frame_indices = np.linspace(start_frame, end_frame, self.zhen_geshu, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            frame_name = f"{video_name}_frame_{frame_idx:05d}.png"
            frame_path = self.zhen_mulu / frame_name
            
            # 读取并预处理帧
            image = cv2.imread(str(frame_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 转换为灰度图以减少计算量
            image = cv2.resize(image, self.tupian_chicun)
            frames.append(image)
        
        # 将多帧堆叠为一个张量
        # (N, H, W) -> (N, H, W)
        frames_np = np.stack(frames, axis=0)
        
        # 转换为 PyTorch 张量并归一化
        # (N, H, W) -> (N, H, W)
        frames_tensor = torch.from_numpy(frames_np).float() / 255.0
        
        # 获取动作标签
        label = self.action_to_id[action]
        
        return frames_tensor, torch.tensor(label, dtype=torch.long)