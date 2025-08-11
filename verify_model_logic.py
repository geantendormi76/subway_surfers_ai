# verify_model_logic.py (V5 - Robust Path Finding)
import torch
import torch.nn.functional as F
import pickle
import lzma
import numpy as np
from pathlib import Path
import sys
import json
from tqdm import tqdm

# --- 导入项目核心模块 ---
try:
    from subway_surfers_ai.decision.model import StARformer
    from subway_surfers_ai.decision.config import ModelConfig, TrainConfig
    from subway_surfers_ai.perception.state_builder import build_state_tensor
except ImportError:
    PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parent
    if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))
    from subway_surfers_ai.decision.model import StARformer
    from subway_surfers_ai.decision.config import ModelConfig, TrainConfig
    from subway_surfers_ai.perception.state_builder import build_state_tensor

# --- 配置 ---
PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_ROOT / "models" / "decision_model_weights.pt"
CONFIG_PATH = PROJECT_ROOT / "models" / "decision_model_config.json"
TRAIN_CONFIG_PATH = PROJECT_ROOT / "models" / "decision_train_config.json"

# --- 核心修正：自动查找验证集中的第一个轨迹文件 ---
VAL_TRAJECTORIES_DIR = PROJECT_ROOT / "data" / "val_trajectories"
try:
    # 查找所有 .pkl.xz 文件并排序，取第一个
    TRAJECTORY_PATH = sorted(list(VAL_TRAJECTORIES_DIR.glob("*.pkl.xz")))[0]
except IndexError:
    print(f"错误: 在验证集目录 '{VAL_TRAJECTORIES_DIR}' 中没有找到任何 .pkl.xz 轨迹文件。")
    print("请确保您已经运行了 '12_generate_final_trajectories.py' 和 '14_split_trajectories.py'。")
    sys.exit(1)
# --- 修正结束 ---


class OfflineAgent:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"将使用设备: {self.device}")

        with open(CONFIG_PATH, 'r') as f:
            model_params = json.load(f)
        self.model_config = ModelConfig(**model_params)
        
        with open(TRAIN_CONFIG_PATH, 'r') as f:
            train_params = json.load(f)
        self.train_config = TrainConfig(**train_params)

        self.decision_model = StARformer(self.model_config).to(self.device)
        state_dict = torch.load(WEIGHTS_PATH, map_location=self.device)
        self.decision_model.load_state_dict(state_dict)
        self.decision_model.eval()
        print("决策模型加载成功！")
        
        self.sequence_length = self.train_config.sequence_length
        self.states, self.actions, self.rtgs, self.timesteps = [], [], [], []

    @torch.no_grad()
    def get_action(self):
        # 这个方法与 play.py (v5) 中的 get_action 完全一致
        states = torch.from_numpy(np.array(self.states)).unsqueeze(0).to(self.device, dtype=torch.float32)
        actions = torch.tensor(self.actions, dtype=torch.long).unsqueeze(0).to(self.device)
        rtgs = torch.from_numpy(np.array(self.rtgs)).unsqueeze(0).to(self.device, dtype=torch.float32)
        timesteps = torch.tensor(self.timesteps, dtype=torch.long).unsqueeze(0).to(self.device)

        states = states.permute(0, 1, 4, 2, 3)
        current_len = states.shape[1]
        
        if actions.shape[1] < current_len:
            actions = F.pad(actions, (0, 1), value=0)

        pad_len = self.sequence_length - current_len
        if pad_len > 0:
            states = F.pad(states, (0, 0, 0, 0, 0, 0, 0, pad_len))
            actions = F.pad(actions, (0, pad_len), value=0)
            rtgs = F.pad(rtgs, (0, pad_len), value=0)
            timesteps = F.pad(timesteps, (0, pad_len), value=0)
        
        action_logits = self.decision_model(states, actions, rtgs.unsqueeze(-1), timesteps)
        pred = action_logits[0, current_len - 1, :]
        return torch.argmax(pred).item()

    def update_history(self, state, action, rtg, timestep):
        self.states.append(state)
        self.actions.append(action)
        self.rtgs.append(rtg)
        self.timesteps.append(timestep)
        
        while len(self.states) > self.sequence_length:
            self.states.pop(0)
            self.actions.pop(0)
            self.rtgs.pop(0)
            self.timesteps.pop(0)

def main():
    print(f"--- 模型逻辑离线回放诊断 ---")
    agent = OfflineAgent()

    print(f"正在加载轨迹数据: {TRAJECTORY_PATH}")
    with lzma.open(TRAJECTORY_PATH, 'rb') as f:
        trajectory = pickle.load(f)
    print(f"轨迹数据加载成功，总共 {len(trajectory)} 步。")

    action_map = {0: 'NOOP', 1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT'}
    
    # 模拟 play.py 的主循环
    for i in tqdm(range(len(trajectory)), desc="轨迹回放中"):
        step_data = trajectory[i]
        
        # 注意：轨迹文件中保存的是YOLO对象列表，需要先构建成状态张量
        current_state = build_state_tensor(step_data['state'])
        current_rtg = step_data['rtg']
        current_timestep = step_data['timestep']
        expert_action = step_data['action']
        
        # 更新历史记录 (用的是专家上一步的真实动作)
        last_action = trajectory[i-1]['action'] if i > 0 else 0
        agent.update_history(current_state, last_action, current_rtg, current_timestep)
        
        # 模型根据当前状态和历史进行预测
        predicted_action = agent.get_action()
        
        # 只在专家做出动作时，或模型做出动作时打印，避免刷屏
        if expert_action != 0 or predicted_action != 0:
            expert_str = action_map.get(expert_action, '未知')
            pred_str = action_map.get(predicted_action, '未知')
            
            # 使用颜色区分
            if expert_action == predicted_action:
                # 绿色: 预测正确
                print(f"\033[92m[帧 {current_timestep:4d}] 专家动作: {expert_str:<6} | 模型预测: {pred_str:<6} (✅ 一致)\033[0m")
            else:
                # 红色: 预测错误
                print(f"\033[91m[帧 {current_timestep:4d}] 专家动作: {expert_str:<6} | 模型预测: {pred_str:<6} (❌ 不一致)\033[0m")

if __name__ == '__main__':
    main()