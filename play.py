# C:\Users\zhz\Deepl\subway_surfers_ai\play.py (v5 - 最终稳定版)

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time
import onnxruntime as ort
import uiautomator2 as u2
import subprocess
import sys
from pathlib import Path

# --- 导入项目核心模块 ---
# 这个 try-except 结构可以增强脚本的兼容性
try:
    from subway_surfers_ai.decision.model import StARformer
    from subway_surfers_ai.decision.config import ModelConfig, TrainConfig
    from subway_surfers_ai.perception.state_builder import build_state_tensor
    from subway_surfers_ai.utils import constants
except ImportError:
    PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parent
    if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))
    from subway_surfers_ai.decision.model import StARformer
    from subway_surfers_ai.decision.config import ModelConfig, TrainConfig
    from subway_surfers_ai.perception.state_builder import build_state_tensor
    from subway_surfers_ai.utils import constants

# --- Windows端配置 ---
PROJECT_ROOT = Path(__file__).resolve().parent
ONNX_MODEL_PATH = PROJECT_ROOT / "models" / "one_best_v2.onnx"
DECISION_MODEL_PATH = PROJECT_ROOT / "models" / "decision_model.pt"
DEVICE_IP = "192.168.3.17:44339" 

def robust_connect(device_ip):
    """
    一个更健壮的连接函数，对齐业界实践。
    """
    print(f"正在尝试连接手机: {device_ip}...")
    try:
        d = u2.connect(device_ip)
        d.info 
        print("直接连接成功！")
        return d
    except Exception:
        print("直接连接失败。请确保：")
        print(f"1. 手机的无线调试已开启，且IP与端口为: {device_ip}")
        print(f"2. PC与手机在同一WiFi网络下。")
        print(f"3. 已通过USB线缆执行过一次 'python -m uiautomator2 init'。")
        sys.exit(1)

class Agent:
    def __init__(self, device_ip):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"决策模型将使用设备: {self.device}")
        
        print("正在加载感知模型...")
        providers = ['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=providers)
        self.input_name = self.ort_session.get_inputs()[0].name
        _, _, self.model_height, self.model_width = self.ort_session.get_inputs()[0].shape
        print("感知模型加载成功！")

        print("正在加载决策模型...")
        checkpoint = torch.load(DECISION_MODEL_PATH, map_location=self.device, weights_only=False)
        self.model_config = checkpoint['model_config']
        self.train_config = checkpoint['train_config']
        self.decision_model = StARformer(self.model_config).to(self.device)
        self.decision_model.load_state_dict(checkpoint['model_state_dict'])
        self.decision_model.eval()
        print("决策模型加载成功！")

        self.d = robust_connect(device_ip)
        self.screen_width, self.screen_height = self.d.window_size()
        print(f"手机连接成功！屏幕尺寸: {self.screen_width}x{self.screen_height}")

        self.sequence_length = self.train_config.sequence_length
        self.states, self.actions, self.rtgs, self.timesteps = [], [], [], []

    def get_yolo_objects(self, frame):
        height, width, _ = frame.shape
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.model_width, self.model_height))
        img = img.astype(np.float32) / 255.0
        blob = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
        outputs = self.ort_session.run(None, {self.input_name: blob})[0]
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]
        boxes, scores, class_ids = [], [], []
        for j in range(rows):
            classes_scores = outputs[0][j][4:]
            (_, maxScore, _, (_, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore > 0.25:
                scores.append(maxScore)
                class_ids.append(maxClassIndex)
                x_center, y_center, w, h = outputs[0][j][:4]
                x_scale, y_scale = width / self.model_width, height / self.model_height
                x1, y1 = (x_center - w / 2) * x_scale, (y_center - h / 2) * y_scale
                box_w, box_h = w * x_scale, h * y_scale
                boxes.append([x1, y1, box_w, box_h])
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)
        yolo_objects = []
        if len(indices) > 0:
            for k in indices.flatten():
                box = boxes[k]
                yolo_objects.append([
                    class_ids[k], (box[0] + box[2] / 2) / width, (box[1] + box[3] / 2) / height,
                    box[2] / width, box[3] / height,
                ])
        return yolo_objects

    @torch.no_grad()
    def get_action(self):
        # --- [核心修正] 采用最标准、最简洁的方式构建输入张量 ---
        
        # 1. 将历史列表转换为Numpy数组
        states_np = np.array(self.states)       # Shape: (T, H, W, C)
        actions_np = np.array(self.actions)     # Shape: (T,)
        rtgs_np = np.array(self.rtgs)           # Shape: (T,)
        timesteps_np = np.array(self.timesteps) # Shape: (T,)

        # 2. 转换为PyTorch张量并添加Batch维度 (B=1)
        states = torch.from_numpy(states_np).unsqueeze(0).to(self.device).float()
        actions = torch.from_numpy(actions_np).unsqueeze(0).to(self.device).long()
        rtgs = torch.from_numpy(rtgs_np).unsqueeze(0).to(self.device).float()
        timesteps = torch.from_numpy(timesteps_np).unsqueeze(0).to(self.device).long()

        # 3. 调整状态张量的维度以符合PyTorch CNN的期望 (B, T, H, W, C) -> (B, T, C, H, W)
        states = states.permute(0, 1, 4, 2, 3)

        # 4. 填充序列到固定长度
        current_len = states.shape[1]
        pad_len = self.sequence_length - current_len
        if pad_len > 0:
            states = F.pad(states, (0, 0, 0, 0, 0, 0, 0, pad_len))
            actions = F.pad(actions, (0, pad_len), value=0)
            rtgs = F.pad(rtgs, (0, pad_len), value=0)
            timesteps = F.pad(timesteps, (0, pad_len), value=0)

        # 5. 模型推理
        action_logits = self.decision_model(states, actions, rtgs.unsqueeze(-1), timesteps)
        
        # 6. 获取当前最后一个时间步的预测
        pred = action_logits[0, current_len - 1, :]
        action = torch.argmax(pred).item()
        return action

    def execute_action(self, action):
        center_x, center_y = self.screen_width // 2, self.screen_height // 2
        swipe_dist = self.screen_height // 6
        action_map = {1: 'up', 2: 'down', 3: 'left', 4: 'right'}
        action_str = action_map.get(action)
        if action_str:
            print(f"执行动作: {action_str.upper()}")
            if action_str == 'up': self.d.swipe(center_x, center_y, center_x, center_y - swipe_dist, 0.1)
            elif action_str == 'down': self.d.swipe(center_x, center_y, center_x, center_y + swipe_dist, 0.1)
            elif action_str == 'left': self.d.swipe(center_x, center_y, center_x - swipe_dist, center_y, 0.1)
            elif action_str == 'right': self.d.swipe(center_x, center_y, center_x + swipe_dist, center_y, 0.1)

    def run(self):
        print("\n--- AI已启动，按 Ctrl+C 停止 ---")
        
        timestep = 0
        current_rtg = 3600
        
        while True:
            try:
                start_time = time.time()
                
                # 1. 感知当前状态
                screenshot = self.d.screenshot(format="opencv")
                yolo_objects = self.get_yolo_objects(screenshot)
                state_tensor = build_state_tensor(yolo_objects)
                
                # 2. 将新状态加入历史
                self.states.append(state_tensor)
                self.timesteps.append(timestep)
                
                # 3. 如果历史序列为空，则填充初始动作和RTG
                if not self.actions:
                    self.actions.append(0)
                    self.rtgs.append(current_rtg)

                # 4. 决策
                action = self.get_action()
                
                # 5. 执行
                if action != 0:
                    self.execute_action(action)

                # 6. 将刚执行的动作和用于决策的RTG更新到历史中，为下一次循环准备
                self.actions.append(action)
                self.rtgs.append(current_rtg)
                
                # 7. 更新下一个时间步的变量
                timestep += 1
                current_rtg -= 0.01

                # 8. 统一维护所有序列长度
                while len(self.states) > self.sequence_length:
                    self.states.pop(0)
                    self.actions.pop(0)
                    self.rtgs.pop(0)
                    self.timesteps.pop(0)
                
                elapsed = time.time() - start_time
                print(f"循环耗时: {elapsed:.3f}s, 序列长度 (S/A/R/T): {len(self.states)}/{len(self.actions)}/{len(self.rtgs)}/{len(self.timesteps)}")
                time.sleep(max(0, 0.1 - elapsed))

            except KeyboardInterrupt:
                print("\n--- AI已停止 ---")
                break
            except Exception as e:
                print(f"\n--- 发生严重错误，AI停止 ---")
                print(f"错误详情: {e}")
                break

if __name__ == '__main__':
    agent = Agent(device_ip=DEVICE_IP)
    agent.run()