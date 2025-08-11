# C:\Users\zhz\Deepl\subway_surfers_ai\play.py (v5 - 最终稳定版)

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time
import onnxruntime as ort
import uiautomator2 as u2
import sys
from pathlib import Path
import json

# --- 导入项目核心模块 ---
# ... (这部分保持不变) ...
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
# ... (这部分保持不变) ...
PROJECT_ROOT = Path(__file__).resolve().parent
ONNX_MODEL_PATH = PROJECT_ROOT / "models" / "one_best_v3.onnx"
WEIGHTS_PATH = PROJECT_ROOT / "models" / "decision_model_weights.pt"
CONFIG_PATH = PROJECT_ROOT / "models" / "decision_model_config.json"
TRAIN_CONFIG_PATH = PROJECT_ROOT / "models" / "decision_train_config.json"
DEVICE_IP = "192.168.3.17:35243" # 请确保这里的IP和端口是最新的

# ... (robust_connect 方法保持不变) ...
def robust_connect(device_ip):
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
        # ... (init 方法保持 v3 - 权重配置分离版 不变) ...
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"决策模型将使用设备: {self.device}")
        print("正在加载感知模型...")
        providers = ['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=providers)
        self.input_name = self.ort_session.get_inputs()[0].name
        _, _, self.model_height, self.model_width = self.ort_session.get_inputs()[0].shape
        print("感知模型加载成功！")
        print("正在加载决策模型...")
        with open(CONFIG_PATH, 'r') as f:
            model_params = json.load(f)
        self.model_config = ModelConfig(**model_params)
        with open(TRAIN_CONFIG_PATH, 'r') as f:
            train_params = json.load(f)
        self.train_config = TrainConfig(**train_params)
        print("模型配置加载成功！")
        self.decision_model = StARformer(self.model_config).to(self.device)
        print(f"正在加载模型权重: {WEIGHTS_PATH}")
        state_dict = torch.load(WEIGHTS_PATH, map_location=self.device)
        self.decision_model.load_state_dict(state_dict)
        self.decision_model.eval()
        print("决策模型加载成功！")
        self.d = robust_connect(device_ip)
        self.screen_width, self.screen_height = self.d.window_size()
        print(f"手机连接成功！屏幕尺寸: {self.screen_width}x{self.screen_height}")
        self.sequence_length = self.train_config.sequence_length
        self.states, self.actions, self.rtgs, self.timesteps = [], [], [], []

    def get_yolo_objects(self, frame):
        # ... (此方法不变) ...
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
        # [核心修改] get_action不再负责填充，只负责转换和推理
        states = torch.from_numpy(np.array(self.states)).unsqueeze(0).to(self.device, dtype=torch.float32)
        actions = torch.tensor(self.actions, dtype=torch.long).unsqueeze(0).to(self.device)
        rtgs = torch.from_numpy(np.array(self.rtgs)).unsqueeze(0).to(self.device, dtype=torch.float32)
        timesteps = torch.tensor(self.timesteps, dtype=torch.long).unsqueeze(0).to(self.device)

        states = states.permute(0, 1, 4, 2, 3)
        
        action_logits = self.decision_model(states, actions, rtgs.unsqueeze(-1), timesteps)
        
        pred = action_logits[0, -1, :] # 直接取最后一个时间步的输出
        action = torch.argmax(pred).item()
        return action

    def execute_action(self, action):
        # ... (此方法不变) ...
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
        
        self.states, self.actions, self.rtgs, self.timesteps = [], [], [], []
        no_player_frames = 0
        
        while True:
            try:
                start_time = time.time()
                
                # 1. 感知
                screenshot = self.d.screenshot(format="opencv")
                yolo_objects = self.get_yolo_objects(screenshot)

                player_detected = any(obj[0] == constants.CLASS_TO_ID['player'] for obj in yolo_objects)
                if player_detected:
                    no_player_frames = 0
                else:
                    no_player_frames += 1

                if no_player_frames > 15:
                    print("\n--- [状态] 连续未检测到玩家，判定游戏结束。AI停止。 ---")
                    break

                # 2. 准备模型输入序列
                # [核心修改] 先维护长度，再添加新数据，最后填充
                current_state = build_state_tensor(yolo_objects)
                
                # 如果是第一帧，需要一个虚拟的前置动作
                if not self.actions:
                    self.actions.append(0)

                # 统一添加
                self.states.append(current_state)
                self.rtgs.append(current_rtg)
                self.timesteps.append(timestep)
                
                # 统一维护长度
                while len(self.states) > self.sequence_length:
                    self.states.pop(0)
                    self.actions.pop(0)
                    self.rtgs.pop(0)
                    self.timesteps.pop(0)

                # [核心修改] 在这里进行填充，确保送入模型的长度是正确的
                temp_states = np.array(self.states)
                temp_actions = np.array(self.actions)
                temp_rtgs = np.array(self.rtgs)
                temp_timesteps = np.array(self.timesteps)

                pad_len = self.sequence_length - len(temp_states)
                if pad_len > 0:
                    temp_states = np.pad(temp_states, ((0, pad_len), (0, 0), (0, 0), (0, 0)), 'constant')
                    temp_actions = np.pad(temp_actions, (0, pad_len), 'constant')
                    temp_rtgs = np.pad(temp_rtgs, (0, pad_len), 'constant')
                    temp_timesteps = np.pad(temp_timesteps, (0, pad_len), 'constant')

                # 3. 决策
                action = self.get_action() # get_action现在只负责推理
                
                # 4. 执行
                if action != 0:
                    self.execute_action(action)

                # 5. 更新历史 (用真实的、未填充的action)
                self.actions.append(action)
                
                # 6. 更新时间步
                timestep += 1
                current_rtg -= 0.01

                elapsed = time.time() - start_time
                print(f"循环耗时: {elapsed:.3f}s, 真实序列长度 (S/A/R/T): {len(self.states)}/{len(self.actions)}/{len(self.rtgs)}/{len(self.timesteps)}")
                time.sleep(max(0, 0.1 - elapsed))

            # ... (except 块保持不变) ...
            except KeyboardInterrupt:
                print("\n--- AI已停止 ---")
                break
            except Exception as e:
                print(f"\n--- 发生严重错误，AI停止 ---")
                print(f"错误详情: {e}")
                import traceback
                traceback.print_exc()
                break

if __name__ == '__main__':
    agent = Agent(device_ip=DEVICE_IP)
    agent.run()