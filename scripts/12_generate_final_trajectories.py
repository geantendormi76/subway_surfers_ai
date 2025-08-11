# scripts/generate_final_trajectories.py
import cv2
import numpy as np
import sys
import pickle
import lzma
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm
import json

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subway_surfers_ai.utils import constants

# --- Configuration ---
class TrajectoryConfig:
    VIDEO_DIR = PROJECT_ROOT / "data" / "internet_videos" / "processed"
    ALIGNED_ACTION_DIR = PROJECT_ROOT / "data" / "final_aligned_trajectories"
    OUTPUT_DIR = PROJECT_ROOT / "data" / "trajectories" # 新的输出目录
    
    YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "one_best_v3.onnx"
    
    CONF_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.45

def load_aligned_actions(action_path):
    """加载精准对齐的动作JSON文件"""
    with open(action_path, 'r') as f:
        # JSON的键是字符串，需要转为整数帧号
        return {int(k): v for k, v in json.load(f).items()}

def extract_states_from_video(video_path, ort_session, config):
    """
    从视频中逐帧提取YOLO检测到的状态。
    这部分逻辑与旧的08脚本完全一致。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_name = ort_session.get_inputs()[0].name
    _, _, model_height, model_width = ort_session.get_inputs()[0].shape
    
    all_states = []
    pbar = tqdm(total=frame_count, desc=f"状态提取 (YOLO): {video_path.name}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width, _ = frame.shape
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (model_width, model_height))
        img = img.astype(np.float32) / 255.0
        blob = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
        
        outputs = ort_session.run(None, {input_name: blob})[0]
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]
        
        boxes, scores, class_ids = [], [], []
        for j in range(rows):
            classes_scores = outputs[0][j][4:]
            max_score = np.max(classes_scores)
            if max_score > config.CONF_THRESHOLD:
                class_id = np.argmax(classes_scores)
                scores.append(max_score)
                class_ids.append(class_id)
                x_center, y_center, w, h = outputs[0][j][:4]
                x_scale, y_scale = width / model_width, height / model_height
                x1 = (x_center - w / 2) * x_scale
                y1 = (y_center - h / 2) * y_scale
                box_w, box_h = w * x_scale, h * y_scale
                boxes.append([x1, y1, box_w, box_h])

        indices = cv2.dnn.NMSBoxes(boxes, scores, config.CONF_THRESHOLD, config.NMS_THRESHOLD)
        frame_state = []
        if len(indices) > 0:
            for k in indices.flatten():
                box = boxes[k]
                frame_state.append([
                    class_ids[k],
                    (box[0] + box[2] / 2) / width,
                    (box[1] + box[3] / 2) / height,
                    box[2] / width,
                    box[3] / height
                ])
        all_states.append(frame_state)
        pbar.update(1)
        
    pbar.close()
    cap.release()
    return all_states

def main():
    config = TrajectoryConfig()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    ort_session = ort.InferenceSession(str(config.YOLO_MODEL_PATH), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print(f"YOLO模型已从 {config.YOLO_MODEL_PATH} 加载。")
    
    action_files = list(config.ALIGNED_ACTION_DIR.glob("*_aligned_actions.json"))
    if not action_files:
        print(f"错误：在 {config.ALIGNED_ACTION_DIR} 目录中未找到任何对齐后的动作文件。")
        return

    for action_file in action_files:
        video_name_base = action_file.stem.replace("_aligned_actions", "")
        video_path = config.VIDEO_DIR / f"{video_name_base}.mp4"
        
        if not video_path.exists():
            print(f"警告：找不到与动作文件 {action_file.name} 对应的视频文件，跳过。")
            continue

        print(f"\n--- 正在处理视频: {video_path.name} ---")
        
        # 1. 提取所有帧的状态
        all_states = extract_states_from_video(video_path, ort_session, config)
        
        # 2. 加载对齐后的动作
        aligned_actions = load_aligned_actions(action_file)
        action_map_rev = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4}
        
        # 3. 组装轨迹并进行奖励塑形 (逻辑与旧08脚本的核心一致)
        trajectory = []
        last_coin_count = 0
        for frame_idx, state in enumerate(tqdm(all_states, desc="组装轨迹并进行奖励塑形")):
            action_str = aligned_actions.get(frame_idx, 'NOOP')
            current_action = action_map_rev.get(action_str, 0)
            
            player_detected = any(obj[0] == constants.CLASS_TO_ID['player'] for obj in state)
            
            # 奖励塑形
            reward = 0.01 if player_detected else -1.0 # 生存奖励/惩罚
            
            current_coin_count = sum(1 for obj in state if obj[0] == constants.CLASS_TO_ID['coin'])
            if frame_idx > 0 and player_detected:
                # 简化：假设金币是连续的，数量减少意味着吃到了
                if current_coin_count < last_coin_count:
                    reward += 0.1 * (last_coin_count - current_coin_count)
            last_coin_count = current_coin_count
            
            # 靠近障碍物的惩罚
            player_y = -1
            min_obstacle_dist = float('inf')
            for obj in state:
                if obj[0] == constants.CLASS_TO_ID['player']:
                    player_y = obj[2]
                    break
            if player_y != -1:
                for obj in state:
                    # 假设障碍物的类别ID在2到5之间
                    if constants.CLASS_TO_ID['train1'] <= obj[0] <= constants.CLASS_TO_ID['high_barrier']:
                        if obj[2] > player_y: # 只考虑前方的障碍物
                            min_obstacle_dist = min(min_obstacle_dist, obj[2] - player_y)
            
            if min_obstacle_dist < 0.2:
                reward -= 0.05
            
            trajectory.append({'state': state, 'action': current_action, 'reward': reward, 'timestep': frame_idx})
        
        # 4. 计算Return-to-Go (RTG)
        if trajectory:
            # 最后一个有效帧通常意味着失败
            trajectory[-1]['reward'] = -1.0
            
            running_rtg = 0.0
            for i in reversed(range(len(trajectory))):
                running_rtg += trajectory[i]['reward']
                trajectory[i]['rtg'] = running_rtg
        
        # 5. 保存轨迹文件
        output_path = config.OUTPUT_DIR / f"{video_name_base}.pkl.xz"
        with lzma.open(output_path, 'wb') as f:
            pickle.dump(trajectory, f)
        print(f"✅ 轨迹数据组装完成，共 {len(trajectory)} 个时间步。")
        print(f"   已成功保存到: {output_path}")

if __name__ == "__main__":
    main()