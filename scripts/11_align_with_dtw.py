# scripts/align_actions_with_dtw.py (V2.1 - Robust, YOLO-Integrated)
import numpy as np
import cv2
from pathlib import Path
import sys
import json
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import onnxruntime as ort
import argparse

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subway_surfers_ai.utils import constants

# --- Configuration ---
class DTWConfig:
    ANNOTATED_ACTION_DIR = PROJECT_ROOT / "data" / "annotated_actions"
    YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "one_best_v3.onnx"
    OUTPUT_DIR = PROJECT_ROOT / "data" / "final_aligned_trajectories"
    # 动作ID到视觉特征的映射 (跳跃 -> Y坐标减小, 下滚 -> Y坐标增大)
    ACTION_TO_FEATURE_MAP = {'UP': -1.0, 'DOWN': 1.0}

class PlayerDetector:
    """一个专门用于检测玩家并提取其Y坐标的类"""
    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(str(model_path), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.ort_session.get_inputs()[0].name
        _, _, self.model_height, self.model_width = self.ort_session.get_inputs()[0].shape
        self.last_known_y = 0.5 # 初始位置在屏幕中间

    def get_player_y(self, frame):
        height, width, _ = frame.shape
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.model_width, self.model_height))
        img = img.astype(np.float32) / 255.0
        blob = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
        
        outputs = self.ort_session.run(None, {self.input_name: blob})[0]
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        player_boxes = []
        for j in range(rows):
            classes_scores = outputs[0][j][4:]
            class_id = np.argmax(classes_scores)
            max_score = np.max(classes_scores)
            if max_score > 0.25 and class_id == constants.CLASS_TO_ID['player']:
                _, y_center, _, _ = outputs[0][j][:4]
                player_boxes.append(y_center / self.model_height)
        
        if player_boxes:
            self.last_known_y = np.mean(player_boxes)
            return self.last_known_y
        else:
            return self.last_known_y

def get_player_y_coords(config, video_path):
    """使用YOLO模型从视频流中提取玩家的Y坐标序列"""
    print(f"正在为视频 '{video_path.name}' 提取玩家Y坐标序列...")
    detector = PlayerDetector(config.YOLO_MODEL_PATH)
    y_coords = []
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="提取视觉特征 (YOLO)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        player_y = detector.get_player_y(frame)
        y_coords.append(player_y)
        pbar.update(1)
        
    pbar.close()
    cap.release()
            
    return np.array(y_coords)

def main(video_path):
    config = DTWConfig()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    video_name_base = video_path.stem
    action_file = config.ANNOTATED_ACTION_DIR / f"{video_name_base}_corrected.json"

    if not action_file.exists():
        print(f"错误：找不到与视频 {video_path.name} 对应的已校对动作文件，跳过。")
        return

    # 1. 加载视觉特征序列 (玩家Y坐标)
    visual_sequence = get_player_y_coords(config, video_path)
    if visual_sequence is None:
        return
    
    # 2. 加载动作标签序列
    with open(action_file, 'r') as f:
        annotations = json.load(f)
    
    num_frames = len(visual_sequence)
    action_sequence = np.zeros(num_frames)
    for ann in annotations:
        action_type = ann['action']
        if action_type in config.ACTION_TO_FEATURE_MAP:
            feature_value = config.ACTION_TO_FEATURE_MAP[action_type]
            start = ann['start_frame']
            end = min(ann['end_frame'], num_frames) # 确保不越界
            action_sequence[start:end] = feature_value

    # 3. 执行DTW对齐
    print("正在执行DTW对齐...")
    visual_diff = np.diff(visual_sequence, prepend=visual_sequence[0]).reshape(-1, 1)
    action_sequence_reshaped = action_sequence.reshape(-1, 1)
    
    distance, path = fastdtw(visual_diff, action_sequence_reshaped, dist=euclidean)
    print(f"DTW对齐完成，路径长度: {len(path)}, 对齐距离: {distance:.2f}")
    
    # 4. 生成最终的精准轨迹
    final_actions = {}
    
    # 创建一个从动作帧到视频帧的映射
    act_to_viz_map = {}
    for viz_idx, act_idx in path:
        if act_idx not in act_to_viz_map:
            act_to_viz_map[act_idx] = []
        act_to_viz_map[act_idx].append(viz_idx)

    for ann in tqdm(annotations, desc="应用DTW对齐结果"):
        action_type = ann['action']
        original_start, original_end = ann['start_frame'], ann['end_frame']
        
        mapped_frames = []
        for act_idx in range(original_start, original_end):
            if act_idx in act_to_viz_map:
                mapped_frames.extend(act_to_viz_map[act_idx])
        
        if not mapped_frames: continue
        
        final_start_frame = min(mapped_frames)
        # 动作持续时间保持不变
        duration = original_end - original_start
        final_end_frame = final_start_frame + duration

        for frame_num in range(final_start_frame, final_end_frame):
            final_actions[str(frame_num)] = action_type

    output_path = config.OUTPUT_DIR / f"{video_name_base}_aligned_actions.json"
    with open(output_path, 'w') as f:
        json.dump(final_actions, f, indent=4, sort_keys=True)
    print(f"✅ 精准对齐的动作轨迹已保存到: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align actions with DTW using visual features.")
    parser.add_argument("--video_path", type=Path, required=True,
                        help="Path to the single CFR video file to be processed.")
    args = parser.parse_args()

    if not args.video_path.exists():
        print(f"错误: 视频文件不存在: {args.video_path}")
    else:
        main(args.video_path)