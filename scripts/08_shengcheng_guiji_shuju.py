# /home/zhz/Deepl/subway_surfers_ai/scripts/08_shengcheng_guiji_shuju.py (v5 - 锚点同步最终版)

import cv2
import numpy as np
import sys
import pickle
import lzma
import argparse
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from subway_surfers_ai.utils import constants

ONNX_MODEL_PATH = PROJECT_ROOT / "models" / "one_best_v2.onnx"
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
# 注意：此版本不再需要全局延迟变量，延迟是通过锚点动态计算的

def parse_actions(actions_path, fps, first_action_effect_frame):
    """
    [核心修正 V5] 使用“首个动作锚点”来同步视频和动作两条时间线。
    """
    with open(actions_path, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        return {}

    first_action_timestamp = float(lines[0].strip().split(',')[0])
    t_effect_real = first_action_effect_frame / fps
    time_offset = t_effect_real - first_action_timestamp
    
    print(f"时间线同步：首个动作日志时间戳={first_action_timestamp:.2f}s, "
          f"首个动作生效帧={first_action_effect_frame} ({t_effect_real:.2f}s), "
          f"计算出的时间偏移量={time_offset:.2f}s")

    actions = {}
    for line in lines:
        timestamp_str, action = line.strip().split(',')
        keypress_timestamp = float(timestamp_str)
        aligned_timestamp = keypress_timestamp + time_offset
        if aligned_timestamp < 0: continue
        frame_index = int(aligned_timestamp * fps)
        actions[frame_index] = action
        
    return actions

def extract_states_from_video(video_path, ort_session):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise IOError(f"无法打开视频文件: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    input_name = ort_session.get_inputs()[0].name
    model_height, model_width = ort_session.get_inputs()[0].shape[2:]
    all_states = []
    pbar = tqdm(total=frame_count, desc=f"状态提取 (GPU): {video_path.name}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
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
            (_, maxScore, _, (_, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore > CONF_THRESHOLD:
                scores.append(maxScore)
                class_ids.append(maxClassIndex)
                x_center, y_center, w, h = outputs[0][j][:4]
                x_scale, y_scale = width / model_width, height / model_height
                x1 = (x_center - w / 2) * x_scale
                y1 = (y_center - h / 2) * y_scale
                box_w, box_h = w * x_scale, h * y_scale
                boxes.append([x1, y1, box_w, box_h])
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)
        frame_state = []
        if len(indices) > 0:
            for k in indices.flatten():
                box = boxes[k]
                frame_state.append([
                    class_ids[k], (box[0] + box[2] / 2) / width, (box[1] + box[3] / 2) / height,
                    box[2] / width, box[3] / height,
                ])
        all_states.append(frame_state)
        pbar.update(1)
    pbar.close()
    cap.release()
    return all_states, fps

def main():
    parser = argparse.ArgumentParser(description="从专家视频和动作日志生成轨迹数据 (v5 - 锚点同步)")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--actions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--first-action-frame", required=True, type=int, 
                        help="视频中第一个动作真实生效的帧号")
    args = parser.parse_args()

    print(f"正在加载感知模型...")
    providers = ['CUDAExecutionProvider']
    ort_session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=providers)
    print("模型加载成功并已分配到GPU！")

    states, fps = extract_states_from_video(args.video, ort_session)
    print(f"视频帧率: {fps:.2f} FPS")

    actions_map = parse_actions(args.actions, fps, args.first_action_frame)
    print(f"解析并对齐了 {len(actions_map)} 个动作。")

    trajectory = []
    action_map_rev = {'up': 1, 'down': 2, 'left': 3, 'right': 4}
    for frame_idx, state in enumerate(tqdm(states, desc="组装轨迹")):
        current_state = state
        action_str = actions_map.get(frame_idx, 'noop')
        current_action = action_map_rev.get(action_str, 0)
        player_detected = any(obj[0] == constants.CLASS_TO_ID['player'] for obj in current_state)
        current_reward = 0.01 if player_detected else -1.0
        current_timestep = frame_idx
        trajectory.append({
            'state': current_state, 'action': current_action,
            'reward': current_reward, 'timestep': current_timestep
        })
    running_rtg = 0.0
    for i in reversed(range(len(trajectory))):
        running_rtg += trajectory[i]['reward']
        trajectory[i]['rtg'] = running_rtg
    print(f"轨迹数据组装完成，总共 {len(trajectory)} 个时间步。")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(args.output, 'wb') as f:
        pickle.dump(trajectory, f)
    print(f"✅ 成功！轨迹数据已保存到: {args.output}")

if __name__ == '__main__':
    main()