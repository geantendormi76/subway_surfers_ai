# /home/zhz/Deepl/subway_surfers_ai/scripts/09_yanzheng_guiji.py (v2 - 动作对齐验证增强版)

import cv2
import numpy as np
import sys
import pickle
import lzma
import argparse
from pathlib import Path

# --- [核心] 确保能找到项目根目录下的模块 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from subway_surfers_ai.utils import constants

def find_next_action(trajectory_data, start_frame_idx):
    """
    从指定帧开始，向后查找第一个非NOOP的动作。
    """
    for i in range(start_frame_idx, len(trajectory_data)):
        if trajectory_data[i]['action'] != 0:
            return trajectory_data[i]['action'], i
    return None, -1

def draw_predictions(frame, current_data, next_action_info):
    """
    在视频帧上绘制状态、当前动作、以及下一个即将发生的动作信息。
    """
    height, width, _ = frame.shape
    
    # --- 绘制状态（检测框） ---
    for obj in current_data['state']:
        class_id, x_center, y_center, w, h = obj
        class_name = constants.ID_TO_CLASS.get(int(class_id), "UNKNOWN")
        
        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- 绘制左上角的信息面板 ---
    action_map = {0: 'NOOP', 1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT'}
    
    # 当前帧信息
    action_str = action_map.get(current_data['action'], "INVALID")
    rtg_text = f"RTG: {current_data['rtg']:.2f}"
    frame_text = f"Frame: {current_data['timestep']}"
    
    cv2.putText(frame, f"Action: {action_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, rtg_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, frame_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # [核心功能] 预告下一个动作
    next_action, next_action_frame = next_action_info
    if next_action is not None:
        next_action_str = action_map.get(next_action, "INVALID")
        frames_until = next_action_frame - current_data['timestep']
        next_action_text = f"Next: {next_action_str} in {frames_until}f"
        cv2.putText(frame, next_action_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 205, 255), 2) # 黄色文字

    # 如果当前帧就是动作帧，则在屏幕中央高亮显示
    if current_data['action'] != 0:
        cv2.putText(frame, f"ACTION: {action_str}", (width // 2 - 150, height // 2), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 3)
    
    return frame

def main():
    parser = argparse.ArgumentParser(description="可视化并验证生成的轨迹数据 (v2)")
    parser.add_argument("--video", required=True, type=Path, help="用于验证的原始视频文件")
    parser.add_argument("--trajectory", required=True, type=Path, help="已生成的轨迹数据文件 (.pkl.xz)")
    args = parser.parse_args()

    # --- 1. 加载轨迹数据 ---
    print(f"正在加载轨迹数据: {args.trajectory}")
    with lzma.open(args.trajectory, 'rb') as f:
        trajectory_data = pickle.load(f)
    print("加载成功！")
    
    # --- 2. 打开视频文件 ---
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {args.video}")
    
    print("\n--- 操作指南 ---")
    print("按 [空格键] 暂停/播放")
    print("按 [D]     快进到下一个动作前")
    print("按 [A]     后退10帧")
    print("按 [Q]     退出")
    print("------------------\n")
    
    paused = True
    frame_idx = 0
    next_action_info = (None, -1)

    while 0 <= frame_idx < len(trajectory_data):
        if not paused or 'frame' not in locals(): # 首次进入循环时强制读取第一帧
            ret, frame = cap.read()
            if not ret: break
        
        # 无论是否暂停，都更新信息
        current_data = trajectory_data[frame_idx]
        
        # 查找下一个动作
        if next_action_info[1] <= frame_idx:
            next_action_info = find_next_action(trajectory_data, frame_idx + 1)

        # 绘制信息并显示
        annotated_frame = draw_predictions(frame, current_data, next_action_info)
        cv2.imshow("Trajectory Validation", annotated_frame)
        
        if not paused:
            frame_idx += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('a'): # 后退
            frame_idx = max(0, frame_idx - 10)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read() # 必须重新读取帧
        elif key == ord('d'): # 快进到下一个动作
            if next_action_info[1] != -1:
                # 快进到动作发生前10帧，方便观察
                frame_idx = max(frame_idx + 1, next_action_info[1] - 10)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read() # 必须重新读取帧

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()