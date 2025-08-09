# scripts/01_video_qiege.py (v2 - 命令行工具版)

import cv2
import os
from pathlib import Path
import argparse # [核心修改] 导入argparse库

def slice_video(video_path, output_dir, interval_ms=100):
    # ... (函数内部逻辑保持不变) ...
    if not os.path.exists(video_path):
        print(f"错误：找不到视频文件 -> {video_path}")
        return
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 -> {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"处理视频: {Path(video_path).name}, 帧率: {fps:.2f} FPS")
    frame_interval = int(fps * interval_ms / 1000)
    if frame_interval == 0: frame_interval = 1 # 避免高帧率视频间隔过小导致跳帧
    frame_count = 0
    saved_frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            video_name = Path(video_path).stem
            save_path = os.path.join(output_dir, f"{video_name}_frame_{saved_frame_count:05d}.png")
            cv2.imwrite(save_path, frame)
            saved_frame_count += 1
        frame_count += 1
    cap.release()
    print(f"处理完成！提取了 {saved_frame_count} 帧图片，保存在: {output_dir}")

# [核心修改] 修改主执行逻辑
if __name__ == '__main__':
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="将视频文件按指定时间间隔切成图片帧。")
    parser.add_argument("--video", required=True, type=Path, help="输入视频文件的路径。")
    parser.add_argument("--output_dir", required=True, type=Path, help="保存图片帧的目录路径。")
    parser.add_argument("--interval", type=int, default=100, help="切片的时间间隔（毫秒）。")
    args = parser.parse_args()

    print("--- 开始执行视频切片任务 ---")
    slice_video(args.video, args.output_dir, args.interval)