# scripts/video_slicer.py

# 导入我们需要的工具包
import cv2  # OpenCV库，用于处理视频
import os   # 用于处理文件和目录路径
from pathlib import Path # 更现代化的路径处理工具

# 获取我们项目的根目录路径
# Path(__file__) 指的是当前这个脚本文件 (video_slicer.py)
# .resolve() 会获取它的绝对路径
# .parents[1] 会向上走一级目录 (从 scripts/ 到 subway_surfers_ai/)
# 这样，无论我们在哪里运行这个脚本，它都能准确地找到项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def slice_video(video_path, output_dir, interval_ms=500):
    """
    将视频文件按指定的时间间隔切成图片帧。

    :param video_path: 输入视频文件的完整路径。
    :param output_dir: 保存图片帧的目录路径。
    :param interval_ms: 切片的时间间隔，单位是毫秒 (ms)。默认500ms，即每秒2张。
    """
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：找不到视频文件 -> {video_path}")
        return

    # 确保输出目录存在，如果不存在就创建它
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用OpenCV打开视频文件
    cap = cv2.VideoCapture(str(video_path))
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 -> {video_path}")
        return

    # 获取视频的帧率(FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频信息: {Path(video_path).name}, 帧率: {fps:.2f} FPS")
    
    # 计算需要跳过的帧数
    # (fps * interval_ms / 1000) 算出了间隔时间内的总帧数
    # 视频帧率是30帧/秒，我们想每500ms（0.5秒）截一张图
    # 那么需要跳过的帧数 = 30 * (500 / 1000) = 30 * 0.5 = 15 帧
    frame_interval = int(fps * interval_ms / 1000)
    
    frame_count = 0        # 用来记录这是视频的第几帧 (从0开始)
    saved_frame_count = 0  # 用来记录我们已经保存了多少张图片 (用于命名)

    while True:
        # read() 方法会读取一帧
        # ret 是一个布尔值，如果成功读取到帧，则为True
        # frame 是读取到的那一帧画面 (一个numpy array)
        ret, frame = cap.read()
        
        # 如果 ret 是 False，说明视频已经播放完毕
        if not ret:
            break
        
        # 检查当前是否是我们需要保存的帧
        if frame_count % frame_interval == 0:
            # 构建保存文件的路径和文件名
            # 例如: .../images/gameplay_01_frame_00000.png
            video_name = Path(video_path).stem # 获取不带后缀的文件名
            save_path = os.path.join(output_dir, f"{video_name}_frame_{saved_frame_count:05d}.png")
            
            # 保存这一帧画面
            cv2.imwrite(save_path, frame)
            saved_frame_count += 1
            
        frame_count += 1
    
    # 释放视频捕获对象，关闭文件
    cap.release()
    print(f"处理完成！总共从视频中提取了 {saved_frame_count} 帧图片，保存在: {output_dir}")


# 同样，使用 if __name__ == '__main__' 来作为脚本的执行入口
if __name__ == '__main__':
    # --- 配置区 ---
    # 定义我们的输入视频文件名
    INPUT_VIDEO_NAME = "clean_run_01.mp4"
    # 定义我们希望的截图间隔（毫秒）
    SLICE_INTERVAL_MS = 100

    # --- 执行区 ---
    # 使用我们之前定义的 PROJECT_ROOT 来构建绝对路径，这样更健壮
    video_input_path = PROJECT_ROOT / "data" / "raw_videos" / INPUT_VIDEO_NAME
    image_output_dir = PROJECT_ROOT / "data" / "frames"

    print("--- 开始执行视频切片任务 ---")
    slice_video(video_input_path, image_output_dir, SLICE_INTERVAL_MS)