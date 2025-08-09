# /home/zhz/Deepl/subway_surfers_ai/scripts/evaluate_model_on_video.py (v_final)

import cv2
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from subway_surfers_ai.utils import constants
from scripts.validate_generator import draw_predictions

# --- [核心] 配置区 ---
# 使用我们充分训练的V2模型
ONNX_MODEL_PATH = PROJECT_ROOT / "models" / "one_best_v2.onnx"
# 使用你的新测试视频
INPUT_VIDEO_PATH = PROJECT_ROOT / "data" / "raw_videos" / "test_02.mp4"
EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "evaluation_output"

# --- 【核心修正】---
# 将置信度阈值从0.3降低到一个更合理的值，比如0.25。
# 这是一个超参数，你可以根据最终效果在0.1到0.4之间进行调整，以找到最佳平衡点。
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
# --------------------

def evaluate_on_video():
    """
    加载ONNX模型，对一个视频文件进行逐帧预测，并生成一个新的带标注的视频。
    """
    print("--- 启动模型视频评估程序 (最终版) ---")

    if not INPUT_VIDEO_PATH.exists():
        print(f"错误: 输入视频文件 '{INPUT_VIDEO_PATH}' 不存在！")
        return
    
    EVALUATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"正在从 {ONNX_MODEL_PATH} 加载模型...")
    if not ONNX_MODEL_PATH.exists():
        print(f"错误: 找不到ONNX模型文件！请确认 '{ONNX_MODEL_PATH}' 存在。")
        return
    net = cv2.dnn.readNetFromONNX(str(ONNX_MODEL_PATH))
    print("✅ 模型加载成功！")

    cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {INPUT_VIDEO_PATH}")
        return
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_video_name = f"evaluated_final_{INPUT_VIDEO_PATH.name}"
    output_video_path = EVALUATION_OUTPUT_DIR / output_video_name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))
    
    print(f"视频处理开始，总帧数: {frame_count}，帧率: {fps:.2f}")

    pbar = tqdm(total=frame_count, desc="视频评估进度")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        
        net.setInput(blob)
        outputs = net.forward()
        
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]
        boxes, scores, class_ids = [], [], []

        for j in range(rows):
            classes_scores = outputs[0][j][4:]
            (_, maxScore, _, (_, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            
            if maxScore > CONF_THRESHOLD: # 使用我们新的、更低的阈值
                scores.append(maxScore)
                class_ids.append(maxClassIndex)
                x_center, y_center, w, h = outputs[0][j][0], outputs[0][j][1], outputs[0][j][2], outputs[0][j][3]
                x_scale, y_scale = width / 640, height / 640
                x = int((x_center - w / 2) * x_scale)
                y = int((y_center - h / 2) * y_scale)
                box_w, box_h = int(w * x_scale), int(h * y_scale)
                boxes.append([x, y, box_w, box_h])

        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)
        
        final_boxes, final_confidences, final_class_ids = [], [], []
        if len(indices) > 0:
            for k in indices.flatten():
                final_boxes.append(boxes[k])
                final_confidences.append(scores[k])
                final_class_ids.append(class_ids[k])

        annotated_frame = draw_predictions(frame.copy(), final_boxes, final_confidences, final_class_ids)
        
        video_writer.write(annotated_frame)
        
        pbar.update(1)

    pbar.close()
    cap.release()
    video_writer.release()
    
    print("--- ✅ 视频评估全部完成！---")
    print(f"带标注的视频已保存到: '{output_video_path}'")

if __name__ == '__main__':
    evaluate_on_video()