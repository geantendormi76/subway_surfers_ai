# scripts/09_pinggu_shipin.py (v1 - 模型视频评估工具)

import cv2
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

# 确保能导入项目内的模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from subway_surfers_ai.utils import constants

# --- 配置区 ---
ONNX_MODEL_PATH = PROJECT_ROOT / "models" / "one_best_v2.onnx" # [关键] 我们要评估V2模型
INPUT_VIDEO_PATH = PROJECT_ROOT / "data" / "raw_videos" / "gameplay_01_actions.mp4" # 选择一个视频进行测试
EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "evaluation_output"

CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

def draw_predictions(frame, boxes, confidences, class_ids):
    """在帧上绘制预测框、类别和置信度"""
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        class_id = class_ids[i]
        confidence = confidences[i]
        
        class_name = constants.ID_TO_CLASS.get(class_id, "UNKNOWN")
        label = f"{class_name}: {confidence:.2f}"
        
        # 简单的颜色映射
        color = (0, 255, 0) if class_name == 'player' else (255, 0, 0)

        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        cv2.putText(frame, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def evaluate_on_video():
    print("--- 启动模型视频评估程序 ---")
    EVALUATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    net = cv2.dnn.readNetFromONNX(str(ONNX_MODEL_PATH))
    print(f"✅ 模型加载成功: {ONNX_MODEL_PATH}")

    cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_video_name = f"evaluated_v2_{INPUT_VIDEO_PATH.name}"
    output_video_path = EVALUATION_OUTPUT_DIR / output_video_name
    video_writer = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    pbar = tqdm(total=frame_count, desc="视频评估进度")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
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
            if maxScore > CONF_THRESHOLD:
                scores.append(maxScore)
                class_ids.append(maxClassIndex)
                x_center, y_center, w, h = outputs[0][j][:4]
                x_scale, y_scale = width / 640, height / 640
                x = (x_center - w / 2) * x_scale
                y = (y_center - h / 2) * y_scale
                box_w, box_h = w * x_scale, h * y_scale
                boxes.append([x, y, box_w, box_h])

        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)
        
        annotated_frame = frame.copy()
        if len(indices) > 0:
            final_boxes = [boxes[k] for k in indices.flatten()]
            final_confidences = [scores[k] for k in indices.flatten()]
            final_class_ids = [class_ids[k] for k in indices.flatten()]
            annotated_frame = draw_predictions(annotated_frame, final_boxes, final_confidences, final_class_ids)
        
        video_writer.write(annotated_frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    video_writer.release()
    
    print(f"--- ✅ 视频评估完成！带标注的视频已保存到: '{output_video_path}' ---")

if __name__ == '__main__':
    evaluate_on_video()