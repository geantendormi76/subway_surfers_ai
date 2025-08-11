# scripts/tool_model_assisted_annotation.py (V2.0 - Robust Video-Direct Processing)
import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import json
from tqdm import tqdm
import argparse

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subway_surfers_ai.action_recognition.model import ActionRecognitionCNN

# --- Configuration ---
class AnnotatorConfig:
    MODEL_PATH = PROJECT_ROOT / "models" / "action_recognizer_best.pt"
    OUTPUT_DIR = PROJECT_ROOT / "data" / "annotated_actions"
    
    # 模型相关参数 (必须与训练时一致)
    NUM_FRAMES = 8
    IMAGE_SIZE = (64, 64)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ID_TO_ACTION = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
    
    # 后处理参数
    ACTION_DURATION_FRAMES = 8 # 假设一个动作的平均持续时间（帧）
    CONFIDENCE_THRESHOLD = 0.8 # 模型输出概率的置信度阈值
    MIN_CONSECUTIVE_FRAMES = 3 # 需要连续多少帧预测为同一动作才认为有效

class ModelAssistedAnnotator:
    def __init__(self, config):
        self.config = config
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 1. 加载训练好的动作识别模型
        self.model = ActionRecognitionCNN(
            num_frames=self.config.NUM_FRAMES, 
            num_classes=len(self.config.ID_TO_ACTION)
        )
        self.model.load_state_dict(torch.load(self.config.MODEL_PATH, map_location=self.config.DEVICE))
        self.model.to(self.config.DEVICE)
        self.model.eval()
        print(f"动作识别模型已从 {self.config.MODEL_PATH} 加载。")

    def _prepare_frame_buffer(self, frame):
        """对单帧图像进行预处理"""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, self.config.IMAGE_SIZE)
        return image

    def generate_draft_annotations(self, video_path: Path):
        """为单个视频生成草稿标注，直接处理视频流"""
        print(f"\n正在为视频 '{video_path.name}' 生成草稿标注...")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_buffer = []
        raw_predictions = {} # 存储每一帧的原始预测结果 (action_id, probability)

        with torch.no_grad():
            pbar = tqdm(total=total_frames, desc="模型预测中")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame = self._prepare_frame_buffer(frame)
                frame_buffer.append(processed_frame)
                
                if len(frame_buffer) >= self.config.NUM_FRAMES:
                    # 从缓冲区末尾取N帧作为模型输入
                    input_frames = np.stack(frame_buffer[-self.config.NUM_FRAMES:], axis=0)
                    input_tensor = torch.from_numpy(input_frames).float().unsqueeze(0) / 255.0
                    input_tensor = input_tensor.to(self.config.DEVICE)
                    
                    outputs = self.model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    max_prob, predicted_id = torch.max(probs.data, 1)
                    
                    current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    if max_prob.item() > self.config.CONFIDENCE_THRESHOLD:
                        raw_predictions[current_frame_idx] = predicted_id.item()

                pbar.update(1)
            pbar.close()

        cap.release()
        
        # 后处理，将原始预测合并为稳定的动作事件
        draft_annotations = self._postprocess_predictions(raw_predictions, video_path.stem)
        
        print(f"草稿标注生成完毕，共发现 {len(draft_annotations)} 个潜在动作。")
        return draft_annotations

    def _postprocess_predictions(self, raw_predictions, video_name):
        """
        将逐帧的原始预测转换为结构化的动作事件列表。
        逻辑：查找连续N帧以上的相同预测，并将其合并为一个动作事件。
        """
        annotations = []
        sorted_frames = sorted(raw_predictions.keys())
        
        i = 0
        while i < len(sorted_frames):
            start_frame = sorted_frames[i]
            current_action_id = raw_predictions[start_frame]
            
            j = i
            # 查找连续预测的结束点
            while (j + 1 < len(sorted_frames) and
                   sorted_frames[j+1] == sorted_frames[j] + 1 and
                   raw_predictions[sorted_frames[j+1]] == current_action_id):
                j += 1
            
            consecutive_count = j - i + 1
            
            if consecutive_count >= self.config.MIN_CONSECUTIVE_FRAMES:
                # 找到了一个稳定的动作事件
                end_frame = sorted_frames[j] + self.config.ACTION_DURATION_FRAMES
                annotations.append({
                    "video_name": video_name,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "action": self.config.ID_TO_ACTION[current_action_id],
                    "status": "draft"
                })
            
            i = j + 1
        return annotations

    def review_and_correct(self, video_name, draft_annotations):
        """
        启动GUI界面供人工审核和修正。
        在本次实现中，我们暂时跳过GUI，直接保存草稿。
        """
        print("在完整的实现中，这里会有一个GUI界面供您校对。")
        print("为简化流程，我们暂时将草稿标注视为已校对。")
        
        corrected_annotations = draft_annotations
        output_path = self.config.OUTPUT_DIR / f"{video_name}_corrected.json"
        with open(output_path, 'w') as f:
            json.dump(corrected_annotations, f, indent=4)
        print(f"已校对的标注已保存到: {output_path}")

    def run_for_single_video(self, video_path):
        """对单个视频执行完整的标注流程"""
        draft = self.generate_draft_annotations(video_path)
        if draft is not None:
            self.review_and_correct(video_path.stem, draft)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model-Assisted Annotation Tool for Subway Surfers")
    parser.add_argument("--video_path", type=Path, required=True,
                        help="Path to the single CFR video file to be processed.")
    args = parser.parse_args()

    if not args.video_path.exists():
        print(f"错误: 视频文件不存在: {args.video_path}")
    else:
        config = AnnotatorConfig()
        annotator = ModelAssistedAnnotator(config)
        annotator.run_for_single_video(args.video_path)