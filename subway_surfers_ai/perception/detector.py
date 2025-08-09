# 文件: subway_surfers_ai/perception/detector.py

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes

# 这是我们对追踪器返回结果的一种类型标注，让代码更清晰
# 它是一个列表，列表里每个元素是一个包含7个数字的numpy数组
# 这7个数字分别是：x1, y1, x2, y2, track_id, confidence, class_id
TrackedObjects = np.ndarray # Shape: (N, 7)

class Detector:
    """
    一个集成了YOLOv11目标检测器和ByteTrack追踪器的感知类。
    这是我们AI的“眼睛”，负责从游戏画面中“看”到所有物体。
    """
    def __init__(self, model_path: str):
        """
        初始化检测器。

        Args:
            model_path (str): 训练好的YOLOv8模型文件路径 (.pt)。
        """
        # 加载我们训练好的 YOLOv11 核心检测模型
        self.model = YOLO(model_path)
        print(f"✅ [Detector] 成功从 {model_path} 加载YOLOv11模型。")

    