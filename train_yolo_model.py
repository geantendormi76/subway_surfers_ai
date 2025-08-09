# /home/zhz/Deepl/subway_surfers_ai/train.py (最终正确版)

import sys
from pathlib import Path
from ultralytics import YOLO
import yaml  # 导入YAML库来读取配置文件

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def train_detector():
    """
    一个由YAML配置文件驱动的、更专业、更健壮的YOLOv8训练脚本。
    """
    print("--- 启动YOLOv8训练任务 (配置文件驱动) ---")
    
    # 1. 定义我们的主配置文件路径
    config_path = PROJECT_ROOT / "yolo_model_config.yaml"

    if not config_path.exists():
        print(f"错误: 找不到主配置文件 '{config_path}'。")
        return
        
    print(f"\n[配置加载]")
    print(f"  - 主训练配置: {config_path}")

    # 2. 加载一个基础模型
    # 我们从配置文件中读取要使用的基础模型
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        base_model_name = config.get('model', 'yolo11n.pt') # 默认使用yolov8n.pt
    
    model = YOLO(base_model_name)

    # 3. 【核心修正】启动训练
    # YOLOv8的 model.train() 函数本身就可以接收所有配置参数
    # 我们不再将配置文件路径传给 'data' 参数，而是直接将整个配置文件作为参数传入
    print("--- 开始模型训练 ---")
    model.train(**config) # 使用 **kwargs 的方式将配置文件中的所有参数解包传入
    print("--- 训练完成 ---")
    
    # 4. 导出最佳模型 (逻辑不变)
    best_model_path = model.trainer.best
    print(f"\n训练出的最佳模型保存在: {best_model_path}")
    
    best_model = YOLO(best_model_path)
    
    print("正在导出最佳模型为ONNX格式...")
    try:
        onnx_path = best_model.export(format='onnx')
        print(f"--- 导出成功！ONNX模型保存在: {onnx_path} ---")
        print(f"请手动将最终的 .onnx 文件移动到 'models/' 目录下，并重命名。")
    except Exception as e:
        print(f"\n--- 导出失败！错误: {e} ---")

if __name__ == '__main__':
    train_detector()