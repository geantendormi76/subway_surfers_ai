# /home/zhz/Deepl/subway_surfers_ai/scripts/convert_voc_to_yolo.py

import xml.etree.ElementTree as ET
import glob
import os
import sys
from pathlib import Path
from tqdm import tqdm

# --- [核心] 确保脚本能找到项目根目录下的模块 ---
# 这部分和你在 train.py 中学到的是完全一样的逻辑
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 从我们的常量模块中导入类别到ID的映射
# 这确保了我们生成的标签ID和训练时使用的ID是完全一致的
from subway_surfers_ai.utils import constants

def convert_box(size, box):
    """
    将 PascalVOC 格式的边界框 (xmin, ymin, xmax, ymax) 转换为 YOLO 格式。
    YOLO 格式是归一化的 (x_center, y_center, width, height)。

    Args:
        size (tuple): 图像的 (宽度, 高度)。
        box (tuple): PascalVOC 格式的 (xmin, ymin, xmax, ymax)。

    Returns:
        tuple: YOLO 格式的 (x_center, y_center, width, height)。
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    width = box[2] - box[0]
    height = box[3] - box[1]
    
    x_center = x_center * dw
    width = width * dw
    y_center = y_center * dh
    height = height * dh
    
    return (x_center, y_center, width, height)

def convert_voc_to_yolo(xml_dir: Path, labels_dir: Path):
    """
    批量将一个目录下的所有 .xml 标注文件转换为 .txt (YOLO) 格式。
    """
    # 确保输出目录存在
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有 .xml 文件
    xml_files = glob.glob(str(xml_dir / '*.xml'))
    
    if not xml_files:
        print(f"警告：在目录 '{xml_dir}' 中没有找到任何 .xml 文件。")
        return

    print(f"开始转换 {len(xml_files)} 个XML文件...")
    
    # 使用 tqdm 创建一个进度条，方便观察进度
    for xml_file in tqdm(xml_files, desc="转换进度"):
        xml_path = Path(xml_file)
        # 解析XML文件
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 获取图像尺寸
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # 创建对应的 .txt 文件路径
        txt_path = labels_dir / xml_path.with_suffix('.txt').name
        
        yolo_labels = []
        # 遍历XML中的每一个'object'
        for obj in root.iter('object'):
            # 检查是否是困难样本，我们暂时不处理
            difficult = obj.find('difficult')
            if difficult is not None and int(difficult.text) == 1:
                continue
            
            # 获取类别名称和边界框
            cls_name = obj.find('name').text
            if cls_name not in constants.CLASS_TO_ID:
                print(f"警告: 在文件 {xml_file} 中找到未知类别 '{cls_name}'，已跳过。")
                continue
                
            class_id = constants.CLASS_TO_ID[cls_name]
            
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), 
                 float(xmlbox.find('ymin').text), 
                 float(xmlbox.find('xmax').text), 
                 float(xmlbox.find('ymax').text))
            
            # 转换格式并添加到列表中
            bb = convert_box((width, height), b)
            yolo_labels.append(f"{class_id} " + " ".join([f"{a:.6f}" for a in bb]))
            
        # 将所有标签写入 .txt 文件
        if yolo_labels:
            with open(txt_path, 'w') as f:
                f.write("\n".join(yolo_labels))
                
    print(f"转换完成！所有 .txt 标签文件已保存到: {labels_dir}")

# --- 脚本执行入口 ---
if __name__ == '__main__':
    # 【重要】定义你的图片和XML所在的目录
    # 这个脚本专门用来处理你的手工标注数据
    image_xml_directory = PROJECT_ROOT / "data" / "images"
    
    # 【重要】定义你希望保存YOLO格式标签的目录
    yolo_labels_directory = PROJECT_ROOT / "data" / "labels"

    print("--- 启动 PascalVOC (.xml) 到 YOLO (.txt) 的格式转换任务 ---")
    print(f"读取源: {image_xml_directory}")
    print(f"写入到: {yolo_labels_directory}")
    convert_voc_to_yolo(image_xml_directory, yolo_labels_directory)