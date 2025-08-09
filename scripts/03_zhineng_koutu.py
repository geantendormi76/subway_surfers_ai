# scripts/slicer_with_sam.py 

import os
import cv2
import torch
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from segment_anything import sam_model_registry, SamPredictor

# --- 配置区 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAM_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "sam_vit_h_4b8939.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def slice_with_sam(image_path, xml_path, slices_output_dir, predictor):
    """
    使用SAM模型，根据XML标注文件，从图片中抠出物体。
    """
    # 1. 读取图片和XML文件
    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 2. 从XML中解析出所有的边界框和类别
    #    我们把它们存在一个列表里，每个元素是一个包含名字和框的字典
    all_objects = []
    for member in root.findall('object'):
        class_name = member.find('name').text
        bndbox = member.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        all_objects.append({'name': class_name, 'box': [xmin, ymin, xmax, ymax]})
    
    if not all_objects:
        return

    # 3. 【核心修正】让SAM“看”一次完整的图片，为后续多次预测做准备
    #    这一步会计算图像的嵌入特征，可以被后续的预测复用，效率很高
    predictor.set_image(image_rgb)
    
    # 4. 【对标黄金标准】使用 for 循环，一次只处理一个物体
    for i, obj in enumerate(all_objects):
        class_name = obj['name']
        box_np = np.array([obj['box']]) # 注意这里，我们把单个框包装成numpy数组

        # 5. 调用SAM进行单次预测，输入是单个框
        masks, scores, logits = predictor.predict(
            box=box_np,
            multimask_output=False
        )
        # 此时返回的 masks 是一个只包含一个蒙版的数组
        mask = masks[0]

        # 6. 根据返回的单个蒙版，进行抠图和保存
        segmented_image = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4), dtype=np.uint8)
        segmented_image[:, :, :3] = image_rgb
        segmented_image[:, :, 3] = mask * 255
        
        xmin, ymin, xmax, ymax = obj['box']
        cropped_slice = segmented_image[ymin:ymax, xmin:xmax]

        # 7. 保存到对应的类别文件夹
        class_dir = slices_output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        image_stem = Path(image_path).stem
        slice_save_path = class_dir / f"{image_stem}_{i}.png"
        
        cropped_slice_bgra = cv2.cvtColor(cropped_slice, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(slice_save_path), cropped_slice_bgra)

if __name__ == '__main__':
    print("--- 开始智能抠图任务 ---")
    
    # 1. 加载SAM模型到GPU
    print("正在加载SAM模型...")
    if not SAM_CHECKPOINT_PATH.exists():
        print(f"错误：找不到SAM模型权重文件，请确认它位于: {SAM_CHECKPOINT_PATH}")
    else:
        sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        predictor = SamPredictor(sam)
        print("SAM模型加载完毕！")

        # 2. 定义输入和输出目录
        IMAGES_DIR = PROJECT_ROOT / "data" / "images"
        SLICES_OUTPUT_DIR = PROJECT_ROOT / "data" / "slices"

        # 3. 找到所有标注好的xml文件
        xml_files = sorted(list(IMAGES_DIR.glob("*.xml")))
        if not xml_files:
            print("错误：在 data/images/ 目录下没有找到任何.xml标注文件！")
        else:
            # 4. 遍历所有文件，执行抠图
            for xml_path in xml_files:
                # 兼容.png和.jpg两种格式
                image_path = xml_path.with_suffix(".png")
                if not image_path.exists():
                    image_path = xml_path.with_suffix(".jpg")
                
                if image_path.exists():
                    print(f"正在处理: {image_path.name}")
                    slice_with_sam(image_path, xml_path, SLICES_OUTPUT_DIR, predictor)
                else:
                    print(f"警告：找不到与 {xml_path.name} 匹配的图片文件，跳过。")
        
        print("--- 智能抠图任务完成！请检查 data/slices/ 目录。---")