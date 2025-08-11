# scripts/generator.py (v9.3 - 智能寻路版)

import os
import random
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from subway_surfers_ai.utils import constants

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ... (渲染参数、游戏逻辑、深度范围、投影函数都保持不变)
VANISHING_POINT_Y_RATIO = 0.4
PERSPECTIVE_STRENGTH = 1.8
CURVE_FACTOR = 1.5
LANE_WIDTH_STRETCH = 1.2
NUM_LANES = 3
GRID_DEPTH = 32
PLAYER_DEPTH_CELLS = [0, 1, 2] 
OBSTACLE_DEPTH_CELLS = range(10, GRID_DEPTH - 5) 

def project_3d_to_2d(world_z, bg_h):
    vanishing_point_y = bg_h * VANISHING_POINT_Y_RATIO
    normalized_depth = world_z / GRID_DEPTH
    depth_factor = normalized_depth ** CURVE_FACTOR
    screen_y = int(bg_h - (bg_h - vanishing_point_y) * depth_factor)
    scale = max(0.05, (1.0 - normalized_depth) ** PERSPECTIVE_STRENGTH)
    return screen_y, scale

def project_lane_to_screen_x(lane_idx, screen_y, bg_w, bg_h):
    vanishing_point_y = bg_h * VANISHING_POINT_Y_RATIO
    if (bg_h - vanishing_point_y) == 0:
        return bg_w // NUM_LANES * (lane_idx + 0.5)
    y_progress = (bg_h - screen_y) / (bg_h - vanishing_point_y)
    current_lane_width = bg_w * (1 + (LANE_WIDTH_STRETCH - 1) * y_progress)
    lane_center_x = (lane_idx + 0.5) * (current_lane_width / NUM_LANES)
    offset = (current_lane_width - bg_w) / 2
    screen_x = int(lane_center_x - offset)
    return screen_x

def generate_training_image(slices_dir, background_path, output_dir, image_idx):
    try:
        background = Image.open(background_path).convert("RGBA")
    except Exception as e:
        print(f"错误：无法打开背景帧 {background_path}: {e}")
        return
        
    bg_w, bg_h = background.size
    yolo_labels = []
    
    world_grid = np.full((GRID_DEPTH, NUM_LANES), -1, dtype=int)
    objects_to_render = []
    available_classes = [d.name for d in slices_dir.iterdir() if d.is_dir() and any(d.glob("*.png"))]
    if not available_classes: return

    # --- 注入逻辑 (保持v9.2的逻辑，不生成玩家) ---
    num_obstacle_groups = random.randint(1, 3)
    
    possible_depths = list(OBSTACLE_DEPTH_CELLS)
    if len(possible_depths) < num_obstacle_groups: return
    chosen_depths = sorted(random.sample(possible_depths, k=num_obstacle_groups))

    lanes = [0, 1, 2]
    occupied_lanes_for_coins = set()
    
    for depth in chosen_depths:
        num_obstacles_at_this_depth = random.choice([1, 1, 1, 2])
        lanes_for_this_depth = random.sample(lanes, k=num_obstacles_at_this_depth)
        
        for lane in lanes_for_this_depth:
            obstacle_types = ['train', 'train1', 'low_barrier', 'high_barrier']
            valid_obstacles = [o for o in obstacle_types if o in available_classes]
            if valid_obstacles:
                obstacle_type = random.choice(valid_obstacles)
                world_grid[depth, lane] = constants.CLASS_TO_ID[obstacle_type]
                occupied_lanes_for_coins.add(lane)

    if 'coin' in available_classes:
        safe_lanes = [l for l in lanes if l not in occupied_lanes_for_coins]
        if safe_lanes:
            coin_lane = random.choice(safe_lanes)
            start_depth = random.choice(range(15, GRID_DEPTH - 8))
            for i in range(random.randint(5, 10)):
                depth_offset = i * 2
                if start_depth + depth_offset < GRID_DEPTH:
                    world_grid[start_depth + depth_offset, coin_lane] = constants.CLASS_TO_ID['coin']

    # --- 【v9.3 核心修正】准备渲染列表 ---
    for r in range(GRID_DEPTH):
        for c in range(NUM_LANES):
            class_id = world_grid[r, c]
            if class_id != -1:
                class_name = constants.ID_TO_CLASS.get(class_id, 'unknown')
                class_slices_dir = slices_dir / class_name
                if not class_slices_dir.exists(): continue
                
                # 【核心修正】不再假设文件名格式，而是直接获取所有png文件
                all_slices = list(class_slices_dir.glob("*.png"))
                if not all_slices: continue # 如果目录为空，跳过
                
                # 从找到的所有切片中，随机选择一个
                slice_path = random.choice(all_slices)
                
                try:
                    slice_img = Image.open(slice_path).convert("RGBA")
                    objects_to_render.append([class_name, c, r, slice_img])
                except FileNotFoundError:
                    # 这个异常理论上不会再发生，但保留以策万全
                    print(f"警告: 文件未找到: {slice_path}")
                    continue
    
    # --- 渲染阶段 (不变) ---
    objects_to_render.sort(key=lambda obj: obj[2]) 

    for obj_data in objects_to_render:
        class_name, lane_idx, depth_cell, slice_img = obj_data
        screen_y, scale = project_3d_to_2d(depth_cell, bg_h)
        if screen_y is None: continue
        new_w, new_h = int(slice_img.width * scale), int(slice_img.height * scale)
        if new_w < 2 or new_h < 2: continue
        resized_slice = slice_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        screen_x = project_lane_to_screen_x(lane_idx, screen_y, bg_w, bg_h)
        paste_x, paste_y = int(screen_x - new_w / 2), screen_y - new_h
        background.paste(resized_slice, (paste_x, paste_y), resized_slice)
        box = [paste_x, paste_y, paste_x + new_w, paste_y + new_h]
        class_id = constants.CLASS_TO_ID[class_name]
        yolo_labels.append(f"{class_id} {(box[0]+box[2])/2/bg_w} {(box[1]+box[3])/2/bg_h} {new_w/bg_w} {new_h/bg_h}")

    # --- 保存 (不变) ---
    final_image = background.convert("RGB")
    image_save_path = output_dir / "images" / f"synthetic_{image_idx:05d}.jpg"
    label_save_path = output_dir / "labels" / f"synthetic_{image_idx:05d}.txt"
    final_image.save(image_save_path)
    if yolo_labels:
        with open(label_save_path, 'w') as f:
            f.write("\n".join(yolo_labels))

# --- 脚本执行入口 (不变) ---
if __name__ == '__main__':
    # 【重要】确保你使用的是标准化后的切片库
    SLICES_DIR = PROJECT_ROOT / "data" / "normalized_slices"
    # 【重要】确保你的背景帧目录名为 frames
    BACKGROUND_FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
    OUTPUT_DATASET_DIR = PROJECT_ROOT / "data" / "yolo_dataset"
    NUM_IMAGES_TO_GENERATE = 5000

    (OUTPUT_DATASET_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DATASET_DIR / "labels").mkdir(parents=True, exist_ok=True)
    
    background_files = [f for f in BACKGROUND_FRAMES_DIR.iterdir() if f.suffix in ['.png', '.jpg']]
    if not background_files:
        print(f"错误：请先运行 video_slicer.py 在 '{BACKGROUND_FRAMES_DIR}' 目录下生成背景帧！")
    else:
        print(f"--- 开始生成合成数据集v9.3 (智能寻路版)，共 {NUM_IMAGES_TO_GENERATE} 张 ---")
        for i in tqdm(range(NUM_IMAGES_TO_GENERATE), desc="生成合成数据"):
            random_background_frame = random.choice(background_files)
            generate_training_image(SLICES_DIR, random_background_frame, OUTPUT_DATASET_DIR, i)
        print("--- 生成完毕！---")