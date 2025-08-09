# /home/zhz/Deepl/subway_surfers_ai/subway_surfers_ai/perception/state_builder.py (v2 - 对齐katacr)

import numpy as np
import scipy.spatial

# --- [对齐] 定义更丰富的特征维度 ---
# 0: class_id, 1: x_offset, 2: y_offset, 3: width, 4: height
# 5: is_player (one-hot), 6: is_obstacle, 7: is_collectible
FEATURE_DIM = 8
GRID_HEIGHT = 32
GRID_WIDTH = 18

class PositionFinder:
    """
    对齐 katacr/policy/perceptron/utils.py 中的 PositionFinder
    用于智能地寻找网格中的可用位置。
    """
    def __init__(self, grid_height, grid_width):
        self.grid_shape = (grid_height, grid_width)
        self.reset()

    def reset(self):
        self.used = np.zeros(self.grid_shape, dtype=bool)
        self.grid_coords = np.indices(self.grid_shape).transpose(1, 2, 0).astype(np.float32) + 0.5
    
    def find_near_pos(self, xy_coord):
        # xy_coord 是 (x, y) 格式的浮点坐标
        yx_coord = xy_coord[::-1]
        y, x = np.clip(yx_coord, 0, np.array(self.grid_shape) - 1).astype(int)

        if not self.used[y, x]:
            self.used[y, x] = True
            return y, x
        
        # 如果已被占用，则寻找最近的空位
        avail_coords = self.grid_coords[~self.used]
        if len(avail_coords) == 0:
            return None # 没有可用位置
            
        map_indices = np.argwhere(~self.used)
        distances = scipy.spatial.distance.cdist(yx_coord.reshape(1, 2), avail_coords)
        best_idx = np.argmin(distances)
        
        y_new, x_new = map_indices[best_idx]
        self.used[y_new, x_new] = True
        return y_new, x_new

def build_state_tensor(yolo_objects, grid_height=GRID_HEIGHT, grid_width=GRID_WIDTH):
    """
    [核心对齐] 将YOLO输出转换为结构化状态张量，使用PositionFinder并填充更丰富的特征。
    """
    state_grid = np.zeros((grid_height, grid_width, FEATURE_DIM), dtype=np.float32)
    pos_finder = PositionFinder(grid_height, grid_width)

    if not yolo_objects:
        return state_grid

    # 玩家应该被最先处理，以确保它能占据最准确的位置
    yolo_objects.sort(key=lambda obj: 1 if obj[0] != 0 else 0)

    for obj in yolo_objects:
        class_id, x_center, y_center, width, height = obj
        
        grid_pos_float = np.array([x_center * grid_width, y_center * grid_height])
        pos = pos_finder.find_near_pos(grid_pos_float)
        
        if pos is not None:
            grid_y, grid_x = pos
            
            # --- [对齐] 填充更丰富的特征 ---
            state_grid[grid_y, grid_x, 0] = class_id
            state_grid[grid_y, grid_x, 1] = grid_pos_float[0] - grid_x
            state_grid[grid_y, grid_x, 2] = grid_pos_float[1] - grid_y
            state_grid[grid_y, grid_x, 3] = width * grid_width
            state_grid[grid_y, grid_x, 4] = height * grid_height
            
            # 简单的 one-hot 特征示例
            if class_id == 0: # player
                state_grid[grid_y, grid_x, 5] = 1.0
            elif class_id in [2, 3, 4, 5]: # train, low_barrier, high_barrier
                state_grid[grid_y, grid_x, 6] = 1.0
            elif class_id == 1: # coin
                state_grid[grid_y, grid_x, 7] = 1.0
    
    return state_grid