# subway_surfers_ai/subway_surfers_ai/utils/constants.py

"""
这个文件定义了整个项目中所有模块都会用到的常量。
这是一种非常好的工程实践，可以避免在代码中使用“魔法字符串”或“魔法数字”，
使得代码更易于维护和理解。
"""

# --- 物体类别定义 ---

# 这是一个列表(list)，定义了我们第一版AI需要识别的所有物体类别名称。
# 列表的“顺序”非常重要，它决定了每个类别对应的ID号。
# 比如，'player'的ID就是0，'coin'的ID就是1，以此类推。
OBJECT_CLASSES = [
    'player',               # 玩家角色
    'coin',                 # 金币
    'train1',               # 可以上去的火车
    'train',                # 火车障碍物需要避开
    'low_barrier',          # 低空栏杆 (需要下滑)
    'high_barrier',         # 限高栅栏 (需要跳跃)
    'double_score_powerup', # 道具：双倍得分
    'jetpack_powerup',      # 道具：喷射背包
    'shoes_powerup',        # 道具：超级跑鞋
    'magnet_powerup'        # 道具：磁铁w
]

# 为了方便在代码中通过名字快速查找ID，我们创建一个字典(dictionary)。
# 这叫做“从列表生成字典”的推导式写法，是一种非常Pythonic的技巧。
CLASS_TO_ID = {class_name: i for i, class_name in enumerate(OBJECT_CLASSES)}


# 同样，为了方便从ID反向查找名字（比如在画图显示结果时），
# 我们也创建一个ID到名字的字典。
ID_TO_CLASS = {i: class_name for i, class_name in enumerate(OBJECT_CLASSES)}

# 获取我们总共有多少个类别，len()函数可以计算列表的长度。
NUM_CLASSES = len(OBJECT_CLASSES)


# --- 打印一下，确保我们的定义是正确的 ---
# if __name__ == '__main__' 的技巧，让这段代码只在直接运行本文件时执行。
if __name__ == '__main__':
    print("--- 地铁跑酷AI 物体类别定义 v1.0 ---")
    print(f"总共定义了 {NUM_CLASSES} 个类别。")
    print("\n类别列表 (ID由顺序决定):")
    print(OBJECT_CLASSES)
    print("\n从类别名到ID的映射:")
    print(CLASS_TO_ID)
    print("\n从ID到类别名的映射:")
    print(ID_TO_CLASS)

    # 测试一下如何使用
    print("\n--- 使用示例 ---")
    player_id = CLASS_TO_ID['player']
    print(f" 'player' 对应的ID是: {player_id}")
    
    class_name_of_id_3 = ID_TO_CLASS[3]
    print(f" ID为3对应的类别名是: '{class_name_of_id_3}'")