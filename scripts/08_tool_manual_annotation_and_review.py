# scripts/gongju_shengcheng_dongzuo_shuju.py (V2.0 - 支持追加和撤销)

import cv2
import os
from pathlib import Path
import json

# --- 配置区 ---
XIANGMU_GEN_MULU = Path(__file__).resolve().parents[1]
YUAN_ZHEN_MULU = XIANGMU_GEN_MULU / "data" / "dongzuo_shibie_shuju" / "raw_frames"
BIAOZHU_SHUCHU_LUJING = XIANGMU_GEN_MULU / "data" / "dongzuo_shibie_shuju" / "annotations" / "dongzuo_biaozhu.json"

# --- 主逻辑 ---
class DongzuoBiaozhuGongju:
    def __init__(self, zhen_mulu):
        self.zhen_mulu = Path(zhen_mulu)
        self.zhen_wenjian_liebiao = sorted([f for f in self.zhen_mulu.iterdir() if f.suffix.lower() in ['.png', '.jpg']])
        if not self.zhen_wenjian_liebiao:
            raise FileNotFoundError(f"在目录 {self.zhen_mulu} 中未找到任何图片帧！")

        self.dangqian_zhen_suoyin = 0
        self.linshi_biaozhu = {}
        
        self.action_map = {
            ord('w'): 'UP',
            ord('s'): 'DOWN',
            ord('q'): 'LEFT',
            ord('e'): 'RIGHT'
        }
        BIAOZHU_SHUCHU_LUJING.parent.mkdir(parents=True, exist_ok=True)
        self.zai_ru_jiu_biaozhu() # V2.0 新增：加载旧标注

    def zai_ru_jiu_biaozhu(self):
        """V2.0 新增功能：如果存在旧的标注文件，则加载它。"""
        if BIAOZHU_SHUCHU_LUJING.exists():
            try:
                with open(BIAOZHU_SHUCHU_LUJING, 'r') as f:
                    self.biaozhu_jieguo = json.load(f)
                print(f"成功加载了 {len(self.biaozhu_jieguo)} 条已有的标注。")
            except (json.JSONDecodeError, IOError) as e:
                print(f"警告：无法加载旧的标注文件 {BIAOZHU_SHUCHU_LUJING}。将创建一个新文件。错误: {e}")
                self.biaozhu_jieguo = []
        else:
            self.biaozhu_jieguo = []

    def chexiao_shangyici(self):
        """V2.0 新增功能：撤销上一个保存的标注。"""
        if self.biaozhu_jieguo:
            last_annotation = self.biaozhu_jieguo.pop()
            print(f"\033[91m已撤销上一个标注: {last_annotation}\033[0m")
        else:
            print("\033[93m警告: 没有可供撤销的标注。\033[0m")

    def yunxing(self):
        cv2.namedWindow("Dongzuo Biaozhu", cv2.WINDOW_NORMAL)
        print("\n--- 动作标注工具 V2.0 ---")
        print(" a: 上一帧 | d: 下一帧")
        print(" w/s/q/e: 标记动作起始帧 (UP/DOWN/LEFT/RIGHT)")
        print(" z: 撤销上一个已保存的样本")
        print(" 空格键: 标记动作结束帧并保存样本")
        print(" ESC: 退出并保存所有标注")
        print("---------------------------\n")

        while True:
            zhen_lujing = self.zhen_wenjian_liebiao[self.dangqian_zhen_suoyin]
            image = cv2.imread(str(zhen_lujing))
            
            # 在图像上显示信息
            info_text = f"Frame: {zhen_lujing.name} ({self.dangqian_zhen_suoyin + 1}/{len(self.zhen_wenjian_liebiao)})"
            cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # V2.0 优化：显示已标注数量
            count_text = f"Yi Biaozhu: {len(self.biaozhu_jieguo)}"
            cv2.putText(image, count_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if 'start_frame' in self.linshi_biaozhu:
                status_text = f"Biaozhu Zhong: {self.linshi_biaozhu['action']} from frame {self.linshi_biaozhu['start_frame_name']}"
                cv2.putText(image, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Dongzuo Biaozhu", image)
            
            key = cv2.waitKey(0) & 0xFF

            if key == ord('d'):
                self.dangqian_zhen_suoyin = min(self.dangqian_zhen_suoyin + 10, len(self.zhen_wenjian_liebiao) - 1) # V2.0 优化：一次快进10帧
            elif key == ord('a'):
                self.dangqian_zhen_suoyin = max(self.dangqian_zhen_suoyin - 10, 0) # V2.0 优化：一次快退10帧
            elif key == ord('l'): # V2.0 新增：单帧前进
                self.dangqian_zhen_suoyin = min(self.dangqian_zhen_suoyin + 1, len(self.zhen_wenjian_liebiao) - 1)
            elif key == ord('k'): # V2.0 新增：单帧后退
                self.dangqian_zhen_suoyin = max(self.dangqian_zhen_suoyin - 1, 0)
            elif key in self.action_map:
                self.linshi_biaozhu = {
                    'video_name': zhen_lujing.stem.rsplit('_frame_', 1)[0],
                    'start_frame': self.dangqian_zhen_suoyin,
                    'start_frame_name': zhen_lujing.name,
                    'action': self.action_map[key]
                }
                print(f"标记动作 '{self.linshi_biaozhu['action']}' 起始于: {self.linshi_biaozhu['start_frame_name']}")
            elif key == ord(' '):
                if 'start_frame' in self.linshi_biaozhu:
                    self.linshi_biaozhu['end_frame'] = self.dangqian_zhen_suoyin
                    self.biaozhu_jieguo.append(self.linshi_biaozhu)
                    print(f"\033[92m成功保存样本 ({len(self.biaozhu_jieguo)}): {self.linshi_biaozhu}\033[0m")
                    self.linshi_biaozhu = {}
                else:
                    print("\033[93m警告: 请先用 w/s/q/e 键标记一个动作的起始帧。\033[0m")
            elif key == ord('z'): # V2.0 新增：撤销
                self.chexiao_shangyici()
            elif key == 27:
                break
        
        cv2.destroyAllWindows()
        self.baocun_biaozhu()

    def baocun_biaozhu(self):
        # V2.0 修正：始终使用 'w' 模式，因为 self.biaozhu_jieguo 已经包含了所有新旧数据
        with open(BIAOZHU_SHUCHU_LUJING, 'w') as f:
            json.dump(self.biaozhu_jieguo, f, indent=4)
        print(f"\n--- 标注完成！共 {len(self.biaozhu_jieguo)} 个动作样本已保存到: {BIAOZHU_SHUCHU_LUJING} ---")

if __name__ == "__main__":
    # 移除了所有文件重命名逻辑，只保留核心功能调用
    tool = DongzuoBiaozhuGongju(YUAN_ZHEN_MULU)
    tool.yunxing()