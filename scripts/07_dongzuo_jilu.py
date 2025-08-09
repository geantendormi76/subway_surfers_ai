# /home/zhz/Deepl/subway_surfers_ai/scripts/07_dongzuo_jilu.py (uiautomator2 最终版)

import time
import sys
from pathlib import Path
from pynput import keyboard
import uiautomator2 as u2

# --- 配置区 ---
ACTION_MAP = {
    'w': 'up',
    's': 'down',
    'a': 'left',
    'd': 'right'
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_LOG_FILE = "gameplay_01_actions.txt"
OUTPUT_LOG_PATH = PROJECT_ROOT / OUTPUT_LOG_FILE

def record_and_control_session():
    """
    使用 pynput 监听键盘，并使用 uiautomator2 实时控制手机。
    """
    print("--- 动作记录与u2实时控制程序 ---")

    # 1. 连接到设备
    try:
        print("正在通过 uiautomator2 连接到手机...")
        # u2.connect() 会自动查找通过ADB连接的设备
        d = u2.connect('192.168.3.17:34781')
        print(f"✅ 成功连接到设备: {d.device_info['serial']}")
        # 获取屏幕尺寸，用于计算滑动坐标
        width, height = d.window_size()
        print(f"屏幕尺寸: {width}x{height}")
    except Exception as e:
        print(f"❌ 连接设备失败: {e}")
        print("请确保手机已通过ADB连接，并且uiautomator2守护进程已正确安装运行。")
        return

    # 2. 定义滑动函数
    def perform_swipe(action):
        center_x, center_y = width // 2, height // 2
        swipe_dist = height // 6

        try:
            if action == 'up':
                d.swipe(center_x, center_y, center_x, center_y - swipe_dist, 0.1)
            elif action == 'down':
                d.swipe(center_x, center_y, center_x, center_y + swipe_dist, 0.1)
            elif action == 'left':
                d.swipe(center_x, center_y, center_x - swipe_dist, center_y, 0.1)
            elif action == 'right':
                d.swipe(center_x, center_y, center_x + swipe_dist, center_y, 0.1)
            print(f"    -> 已通过u2发送指令: '{action}'")
        except Exception as e:
            print(f"    -> [错误] u2发送指令 '{action}' 失败: {e}")


    # 3. 启动键盘监听和记录
    print("\n请将鼠标焦点置于Scrcpy投屏窗口。")
    print(f"所有动作将被记录到: {OUTPUT_LOG_PATH}")
    print("按下 'Esc' 键停止。")

    log_file = open(OUTPUT_LOG_PATH, 'w')
    start_time = time.time()

    def on_press(key):
        try:
            char_key = key.char
            if char_key in ACTION_MAP:
                timestamp = time.time() - start_time
                action = ACTION_MAP[char_key]
                
                # 1. 记录
                log_entry = f"{timestamp:.4f},{action}\n"
                log_file.write(log_entry)
                log_file.flush()
                print(f"记录: {log_entry.strip()}")

                # 2. 通过 uiautomator2 控制
                perform_swipe(action)

        except AttributeError:
            if key == keyboard.Key.esc:
                print("检测到 Esc 键，停止...")
                log_file.close()
                return False # 停止监听

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    listener.join()

    print(f"--- 程序结束，日志已保存到 {OUTPUT_LOG_PATH} ---")

if __name__ == '__main__':
    record_and_control_session()