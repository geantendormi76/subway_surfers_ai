# scripts/07_dongzuo_jilu.py (v2 - UIA2注入对齐版)

import time
import sys
from pathlib import Path
from pynput import keyboard
import uiautomator2 as u2

# --- [新增] 配置区 ---
# 【重要】确保此处的设备IP与您 play.py 中的一致
DEVICE_IP = "192.168.3.17:37169" # 请替换为您手机的IP和端口
# 【重要】定义输出文件路径
OUTPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "actions" / "gameplay_01_actions.txt"

# --- [新增] 全局变量 ---
d = None
action_file = None
start_time = None
screen_width = 0
screen_height = 0

# --- [新增] 健壮的手机连接函数 (与play.py保持一致) ---
def robust_connect(device_ip):
    """
    连接到安卓设备，如果失败则提供清晰的指引并退出。
    """
    global d, screen_width, screen_height
    print(f"正在尝试连接手机: {device_ip}...")
    try:
        d = u2.connect(device_ip)
        d.info 
        screen_width, screen_height = d.window_size()
        print(f"手机连接成功！屏幕尺寸: {screen_width}x{screen_height}")
        return True
    except Exception as e:
        print("\n--- 连接失败！---")
        print(f"错误详情: {e}")
        print("请确保：")
        print(f"1. 手机的无线调试已开启，且IP与端口为: {device_ip}")
        print(f"2. PC与手机在同一WiFi网络下。")
        print(f"3. 已通过USB线缆执行过一次 'python -m uiautomator2 init'。")
        print("4. scrcpy投屏窗口已打开并能正常显示。")
        return False

# --- [新增] 动作执行与记录函数 ---
def execute_and_record(action_str):
    """
    执行一个动作，并在动作完成后记录时间戳。
    """
    global d, action_file, start_time, screen_width, screen_height
    if not d or not action_file:
        return

    center_x, center_y = screen_width // 2, screen_height // 2
    swipe_dist = screen_height // 6
    
    print(f"执行动作: {action_str.upper()}")
    
    # 执行滑动操作
    if action_str == 'up':
        d.swipe(center_x, center_y, center_x, center_y - swipe_dist, 0.1)
    elif action_str == 'down':
        d.swipe(center_x, center_y, center_x, center_y + swipe_dist, 0.1)
    elif action_str == 'left':
        d.swipe(center_x, center_y, center_x - swipe_dist, center_y, 0.1)
    elif action_str == 'right':
        d.swipe(center_x, center_y, center_x + swipe_dist, center_y, 0.1)
    
    # [核心] 动作执行后，记录当前时间
    current_time = time.time() - start_time
    log_entry = f"{current_time:.4f},{action_str}\n"
    action_file.write(log_entry)
    action_file.flush() # 确保立即写入文件
    print(f"记录: {log_entry.strip()}")


# --- [修改] 键盘监听回调函数 ---
def on_press(key):
    try:
        # 将 pynput 的 Key 对象或字符转换为统一的字符串
        key_char = key.char if hasattr(key, 'char') else key.name
        
        action_map = {
            'w': 'up',
            's': 'down',
            'a': 'left',
            'd': 'right',
            # 如果您使用方向键，也可以添加映射
            'up': 'up',
            'down': 'down',
            'left': 'left',
            'right': 'right'
        }
        
        if key_char in action_map:
            execute_and_record(action_map[key_char])

    except Exception as e:
        print(f"处理按键时发生错误: {e}")

def on_release(key):
    # 按下ESC键停止监听
    if key == keyboard.Key.esc:
        print("\n--- 停止记录 ---")
        return False

# --- [修改] 主程序逻辑 ---
def main():
    global action_file, start_time

    # 1. 连接手机
    if not robust_connect(DEVICE_IP):
        sys.exit(1)

    # 2. 准备文件和计时器
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(OUTPUT_FILE, 'w') as f:
            action_file = f
            start_time = time.time()
            
            print("\n" + "="*40)
            print("--- 动作记录已开始 ---")
            print(f"设备: {DEVICE_IP}")
            print(f"记录文件: {OUTPUT_FILE}")
            print("请将焦点置于 scrcpy 投屏窗口，并开始游戏。")
            print("使用 W/A/S/D 或方向键进行操作。")
            print("按 [ESC] 键结束录制。")
            print("="*40 + "\n")

            # 3. 启动键盘监听器
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                listener.join()
                
    except IOError as e:
        print(f"错误：无法写入文件 {OUTPUT_FILE}。请检查权限。")
        print(f"详情: {e}")
    finally:
        if action_file:
            action_file.close()

if __name__ == '__main__':
    main()