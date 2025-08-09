# /home/zhz/Deepl/subway_surfers_ai/utils/controller.py

import socket
import time

# --- 配置区 ---
# HOST '127.0.0.1' (或 'localhost') 代表本机。
# 因为我们使用了 adb forward 进行端口转发，
# 所以从PC的角度看，手机App的端口就像是在本机上一样。
HOST = '127.0.0.1'
# PORT 必须与我们在安卓App (TouchAccessibilityService.kt) 中定义的端口完全一致
PORT = 12345

def send_action(action: str):
    """
    通过TCP Socket向手机App发送一个动作指令。

    Args:
        action (str): 要发送的动作指令，如 "up", "down", "left", "right"。
    """
    # 打印将要执行的动作，方便调试
    print(f"发送指令: {action}")
    try:
        # 创建一个TCP/IP套接字
        # with语句可以确保socket在使用完毕后被自动关闭
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # 连接到服务器 (我们的手机App)
            s.connect((HOST, PORT))
            # 将字符串指令编码为字节流并发送
            # 我们在指令末尾添加一个换行符 '\n'，因为App中的BufferedReader.readLine()需要它来识别一行结束
            s.sendall(f"{action}\n".encode('utf-8'))
    except ConnectionRefusedError:
        print(f"错误: 连接被拒绝。请确保：")
        print(f"  1. 手机上的 'SubwaySurfersHelper' 辅助功能已开启。")
        print(f"  2. 'adb forward tcp:{PORT} tcp:{PORT}' 命令已成功执行。")
    except Exception as e:
        print(f"发送指令 '{action}' 时发生未知错误: {e}")

# --- 测试区 ---
if __name__ == '__main__':
    # 这个部分用于独立测试脚本是否能正常工作
    print("将在3秒后开始测试辅助功能注入...")
    print("请确保游戏投屏窗口在前台并已开始一局游戏。")
    time.sleep(3)
    
    print("\n--- 测试开始 ---")
    
    # 模拟一系列游戏操作
    send_action('up')    # 跳跃
    time.sleep(1)        # 等待1秒
    
    send_action('down')  # 翻滚
    time.sleep(1)
    
    send_action('left')  # 向左
    time.sleep(1)
    
    send_action('right') # 向右
    time.sleep(1)
    
    send_action('up')    # 再次跳跃
    time.sleep(1)
    
    print("--- 控制测试完成 ---")