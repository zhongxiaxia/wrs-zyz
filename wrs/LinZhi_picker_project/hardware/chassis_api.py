# hardware/chassis_api.py
import time

class ChassisAPI:
    def __init__(self, ip):
        self.ip = ip
        print(f"[硬件模块] 底盘初始化完毕 (IP: {self.ip})")

    def move_to(self, target_node):
        print(f"[底盘 API] 开始移动到停靠点 {target_node}...")
        time.sleep(2) # 模拟移动耗时
        print("[底盘 API] 已到达目标位置。")
        return True

    def set_diagonal_mode(self, enabled):
        state = "开启" if enabled else "关闭"
        print(f"[底盘 API] 轮子斜向模式已 {state}。")