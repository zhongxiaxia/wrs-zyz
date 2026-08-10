import os
import json
import time
import numpy as np

from wrs import rrtc
from wrs.robot_con.realman.realman import RealmanArmController

# ======== 配置参数 ========
IP = "192.168.1.18"
PORT = 8080
CONTROL_FREQ = 0.6
SPEED = 5
DIST_THRESHOLD = 1.0
JSON_PATH = "recorded_motions.json"

# ======== 初始化对象 ========
robot = RealmanArmController(ip=IP, port=PORT, has_gripper=False)
planner = rrtc.RRTConnect(robot)

# ======== 工具函数 ========
def distance_between(joints1, joints2):
    return np.linalg.norm(np.array(joints1) - np.array(joints2))


def plan_and_execute(start_conf_deg, end_conf_deg):
    dist = distance_between(start_conf_deg, end_conf_deg)
    if dist < DIST_THRESHOLD:
        print(f"[直接] 起止位置非常接近（{dist:.2f}°），直接移动")
        robot.move_j(end_conf_deg, speed=SPEED)
        return

    print(f"[规划] 从 {np.round(start_conf_deg, 1)} 到 {np.round(end_conf_deg, 1)}，距离：{dist:.2f}°")
    path_result = planner.plan(
        start_conf=np.radians(start_conf_deg),
        goal_conf=np.radians(end_conf_deg),
        ext_dist=0.1,
        max_time=300
    )

    if path_result is None:
        print("[失败] 未找到路径，跳过该段")
        return

    path_deg = [list(np.rad2deg(jv)) for jv in path_result.jv_list]
    print(f"[轨迹] 路径点数量: {len(path_deg)}，开始逐点执行...")

    for idx, point in enumerate(path_deg):
        ret = robot.movej_canfd(
            joint=point,
            follow=True,
            expand=0,
            trajectory_mode=1,
            radio=50
        )
        if ret != 0:
            print(f"[错误] 第 {idx} 个点执行失败，状态码: {ret}")
        time.sleep(CONTROL_FREQ)

    print("[完成] 全路径执行完成")


# ======== 主执行函数 ========
def execute_from_file(json_path):
    if not os.path.exists(json_path):
        print(f"[错误] 找不到文件: {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    motion_sequence = data.get("motion_sequence", [])
    if not motion_sequence:
        print("[提示] 文件中没有任何动作序列")
        return

    for i, motion in enumerate(motion_sequence):
        start = motion.get("start_joint")
        end = motion.get("end_joint")
        if start is None or end is None:
            print(f"[跳过] 动作 ID={motion.get('id')} 缺少 start 或 end")
            continue

        if i > 0:
            prev_end = motion_sequence[i - 1].get("end_joint")
            if prev_end is not None:
                dist = distance_between(prev_end, start)
                if dist < DIST_THRESHOLD:
                    print(f"[连贯] 与上段终点接近（{dist:.2f}°），直接移动到 start")
                    robot.move_j(start, speed=SPEED)
                    time.sleep(0.5)

        plan_and_execute(start, end)
        time.sleep(1)


# ======== 执行入口 ========
if __name__ == "__main__":
    execute_from_file(JSON_PATH)
