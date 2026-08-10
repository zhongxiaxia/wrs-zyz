import os
import json
import time
import numpy as np
import keyboard

from wrs import rrtc
from wrs.robot_con.realman.realman import RealmanArmController

# ========== 配置 ==========
IP = "192.168.1.18"
PORT = 8080
CONTROL_FREQ = 0.6
SPEED = 5  # 直接移动速度
DIST_THRESHOLD = 1.0  # 如果起止角小于此值（角度），则认为“很近”不规划
JSON_PATH = "recorded_motions.json"

robot = RealmanArmController(ip=IP, port=PORT,has_gripper=False)
planner = rrtc.RRTConnect(robot)

motion_sequence = []  # 暂存每对 start-end
current_motion = {}


def save_motion_sequence():
    with open(JSON_PATH, "w") as f:
        json.dump({"motion_sequence": motion_sequence}, f, indent=2)
    print(f"[保存] 所有记录已写入 {JSON_PATH}")


def distance_between(joints1, joints2):
    return np.linalg.norm(np.array(joints1) - np.array(joints2))


def plan_and_execute(start_conf_deg, end_conf_deg):
    dist = distance_between(start_conf_deg, end_conf_deg)
    if dist < DIST_THRESHOLD:
        print(f"\n[直接] 起止位置非常接近（{dist:.2f}°），直接移动")
        robot.move_j(end_conf_deg, speed=SPEED)
        return

    print(f"\n[规划] 从 {np.round(start_conf_deg, 1)} 到 {np.round(end_conf_deg, 1)}，距离：{dist:.2f}°")
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
            follow=True,          # 使用高跟随模式
            expand=0,
            trajectory_mode=1,    # 曲线拟合模式
            radio=50              # 平滑系数
        )
        if ret != 0:
            print(f"[错误] 第 {idx} 个点执行失败，状态码: {ret}")
        time.sleep(CONTROL_FREQ)

    print("[完成] 全路径执行完成")


def execute_all_from_json(json_path):
    if not os.path.exists(json_path):
        print(f"[错误] 找不到文件: {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    motion_sequence = data.get("motion_sequence", [])
    for i, motion in enumerate(motion_sequence):
        start = motion.get("start_joint")
        end = motion.get("end_joint")
        if start is None or end is None:
            print(f"[跳过] 有不完整的 start 或 end，id={motion.get('id')}")
            continue

        # 选择是否直接移动
        if i > 0:
            prev_end = motion_sequence[i - 1].get("end_joint")
            if prev_end is not None:
                dist = distance_between(prev_end, start)
                if dist < DIST_THRESHOLD:
                    print(f"[前后连贯] 与上一个动作末尾角度接近（{dist:.2f}°），跳过规划，直接跳到 start")
                    robot.move_j(start, speed=SPEED)
                    time.sleep(0.5)

        plan_and_execute(start, end)
        time.sleep(1)


# ========== 主逻辑循环 ==========
print("⚙️ 记录模式启动：按 's' 记录 start_joint，按 'w' 记录 end_joint")
print("⏹ 按 'p' 停止记录并执行轨迹，按 'esc' 退出")

motion_id = 0

while True:
    if keyboard.is_pressed("s"):
        joint_deg = robot.get_joint_values()[1]
        current_motion["start_joint"] = list(joint_deg)
        print(f"[记录] Start_joint (ID={motion_id}): {np.round(joint_deg, 1)}")
        time.sleep(0.5)

    elif keyboard.is_pressed("w"):
        joint_deg = robot.get_joint_values()[1]
        current_motion["end_joint"] = list(joint_deg)
        print(f"[记录] End_joint (ID={motion_id}): {np.round(joint_deg, 1)}")

        if "start_joint" in current_motion:
            current_motion["id"] = motion_id
            motion_sequence.append(current_motion.copy())
            motion_id += 1
            current_motion = {}
            print(f"[✓] 成功记录第 {motion_id} 段 start-end 配对")
        else:
            print("[⚠️] 请先按 's' 记录 start_joint")
        time.sleep(0.5)

    elif keyboard.is_pressed("p"):
        print("📝 停止记录，准备执行")
        save_motion_sequence()
        execute_all_from_json(JSON_PATH)
        break

    elif keyboard.is_pressed("esc"):
        print("🛑 手动退出")
        break

    time.sleep(0.1)
