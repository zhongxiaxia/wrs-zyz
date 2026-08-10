#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""连续记录 A1：机械臂法兰末端和夹爪末端的世界坐标。

本程序使用 ``pick/pepper_picker.py`` 中相同的 Realman 机器人模型：

* 机械臂末端：第六轴之后的法兰中心；
* 夹爪末端：模型定义的夹爪作用中心（TCP）。

程序只读取当前关节角，不使用相机或 YOLO，也不会移动机械臂。
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
WORKSPACE_ROOT = THIS_FILE.parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from wrs.robot_con.realman.realman import RealmanArmController
from wrs.robot_sim.robots.realman.realman_single_arm import Realman


# ======================== 现场配置 ========================
RECORD_LABEL = "A1"
ROBOT_IP = "192.168.1.18"
ROBOT_PORT = 8080
OUTPUT_DIR = THIS_FILE.parent / "A1_records"
# =========================================================


def calculate_end_points(
    robot: Realman,
    joint_values_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """返回（机械臂法兰世界坐标，夹爪末端世界坐标）。"""
    gripper_tip_position, gripper_tip_rotation = robot.fk(
        joint_values_rad,
        update=False,
    )
    gripper_tip_position = np.asarray(gripper_tip_position, dtype=float)
    gripper_tip_rotation = np.asarray(gripper_tip_rotation, dtype=float)

    # robot.fk() 返回夹爪 TCP。根据模型中的局部 TCP 偏置反算机械臂法兰。
    tcp_local_position = np.asarray(
        robot.manipulator.loc_tcp_pos,
        dtype=float,
    )
    tcp_local_rotation = np.asarray(
        robot.manipulator.loc_tcp_rotmat,
        dtype=float,
    )
    flange_world_rotation = gripper_tip_rotation @ tcp_local_rotation.T
    flange_world_position = (
        gripper_tip_position
        - flange_world_rotation @ tcp_local_position
    )
    return flange_world_position, gripper_tip_position


def save_result(result: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = OUTPUT_DIR / f"{RECORD_LABEL}_{timestamp}.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    controller = None

    try:
        print(f"========== 记录 {RECORD_LABEL} ==========")
        controller = RealmanArmController(
            ip=ROBOT_IP,
            port=ROBOT_PORT,
            has_gripper=False,
        )
        robot = Realman(enable_cc=False)
        record_count = 0

        while True:
            print("\n请将机械臂停在需要记录的位置，并保持静止。")
            command = input(
                "按 Enter 读取并保存；输入 q 后按 Enter 退出："
            ).strip().lower()
            if command in {"q", "quit", "exit"}:
                break

            try:
                controller.wait_until_idle(timeout=30.0)
                joint_values_rad = np.asarray(
                    controller.get_joint_values(),
                    dtype=float,
                )
                if joint_values_rad.shape != (6,):
                    raise RuntimeError(
                        "关节角数量错误："
                        f"期望 6，实际 {joint_values_rad.shape}"
                    )

                arm_flange_world, gripper_tip_world = calculate_end_points(
                    robot,
                    joint_values_rad,
                )
                flange_to_gripper_distance = float(
                    np.linalg.norm(gripper_tip_world - arm_flange_world)
                )

                result = {
                    "label": RECORD_LABEL,
                    "recorded_at": time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(),
                    ),
                    "coordinate_frame": "pepper_picker world",
                    "coordinate_unit": "m",
                    "arm_flange_world_m": arm_flange_world.tolist(),
                    "gripper_tip_world_m": gripper_tip_world.tolist(),
                    "flange_to_gripper_distance_m": (
                        flange_to_gripper_distance
                    ),
                    "gripper_tcp_local_offset_m": np.asarray(
                        robot.manipulator.loc_tcp_pos,
                        dtype=float,
                    ).tolist(),
                    "joint_values_rad": joint_values_rad.tolist(),
                    "joint_values_deg": np.degrees(
                        joint_values_rad
                    ).tolist(),
                }
                output_path = save_result(result)
                record_count += 1

                print(f"\n第 {record_count} 条记录完成：")
                print(
                    "机械臂法兰末端世界坐标（m）："
                    f"{np.round(arm_flange_world, 6)}"
                )
                print(
                    "夹爪末端世界坐标（m）：    "
                    f"{np.round(gripper_tip_world, 6)}"
                )
                print(
                    "法兰到夹爪末端距离（m）：  "
                    f"{flange_to_gripper_distance:.6f}"
                )
                print(f"保存位置：{output_path}")
            except Exception as exc:
                print(
                    f"[本次记录失败] {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

        print(f"已退出，共成功记录 {record_count} 条数据。")
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\n已停止记录。")
        return 0
    except Exception as exc:
        print(f"[初始化失败] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if controller is not None:
            try:
                controller.close_connection()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
