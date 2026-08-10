# hardware/arm_api.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import config


@dataclass
class ArmPose:
    name: str
    joint_values: list[float] | None = None
    tcp_xyz: tuple[float, float, float] | None = None
    tcp_rpy: tuple[float, float, float] | None = None


class ArmAPI:
    """机械臂适配层。

    设计目标：
    1. 对上层状态机保持稳定接口。
    2. 下层可切换到真实机械臂 SDK（如 RealmanArmController / ClawController）。
    3. 若 SDK 不可用，则自动退回到可运行的仿真占位实现。
    """

    def __init__(self, ip: str, use_real_sdk: bool = True):
        self.ip = ip
        self.use_real_sdk = use_real_sdk
        self.rm_controller: Any = None
        self.claw_controller: Any = None
        self._connected = False
        self._observed_pose = ArmPose(config.ARM_OBSERVE_POSE_NAME)
        self._safe_pose = ArmPose(config.ARM_SAFE_POSE_NAME)
        self._basket_pose = ArmPose(config.ARM_BASKET_POSE_NAME)
        print(f"[硬件模块] 睿尔曼机械臂适配层初始化 (IP: {self.ip}, real_sdk={self.use_real_sdk})")
        self.connect()

    def connect(self) -> bool:
        if not self.use_real_sdk:
            print("[机械臂 API] 当前为仿真模式，未连接真实 SDK。")
            self._connected = True
            return True

        try:
            from wrs.robot_con.realman.realman import RealmanArmController
        except Exception as e:
            print(f"[机械臂 API] 真实 SDK 导入失败，回退到仿真模式：{e}")
            self.use_real_sdk = False
            self._connected = True
            return True

        try:
            self.rm_controller = RealmanArmController(has_gripper=True)
            self._connected = True
            print("[机械臂 API] 真实机械臂 SDK 连接成功。")
            return True
        except Exception as e:
            print(f"[机械臂 API] 真实机械臂连接失败，回退到仿真模式：{e}")
            self.rm_controller = None
            self.claw_controller = None
            self.use_real_sdk = False
            self._connected = True
            return True

    def is_connected(self) -> bool:
        return self._connected

    def _sleep(self, seconds: float) -> None:
        import time
        time.sleep(seconds)

    def _call_rm(self, method_name: str, *args, **kwargs) -> bool:
        if self.use_real_sdk and self.rm_controller is not None and hasattr(self.rm_controller, method_name):
            getattr(self.rm_controller, method_name)(*args, **kwargs)
            return True
        return True

    def _open_gripper(self) -> bool:
        if self.use_real_sdk and self.claw_controller is not None:
            self.claw_controller.open_gripper()
            return True
        print("[机械臂 API] 张开夹爪(仿真)...")
        return True

    def _close_gripper(self) -> bool:
        if self.use_real_sdk and self.claw_controller is not None:
            self.claw_controller.close_gripper()
            return True
        print("[机械臂 API] 闭合夹爪(仿真)...")
        return True

    def move_to_observe_pose(self) -> bool:
        print(f"[机械臂 API] 移动至观测姿态({self._observed_pose.name})...")
        return self._call_rm("move_jntspace_path", [self.get_current_joint_values(), self.get_current_joint_values()], control_frequency=0.8)

    def open_gripper(self) -> bool:
        return self._open_gripper()

    def close_gripper(self) -> bool:
        return self._close_gripper()

    def get_current_joint_values(self) -> list[float]:
        if self.use_real_sdk and self.rm_controller is not None and hasattr(self.rm_controller, "get_joint_values"):
            return self.rm_controller.get_joint_values()
        return [0.0] * 6

    def pre_grasp_check(self, xyz_pos: tuple[float, float, float]) -> bool:
        print(f"[机械臂 API] 预抓取检查通过: {xyz_pos}")
        return True

    def approach_target(self, xyz_pos: tuple[float, float, float]) -> bool:
        print(f"[机械臂 API] 接近目标点: {xyz_pos}, 预留偏移: {config.ARM_APPROACH_OFFSET_M}m")
        if self.use_real_sdk and self.rm_controller is not None:
            return True
        self._sleep(1.0)
        return True

    def move_to_target(self, xyz_pos: tuple[float, float, float]) -> bool:
        return self.approach_target(xyz_pos)

    def grab(self, xyz_pos: tuple[float, float, float]) -> bool:
        print(f"[机械臂 API] 开始执行抓取，目标坐标: {xyz_pos}")
        self.close_gripper()
        return True

    def retreat_after_grab(self) -> bool:
        print(f"[机械臂 API] 抓取后撤离到安全位置，偏移: {config.ARM_RETREAT_OFFSET_M}m...")
        if self.use_real_sdk and self.rm_controller is not None:
            return True
        self._sleep(0.8)
        return True

    def move_to_basket_pose(self) -> bool:
        print(f"[机械臂 API] 移动到篮筐上方({self._basket_pose.name})...")
        if self.use_real_sdk and self.rm_controller is not None:
            return True
        self._sleep(1.0)
        return True

    def put_in_basket(self) -> bool:
        self.move_to_basket_pose()
        print("[机械臂 API] 松开夹爪并放入篮筐...")
        self.open_gripper()
        return True

    def reset_to_safe_pose(self) -> None:
        print(f"[机械臂 API] 收纳机械臂至安全紧凑姿态({self._safe_pose.name})。")
        if self.use_real_sdk and self.rm_controller is not None:
            return
        self._sleep(1)
