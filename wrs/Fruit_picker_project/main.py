# main.py
from __future__ import annotations

import sys
import time
from pathlib import Path

WRS_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PEPPER_ROOT = WRS_ROOT / "Pepper_"
if str(LOCAL_PEPPER_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_PEPPER_ROOT))

import config
from fsm.robot_context import FruitPickerRobot
from pick.pepper_picker import RealmanPepperPickerFast


def build_robot() -> FruitPickerRobot:
    robot = FruitPickerRobot()
    robot.configure_lift_layer(
        step_mm=config.LIFT_STEP_MM,
        min_height_mm=config.LIFT_MIN_HEIGHT_MM,
        max_height_mm=config.LIFT_MAX_HEIGHT_MM,
    )
    return robot


def init_hardware_for_pick() -> RealmanPepperPickerFast:
    picker = RealmanPepperPickerFast(real_robot=True, defer_init=True)
    picker.initialize()
    return picker


def main():
    print("=== 采摘机器人控制系统启动 ===")
    picker = init_hardware_for_pick()
    robot = build_robot()
    robot.real_picker = picker
    robot.use_external_picker = True
    robot.camera_api.set_backend(picker)
    robot.change_state("INIT")

    try:
        while True:
            robot.update()
            time.sleep(0.1)
            if robot.current_state_name in ["SHUTDOWN", "ERROR_STOP"]:
                break
    except KeyboardInterrupt:
        print("\n[MAIN] 检测到手动终止 (Ctrl+C)，正在安全关机...")
        robot.change_state("SHUTDOWN")


if __name__ == "__main__":
    main()