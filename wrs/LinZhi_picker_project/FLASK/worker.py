from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# Make the original project importable without modifying its code.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WRS_ROOT = PROJECT_ROOT.parents[1]
PEPPER_ROOT = WRS_ROOT / "Pepper_"
for path in (str(PROJECT_ROOT), str(PEPPER_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import config  # noqa: E402
from fsm.robot_context import FruitPickerRobot  # noqa: E402
from pick.pepper_picker import RealmanPepperPickerFast  # noqa: E402


def _arm_state_label(robot: FruitPickerRobot) -> str:
    mapping = {
        "OBSERVE": "相机观察目标",
        "TARGET_SELECT": "目标筛选与预检查",
        "APPROACH": "机械臂接近目标",
        "GRASP": "执行抓取动作",
        "RETREAT": "抓取后退回",
        "DROP": "投放到篮筐",
        "RESCAN": "重新观察下一目标",
        "IDLE": "机械臂空闲",
    }
    return mapping.get(getattr(robot, "arm_state", None), "机械臂执行中")


def _lift_state_label(robot: FruitPickerRobot) -> str:
    if getattr(robot, "lift_return_required", False):
        return "升降机回到底部"
    if getattr(robot, "lift_advance_required", False):
        return "升降机准备进入下一层"
    if getattr(robot, "lift_step_index", 0) == 0:
        return "升降机准备定位当前层"
    return "升降机执行当前层采摘"


def _infer_task(robot: FruitPickerRobot | None) -> tuple[str | None, str | None]:
    if robot is None:
        return None, None

    state_name = getattr(robot, "current_state_name", None)
    if state_name == "INIT":
        return "初始化", "正在检查硬件并复位机器人"
    if state_name == "MOVE":
        repeat = getattr(robot, "move_repeat_count", 0)
        limit = getattr(robot, "move_repeat_limit", 0)
        return "移动调度", f"外层移动第 {repeat}/{limit} 次循环"
    if state_name == "LIFT":
        target = getattr(robot, "lift_current_target", None)
        if getattr(robot, "lift_return_required", False):
            return "升降", f"到达最高层，正在回到底部（当前目标 {target}mm）"
        return "升降", f"当前层位 {target}mm，{_lift_state_label(robot)}"
    if state_name == "ARM":
        arm_state = getattr(robot, "arm_state", None)
        return "采摘", _arm_state_label(robot) if arm_state else "机械臂执行中"
    if state_name == "IDLE":
        return "待机", "机器人处于空闲状态"
    if state_name == "ERROR":
        return "故障", getattr(robot, "error_info", "机器人发生错误") or "机器人发生错误"
    if state_name == "SHUTDOWN":
        return "关机", "正在执行安全收尾"
    return state_name, None


def _status_dict(
    robot: FruitPickerRobot | None,
    *,
    running: bool,
    stop_requested: bool,
    last_action: str | None,
    last_error: str | None,
    last_task: str | None,
    last_detail: str | None,
    last_transition_at: float | None,
) -> dict[str, Any]:
    task, detail = _infer_task(robot)
    if task is None and last_task is not None:
        task = last_task
        detail = last_detail

    return {
        "running": running,
        "stop_requested": stop_requested,
        "current_state": getattr(robot, "current_state_name", None),
        "task": task,
        "detail": detail,
        "last_action": last_action,
        "last_error": last_error,
        "last_transition_at": last_transition_at,
        "lift": {
            "current_target_mm": getattr(robot, "lift_current_target", None),
            "min_height_mm": getattr(robot, "lift_min_height", None),
            "max_height_mm": getattr(robot, "lift_max_height", None),
            "step_mm": getattr(robot, "lift_step_mm", None),
            "lift_inited": getattr(robot, "lift_inited", None),
            "lift_return_required": getattr(robot, "lift_return_required", None),
            "lift_advance_required": getattr(robot, "lift_advance_required", None),
            "lift_step_index": getattr(robot, "lift_step_index", None),
        },
        "arm": {
            "arm_state": getattr(robot, "arm_state", None),
            "arm_state_label": _arm_state_label(robot) if robot else None,
            "arm_no_target_frames": getattr(robot, "arm_no_target_frames", None),
            "arm_layer_no_target_frames": getattr(robot, "arm_layer_no_target_frames", None),
            "arm_pick_done": getattr(robot, "arm_pick_done", None),
            "arm_retry_count": getattr(robot, "arm_retry_count", None),
        },
        "move": {
            "move_repeat_count": getattr(robot, "move_repeat_count", None),
            "move_repeat_limit": getattr(robot, "move_repeat_limit", None),
        },
    }


def _build_robot() -> FruitPickerRobot:
    robot = FruitPickerRobot()
    robot.configure_lift_layer(
        step_mm=config.LIFT_STEP_MM,
        min_height_mm=config.LIFT_MIN_HEIGHT_MM,
        max_height_mm=config.LIFT_MAX_HEIGHT_MM,
    )
    return robot


def _init_picker() -> RealmanPepperPickerFast:
    picker = RealmanPepperPickerFast(real_robot=True, defer_init=True)
    picker.initialize()
    return picker


def _cleanup_robot(robot: FruitPickerRobot | None, picker: RealmanPepperPickerFast | None) -> None:
    if robot is not None:
        for action in (
            lambda: robot.chassis_api.set_diagonal_mode(False),
            lambda: robot.arm_api.reset_to_safe_pose(),
            lambda: robot.lift_api.reset_to_bottom(),
        ):
            try:
                action()
            except Exception:
                pass

    if picker is not None:
        try:
            picker.shutdown()
        except Exception:
            pass


def run_robot_worker(control_queue, status_queue) -> None:
    robot: FruitPickerRobot | None = None
    picker: RealmanPepperPickerFast | None = None
    running = True
    stop_requested = False
    last_action = "start"
    last_error: str | None = None
    last_task: str | None = None
    last_detail: str | None = None
    last_transition_at: float | None = time.time()

    def publish() -> None:
        payload = _status_dict(
            robot,
            running=running,
            stop_requested=stop_requested,
            last_action=last_action,
            last_error=last_error,
            last_task=last_task,
            last_detail=last_detail,
            last_transition_at=last_transition_at,
        )
        try:
            while True:
                status_queue.get_nowait()
        except Exception:
            pass
        try:
            status_queue.put_nowait(payload)
        except Exception:
            pass

    try:
        picker = _init_picker()
        robot = _build_robot()
        robot.real_picker = picker
        robot.use_external_picker = True
        robot.camera_api.set_backend(picker)
        robot.change_state("INIT")
        last_task, last_detail = "初始化", "准备启动机器人"
        publish()

        while True:
            publish()

            try:
                command = control_queue.get(timeout=0.1)
            except Exception:
                command = None

            if command is not None:
                cmd_type = command.get("type")
                last_action = str(cmd_type or last_action)
                last_transition_at = time.time()

                if cmd_type == "stop":
                    stop_requested = True
                    if robot is not None:
                        try:
                            robot.change_state("SHUTDOWN")
                        except Exception as exc:
                            last_error = str(exc)
                    break

                if cmd_type == "step" and robot is not None:
                    try:
                        robot.update()
                    except Exception as exc:
                        last_error = str(exc)
                        try:
                            robot.error_info = str(exc)
                        except Exception:
                            pass
                        try:
                            robot.change_state("ERROR")
                        except Exception:
                            pass
                    continue

                if cmd_type == "state" and robot is not None:
                    state_name = str(command.get("state") or "").strip().upper()
                    if state_name:
                        try:
                            robot.change_state(state_name)
                            last_task, last_detail = state_name, f"切换到状态 {state_name}"
                        except Exception as exc:
                            last_error = str(exc)
                    continue

                if cmd_type == "reset":
                    stop_requested = True
                    break

                if cmd_type == "start":
                    continue

            if robot is None:
                break

            try:
                robot.update()
            except Exception as exc:
                last_error = str(exc)
                try:
                    robot.error_info = str(exc)
                    robot.change_state("ERROR")
                except Exception:
                    pass
                break

            state_name = getattr(robot, "current_state_name", None)
            last_task, last_detail = _infer_task(robot)
            running = state_name not in ("SHUTDOWN", "ERROR_STOP")
            if state_name in ("SHUTDOWN", "ERROR_STOP"):
                break

            time.sleep(0.1)

    except Exception as exc:
        last_error = str(exc)
    finally:
        running = False
        stop_requested = True
        try:
            if robot is not None and getattr(robot, "current_state_name", None) != "SHUTDOWN":
                try:
                    robot.change_state("SHUTDOWN")
                except Exception:
                    pass
        except Exception:
            pass
        _cleanup_robot(robot, picker)
        try:
            while True:
                status_queue.get_nowait()
        except Exception:
            pass
        try:
            status_queue.put_nowait(
                {
                    "running": False,
                    "stop_requested": True,
                    "current_state": getattr(robot, "current_state_name", None) if robot else None,
                    "task": last_task or "关机",
                    "detail": last_detail or "机器人资源已释放",
                    "last_action": last_action,
                    "last_error": last_error,
                    "last_transition_at": time.time(),
                    "lift": {
                        "current_target_mm": getattr(robot, "lift_current_target", None) if robot else None,
                        "min_height_mm": getattr(robot, "lift_min_height", None) if robot else None,
                        "max_height_mm": getattr(robot, "lift_max_height", None) if robot else None,
                        "step_mm": getattr(robot, "lift_step_mm", None) if robot else None,
                        "lift_inited": getattr(robot, "lift_inited", None) if robot else None,
                        "lift_return_required": getattr(robot, "lift_return_required", None) if robot else None,
                        "lift_advance_required": getattr(robot, "lift_advance_required", None) if robot else None,
                        "lift_step_index": getattr(robot, "lift_step_index", None) if robot else None,
                    },
                    "arm": {
                        "arm_state": getattr(robot, "arm_state", None) if robot else None,
                        "arm_state_label": _arm_state_label(robot) if robot else None,
                        "arm_no_target_frames": getattr(robot, "arm_no_target_frames", None) if robot else None,
                        "arm_layer_no_target_frames": getattr(robot, "arm_layer_no_target_frames", None) if robot else None,
                        "arm_pick_done": getattr(robot, "arm_pick_done", None) if robot else None,
                        "arm_retry_count": getattr(robot, "arm_retry_count", None) if robot else None,
                    },
                    "move": {
                        "move_repeat_count": getattr(robot, "move_repeat_count", None) if robot else None,
                        "move_repeat_limit": getattr(robot, "move_repeat_limit", None) if robot else None,
                    },
                }
            )
        except Exception:
            pass
