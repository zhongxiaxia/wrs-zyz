#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""辣椒采摘主流程：YOLO 检测、IK 求解与运动规划。

整体流程：
- 使用 YOLO 关键点检测目标辣椒
- 将图像中的检测结果转换为机器人世界坐标
- 通过 IK 搜索可行的抓取姿态
- 使用 RRT 规划观察位、预抓取、放置位和返回路径
- 使用直线运动完成最后一小段抓取靠近动作
- 在执行前检查与箱体模型的碰撞
"""

from __future__ import annotations

import importlib.util
import csv
import json
import os
import sys
import time
from pathlib import Path

WRS_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PEPPER_ROOT = WRS_ROOT / "Pepper_"
if str(LOCAL_PEPPER_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_PEPPER_ROOT))

# 当系统缺少字体时，提前指定 Qt 字体目录，避免 OpenCV 窗口报错。
if not os.environ.get("QT_QPA_FONTDIR"):
    for _qt_font in (
            "/usr/share/fonts/truetype/dejavu",
            "/usr/share/fonts/truetype/liberation",
            "/usr/share/fonts/truetype/noto",
            "/usr/share/fonts",
    ):
        if os.path.isdir(_qt_font):
            os.environ["QT_QPA_FONTDIR"] = _qt_font
            break


def _prepare_cv2_qt_fonts() -> None:
    """为 OpenCV 的 Qt 后端准备字体目录。

    某些嵌入式或最小化系统里，OpenCV 的图形窗口会因为找不到
    字体而出现警告。这里尝试把系统已有字体链接到 cv2 的 Qt
    字体目录中，尽量保证调试窗口正常显示中文和英文。
    """
    try:
        spec = importlib.util.find_spec("cv2")
        if spec is None:
            return
        cv2_root = None
        locs = getattr(spec, "submodule_search_locations", None) or ()
        if locs:
            cv2_root = locs[0]
        elif getattr(spec, "origin", None):
            cv2_root = os.path.dirname(spec.origin)
            if os.path.basename(cv2_root) != "cv2":
                alt = os.path.join(cv2_root, "cv2")
                if os.path.isdir(alt):
                    cv2_root = alt
        if not cv2_root or not os.path.isdir(cv2_root):
            py = f"python{sys.version_info.major}.{sys.version_info.minor}"
            for lib in ("lib", "lib64"):
                alt = os.path.join(sys.prefix, lib, py, "site-packages", "cv2")
                if os.path.isdir(alt):
                    cv2_root = alt
                    break
        if not cv2_root or not os.path.isdir(cv2_root):
            return
        qt_fonts = os.path.join(cv2_root, "qt", "fonts")
        for dejavu in (
                "/usr/share/fonts/truetype/dejavu",
                "/usr/share/fonts/truetype/liberation",
                "/usr/share/fonts/truetype/noto",
        ):
            if not os.path.isdir(dejavu):
                continue
            try:
                os.makedirs(qt_fonts, exist_ok=True)
            except OSError:
                return
            for name in os.listdir(dejavu):
                low = name.lower()
                if not (low.endswith(".ttf") or low.endswith(".otf") or low.endswith(".ttc")):
                    continue
                src = os.path.join(dejavu, name)
                dst = os.path.join(qt_fonts, name)
                if os.path.lexists(dst):
                    continue
                try:
                    os.symlink(os.path.realpath(src), dst)
                except OSError:
                    pass
            return
    except Exception:
        return


_prepare_cv2_qt_fonts()

import numpy as np
import cv2
import pyrealsense2 as rs

import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.modeling.geometric_model as mgm
import wrs.motion.primitives.approach_depart_planner as adp
import wrs.motion.probabilistic.rrt_connect as rrtc
import wrs.visualization.panda.world as wd
from Robotic_Arm.rm_ctypes_wrap import rm_peripheral_read_write_params_t
from Robotic_Arm.rm_robot_interface import *
from wrs.Fruit_picker_project import config
from wrs.Pepper.vision_io import safe_imshow
from wrs.Fruit_picker_project.pick.pepper_path_io import save_pick_path_record
from wrs.robot_con.realman.realman import RealmanArmController
from wrs.robot_sim.robots.realman.realman_single_arm import Realman

# 当前文件位于 `wrs/Fruit_picker_project/pick/...` 下，
# 所以工作区根目录需要向上回退两层。
WRS_ROOT = Path(__file__).resolve().parents[2]
PEPPER_DIR = WRS_ROOT / "Pepper"
PEPPER_NBV_DIR = WRS_ROOT / "Pepper_NBV"

CLAW_ID = 9
MODBUS_PORT = 1
GRIPPER_VOLTAGE_TYPE = 3  # 24V
GRIPPER_BAUDRATE = 115200
GRIPPER_TIMEOUT = 2

JOINT_LIMIT_MARGIN = 0.08
JOINT3_EXTRA_MARGIN = 0.18
RRT_PATH_MARGIN = 0.03
# A 10-15 cm Cartesian approach should stay on one local IK branch.  A larger
# per-joint endpoint change indicates that the IK solver selected a different
# arm configuration even though it was seeded with the neighbouring pose.
APPROACH_GRASP_MAX_JOINT_DELTA_DEG = 60.0
LINEAR_PATH_MAX_JOINT_STEP_DEG = 30.0
LINEAR_PATH_ENDPOINT_TOLERANCE_DEG = 1.0
# Independently generated adjacent paths can differ at their shared endpoint
# by a tiny floating-point/IK residual.  Snap only negligible gaps; a larger
# discontinuity remains an execution error.
CONTINUOUS_PATH_JOIN_SNAP_TOLERANCE_DEG = 0.05

# TOPPRA speed profiles.  Velocity is in deg/s and acceleration is in deg/s^2
# here; they are converted to radians before being passed to the trajectory
# generator.  Keep the sample period at 0.1 s because the current RealMan
# controller wrapper sends one CAN-FD point every 0.1 s.
TOPPRA_SAMPLE_PERIOD_S = 0.1
TRANSFER_MAX_JNTVEL_DEG_S = np.array([15, 15, 15, 20, 20, 25], dtype=float)
TRANSFER_MAX_JNTACC_DEG_S2 = np.array([30, 30, 30, 40, 40, 50], dtype=float)
APPROACH_MAX_JNTVEL_DEG_S = np.array([8, 8, 8, 10, 10, 12], dtype=float)
APPROACH_MAX_JNTACC_DEG_S2 = np.array([15, 15, 15, 20, 20, 25], dtype=float)


def _make_gripper_params(port: int, address: int, device: int, num: int = 0):
    params = rm_peripheral_read_write_params_t()
    params.port = port
    params.address = address
    params.device = device
    params.num = num
    return params


class RealManGripperController:
    def __init__(self, arm: RoboticArm, device_id: int = CLAW_ID, modbus_port: int = MODBUS_PORT):
        self.arm = arm
        self.device_id = device_id
        self.modbus_port = modbus_port

    def set_tool_voltage_24v(self):
        result = self.arm.rm_set_tool_voltage(GRIPPER_VOLTAGE_TYPE)
        print(f"[RealMan] set_tool_voltage result={result}")
        return result

    def set_modbus_mode(self):
        result = self.arm.rm_set_modbus_mode(self.modbus_port, GRIPPER_BAUDRATE, GRIPPER_TIMEOUT)
        print(f"[RealMan] set_modbus_mode result={result}")
        return result

    def write_registers(self, address: int, num: int, data: list[int]):
        params = _make_gripper_params(self.modbus_port, address, self.device_id, num=num)
        result = self.arm.rm_write_registers(params, data)
        print(f"[RealMan] write_registers address={address}, num={num}, data={data}, result={result}")
        return result

    def read_input_registers(self, address: int):
        params = _make_gripper_params(self.modbus_port, address, self.device_id)
        result = self.arm.rm_read_input_registers(params)
        print(f"[RealMan] read_input_registers address={address}, result={result}")
        return result

    def read_holding_registers(self, address: int, num: int):
        params = _make_gripper_params(self.modbus_port, address, self.device_id, num=num)
        result = self.arm.rm_read_multiple_holding_registers(params)
        print(f"[RealMan] read_holding_registers address={address}, num={num}, result={result}")
        return result

    def activate_gripper(self, timeout_s: float = 10.0, poll_interval_s: float = 0.2):
        self.write_registers(address=1000, num=1, data=[0, 0])
        time.sleep(0.2)
        self.write_registers(address=1000, num=1, data=[0, 1])
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            status, value = self.read_input_registers(2000)
            print(f"[RealMan] activation status={status}, value={value}")
            # 0xD1 (209) 仍可能处于激活/回零过程中；若此时发送张开
            # 命令，后续回零动作可能再次把夹爪带回闭合位置。
            # 仅在 0xF1 (241) 表示激活真正完成后返回。
            if status == 0 and value == 0xF1:
                return True
            time.sleep(poll_interval_s)
        raise TimeoutError("Gripper activation timeout")

    def close_gripper(self):
        return self.write_registers(address=1000, num=3, data=[0, 9, 255, 0, 255, 255])

    def open_gripper(self):
        return self.write_registers(address=1000, num=3, data=[0, 9, 0, 0, 255, 255])


claw_controller = None


def init_claw_hardware(arm: Optional[RoboticArm] = None):
    """初始化夹爪硬件控制器。

    如果当前机械臂已经安装了夹爪，就会通过该函数完成供电、
    Modbus 配置和夹爪激活状态检查。激活完成后发送一次张开命令，
    但不读取夹爪位置寄存器，也不检测张开是否成功。
    如果你现在已经把夹爪拆掉了，
    那么这部分逻辑需要暂时跳过，否则程序会在初始化阶段等待
    夹爪响应。
    """
    global claw_controller
    if arm is None:
        raise RuntimeError("RealMan arm is required to initialize gripper controller")

    reuse_controller = claw_controller is not None and claw_controller.arm is arm
    controller = (
        claw_controller
        if reuse_controller
        else RealManGripperController(arm, device_id=CLAW_ID, modbus_port=MODBUS_PORT)
    )
    try:
        if not reuse_controller:
            controller.set_tool_voltage_24v()
            controller.set_modbus_mode()
            controller.activate_gripper()
            # 部分夹爪在激活/回零后会停在闭合位置。这里直接给定张开目标，
            # 不读取位置寄存器，也不等待或判断张开是否成功。
            controller.open_gripper()
        print("[夹爪初始化] 夹爪已激活并发送张开命令（不检测张开结果）。")
    except Exception as exc:
        try:
            stop_result = arm.rm_set_arm_slow_stop()
            print(f"[安全停止] 夹爪初始化失败，机械臂慢停 result={stop_result}")
        except Exception as stop_exc:
            print(f"[安全停止][警告] 机械臂慢停调用失败：{stop_exc}")
        if not reuse_controller:
            claw_controller = None
        raise RuntimeError(f"夹爪初始化失败：{exc}") from exc

    claw_controller = controller
    return claw_controller


class RealmanPepperPickerFast:
    """High-level pepper picking controller.

    This class bundles together:
    - scene and robot initialization
    - RealSense and YOLO setup
    - IK feasibility checks
    - RRT / linear motion planning
    - gripper control and execution logging
    """

    def __init__(self, real_robot: bool = True, defer_init: bool = False, use_gripper: bool = True):
        self._reset_timing_records()
        self.real_robot = real_robot
        self.defer_init = defer_init
        self.use_gripper = use_gripper
        self._inited = False
        self.base = None
        self.shelf_model = None
        self.box_model = None
        self.robot = None
        self.rrtc_planner = None
        self.adp_planner = None
        self.rm_controller = None
        self.rs_pipeline = None
        self.rs_align = None
        self.HANDEYE = np.eye(4)
        self.yolo_model = None
        self.claw_controller = None
        self.rs_config = None
        self.last_pick_path_record = None

    def initialize(self):
        """Build the simulation scene, connect hardware, and load perception models.

        Initialization is intentionally separated from `__init__` so the caller
        can choose to defer expensive hardware startup when needed.
        """
        if self._inited:
            return
        self._reset_timing_records()
        _init_total_t0 = time.perf_counter()
        _init_t0 = time.perf_counter()

        self.base = wd.World(cam_pos=rm.vec(2, 0, 1), lookat_pos=rm.vec(0, 0, 0.5))
        mgm.gen_frame().attach_to(self.base)
        self._record_timing("init_scene", time.perf_counter() - _init_t0, "world/frame attached")
        _init_t0 = time.perf_counter()

        # self.shelf_model = mcm.gen_box(
        #     xyz_lengths=np.array([0.02, 0.7, 1.55]),
        #     pos=rm.vec(-1, 0, 0),
        #     rotmat=np.eye(3),
        #     rgb=[0.1, 0.1, 0.2],
        #     alpha=0.8,
        # )
        # self.shelf_model.attach_to(self.base)
        # self._record_timing("init_models", time.time() - _init_t0, "shelf attached")
        # _init_t0 = time.time()

        self.box_model = mcm.CollisionModel(
            str(WRS_ROOT / "LinZhi_picker_project" / "model" / "car.stl"),
            rgb=np.array([0.6, 0.6, 0.6]),
            alpha=0.9,
        )
        self.box_model.pos = np.array([0.16, 0.0, 0.0])
        # self.box_model.rotmat = rm.rotmat_from_axangle(np.array([0, 0, 1]), np.pi * 90 / 180)
        self.box_model.attach_to(self.base)

        self.robot = Realman(enable_cc=True)
        self.rrtc_planner = rrtc.RRTConnect(self.robot)
        self.adp_planner = adp.ADPlanner(self.robot)
        self.rm_controller = None
        self.rs_pipeline = None
        self.rs_align = None
        self.HANDEYE = self._load_handeye_matrix()
        self.yolo_model = self._init_yolo()
        self.claw_controller = None
        self._init_hardware()
        if self.use_gripper and self.real_robot:
            self.claw_controller = init_claw_hardware(self.rm_controller.arm)
        else:
            self.claw_controller = None
        self._record_timing("init_robot_planner", time.perf_counter() - _init_t0, "robot/adp ready")
        self._record_timing(
            "init_total",
            time.perf_counter() - _init_total_t0,
            "initialization finished",
            kind="inclusive",
            status="success",
        )
        self._flush_timing_log(total_success=0)
        self._inited = True

    def _load_handeye_matrix(self):
        """加载手眼标定矩阵，把相机坐标映射到机器人世界坐标系。"""
        calib_path = WRS_ROOT / "vision" / "depth_camera" / "manual_calibration_realman22.json"
        try:
            with open(calib_path, "r", encoding="utf-8") as f:
                calib_data = json.load(f)
            print(f"[手眼标定] 已加载: {calib_path}")
            return np.array(calib_data["affine_mat"], dtype=float)
        except Exception:
            print(f"[手眼标定] 读取失败")
            return np.eye(4)

    def _init_yolo(self):
        """初始化用于辣椒定位的 YOLO 关键点模型。"""
        import wrs.Pepper.yoloutils.yolo_utils as yoloutils
        # The active checkpoint is stored under Fruit_picker_project/Weight.
        # model = yoloutils.init_yolo(str(WRS_ROOT / "LinZhi_picker_project" / "Weight" / "best.pt"))
        model = yoloutils.init_yolo(str(WRS_ROOT / "Fruit_picker_project" / "Weight" / "best.pt"))
        if model is None:
            print("[警告] YOLO 初始化失败，将使用仿真/降级模式。")
        return model

    def _record_timing(self, stage, duration, detail="", *, kind="exclusive", status=""):
        """Store one timing row.

        kind='exclusive' is a directly measured component; 'nested' is already
        included in another field; 'inclusive' is a wall-clock subtotal/total;
        and 'summary' is a derived statistic.  Only exclusive components should
        be summed when building a manual breakdown.
        """
        if not hasattr(self, "_timing_records"):
            self._timing_records = []
        duration = float(duration)
        record = {
            "cycle": getattr(self, "_timing_cycle", None),
            "stage": str(stage),
            "duration_s": duration,
            "kind": str(kind),
            "status": str(status),
            "detail": str(detail),
        }
        self._timing_records.append(record)
        cycle_text = "" if record["cycle"] is None else f" | cycle={record['cycle']}"
        meta_text = f" | {kind}"
        if status:
            meta_text += f" | {status}"
        if detail:
            print(f"[计时] {stage}: {duration:.3f} 秒{cycle_text}{meta_text} | {detail}")
        else:
            print(f"[计时] {stage}: {duration:.3f} 秒{cycle_text}{meta_text}")

    def _joint_limit_range(self):
        robot = self.robot
        if robot is None:
            return None
        try:
            ranges = np.asarray(robot.jnt_ranges, dtype=float)
            if ranges.ndim == 2 and ranges.shape[1] == 2:
                return ranges
        except Exception:
            pass
        try:
            ranges = np.asarray(robot.manipulator.jnt_ranges, dtype=float)
            if ranges.ndim == 2 and ranges.shape[1] == 2:
                return ranges
        except Exception:
            pass
        return None

    def _is_conf_within_joint_margin(self, conf, margin: float = JOINT_LIMIT_MARGIN):
        ranges = self._joint_limit_range()
        if ranges is None:
            return True
        conf = np.asarray(conf, dtype=float).reshape(-1)
        if conf.shape[0] != ranges.shape[0]:
            return True
        lower = ranges[:, 0] + margin
        upper = ranges[:, 1] - margin
        return bool(np.all((conf >= lower) & (conf <= upper)))

    def _is_conf_safe_for_joint3(self, conf, extra_margin: float = JOINT3_EXTRA_MARGIN):
        ranges = self._joint_limit_range()
        if ranges is None:
            return True
        conf = np.asarray(conf, dtype=float).reshape(-1)
        if conf.shape[0] != ranges.shape[0] or conf.shape[0] < 3:
            return True
        j3_low, j3_high = ranges[2]
        return bool((conf[2] >= j3_low + extra_margin) and (conf[2] <= j3_high - extra_margin))

    def _path_within_joint_margin(self, path, margin: float = RRT_PATH_MARGIN):
        if path is None:
            return False
        jv_list = getattr(path, "jv_list", path)
        try:
            for conf in jv_list:
                if not self._is_conf_within_joint_margin(conf, margin=margin):
                    return False
                if not self._is_conf_safe_for_joint3(conf):
                    return False
            return True
        except Exception:
            return False

    @staticmethod
    def _joint_delta_deg(conf_a, conf_b):
        """Return raw per-joint differences in degrees without angle wrapping."""
        conf_a = np.asarray(conf_a, dtype=float).reshape(-1)
        conf_b = np.asarray(conf_b, dtype=float).reshape(-1)
        if conf_a.shape != conf_b.shape or conf_a.shape != (6,):
            raise ValueError("expected two 6-DOF joint configurations")
        if not np.all(np.isfinite(conf_a)) or not np.all(np.isfinite(conf_b)):
            raise ValueError("joint configuration contains NaN or Inf")
        return np.abs(np.degrees(conf_b - conf_a))

    def _solve_continuous_approach_grasp_ik(
            self,
            *,
            current_conf,
            target_pos,
            target_rotmat,
            hand_z,
            approach_dis,
    ):
        """Solve approach first, then grasp, and reject IK branch switching."""
        current_conf = np.asarray(current_conf, dtype=float).reshape(-1)
        approach_pos = np.asarray(target_pos, dtype=float) - np.asarray(hand_z, dtype=float) * float(approach_dis)
        approach_conf = self.robot.ik(
            tgt_pos=approach_pos,
            tgt_rotmat=target_rotmat,
            seed_jnt_values=current_conf,
        )
        if approach_conf is None:
            return None, "approach IK failed"
        approach_conf = np.asarray(approach_conf, dtype=float).reshape(-1)
        if not self._is_conf_within_joint_margin(approach_conf):
            return None, "approach configuration is near a joint limit"
        if not self._is_conf_safe_for_joint3(approach_conf):
            return None, "approach joint 3 is near its safety margin"

        # The physical motion is approach -> grasp, so solve in that same order.
        # This gives the IK solver the neighbouring approach pose as its seed.
        grasp_conf = self.robot.ik(
            tgt_pos=target_pos,
            tgt_rotmat=target_rotmat,
            seed_jnt_values=approach_conf,
        )
        if grasp_conf is None:
            return None, "grasp IK failed from approach seed"
        grasp_conf = np.asarray(grasp_conf, dtype=float).reshape(-1)
        if not self._is_conf_within_joint_margin(grasp_conf):
            return None, "grasp configuration is near a joint limit"
        if not self._is_conf_safe_for_joint3(grasp_conf):
            return None, "grasp joint 3 is near its safety margin"

        try:
            delta_deg = self._joint_delta_deg(approach_conf, grasp_conf)
        except ValueError as exc:
            return None, str(exc)
        max_delta_deg = float(np.max(delta_deg))
        if max_delta_deg > APPROACH_GRASP_MAX_JOINT_DELTA_DEG:
            return None, (
                "different IK branch: "
                f"max_joint_delta={max_delta_deg:.3f} deg > "
                f"{APPROACH_GRASP_MAX_JOINT_DELTA_DEG:.3f} deg; "
                f"per_joint_deg={np.round(delta_deg, 3).tolist()}"
            )
        return {
            "approach_conf": approach_conf,
            "grasp_conf": grasp_conf,
            "joint_delta_deg": delta_deg,
            "max_joint_delta_deg": max_delta_deg,
        }, ""

    def _linear_path_continuity_error(self, path, approach_conf, grasp_conf):
        """Return an error string if a Cartesian-linear path changes IK branch."""
        jv_list = getattr(path, "jv_list", path)
        try:
            joints = np.asarray(jv_list, dtype=float)
        except Exception as exc:
            return f"cannot read linear joint path: {exc}"
        if joints.ndim != 2 or joints.shape[0] < 2 or joints.shape[1] != 6:
            return f"linear path has invalid shape {joints.shape}"
        if not np.all(np.isfinite(joints)):
            return "linear path contains NaN or Inf"

        start_error = float(np.max(self._joint_delta_deg(approach_conf, joints[0])))
        goal_error = float(np.max(self._joint_delta_deg(joints[-1], grasp_conf)))
        if start_error > LINEAR_PATH_ENDPOINT_TOLERANCE_DEG:
            return f"linear path start differs from approach by {start_error:.3f} deg"
        if goal_error > LINEAR_PATH_ENDPOINT_TOLERANCE_DEG:
            return f"linear path goal differs from grasp by {goal_error:.3f} deg"

        max_step_deg = float(np.max(np.abs(np.degrees(np.diff(joints, axis=0)))))
        if max_step_deg > LINEAR_PATH_MAX_JOINT_STEP_DEG:
            return (
                f"linear path joint jump={max_step_deg:.3f} deg > "
                f"{LINEAR_PATH_MAX_JOINT_STEP_DEG:.3f} deg"
            )
        return None

    def _flush_timing_log(self, total_success):
        if not hasattr(self, "_timing_records"):
            return
        task_start = getattr(self, "_timing_task_start", time.time())
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(task_start))
        try:
            log_path = Path("pepper_pick_timing_log.csv")
            needs_header = not log_path.exists() or log_path.stat().st_size == 0
            with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                if needs_header:
                    writer.writerow(
                        [
                            "run_id", "started_at", "path_method", "cycle", "stage",
                            "duration_s", "kind", "status", "detail", "task_success_count",
                        ]
                    )
                for record in self._timing_records:
                    writer.writerow(
                        [
                            getattr(self, "_timing_run_id", ""),
                            ts_str,
                            getattr(self, "_timing_path_method", ""),
                            "" if record["cycle"] is None else record["cycle"],
                            record["stage"],
                            f"{record['duration_s']:.6f}",
                            record["kind"],
                            record["status"],
                            record["detail"],
                            total_success,
                        ]
                    )
        except Exception as e:
            print(f"[警告] 写入分阶段计时日志失败：{e}")

    def _reset_timing_records(self):
        self._timing_records = []
        self._timing_task_start = time.time()
        self._timing_run_id = str(time.time_ns())
        self._timing_cycle = None
        self._timing_path_method = ""

    def _init_hardware(self):
        if not self.real_robot:
            return
        controller_t0 = time.perf_counter()
        try:
            self.rm_controller = RealmanArmController(has_gripper=self.use_gripper)
            print("Realman机械臂连接成功。")
            self._record_timing(
                "init_rm_controller",
                time.perf_counter() - controller_t0,
                "realman controller connected",
                status="success",
            )
        except Exception as e:
            self._record_timing(
                "init_rm_controller", time.perf_counter() - controller_t0, str(e), status="failed"
            )
            print(f"[错误] Realman机械臂连接失败：{e}")
            print("[提示] 已切换到仿真模式。")
            self.real_robot = False
            return

        try:
            _t0 = time.perf_counter()
            self.rs_pipeline = rs.pipeline()
            self.rs_config = rs.config()
            self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.rs_pipeline.start(self.rs_config)
            self.rs_align = rs.align(rs.stream.color)
            print("RealSense 相机已启动。")
            self._record_timing(
                "init_realsense",
                time.perf_counter() - _t0,
                "realsense pipeline started",
                status="success",
            )
        except Exception as e:
            self._record_timing(
                "init_realsense", time.perf_counter() - _t0, str(e), status="failed"
            )
            print(f"[错误] RealSense初始化失败：{e}")
            print("[提示] 已切换到仿真模式。")
            self.real_robot = False
            self.rs_pipeline = None
            self.rs_align = None

    def _yolo_predict_device(self) -> str:
        if hasattr(self, "_yolo_pred_device_resolved") and self._yolo_pred_device_resolved is not None:
            return self._yolo_pred_device_resolved

        forced = (os.environ.get("WRS_YOLO_DEVICE") or "").strip()
        if forced:
            self._yolo_pred_device_resolved = forced
            return forced
        import torch
        if torch.cuda.is_available():
            self._yolo_pred_device_resolved = "cuda:0"
            try:
                name = torch.cuda.get_device_name(0)
                major, minor = torch.cuda.get_device_capability(0)
                print(f"[YOLO] 检测到 GPU: {name} (SM {major}.{minor})，将优先使用 CUDA 推理。")
            except Exception:
                print("[YOLO] 检测到 CUDA 可用，将优先使用 GPU 推理。")
            return self._yolo_pred_device_resolved
        self._yolo_pred_device_resolved = "cpu"
        print("[YOLO] 未检测到可用 CUDA，回退到 CPU。")
        return self._yolo_pred_device_resolved

    def get_Keyresult(self, results):
        """Extract class-0 targets with keypoint and box confidence metadata."""
        candidates = []
        target_class_id = 0
        detection_index = 0
        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            for i in range(len(boxes)):
                class_id = int(boxes[i].cls)
                if class_id != target_class_id:
                    continue
                if keypoints is None or len(keypoints) <= i:
                    continue
                kpts = keypoints[i].data[0].cpu().numpy()
                if len(kpts) == 0:
                    continue
                kpt = kpts[0]
                detection_confidence = float(boxes[i].conf.item())
                min_detection_confidence = getattr(
                    self,
                    "_experiment_min_detection_confidence",
                    None,
                )
                if (
                    min_detection_confidence is not None
                    and detection_confidence < float(min_detection_confidence)
                ):
                    print(
                        "[实验置信度过滤] "
                        f"idx={detection_index}, "
                        f"box_confidence={detection_confidence:.3f} "
                        f"< required={float(min_detection_confidence):.3f}，"
                        "按未识别到辣椒处理。"
                    )
                    detection_index += 1
                    continue
                candidates.append({
                    "pixel_x": float(kpt[0]),
                    "pixel_y": float(kpt[1]),
                    "keypoint_confidence": float(kpt[2]),
                    "detection_confidence": detection_confidence,
                    "confidence_score": float(kpt[2]) + detection_confidence,
                    "detection_index": detection_index,
                })
                detection_index += 1
        return candidates

    def _find_ik_feasible_target(
            self,
            target_pos,
            current_conf,
            approach_dis,
            bound_lower,
            bound_upper,
            grad,
    ):
        """Return the first safe, same-branch approach/grasp IK pair."""
        hand_x = np.array([0, 0, -1])
        hand_z = np.array([-1, 0, 0])
        hand_y = np.cross(hand_z, hand_x)
        tgt_rotmat_base = np.array([hand_x, hand_y, hand_z]).T
        for theta in range(bound_lower, bound_upper + 1, grad):
            tgt_rotmat = rm.rotmat_from_axangle(hand_y, np.radians(theta)) @ tgt_rotmat_base
            ik_pair, _ = self._solve_continuous_approach_grasp_ik(
                current_conf=current_conf,
                target_pos=target_pos,
                target_rotmat=tgt_rotmat,
                hand_z=hand_z,
                approach_dis=approach_dis,
            )
            if ik_pair is None:
                continue
            return {
                "theta": theta,
                **ik_pair,
            }
        return None

    def _select_target_candidate(
            self,
            candidates,
            strategy,
            approach_dis,
            bound_lower,
            bound_upper,
            grad,
    ):
        """Filter targets by IK, then deterministically rank the feasible set."""
        valid_strategies = {
            "nearest_tcp",
            "nearest_depth",
            "highest_confidence",
        }
        if strategy not in valid_strategies:
            raise ValueError(
                f"target_selection_strategy must be one of {sorted(valid_strategies)}, got {strategy!r}"
            )
        if self.real_robot and self.rm_controller is not None:
            current_conf = np.asarray(self.rm_controller.get_joint_values(), dtype=float)
        else:
            current_conf = np.asarray(self.robot.get_jnt_values(), dtype=float)
        tcp_pos, _ = self.robot.fk(current_conf)

        feasible_candidates = []
        for candidate in candidates:
            ik_result = self._find_ik_feasible_target(
                target_pos=candidate["world_pos"],
                current_conf=current_conf,
                approach_dis=approach_dis,
                bound_lower=bound_lower,
                bound_upper=bound_upper,
                grad=grad,
            )
            if ik_result is None:
                print(
                    f"[TARGET][skip] idx={candidate['detection_index']} has no joint-limit-safe IK"
                )
                continue
            ranked = dict(candidate)
            ranked.update(ik_result)
            ranked["tcp_distance"] = float(np.linalg.norm(candidate["world_pos"] - tcp_pos))
            feasible_candidates.append(ranked)

        if not feasible_candidates:
            self._last_selected_target = None
            return None

        stable_tail = lambda item: (
            item["pixel_y"],
            item["pixel_x"],
            item["detection_index"],
        )
        if strategy == "nearest_tcp":
            feasible_candidates.sort(
                key=lambda item: (
                    item["tcp_distance"],
                    -item["detection_confidence"],
                    *stable_tail(item),
                )
            )
        elif strategy == "nearest_depth":
            feasible_candidates.sort(
                key=lambda item: (
                    item["depth"],
                    -item["confidence_score"],
                    *stable_tail(item),
                )
            )
        else:
            feasible_candidates.sort(
                key=lambda item: (
                    -item["confidence_score"],
                    item["tcp_distance"],
                    *stable_tail(item),
                )
            )

        selected = feasible_candidates[0]
        self._last_selected_target = selected
        print(
            f"[TARGET] strategy={strategy}, idx={selected['detection_index']}, "
            f"box_confidence={selected['detection_confidence']:.3f}, "
            f"keypoint_confidence={selected['keypoint_confidence']:.3f}, "
            f"confidence_score={selected['confidence_score']:.3f}, "
            f"tcp_distance={selected['tcp_distance']:.3f}m, theta={selected['theta']}"
        )
        return selected

    def scan_and_detect(
            self,
            target_selection_strategy="nearest_tcp",
            approach_dis=0.15,
            bound_lower=-30,
            bound_upper=30,
            grad=3,
    ):
        """Capture an image, run YOLO, and return the pepper target in world coordinates.

        The detector uses keypoints rather than the bounding-box center so that
        the grasp point is more stable and closer to the actual pepper stem / fruit
        location.
        """
        # The picking loop uses this status to distinguish a truly empty view
        # from a visible target that is temporarily unusable (bad depth / IK).
        self._last_scan_status = "started"
        view_name = getattr(self, "_experiment_view_name", "default_view")
        self._experiment_view_started_at = time.perf_counter()
        self._experiment_recognition_completed_at = None
        if self.real_robot:
            if self.rs_pipeline is None or self.rs_align is None:
                self._record_timing(
                    "experiment_capture", 0.0, f"view={view_name}; camera unavailable", status="failed"
                )
                print("[错误] 相机数据流未初始化。")
                self._last_scan_status = "camera_error"
                return None

            capture_t0 = time.perf_counter()
            try:
                frames = self.rs_pipeline.wait_for_frames()
                aligned = self.rs_align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    self._record_timing(
                        "experiment_capture",
                        time.perf_counter() - capture_t0,
                        f"view={view_name}; invalid frames",
                        status="failed",
                    )
                    print("[错误] 获取深度图或彩色图失败。")
                    self._last_scan_status = "camera_error"
                    return None
                color_img = np.asanyarray(color_frame.get_data())
                height, width, _ = color_img.shape
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                self._record_timing(
                    "experiment_capture",
                    time.perf_counter() - capture_t0,
                    f"view={view_name}",
                    status="success",
                )
                frame_callback = getattr(self, "_experiment_frame_callback", None)
                if callable(frame_callback):
                    try:
                        frame_callback(
                            kind="capture",
                            cycle=getattr(self, "_timing_cycle", None),
                            view_name=view_name,
                            color_image=color_img,
                            depth_image=np.asanyarray(depth_frame.get_data()),
                            intrinsics={
                                "width": int(depth_intrin.width),
                                "height": int(depth_intrin.height),
                                "fx": float(depth_intrin.fx),
                                "fy": float(depth_intrin.fy),
                                "ppx": float(depth_intrin.ppx),
                                "ppy": float(depth_intrin.ppy),
                            },
                        )
                    except Exception as callback_error:
                        print(f"[实验记录][警告] 保存相机帧失败：{callback_error}")
            except Exception as e:
                self._record_timing(
                    "experiment_capture",
                    time.perf_counter() - capture_t0,
                    f"view={view_name}; {e}",
                    status="failed",
                )
                print(f"[错误] 相机取帧失败：{e}")
                self._last_scan_status = "camera_error"
                return None

            print("--- 正在执行YOLO推理 ---")
            if self.yolo_model is None:
                self._record_timing(
                    "experiment_recognition", 0.0, f"view={view_name}; model unavailable", status="skipped"
                )
                print("[警告] YOLO 模型未初始化，跳过真实推理，使用降级检测结果。")
                self._last_scan_status = "selected"
                self._experiment_recognition_completed_at = time.perf_counter()
                return np.array([0.45, -0.1, 0.3])
            device = self._yolo_predict_device()
            print(f"[YOLO] 使用设备: {device}")
            _pred_kw = dict(save=False, show_conf=False, device=device)
            recognition_t0 = time.perf_counter()
            results = self.yolo_model.predict(color_img, **_pred_kw)
            image_candidates = self.get_Keyresult(results)
            self._record_timing(
                "experiment_recognition",
                time.perf_counter() - recognition_t0,
                f"view={view_name}; detections={len(image_candidates)}",
                status="success" if image_candidates else "no_target",
            )
            # Keep the recognition boundary available for diagnostics.  The
            # grasp-to-close timer itself starts later, at execution stage E.
            self._experiment_recognition_completed_at = time.perf_counter()
            if not image_candidates:
                self._last_scan_status = "no_visual_target"
                print("[提示] YOLO未检测到任何辣椒。")
                vis_scale = 1.5
                show_img = cv2.resize(color_img, None, fx=vis_scale, fy=vis_scale)
                safe_imshow('YOLO Detection', show_img, wait_ms=1)
                return None

            world_candidates = []
            given_conf = self.rm_controller.get_joint_values() if self.rm_controller else None
            for candidate in image_candidates:
                if candidate["keypoint_confidence"] < 0.5:
                    continue
                ux, uy = int(candidate["pixel_x"]), int(candidate["pixel_y"])
                if not (0 <= ux < depth_intrin.width and 0 <= uy < depth_intrin.height):
                    print(f"[警告] 坐标超出范围 ({ux}, {uy})")
                    continue
                dis = depth_frame.get_distance(ux, uy)
                if dis <= 0:
                    print(f"[警告] 关键点 ({ux}, {uy}) 深度无效")
                    continue
                camera_xyz = np.asarray(
                    rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis),
                    dtype=float,
                )
                world_point_cloud = self.transform_point_cloud_handeye(
                    handeye_mat=self.HANDEYE,
                    pcd=camera_xyz.reshape(1, 3),
                    given_conf=given_conf,
                )
                converted = dict(candidate)
                converted["world_pos"] = np.asarray(world_point_cloud[0], dtype=float)
                converted["depth"] = float(dis)
                world_candidates.append(converted)

            if not world_candidates:
                self._last_scan_status = "invalid_target_data"
                print("[提示] 已检测到辣椒，但关键点置信度或深度数据无效，将重新检测。")
                vis_scale = 1.5
                show_img = cv2.resize(color_img, None, fx=vis_scale, fy=vis_scale)
                safe_imshow('YOLO Detection', show_img, wait_ms=1)
                return None

            selection_t0 = time.perf_counter()
            selected = self._select_target_candidate(
                candidates=world_candidates,
                strategy=target_selection_strategy,
                approach_dis=approach_dis,
                bound_lower=bound_lower,
                bound_upper=bound_upper,
                grad=grad,
            )
            self._record_timing(
                "cycle_target_selection",
                time.perf_counter() - selection_t0,
                (
                    f"strategy={target_selection_strategy}, candidates={len(world_candidates)}, "
                    f"selected={None if selected is None else selected['detection_index']}, "
                    f"box_confidence={None if selected is None else selected['detection_confidence']}, "
                    f"keypoint_confidence={None if selected is None else selected['keypoint_confidence']}, "
                    f"confidence_score={None if selected is None else selected['confidence_score']}, "
                    f"tcp_distance_m={None if selected is None else selected['tcp_distance']}"
                ),
                kind="nested",
                status="success" if selected is not None else "no_feasible_target",
            )
            if selected is None:
                self._last_scan_status = "no_feasible_target"
                print("[提示] 检测到目标，但没有IK可行目标。")
                safe_imshow('YOLO Detection', results[0].plot(), wait_ms=1)
                return None

            vis_scale = 2.0
            im_array = cv2.resize(results[0].plot(), None, fx=vis_scale, fy=vis_scale)
            ux_vis = int(round(selected["pixel_x"] * vis_scale))
            uy_vis = int(round(selected["pixel_y"] * vis_scale))
            cv2.circle(im_array, (ux_vis, uy_vis), 5, (0, 255, 0), -1)
            cv2.putText(
                im_array,
                f"{target_selection_strategy} conf={selected['detection_confidence']:.2f}",
                (ux_vis + 10, uy_vis - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            safe_imshow('YOLO Detection', im_array, wait_ms=1)
            frame_callback = getattr(self, "_experiment_frame_callback", None)
            if callable(frame_callback):
                try:
                    frame_callback(
                        kind="detection",
                        cycle=getattr(self, "_timing_cycle", None),
                        view_name=view_name,
                        annotated_image=im_array,
                    )
                except Exception as callback_error:
                    print(f"[实验记录][警告] 保存检测图失败：{callback_error}")
            self._last_scan_status = "selected"
            return selected["world_pos"]

        else:
            self._record_timing("experiment_capture", 0.0, f"view={view_name}; simulation", status="skipped")
            self._record_timing("experiment_recognition", 0.0, f"view={view_name}; simulation", status="skipped")
            print("当前为仿真模式，使用模拟检测结果。")
            self._last_scan_status = "selected"
            self._experiment_recognition_completed_at = time.perf_counter()
            return np.array([0.45, -0.1, 0.3])

    def transform_point_cloud_handeye(self, handeye_mat, pcd, given_conf=None):
        """Transform camera-space points into the robot world frame using hand-eye calibration."""
        import wrs.Pepper.yoloutils.yolo_utils as yoloutils
        return yoloutils.transform_point_cloud_handeye(self, handeye_mat, pcd, given_conf=given_conf)

    def vis_path(self, mot_data, colormap):
        """Visualize a planned motion path in the 3D scene."""
        if mot_data:
            n = len(mot_data.mesh_list)
            for i, model in enumerate(mot_data.mesh_list):
                model.rgb = colormap(i / n)
                model.alpha = 0.3
                model.attach_to(self.base)

    def pick_place_fast(
            self,
            approach_dis=0.15,
            bound_lower=-30,
            bound_upper=30,
            grad=3,
            # viewpoint_joint=np.array([-2.9569019, 0.12503539, -0.20418608, 3.02282313, 1.46630347, 0.03455752]),
            # viewpoint_joint=np.array([-2.98704382,  0.18640117, -0.67127106,  2.92538126,  1.67439902, 0.04036946]),
            viewpoint_joint=np.array([-2.05959318, 2.01424953, -2.0444089, 2.07856501, 1.55449498, -0.01267109]),
            # place_joint=np.array([-1.62024151, -0.72099554, -0.65041438, 3.06512984, 1.58201875, -0.12065461]),
            place_joint=np.array([-1.62006694, -0.52918382, -0.5931676, 3.06320996, 1.68782069, 0.01120501]),
            continuous_mode=True,
            stop_on_no_detection=True,
            control_frequency=TOPPRA_SAMPLE_PERIOD_S,
            open_close_retries=3,
            cycle_once: bool = False,
            target_selection_strategy: str = "nearest_tcp",
            path_method: str = "rrt",
            transfer_max_jntvel_deg_s=None,
            approach_max_jntvel_deg_s=None,
            transfer_max_jntacc_deg_s2=None,
            approach_max_jntacc_deg_s2=None,
            simulation_target_pos=np.array([-0.45, -0.1, 0.3]),
            target_distance_m=None,
            place_distance_m=None,
            max_successful_picks=None,
            save_path_record: bool = True,
            path_record_dir=None,
    ):
        """Run a complete pepper picking cycle.

        The method is structured into explicit phases so that debugging,
        logging, and behavior tuning are easier:

        Phase 1
            Move the arm to a safe viewpoint.
        Phase 2
            Detect a target pepper using YOLO keypoints.
        Phase 3
            Search for a feasible grasp pose, then plan approach + grasp motion.
        Phase 4
            Plan a motion from the grasp pose to the place pose.
        Phase 5
            Execute the planned motion and operate the gripper.
        Return
            Move back to the viewpoint for the next cycle.
        """
        self._reset_timing_records()
        self.last_pick_path_record = None
        clock = time.perf_counter
        _task_t0 = clock()
        _task_started_at = self._timing_task_start
        robot = self.robot

        def _get_current_joint_values():
            if self.real_robot and self.rm_controller is not None:
                return self.rm_controller.get_joint_values()
            return robot.get_jnt_values()

        def _confirm_viewpoint_stable_before_capture():
            if not self.real_robot or self.rm_controller is None:
                return True
            return self.rm_controller.wait_until_joint_target(
                viewpoint_joint,
                timeout=8.0,
                poll_dt=0.1,
                tolerance_deg=0.5,
                motion_tolerance_deg=0.05,
                stable_count=3,
            )

        def _as_six_positive(name, value, default):
            array = np.asarray(default if value is None else value, dtype=float)
            if array.ndim == 0:
                array = np.full(6, float(array), dtype=float)
            if array.shape != (6,) or not np.all(np.isfinite(array)) or np.any(array <= 0):
                raise ValueError(f"{name} must be a positive scalar or a finite 6-element array")
            return array

        trajectory_dt = float(control_frequency)
        if path_method != "rrt":
            raise ValueError(
                "pepper_picker.py only supports path_method='rrt'; "
                "use pepper_picker_norrt.py for path_method='norrt'"
            )
        self._timing_path_method = path_method
        valid_target_strategies = {
            "nearest_tcp",
            "nearest_depth",
            "highest_confidence",
        }
        if target_selection_strategy not in valid_target_strategies:
            raise ValueError(
                "target_selection_strategy must be one of "
                f"{sorted(valid_target_strategies)}, got {target_selection_strategy!r}"
            )
        if approach_dis <= 0:
            raise ValueError("approach_dis must be positive")
        if grad <= 0 or bound_lower > bound_upper:
            raise ValueError("IK angle range requires grad > 0 and bound_lower <= bound_upper")
        viewpoint_joint = np.asarray(viewpoint_joint, dtype=float)
        place_joint = np.asarray(place_joint, dtype=float)
        simulation_target_pos = np.asarray(simulation_target_pos, dtype=float)
        if viewpoint_joint.shape != (6,) or place_joint.shape != (6,):
            raise ValueError("viewpoint_joint and place_joint must each contain 6 joint values")
        if simulation_target_pos.shape != (3,):
            raise ValueError("simulation_target_pos must contain 3 Cartesian coordinates")
        if target_distance_m is not None and target_distance_m <= 0:
            raise ValueError("target_distance_m must be positive or None")
        if place_distance_m is not None and place_distance_m <= 0:
            raise ValueError("place_distance_m must be positive or None")
        if max_successful_picks is not None:
            if isinstance(max_successful_picks, bool) or int(max_successful_picks) != max_successful_picks:
                raise ValueError("max_successful_picks must be a positive integer or None")
            max_successful_picks = int(max_successful_picks)
            if max_successful_picks <= 0:
                raise ValueError("max_successful_picks must be a positive integer or None")
        if isinstance(open_close_retries, bool) or int(open_close_retries) != open_close_retries:
            raise ValueError("open_close_retries must be a positive integer")
        open_close_retries = int(open_close_retries)
        if open_close_retries <= 0:
            raise ValueError("open_close_retries must be a positive integer")
        if not np.isclose(trajectory_dt, TOPPRA_SAMPLE_PERIOD_S):
            raise ValueError(
                f"control_frequency must be {TOPPRA_SAMPLE_PERIOD_S} s to match "
                "the current RealMan CAN-FD send period; adjust the speed "
                "profiles instead of this value"
            )

        transfer_vel_deg = _as_six_positive(
            "transfer_max_jntvel_deg_s", transfer_max_jntvel_deg_s, TRANSFER_MAX_JNTVEL_DEG_S
        )
        approach_vel_deg = _as_six_positive(
            "approach_max_jntvel_deg_s", approach_max_jntvel_deg_s, APPROACH_MAX_JNTVEL_DEG_S
        )
        transfer_acc_deg = _as_six_positive(
            "transfer_max_jntacc_deg_s2", transfer_max_jntacc_deg_s2, TRANSFER_MAX_JNTACC_DEG_S2
        )
        approach_acc_deg = _as_six_positive(
            "approach_max_jntacc_deg_s2", approach_max_jntacc_deg_s2, APPROACH_MAX_JNTACC_DEG_S2
        )
        transfer_max_vel = np.deg2rad(transfer_vel_deg)
        transfer_max_acc = np.deg2rad(transfer_acc_deg)
        approach_max_vel = np.deg2rad(approach_vel_deg)
        approach_max_acc = np.deg2rad(approach_acc_deg)

        print("\n[实验条件]")
        print(f"  path_method={path_method}, target_selection={target_selection_strategy}")
        print(f"  transfer_velocity_deg_s={transfer_vel_deg.tolist()}")
        print(f"  approach_velocity_deg_s={approach_vel_deg.tolist()}")
        print(f"  transfer_acceleration_deg_s2={transfer_acc_deg.tolist()}")
        print(f"  approach_acceleration_deg_s2={approach_acc_deg.tolist()}")
        print(f"  approach_dis={approach_dis} m, IK=[{bound_lower}, {bound_upper}], grad={grad} deg")
        print(f"  requested_target_distance_m={target_distance_m}")
        print(f"  initial_viewpoint_joint_rad={viewpoint_joint.tolist()}")
        print(f"  nominal_place_joint_rad={place_joint.tolist()}, place_distance_m={place_distance_m}")
        print(f"  max_successful_picks={max_successful_picks}")
        print(f"  save_path_record={bool(save_path_record)}, path_record_dir={path_record_dir}")

        def _execute_path(path, *, approach=False):
            """Execute a path with the selected, explicitly bounded profile."""
            if self.rm_controller is None:
                raise RuntimeError("RealMan controller is not initialized")
            max_vel = approach_max_vel if approach else transfer_max_vel
            max_acc = approach_max_acc if approach else transfer_max_acc
            return self.rm_controller.move_jntspace_path(
                path,
                max_jntvel=max_vel,
                max_jntacc=max_acc,
                control_frequency=trajectory_dt,
                use_toppra=True,
            )

        def _execute_continuous_segments(*segments):
            """Stream adjacent motion sections continuously with per-section limits.

            Each item is ``(path, is_approach, name)``.  TOPPRA still applies
            the slower approach limits near the fruit, but the controller does
            not wait for an idle state at an internal section junction.
            """
            if self.rm_controller is None:
                raise RuntimeError("RealMan controller is not initialized")
            sequence = []
            previous_end = None
            for segment_index, (path, is_approach, name) in enumerate(segments):
                path_array = np.asarray(path, dtype=float).copy()
                if (
                        path_array.ndim != 2
                        or path_array.shape[0] < 2
                        or path_array.shape[1] != 6
                        or not np.all(np.isfinite(path_array))):
                    raise ValueError(
                        f"连续轨迹段 {name} 必须是有限的(N, 6)数组，N>=2"
                    )
                if previous_end is not None:
                    gap_deg = float(np.degrees(np.max(
                        np.abs(previous_end - path_array[0])
                    )))
                    if gap_deg > CONTINUOUS_PATH_JOIN_SNAP_TOLERANCE_DEG:
                        raise RuntimeError(
                            f"连续轨迹段连接 {segment_index - 1}->{segment_index} "
                            f"存在真实断裂：max_gap={gap_deg:.6f} deg，"
                            f"允许值<={CONTINUOUS_PATH_JOIN_SNAP_TOLERANCE_DEG:.6f} deg"
                        )
                    if gap_deg > 0:
                        path_array[0] = previous_end
                        print(
                            f"[连续轨迹] 已对齐公共端点 "
                            f"{segment_index - 1}->{segment_index}："
                            f"原误差={gap_deg:.6f} deg"
                        )
                sequence.append(
                    {
                        "path": path_array,
                        "max_jntvel": approach_max_vel if is_approach else transfer_max_vel,
                        "max_jntacc": approach_max_acc if is_approach else transfer_max_acc,
                        "name": str(name),
                    }
                )
                previous_end = path_array[-1].copy()
            return self.rm_controller.move_jntspace_path_sequence(
                sequence,
                control_frequency=trajectory_dt,
                wait_at_end=True,
            )

        start_conf = _get_current_joint_values()
        robot.goto_given_conf(start_conf)

        print("--- 阶段1：移动到观察位 ---")
        startup_total_t0 = clock()
        start_conf = _get_current_joint_values()
        mot_data_view = None
        if np.linalg.norm(np.array(start_conf) - np.array(viewpoint_joint)) > 0.03:
            startup_plan_t0 = clock()
            mot_data_view = self.rrtc_planner.plan(
                start_conf=start_conf,
                goal_conf=viewpoint_joint,
                obstacle_list=[self.box_model],
                ext_dist=.1,
                max_time=300,
            )
            self._record_timing(
                "startup_view_path_plan",
                clock() - startup_plan_t0,
                status="success" if mot_data_view is not None else "failed",
            )
            if mot_data_view is not None:
                startup_exec_t0 = clock()
                if self.real_robot and self.rm_controller is not None:
                    _execute_path(mot_data_view.jv_list)
                    # time.sleep(3.0)
                else:
                    robot.goto_given_conf(viewpoint_joint)
                self._record_timing(
                    "startup_view_execute",
                    clock() - startup_exec_t0,
                    "includes configured settling wait",
                    status="success",
                )
                self._record_timing(
                    "startup_view_total",
                    clock() - startup_total_t0,
                    kind="inclusive",
                    status="success",
                )
            else:
                self._record_timing(
                    "startup_view_total",
                    clock() - startup_total_t0,
                    kind="inclusive",
                    status="failed",
                )
                self._record_timing(
                    "task_total",
                    clock() - _task_t0,
                    "startup view planning failed",
                    kind="inclusive",
                    status="failed",
                )
                self._flush_timing_log(total_success=0)
                print("[错误] 规划到观察位失败。")
                return
        else:
            self._record_timing("startup_view_path_plan", 0.0, "already at viewpoint", status="skipped")
            self._record_timing("startup_view_execute", 0.0, "already at viewpoint", status="skipped")
            self._record_timing(
                "startup_view_total",
                clock() - startup_total_t0,
                "already at viewpoint",
                kind="inclusive",
                status="skipped",
            )
            print("当前已经位于观察位。")

        claw_controller = self.claw_controller

        pick_count = 0
        attempt_count = 0
        total_success = 0
        no_target_frames = 0
        single_cycle = bool(cycle_once)
        successful_cycle_durations = []
        cycles_t0 = clock()

        while True:
            attempt_count += 1
            cycle_number = attempt_count
            self._timing_cycle = cycle_number
            cycle_t0 = clock()
            success_before_cycle = total_success
            print(f"\n--- 拍照前确认机械臂停稳（第 {cycle_number} 轮）---")
            settle_t0 = clock()
            try:
                _confirm_viewpoint_stable_before_capture()
                self._record_timing(
                    "cycle_pre_capture_settle",
                    clock() - settle_t0,
                    "at viewpoint; joint motion <= 0.05 deg for 3 samples",
                    status="success" if self.real_robot else "skipped",
                )
                print("[拍照准备] 机械臂已到观察位并停稳，允许拍照。")
            except Exception as e:
                self._record_timing(
                    "cycle_pre_capture_settle",
                    clock() - settle_t0,
                    str(e),
                    status="failed",
                )
                self._record_timing(
                    "cycle_total",
                    clock() - cycle_t0,
                    kind="inclusive",
                    status="pre_capture_settle_failed",
                )
                print(f"[拍照禁止] 无法确认机械臂停稳：{e}")
                if self.real_robot and self.rm_controller is not None:
                    try:
                        stop_result = self.rm_controller.arm.rm_set_arm_slow_stop()
                        print(f"[安全停止] 拍照前停稳确认失败，慢停 result={stop_result}")
                    except Exception as stop_exc:
                        print(f"[安全停止][警告] 慢停调用失败：{stop_exc}")
                break

            if self.real_robot and self.use_gripper and self.claw_controller is not None:
                # 每轮正式识别前重新给定张开目标，避免夹爪在初始化回零或
                # 上一轮动作后停留在闭合位置；只发命令，不读取位置或检测结果。
                self.claw_controller.open_gripper()
                print("[夹爪] 拍照前已发送张开命令（不检测张开结果）。")

            print(f"\n--- 阶段2：检测目标（第 {cycle_number} 轮）---")
            detection_t0 = clock()
            if self.real_robot:
                tgt_pos = self.scan_and_detect(
                    target_selection_strategy=target_selection_strategy,
                    approach_dis=approach_dis,
                    bound_lower=bound_lower,
                    bound_upper=bound_upper,
                    grad=grad,
                )
            else:
                tgt_pos = simulation_target_pos.copy()
                if target_distance_m is not None:
                    current_tcp_pos, _ = robot.fk(_get_current_joint_values())
                    target_direction = tgt_pos - current_tcp_pos
                    direction_norm = np.linalg.norm(target_direction)
                    if direction_norm <= 1e-9:
                        raise ValueError(
                            "simulation_target_pos must differ from the current TCP when "
                            "target_distance_m is set"
                        )
                    tgt_pos = (
                            current_tcp_pos
                            + target_direction / direction_norm * float(target_distance_m)
                    )
                print(f"[仿真] 使用模拟目标：{tgt_pos}")
            self._record_timing(
                "cycle_detection_total",
                clock() - detection_t0,
                "includes cycle_target_selection when real_robot=True",
                kind="inclusive",
                status="success" if tgt_pos is not None else "no_target",
            )

            if tgt_pos is None:
                scan_status = getattr(self, "_last_scan_status", "no_visual_target")
                if single_cycle:
                    print(
                        "[提示] 单轮采摘模式未获得可执行目标，"
                        f"status={scan_status}，结束本轮采摘。"
                    )
                    self._record_timing(
                        "cycle_total", clock() - cycle_t0,
                        kind="inclusive", status=scan_status
                    )
                    break

                if scan_status == "no_visual_target":
                    no_target_frames += 1
                    print(
                        f"[提示] 当前视角连续无辣椒帧数：{no_target_frames}/"
                        f"{config.ARM_NO_TARGET_FRAME_LIMIT}"
                    )
                    if no_target_frames >= config.ARM_NO_TARGET_FRAME_LIMIT:
                        print("[提示] 当前视角内没有可见辣椒，结束本次连续采摘。")
                        self._record_timing(
                            "cycle_total", clock() - cycle_t0,
                            kind="inclusive", status="no_target"
                        )
                        break
                    cycle_status = "no_target_retry"
                else:
                    # A pepper is still visible, or image acquisition failed.
                    # This does not satisfy Pepper04's early-stop condition.
                    no_target_frames = 0
                    cycle_status = f"{scan_status}_retry"
                    print(
                        f"[提示] 尚未累计到要求的成功次数，status={scan_status}；"
                        "将重新拍照并继续尝试。"
                    )

                self._record_timing(
                    "cycle_total",
                    clock() - cycle_t0,
                    "retry without arm motion",
                    kind="inclusive",
                    status=cycle_status,
                )
                pick_count += 1
                continue

            no_target_frames = 0

            target_setup_t0 = clock()
            current_tcp_pos, _ = robot.fk(_get_current_joint_values())
            actual_target_distance_m = float(np.linalg.norm(tgt_pos - current_tcp_pos))
            nominal_place_pos, nominal_place_rotmat = robot.fk(place_joint)
            cycle_place_joint = place_joint.copy()
            if place_distance_m is not None:
                place_direction = nominal_place_pos - tgt_pos
                direction_norm = np.linalg.norm(place_direction)
                if direction_norm <= 1e-9:
                    raise ValueError("nominal place TCP position must differ from the target position")
                requested_place_pos = (
                        tgt_pos + place_direction / direction_norm * float(place_distance_m)
                )
                cycle_place_joint = robot.ik(
                    tgt_pos=requested_place_pos,
                    tgt_rotmat=nominal_place_rotmat,
                    seed_jnt_values=place_joint,
                )
                if cycle_place_joint is None or not self._is_conf_within_joint_margin(cycle_place_joint):
                    raise RuntimeError(
                        f"place_distance_m={place_distance_m} has no joint-limit-safe place IK"
                    )
                cycle_place_joint = np.asarray(cycle_place_joint, dtype=float)
            actual_place_pos, _ = robot.fk(cycle_place_joint)
            actual_place_distance_m = float(np.linalg.norm(actual_place_pos - tgt_pos))
            print(
                f"[实验距离] target_tcp_distance={actual_target_distance_m:.4f} m, "
                f"target_to_place_tcp_distance={actual_place_distance_m:.4f} m, "
                f"requested_target_distance={target_distance_m}"
            )
            self._record_timing(
                "cycle_target_setup",
                clock() - target_setup_t0,
                "distance calculation and optional place-position IK",
                status="success",
            )

            tgt_pos_text = np.array2string(
                np.asarray(tgt_pos, dtype=float),
                precision=6,
                floatmode="fixed",
            )
            print(f"--- 阶段3：IK + RRT 规划采摘，目标位置 {tgt_pos_text} ---")
            grasp_plan_t0 = clock()
            is_success = False
            phase3_search_time = 0.0
            phase3_ik_time = 0.0
            phase3_collision_time = 0.0
            phase3_plan_time = 0.0
            found_theta = None
            final_grasp_conf = None
            final_approach_conf = None
            final_linear_path = None
            final_approach_path = None
            approach_start_conf = np.asarray(_get_current_joint_values(), dtype=float)
            robot.goto_given_conf(approach_start_conf)

            # Search over several hand orientation angles to find a feasible IK solution.
            # The loop adjusts the hand pose around the target so the arm can find
            # a collision-free grasp posture.
            for theta in range(bound_lower, bound_upper + 1, grad):
                _t_search = clock()
                # Define a local gripper frame for the grasp:
                #   - hand_z points toward the target
                #   - hand_x / hand_y form the orthogonal grasp frame
                hand_x = np.array([0, 0, -1])
                hand_z = np.array([-1, 0, 0])
                hand_y = np.cross(hand_z, hand_x)
                tgt_rotmat_base = np.array([hand_x, hand_y, hand_z]).T
                # Sweep the pose around the axis to search for a stable orientation.
                tgt_rotmat = rm.rotmat_from_axangle(hand_y, np.radians(theta)) @ tgt_rotmat_base
                phase3_search_time += clock() - _t_search

                _t_ik = clock()
                ik_pair, ik_reason = self._solve_continuous_approach_grasp_ik(
                    current_conf=approach_start_conf,
                    target_pos=tgt_pos,
                    target_rotmat=tgt_rotmat,
                    hand_z=hand_z,
                    approach_dis=approach_dis,
                )
                phase3_ik_time += clock() - _t_ik
                if ik_pair is None:
                    print(f"[IK][skip] theta={theta}: {ik_reason}")
                    continue
                approach_conf = ik_pair["approach_conf"]
                grasp_conf = ik_pair["grasp_conf"]

                _t_collision = clock()
                robot.goto_given_conf(grasp_conf)
                if robot.is_collided(obstacle_list=[self.box_model]):
                    phase3_collision_time += clock() - _t_collision
                    print(f"[COLLISION][fail] theta={theta} grasp")
                    continue

                robot.goto_given_conf(approach_conf)
                if robot.is_collided(obstacle_list=[self.box_model]):
                    phase3_collision_time += clock() - _t_collision
                    print(f"[COLLISION][fail] theta={theta} approach")
                    continue
                phase3_collision_time += clock() - _t_collision

                _t_plan = clock()
                mot_data_rrt = self.rrtc_planner.plan(
                    start_conf=approach_start_conf,
                    goal_conf=approach_conf,
                    obstacle_list=[self.box_model],
                    ext_dist=.1,
                    max_time=300,
                )
                if mot_data_rrt is None:
                    phase3_plan_time += clock() - _t_plan
                    print(f"[RRT][fail] theta={theta} approach")
                    continue
                if not self._path_within_joint_margin(mot_data_rrt):
                    phase3_plan_time += clock() - _t_plan
                    print(f"[LIMIT][skip] theta={theta} RRT path near joint limit")
                    continue

                mot_data_linear = self.adp_planner.gen_linear_motion(
                    start_conf=approach_conf,
                    goal_conf=grasp_conf,
                    granularity=0.01,
                    obstacle_list=[self.box_model],
                )
                phase3_plan_time += clock() - _t_plan
                if mot_data_linear is None:
                    print(f"[LIN][fail] theta={theta} grasp")
                    continue
                if not self._path_within_joint_margin(mot_data_linear):
                    print(f"[LIMIT][skip] theta={theta} linear path near joint limit")
                    continue
                continuity_error = self._linear_path_continuity_error(
                    mot_data_linear,
                    approach_conf,
                    grasp_conf,
                )
                if continuity_error is not None:
                    print(f"[IK_BRANCH][skip] theta={theta}: {continuity_error}")
                    continue

                print(
                    f"[IK][continuous] theta={theta}, "
                    f"approach_to_grasp_max_joint_delta="
                    f"{ik_pair['max_joint_delta_deg']:.3f} deg, "
                    f"per_joint_deg={np.round(ik_pair['joint_delta_deg'], 3).tolist()}"
                )

                found_theta = theta
                final_grasp_conf = grasp_conf
                final_approach_conf = approach_conf
                final_approach_path = mot_data_rrt
                final_linear_path = mot_data_linear
                is_success = True
                self._record_timing("cycle_grasp_pose_setup", phase3_search_time, f"theta={theta}", status="success")
                self._record_timing("cycle_grasp_ik", phase3_ik_time, f"theta={theta}", status="success")
                self._record_timing(
                    "cycle_grasp_collision_check", phase3_collision_time, f"theta={theta}", status="success"
                )
                self._record_timing("cycle_grasp_path_plan", phase3_plan_time, f"theta={theta}", status="success")
                self._record_timing(
                    "cycle_grasp_plan_total",
                    clock() - grasp_plan_t0,
                    f"theta={theta}",
                    kind="inclusive",
                    status="success",
                )
                visualization_t0 = clock()
                self.vis_path(mot_data_rrt, rm.const.winter_map)
                self.vis_path(mot_data_linear, rm.const.wistia_map)
                self._record_timing(
                    "cycle_visualization", clock() - visualization_t0, status="success"
                )
                print(f"角度 {theta}：IK、碰撞检查、RRT 与直线规划全部通过。")
                break

            if not is_success:
                self._record_timing("cycle_grasp_pose_setup", phase3_search_time, status="failed")
                self._record_timing("cycle_grasp_ik", phase3_ik_time, status="failed")
                self._record_timing("cycle_grasp_collision_check", phase3_collision_time, status="failed")
                self._record_timing("cycle_grasp_path_plan", phase3_plan_time, status="failed")
                self._record_timing(
                    "cycle_grasp_plan_total",
                    clock() - grasp_plan_t0,
                    kind="inclusive",
                    status="failed",
                )
                self._record_timing("cycle_total", clock() - cycle_t0, kind="inclusive", status="grasp_plan_failed")
                print("[失败] 未找到满足约束的直接抓取位姿，将重新拍照继续尝试。")
                continue

            print("--- 阶段4：规划放置路径（RRT）---")
            place_plan_t0 = clock()
            if hasattr(final_approach_path, "jv_list"):
                cur_conf = final_approach_path.jv_list[-1]
            else:
                cur_conf = final_approach_path[-1]
            mot_data_place = self.rrtc_planner.plan(
                start_conf=cur_conf,
                goal_conf=cycle_place_joint,
                obstacle_list=[self.box_model],
                ext_dist=.1,
                max_time=300,
            )
            if mot_data_place is not None and not self._path_within_joint_margin(mot_data_place):
                print("[LIMIT][fail] 放置路径接近关节限位，拒绝执行。")
                mot_data_place = None
            if mot_data_place is not None:
                self._record_timing(
                    "cycle_place_path_plan", clock() - place_plan_t0, status="success"
                )
            else:
                self._record_timing(
                    "cycle_place_path_plan", clock() - place_plan_t0, status="failed"
                )
                print("[错误] 规划到放置位失败。")
                is_success = False

            if is_success and mot_data_place is not None:
                print("--- 阶段5：执行抓取与放置动作 ---")
                pick_place_exec_t0 = clock()
                experiment_grasp_t0 = pick_place_exec_t0
                experiment_pick_motion_t0 = pick_place_exec_t0
                experiment_motion_recorded = False
                experiment_grasp_recorded = False
                experiment_place_recorded = False
                experiment_place_t0 = None
                try:
                    if self.real_robot and self.rm_controller is not None:
                        print("[连续轨迹] 观察位 -> 接近位 -> 抓取位")
                        _execute_continuous_segments(
                            (final_approach_path.jv_list, False, "view_to_approach"),
                            (final_linear_path.jv_list, True, "approach_to_grasp"),
                        )
                        if self.use_gripper and claw_controller is not None:
                            # 闭合动作是两条连续轨迹之间唯一必须保留的停点；
                            # 必须给夹爪留出夹紧果实的时间再撤回。
                            close_result = claw_controller.close_gripper()
                            if close_result != 0:
                                raise RuntimeError(f"夹爪闭合命令失败：{close_result}")
                            time.sleep(0.8)
                        self._record_timing(
                            "experiment_grasp_to_close",
                            clock() - experiment_grasp_t0,
                            (
                                f"view={getattr(self, '_experiment_view_name', 'default_view')}; "
                                "starts_at_execution_stage; ends_after_close_wait"
                            ),
                            kind="inclusive",
                            status="success",
                        )
                        experiment_grasp_recorded = True
                        experiment_place_t0 = clock()

                        print("[连续轨迹] 抓取位 -> 退出接近位 -> 放置位")
                        _execute_continuous_segments(
                            (final_linear_path.jv_list[::-1], True, "grasp_to_approach"),
                            (mot_data_place.jv_list, False, "approach_to_place"),
                        )
                        experiment_pick_motion_dur = clock() - experiment_pick_motion_t0
                        view_name = getattr(self, "_experiment_view_name", "default_view")
                        self._record_timing(
                            "experiment_pick_motion",
                            experiment_pick_motion_dur,
                            f"view={view_name}; ends_on_arrival_at_place",
                            status="success",
                        )
                        view_started_at = getattr(self, "_experiment_view_started_at", None)
                        if view_started_at is not None:
                            self._record_timing(
                                "experiment_view_to_place_total",
                                clock() - view_started_at,
                                f"view={view_name}; excludes_drop_and_return",
                                kind="inclusive",
                                status="success",
                            )
                        experiment_motion_recorded = True
                        if self.use_gripper and claw_controller is not None:
                            open_result = claw_controller.open_gripper()
                            if open_result != 0:
                                raise RuntimeError(f"放置点夹爪张开命令失败：{open_result}")
                            self._record_timing(
                                "experiment_place_to_open_command",
                                clock() - experiment_place_t0,
                                f"view={view_name}; ends_when_open_command_returns",
                                kind="inclusive",
                                status="success",
                            )
                            experiment_place_recorded = True
                            print("[夹爪] 放置点已发送张开命令（不检测张开位置）。")
                        else:
                            self._record_timing(
                                "experiment_place_to_open_command",
                                clock() - experiment_place_t0,
                                f"view={view_name}; gripper_disabled",
                                kind="inclusive",
                                status="success",
                            )
                            experiment_place_recorded = True

                        total_success += 1
                        exec_dur = clock() - pick_place_exec_t0
                        self._record_timing(
                            "cycle_pick_place_execute",
                            exec_dur,
                            "includes gripper actions and configured waits",
                            status="success",
                        )
                        print(f"[成功] 已完成第 {total_success} 个辣椒的抓取与放置，"
                              f"执行耗时：{exec_dur:.1f} 秒")
                    else:
                        robot.goto_given_conf(final_approach_conf)
                        robot.goto_given_conf(final_linear_path.jv_list[-1])
                        self._record_timing(
                            "experiment_grasp_to_close",
                            clock() - experiment_grasp_t0,
                            "simulation; starts_at_execution_stage; ends_at_grasp_pose",
                            kind="inclusive",
                            status="success",
                        )
                        experiment_grasp_recorded = True
                        experiment_place_t0 = clock()
                        robot.goto_given_conf(cycle_place_joint)
                        self._record_timing(
                            "experiment_place_to_open_command",
                            clock() - experiment_place_t0,
                            "simulation; ends_at_place_pose",
                            kind="inclusive",
                            status="success",
                        )
                        experiment_place_recorded = True
                        experiment_pick_motion_dur = clock() - experiment_pick_motion_t0
                        view_name = getattr(self, "_experiment_view_name", "default_view")
                        self._record_timing(
                            "experiment_pick_motion",
                            experiment_pick_motion_dur,
                            f"view={view_name}; simulation; ends_on_arrival_at_place",
                            status="success",
                        )
                        view_started_at = getattr(self, "_experiment_view_started_at", None)
                        if view_started_at is not None:
                            self._record_timing(
                                "experiment_view_to_place_total",
                                clock() - view_started_at,
                                f"view={view_name}; simulation; excludes_drop_and_return",
                                kind="inclusive",
                                status="success",
                            )
                        experiment_motion_recorded = True
                        total_success += 1
                        exec_dur = clock() - pick_place_exec_t0
                        self._record_timing(
                            "cycle_pick_place_execute", exec_dur, "simulation", status="success"
                        )
                        print(f"[成功] 仿真模式完成第 {total_success} 个辣椒的抓取与放置。")
                except Exception as e:
                    if not experiment_grasp_recorded:
                        self._record_timing(
                            "experiment_grasp_to_close",
                            clock() - experiment_grasp_t0,
                            str(e),
                            kind="inclusive",
                            status="failed",
                        )
                    elif not experiment_place_recorded and experiment_place_t0 is not None:
                        self._record_timing(
                            "experiment_place_to_open_command",
                            clock() - experiment_place_t0,
                            str(e),
                            kind="inclusive",
                            status="failed",
                        )
                    if not experiment_motion_recorded:
                        self._record_timing(
                            "experiment_pick_motion",
                            clock() - experiment_pick_motion_t0,
                            str(e),
                            status="failed",
                        )
                    self._record_timing(
                        "cycle_pick_place_execute",
                        clock() - pick_place_exec_t0,
                        str(e),
                        status="failed",
                    )
                    self._record_timing(
                        "cycle_total", clock() - cycle_t0, kind="inclusive", status="execution_failed"
                    )
                    print(f"[错误] 执行阶段失败：{e}")
                    if self.real_robot and self.rm_controller is not None:
                        try:
                            stop_result = self.rm_controller.arm.rm_set_arm_slow_stop()
                            print(f"[安全停止] 执行异常，机械臂慢停 result={stop_result}")
                        except Exception as stop_exc:
                            print(f"[安全停止][警告] 机械臂慢停调用失败：{stop_exc}")
                    break

                print("--- 返回观察位（RRT）---")
                return_total_t0 = clock()
                return_succeeded = False
                mot_return = None
                try:
                    cur_conf = _get_current_joint_values()
                    return_plan_t0 = clock()
                    mot_return = self.rrtc_planner.plan(
                        start_conf=cur_conf,
                        goal_conf=viewpoint_joint,
                        obstacle_list=[self.box_model],
                        ext_dist=.1,
                        max_time=300,
                    )
                    self._record_timing(
                        "cycle_return_path_plan",
                        clock() - return_plan_t0,
                        status="success" if mot_return is not None else "failed",
                    )
                    if mot_return is not None:
                        return_exec_t0 = clock()
                        if self.real_robot and self.rm_controller is not None:
                            robot.goto_given_conf(viewpoint_joint)
                            _execute_path(mot_return.jv_list)
                        else:
                            robot.goto_given_conf(viewpoint_joint)
                        self._record_timing(
                            "cycle_return_execute",
                            clock() - return_exec_t0,
                            (
                                "includes configured settling wait"
                                if self.real_robot and self.rm_controller is not None
                                else "simulation"
                            ),
                            status="success",
                        )
                        return_succeeded = True
                    else:
                        self._record_timing(
                            "cycle_return_execute", 0.0, "return path unavailable", status="skipped"
                        )
                        print("[警告] 返回观察位 RRT 规划失败。")
                except Exception as e:
                    print(f"[警告] 返回观察位失败：{e}")
                self._record_timing(
                    "cycle_return_total",
                    clock() - return_total_t0,
                    kind="inclusive",
                    status="success" if return_succeeded else "failed",
                )
                if return_succeeded and save_path_record:
                    path_save_t0 = clock()
                    try:
                        record_segments = [
                            {
                                "name": "view_to_approach",
                                "planner": "rrt_connect",
                                "motion_profile": "transfer",
                                "path": final_approach_path,
                            },
                            {
                                "name": "approach_to_grasp",
                                "planner": "cartesian_linear",
                                "motion_profile": "approach",
                                "path": final_linear_path,
                            },
                            {
                                "name": "grasp_to_approach",
                                "planner": "cartesian_linear_reverse",
                                "motion_profile": "approach",
                                "path": final_linear_path.jv_list[::-1],
                            },
                            {
                                "name": "approach_to_place",
                                "planner": "rrt_connect",
                                "motion_profile": "transfer",
                                "path": mot_data_place,
                            },
                            {
                                "name": "place_to_view",
                                "planner": "rrt_connect",
                                "motion_profile": "transfer",
                                "path": mot_return,
                            },
                        ]
                        if mot_data_view is not None and self.last_pick_path_record is None:
                            record_segments.insert(
                                0,
                                {
                                    "name": "start_to_view",
                                    "planner": "rrt_connect",
                                    "motion_profile": "transfer",
                                    "path": mot_data_view,
                                },
                            )
                        self.last_pick_path_record = save_pick_path_record(
                            robot=robot,
                            segments=record_segments,
                            cycle_number=cycle_number,
                            run_id=self._timing_run_id,
                            path_method=path_method,
                            target_pos=tgt_pos,
                            approach_distance_m=approach_dis,
                            grasp_angle_deg=found_theta,
                            control_period_s=trajectory_dt,
                            obstacle={
                                "name": "robot_car",
                                "mesh_path": "LinZhi_picker_project/model/car.stl",
                                "position_m": self.box_model.pos,
                                "rotation_matrix": self.box_model.rotmat,
                                "rgb": [0.6, 0.6, 0.6],
                                "alpha": 0.9,
                            },
                            real_robot=self.real_robot,
                            output_dir=path_record_dir,
                            extra_metadata={
                                "target_selection_strategy": target_selection_strategy,
                                "actual_target_distance_m": actual_target_distance_m,
                                "actual_place_distance_m": actual_place_distance_m,
                                "successful_pick_index": total_success,
                            },
                        )
                        self._record_timing(
                            "cycle_path_record_save",
                            clock() - path_save_t0,
                            str(self.last_pick_path_record),
                            status="success",
                        )
                        print(f"[路径保存] 完整采摘路径：{self.last_pick_path_record}")
                    except Exception as e:
                        self._record_timing(
                            "cycle_path_record_save",
                            clock() - path_save_t0,
                            str(e),
                            status="failed",
                        )
                        print(f"[路径保存][警告] 完整采摘路径保存失败：{e}")
                if not return_succeeded:
                    self._record_timing(
                        "cycle_total", clock() - cycle_t0, kind="inclusive", status="return_failed"
                    )
                    break

            cycle_duration = clock() - cycle_t0
            cycle_status = "success" if total_success > success_before_cycle else "place_plan_failed"
            self._record_timing("cycle_total", cycle_duration, kind="inclusive", status=cycle_status)
            if cycle_status == "success":
                successful_cycle_durations.append(cycle_duration)
            pick_count += 1
            if max_successful_picks is not None and total_success >= max_successful_picks:
                print(f"[完成] 已达到连续成功采摘数量：{total_success}")
                break
            if single_cycle:
                break
            if not continuous_mode:
                break

        self._timing_cycle = None
        cycles_duration = clock() - cycles_t0
        _duration = clock() - _task_t0
        average_success_cycle = (
            float(np.mean(successful_cycle_durations)) if successful_cycle_durations else None
        )
        self._record_timing(
            "task_cycles_total",
            cycles_duration,
            f"attempts={attempt_count}, success={total_success}",
            kind="inclusive",
        )
        if average_success_cycle is not None:
            self._record_timing(
                "task_success_cycle_average",
                average_success_cycle,
                f"success={len(successful_cycle_durations)}",
                kind="summary",
            )
        self._record_timing(
            "task_total",
            _duration,
            f"success={total_success}",
            kind="inclusive",
            status="success" if total_success > 0 else "failed",
        )
        _result_str = "success" if total_success > 0 else "fail"
        _ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_task_started_at))
        try:
            summary_path = Path("pepper_pick_task_log.csv")
            needs_header = not summary_path.exists() or summary_path.stat().st_size == 0
            with open(summary_path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                if needs_header:
                    writer.writerow(
                        [
                            "run_id", "started_at", "path_method", "task_duration_s",
                            "result", "attempt_count", "success_count",
                            "average_success_cycle_duration_s",
                        ]
                    )
                writer.writerow(
                    [
                        self._timing_run_id,
                        _ts_str,
                        path_method,
                        f"{_duration:.6f}",
                        _result_str,
                        attempt_count,
                        total_success,
                        "" if average_success_cycle is None else f"{average_success_cycle:.6f}",
                    ]
                )
        except Exception as e:
            print(f"[警告] 写入采摘日志失败：{e}")
        self._flush_timing_log(total_success)
        _result_str_cn = "成功" if total_success > 0 else "失败"
        print(f"[日志] 本次采摘任务结果：{_result_str_cn}，总耗时：{_duration:.3f} 秒，成功数量：{total_success}")
        return {
            "success": total_success > 0,
            "picked": total_success,
            "result": _result_str,
            "duration": _duration,
            "average_success_cycle_duration": average_success_cycle,
            "path_record": (
                None
                if self.last_pick_path_record is None
                else str(self.last_pick_path_record)
            ),
        }


if __name__ == "__main__":
    # ======================== 实验条件配置区 ========================
    REAL_ROBOT = True

    # 路径方法：本文件固定为 RRT；非 RRT 实验请运行 pepper_picker_norrt.py。
    PATH_METHOD = "rrt"

    # 关节速度（deg/s）：可以填写一个标量，也可以填写六关节数组。
    TRANSFER_VELOCITY_DEG_S = np.array( [300, 300, 300, 360, 360, 420], dtype=float)
    APPROACH_VELOCITY_DEG_S = np.array( [150, 150, 150, 180, 180, 210], dtype=float)

    # 关节加速度（deg/s^2）：可以填写一个标量，也可以填写六关节数组。
    TRANSFER_ACCELERATION_DEG_S2 = np.array([300, 300, 300, 360, 360, 420], dtype=float)
    APPROACH_ACCELERATION_DEG_S2 = np.array([150, 150, 150, 180, 180, 210], dtype=float)

    # 仿真时按此方向放置目标；真实机械臂的目标位置由相机测量。
    SIMULATION_TARGET_POSITION_M = np.array([-0.45, -0.1, 0.3], dtype=float)
    # 仿真时会精确设置该TCP距离；真实机械臂时作为实验期望值打印对照。
    TARGET_DISTANCE_M = None  # 例如 0.50；None 表示不指定。

    APPROACH_DISTANCE_M = 0.15
    IK_ANGLE_LOWER_DEG = -30
    IK_ANGLE_UPPER_DEG = 30
    IK_ANGLE_STEP_DEG = 3

    # 初始机械臂/观察位和名义放置位，单位为 rad。
    INITIAL_VIEWPOINT_JOINT_RAD = np.array(
        [-1.85877565, 1.94225470, -1.88045259, -1.17633450, -1.54990476, 3.06841112],
        dtype=float,
    )
    NOMINAL_PLACE_JOINT_RAD = np.array(
        [-1.60015275, 1.78765354, -1.01860905, -3.04907284, -1.97429905, 3.15454307],
        dtype=float,
    )
    PLACE_DISTANCE_M = None  # 例如 0.60；None 表示使用名义放置关节位置。

    # 目标选择策略
    TARGET_SELECTION_STRATEGY = "nearest_tcp"
    # TARGET_SELECTION_STRATEGY = "highest_confidence"

    CONTINUOUS_PICK_COUNT = 10  # 按成功采摘次数停止；None 表示不限定。
    # ================================================================

    picker = RealmanPepperPickerFast(
        real_robot=REAL_ROBOT,
        defer_init=False,
        use_gripper=REAL_ROBOT,
    )
    picker.initialize()
    picker.pick_place_fast(
        approach_dis=APPROACH_DISTANCE_M,
        bound_lower=IK_ANGLE_LOWER_DEG,
        bound_upper=IK_ANGLE_UPPER_DEG,
        grad=IK_ANGLE_STEP_DEG,
        viewpoint_joint=INITIAL_VIEWPOINT_JOINT_RAD,
        place_joint=NOMINAL_PLACE_JOINT_RAD,
        continuous_mode=True,
        stop_on_no_detection=True,
        control_frequency=TOPPRA_SAMPLE_PERIOD_S,
        open_close_retries=3,
        target_selection_strategy=TARGET_SELECTION_STRATEGY,
        path_method=PATH_METHOD,
        transfer_max_jntvel_deg_s=TRANSFER_VELOCITY_DEG_S,
        approach_max_jntvel_deg_s=APPROACH_VELOCITY_DEG_S,
        transfer_max_jntacc_deg_s2=TRANSFER_ACCELERATION_DEG_S2,
        approach_max_jntacc_deg_s2=APPROACH_ACCELERATION_DEG_S2,
        simulation_target_pos=SIMULATION_TARGET_POSITION_M,
        target_distance_m=TARGET_DISTANCE_M,
        place_distance_m=PLACE_DISTANCE_M,
        max_successful_picks=CONTINUOUS_PICK_COUNT,
    )
