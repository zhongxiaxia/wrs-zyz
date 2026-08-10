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
import json
import math
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
from wrs.LinZhi_picker_project import config
from wrs.Pepper.vision_io import safe_imshow
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

# 林芝采摘专用关节限制，单位：弧度。
# 这里只额外限制第 1 个关节 J1 的活动范围：[-2.06465465, -1.17106357]。
# 说明：用户给出的范围为 -1.17106357 ~ -2.06465465，实际检查时按较小值作为下限。
CUSTOM_JOINT_LIMITS = {

}

# 林芝目标选择：类别 1 的第 1 个关键点。关键点索引从 0 开始，所以第 1 个关键点为 0。
TARGET_CLASS_ID = 1
TARGET_KEYPOINT_INDEX = 1

# 林芝盖子障碍物检测。若模型类别不同，可在 config.py 中覆盖这些值。
LID_OBSTACLE_CLASS_ID = int(getattr(config, "LID_OBSTACLE_CLASS_ID", 0))
LID_OBSTACLE_MIN_CONF = float(getattr(config, "LID_OBSTACLE_MIN_CONF", 0.35))
LID_OBSTACLE_DEPTH_M = float(getattr(config, "LID_OBSTACLE_DEPTH_M", 0.06))
LID_OBSTACLE_MARGIN_M = float(getattr(config, "LID_OBSTACLE_MARGIN_M", 0.01))
LID_OBSTACLE_MIN_DIM_M = float(getattr(config, "LID_OBSTACLE_MIN_DIM_M", 0.01))
LID_OBSTACLE_PIXEL_STRIDE = int(getattr(config, "LID_OBSTACLE_PIXEL_STRIDE", 6))
LID_OBSTACLE_MAX_POINTS = int(getattr(config, "LID_OBSTACLE_MAX_POINTS", 800))
LID_OBSTACLE_MIN_DEPTH_M = float(getattr(config, "LID_OBSTACLE_MIN_DEPTH_M", 0.05))
LID_OBSTACLE_MAX_DEPTH_M = float(getattr(config, "LID_OBSTACLE_MAX_DEPTH_M", 2.0))


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
            if status == 0 and value in (0xD1, 209, 0xF1, 241):
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
    Modbus 配置和夹爪激活。如果你现在已经把夹爪拆掉了，
    那么这部分逻辑需要暂时跳过，否则程序会在初始化阶段等待
    夹爪响应。
    """
    global claw_controller
    if claw_controller is not None:
        return claw_controller
    if arm is None:
        raise RuntimeError("RealMan arm is required to initialize gripper controller")
    claw_controller = RealManGripperController(arm, device_id=CLAW_ID, modbus_port=MODBUS_PORT)
    claw_controller.set_tool_voltage_24v()
    claw_controller.set_modbus_mode()
    claw_controller.activate_gripper()
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
        self.dynamic_obstacle_models = []

    def initialize(self):
        """Build the simulation scene, connect hardware, and load perception models.

        Initialization is intentionally separated from `__init__` so the caller
        can choose to defer expensive hardware startup when needed.
        """
        if self._inited:
            return
        self._reset_timing_records()
        _init_t0 = time.time()

        self.base = wd.World(cam_pos=rm.vec(2, 0, 1), lookat_pos=rm.vec(0, 0, 0.5))
        mgm.gen_frame().attach_to(self.base)
        self._record_timing("init_scene", time.time() - _init_t0, "world/frame attached")
        _init_t0 = time.time()

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
            # str(PEPPER_NBV_DIR / "newcar_m.stl"),
            r"/home/nvidia/wrs_main/wrs-virtual/wrs/LinZhi_picker_project/model/car.stl",
            rgb=np.array([0.6, 0.6, 0.6]),
            alpha=0.9,
        )
        # self.box_model.pos = np.array([0.16, 0.0, 0.0])
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
        self._record_timing("init_robot_planner", time.time() - _init_t0, "robot/adp ready")
        self._record_timing("init_total", 0.0, "initialization finished")
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
        model = yoloutils.init_yolo(str(WRS_ROOT / "LinZhi_picker_project" / "Weight" / "best.pt"))
        # model = yoloutils.init_yolo(str(WRS_ROOT / "Fruit_picker_project" / "Weight" / "PepperDet-best.pt"))
        if model is None:
            print("[警告] YOLO 初始化失败，将使用仿真/降级模式。")
        return model

    def _record_timing(self, stage, duration, detail=""):
        if not hasattr(self, "_timing_records"):
            self._timing_records = []
        duration = float(duration)
        self._timing_records.append((stage, duration, detail))
        if detail:
            print(f"[计时] {stage}: {duration:.3f} 秒 | {detail}")
        else:
            print(f"[计时] {stage}: {duration:.3f} 秒")

    def _joint_limit_range(self):
        robot = self.robot
        if robot is None:
            return None
        ranges = None
        try:
            candidate = np.asarray(robot.jnt_ranges, dtype=float)
            if candidate.ndim == 2 and candidate.shape[1] == 2:
                ranges = candidate
        except Exception:
            pass
        if ranges is None:
            try:
                candidate = np.asarray(robot.manipulator.jnt_ranges, dtype=float)
                if candidate.ndim == 2 and candidate.shape[1] == 2:
                    ranges = candidate
            except Exception:
                pass
        if ranges is None:
            return None

        ranges = ranges.copy()
        for joint_idx, (custom_low, custom_high) in CUSTOM_JOINT_LIMITS.items():
            if joint_idx >= ranges.shape[0]:
                continue
            low = min(float(custom_low), float(custom_high))
            high = max(float(custom_low), float(custom_high))
            ranges[joint_idx, 0] = max(ranges[joint_idx, 0], low)
            ranges[joint_idx, 1] = min(ranges[joint_idx, 1], high)
        return ranges

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

    def _flush_timing_log(self, total_success):
        if not hasattr(self, "_timing_records"):
            return
        task_start = getattr(self, "_timing_task_start", time.time())
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(task_start))
        try:
            with open("pepper_pick_timing_log.txt", "a", encoding="utf-8") as f:
                f.write(f"==== {ts_str} | picked={total_success} ====\n")
                for stage, duration, detail in self._timing_records:
                    if detail:
                        f.write(f"{stage},{duration:.6f},{detail}\n")
                    else:
                        f.write(f"{stage},{duration:.6f}\n")
        except Exception as e:
            print(f"[警告] 写入分阶段计时日志失败：{e}")

    def _reset_timing_records(self):
        self._timing_records = []
        self._timing_task_start = time.time()

    def _init_hardware(self):
        if not self.real_robot:
            return
        try:
            self.rm_controller = RealmanArmController(has_gripper=self.use_gripper)
            print("Realman机械臂连接成功。")
            self._record_timing("init_rm_controller", 0.0, "realman controller connected")
        except Exception as e:
            print(f"[错误] Realman机械臂连接失败：{e}")
            print("[提示] 已切换到仿真模式。")
            self.real_robot = False
            return

        try:
            _t0 = time.time()
            self.rs_pipeline = rs.pipeline()
            self.rs_config = rs.config()
            self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.rs_pipeline.start(self.rs_config)
            self.rs_align = rs.align(rs.stream.color)
            print("RealSense 相机已启动。")
            self._record_timing("init_realsense", time.time() - _t0, "realsense pipeline started")
        except Exception as e:
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

    def _clear_dynamic_obstacles(self):
        for obstacle in getattr(self, "dynamic_obstacle_models", []):
            try:
                obstacle.detach()
            except Exception:
                pass
        self.dynamic_obstacle_models = []

    def _planning_obstacles(self):
        obstacles = []
        if self.box_model is not None:
            obstacles.append(self.box_model)
        obstacles.extend(getattr(self, "dynamic_obstacle_models", []))
        return obstacles

    def _camera_axes_in_world(self, given_conf=None):
        camera_basis = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)
        world_basis = self.transform_point_cloud_handeye(
            handeye_mat=self.HANDEYE,
            pcd=camera_basis,
            given_conf=given_conf,
        )
        origin = world_basis[0]
        x_axis = rm.unit_vector(world_basis[1] - origin)
        y_axis = rm.unit_vector(world_basis[2] - origin)
        z_axis = rm.unit_vector(world_basis[3] - origin)
        return origin, x_axis, y_axis, z_axis

    def _sample_bbox_point_cloud_camera(self, depth_frame, depth_intrin, box_xyxy):
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        x1 = max(0, int(math.floor(x1)))
        y1 = max(0, int(math.floor(y1)))
        x2 = min(int(depth_intrin.width) - 1, int(math.ceil(x2)))
        y2 = min(int(depth_intrin.height) - 1, int(math.ceil(y2)))
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 3), dtype=float)

        stride = max(1, int(LID_OBSTACLE_PIXEL_STRIDE))
        pixels = [(x, y) for y in range(y1, y2 + 1, stride) for x in range(x1, x2 + 1, stride)]
        if len(pixels) > LID_OBSTACLE_MAX_POINTS:
            pick_ids = np.linspace(0, len(pixels) - 1, LID_OBSTACLE_MAX_POINTS, dtype=int)
            pixels = [pixels[i] for i in pick_ids]

        points = []
        for x, y in pixels:
            depth = float(depth_frame.get_distance(int(x), int(y)))
            if depth < LID_OBSTACLE_MIN_DEPTH_M or depth > LID_OBSTACLE_MAX_DEPTH_M:
                continue
            points.append(rs.rs2_deproject_pixel_to_point(depth_intrin, (int(x), int(y)), depth))
        if not points:
            return np.empty((0, 3), dtype=float)
        return np.asarray(points, dtype=float)

    def _make_lid_obstacle_model(self, target_info, depth_frame, depth_intrin, given_conf, ee_pos):
        camera_points = self._sample_bbox_point_cloud_camera(depth_frame, depth_intrin, target_info["box_xyxy"])
        if camera_points.shape[0] == 0:
            print("[Obstacle] 盖子框内没有有效深度点，跳过该障碍物。")
            return None

        world_points = self.transform_point_cloud_handeye(
            handeye_mat=self.HANDEYE,
            pcd=camera_points,
            given_conf=given_conf,
        )
        if ee_pos is None:
            if given_conf is None:
                given_conf = self.robot.get_jnt_values()
            ee_pos, _ = self.robot.fk(given_conf, update=False)
        ee_pos = np.asarray(ee_pos, dtype=float)
        dists = np.linalg.norm(world_points - ee_pos.reshape(1, 3), axis=1)
        nearest_idx = int(np.argmin(dists))
        nearest_distance = float(dists[nearest_idx])

        _, cam_x_axis, cam_y_axis, cam_z_axis = self._camera_axes_in_world(given_conf=given_conf)
        x1, y1, x2, y2 = target_info["box_xyxy"]
        center_u = 0.5 * (float(x1) + float(x2))
        center_v = 0.5 * (float(y1) + float(y2))
        depth_ref = float(np.median(camera_points[:, 2]))
        face_center_camera = np.asarray(
            rs.rs2_deproject_pixel_to_point(depth_intrin, (center_u, center_v), depth_ref),
            dtype=float,
        )
        face_center_world = self.transform_point_cloud_handeye(
            handeye_mat=self.HANDEYE,
            pcd=face_center_camera.reshape(1, 3),
            given_conf=given_conf,
        )[0]

        depth_axis = face_center_world - ee_pos
        if np.linalg.norm(depth_axis) < 1e-6:
            depth_axis = world_points[nearest_idx] - ee_pos
        if np.linalg.norm(depth_axis) < 1e-6:
            depth_axis = cam_z_axis
        depth_axis = rm.unit_vector(depth_axis)

        x_axis = cam_x_axis - np.dot(cam_x_axis, depth_axis) * depth_axis
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = cam_y_axis - np.dot(cam_y_axis, depth_axis) * depth_axis
        x_axis = rm.unit_vector(x_axis)
        y_axis = rm.unit_vector(np.cross(depth_axis, x_axis))
        if np.dot(y_axis, cam_y_axis) < 0:
            x_axis = -x_axis
            y_axis = -y_axis
        x_axis = rm.unit_vector(np.cross(y_axis, depth_axis))
        rotmat = np.column_stack((x_axis, y_axis, depth_axis))

        box_width_m = max(
            abs(float(target_info["box_width"])) * depth_ref / max(float(depth_intrin.fx), 1e-6),
            LID_OBSTACLE_MIN_DIM_M,
        ) + 2.0 * LID_OBSTACLE_MARGIN_M
        box_height_m = max(
            abs(float(target_info["box_height"])) * depth_ref / max(float(depth_intrin.fy), 1e-6),
            LID_OBSTACLE_MIN_DIM_M,
        ) + 2.0 * LID_OBSTACLE_MARGIN_M
        box_depth_m = max(LID_OBSTACLE_DEPTH_M, LID_OBSTACLE_MIN_DIM_M) + 2.0 * LID_OBSTACLE_MARGIN_M

        # Keep the obstacle face closest to the TCP at the same distance as the closest point in the bbox point cloud.
        nearest_face_center = ee_pos + depth_axis * nearest_distance
        box_center = nearest_face_center + depth_axis * (box_depth_m * 0.5)
        obstacle = mcm.gen_box(
            xyz_lengths=np.array([box_width_m, box_height_m, box_depth_m], dtype=float),
            pos=box_center,
            rotmat=rotmat,
            rgb=np.array([1.0, 0.25, 0.05]),
            alpha=0.35,
        )
        obstacle.obstacle_meta = {
            "class_id": target_info["class_id"],
            "box_conf": target_info["box_conf"],
            "box_xyxy": target_info["box_xyxy"],
            "nearest_distance": nearest_distance,
            "xyz_lengths": [box_width_m, box_height_m, box_depth_m],
            "sample_points": int(camera_points.shape[0]),
        }
        print(
            "[Obstacle] 盖子障碍物: "
            f"class={target_info['class_id']}, conf={target_info['box_conf']:.3f}, "
            f"size=({box_width_m:.3f}, {box_height_m:.3f}, {box_depth_m:.3f}) m, "
            f"nearest_face_dist={nearest_distance:.3f} m, samples={camera_points.shape[0]}"
        )
        return obstacle

    def _update_lid_obstacles_from_yolo(self, results, depth_frame, depth_intrin, given_conf, ee_pos):
        self._clear_dynamic_obstacles()
        if results is None:
            return

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                class_id = int(boxes[i].cls)
                if class_id != LID_OBSTACLE_CLASS_ID:
                    continue
                box_conf = float(boxes[i].conf[0]) if hasattr(boxes[i], "conf") and boxes[i].conf is not None else 0.0
                if box_conf < LID_OBSTACLE_MIN_CONF:
                    continue
                box_xyxy = boxes[i].xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = [float(v) for v in box_xyxy]
                target_info = {
                    "box_xyxy": [x1, y1, x2, y2],
                    "box_width": x2 - x1,
                    "box_height": y2 - y1,
                    "box_conf": box_conf,
                    "class_id": class_id,
                }
                obstacle = self._make_lid_obstacle_model(
                    target_info=target_info,
                    depth_frame=depth_frame,
                    depth_intrin=depth_intrin,
                    given_conf=given_conf,
                    ee_pos=ee_pos,
                )
                if obstacle is None:
                    continue
                self.dynamic_obstacle_models.append(obstacle)
                if self.base is not None:
                    obstacle.attach_to(self.base)

        if self.dynamic_obstacle_models:
            print(f"[Obstacle] 已添加 {len(self.dynamic_obstacle_models)} 个盖子动态障碍物到 RRT。")
        else:
            print("[Obstacle] 未检测到可用盖子动态障碍物。")

    def get_Keyresult(self, results):
        """从 YOLO 结果中提取林芝采摘目标点：类别 1 的第 1 个关键点。"""
        keypoints_array = []
        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            if boxes is None or keypoints is None:
                continue
            for i in range(len(boxes)):
                class_id = int(boxes[i].cls)
                if class_id != TARGET_CLASS_ID:
                    continue
                if len(keypoints) <= i:
                    continue
                kpts = keypoints[i].data[0].cpu().numpy()
                if TARGET_KEYPOINT_INDEX >= len(kpts):
                    continue
                kpt = kpts[TARGET_KEYPOINT_INDEX]
                box_xyxy = boxes[i].xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = [float(v) for v in box_xyxy]
                box_width = x2 - x1
                box_height = y2 - y1
                box_conf = float(boxes[i].conf[0]) if hasattr(boxes[i], "conf") and boxes[i].conf is not None else 0.0
                keypoints_array.append({
                    "keypoint": [kpt[0], kpt[1], kpt[2]],
                    "box_xyxy": [x1, y1, x2, y2],
                    "box_width": box_width,
                    "box_height": box_height,
                    "box_conf": box_conf,
                    "class_id": class_id,
                })
                print(
                    f"[YOLO][box] class={class_id}, conf={box_conf:.3f}, "
                    f"xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), "
                    f"width={box_width:.1f}px, height={box_height:.1f}px"
                )
        return keypoints_array

    def scan_and_detect(self):
        """Capture an image, run YOLO, and return the pepper target in world coordinates.

        The detector uses keypoints rather than the bounding-box center so that
        the grasp point is more stable and closer to the actual pepper stem / fruit
        location.
        """
        if self.real_robot:
            if self.rs_pipeline is None or self.rs_align is None:
                print("[错误] 相机数据流未初始化。")
                return None

            try:
                frames = self.rs_pipeline.wait_for_frames()
                aligned = self.rs_align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    print("[错误] 获取深度图或彩色图失败。")
                    return None
                color_img = np.asanyarray(color_frame.get_data())
                height, width, _ = color_img.shape
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            except Exception as e:
                print(f"[错误] 相机取帧失败：{e}")
                return None

            print("--- 正在执行YOLO推理 ---")
            if self.yolo_model is None:
                print("[警告] YOLO 模型未初始化，跳过真实推理，使用降级检测结果。")
                self._clear_dynamic_obstacles()
                return np.array([0.45, -0.1, 0.3])
            device = self._yolo_predict_device()
            print(f"[YOLO] 使用设备: {device}")
            _pred_kw = dict(save=False, show_conf=False, device=device)
            results = self.yolo_model.predict(color_img, **_pred_kw)
            given_conf = self.rm_controller.get_joint_values() if self.rm_controller else None
            try:
                ee_pos = self.rm_controller.get_pose()[0] if self.rm_controller else None
            except Exception as e:
                print(f"[警告] 获取当前末端位置失败，将退化为选择第一个有效目标：{e}")
                ee_pos = None
            self._update_lid_obstacles_from_yolo(
                results=results,
                depth_frame=depth_frame,
                depth_intrin=depth_intrin,
                given_conf=given_conf,
                ee_pos=ee_pos,
            )
            keypoints_array = self.get_Keyresult(results)

            if len(keypoints_array) > 0:
                candidate_targets = []
                for target_info in keypoints_array:
                    keypoint = target_info["keypoint"]
                    min_box_height_px = float(getattr(config, "TARGET_BOX_MIN_HEIGHT_PX", 0))
                    if target_info["box_height"] < min_box_height_px:
                        print(
                            f"[YOLO] 跳过目标框高度不足：height={target_info['box_height']:.1f}px "
                            f"< min_height={min_box_height_px:.1f}px"
                        )
                        continue
                    if round(keypoint[2]) != 1:
                        print(f"[YOLO] 跳过不可见关键点：{keypoint}")
                        continue

                    ux, uy = int(keypoint[0]), int(keypoint[1])
                    if not (0 <= ux < depth_intrin.width and 0 <= uy < depth_intrin.height):
                        print(f"[警告] 坐标超出范围 ({ux}, {uy})")
                        continue

                    dis = depth_frame.get_distance(ux, uy)
                    print(f"候选采摘点（像素）：{ux}, {uy}, 深度：{dis}")
                    if dis <= 0:
                        print(f"[警告] 关键点 ({ux}, {uy}) 深度无效")
                        continue

                    camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis)
                    camera_xyz = np.array(camera_xyz)
                    camera_point_cloud = camera_xyz.reshape(1, 3)
                    world_point_cloud = self.transform_point_cloud_handeye(
                        handeye_mat=self.HANDEYE,
                        pcd=camera_point_cloud,
                        given_conf=given_conf,
                    )
                    world_pos = world_point_cloud[0]
                    ee_distance = float(np.linalg.norm(world_pos - ee_pos)) if ee_pos is not None else float("inf")
                    candidate_targets.append({
                        "pixel": (ux, uy),
                        "depth": dis,
                        "world_pos": world_pos,
                        "ee_distance": ee_distance,
                        "box_xyxy": target_info["box_xyxy"],
                        "box_width": target_info["box_width"],
                        "box_height": target_info["box_height"],
                        "box_conf": target_info["box_conf"],
                    })
                    if ee_pos is not None:
                        print(f"候选目标世界坐标：{world_pos}，距当前末端：{ee_distance:.4f} m")
                    else:
                        print(f"候选目标世界坐标：{world_pos}")

                if not candidate_targets:
                    print("[警告] 未找到有效的可采摘关键点")
                    if results and len(results) > 0:
                        im_array = results[0].plot()
                        safe_imshow('YOLO Detection', im_array, wait_ms=1)
                    return None

                if ee_pos is not None:
                    selected_target = min(candidate_targets, key=lambda item: item["ee_distance"])
                    print(f"[目标选择] 已选择距离当前末端最近的目标，距离：{selected_target['ee_distance']:.4f} m")
                else:
                    selected_target = candidate_targets[0]
                    print("[目标选择] 当前末端位置不可用，已选择第一个有效目标。")

                target_world_pos = selected_target["world_pos"]
                ux, uy = selected_target["pixel"]
                dis = selected_target["depth"]
                print(f"最终目标世界坐标：{target_world_pos}")

                vis_scale = 2.0
                im_array = results[0].plot()
                im_array = cv2.resize(im_array, None, fx=vis_scale, fy=vis_scale)
                ux_vis = int(round(ux * vis_scale))
                uy_vis = int(round(uy * vis_scale))
                cv2.circle(im_array, (ux_vis, uy_vis), 5, (0, 255, 0), -1)
                cv2.putText(im_array, f"Depth: {dis:.3f}m", (ux_vis + 10, uy_vis - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if ee_pos is not None:
                    cv2.putText(im_array, f"EE Dist: {selected_target['ee_distance']:.3f}m", (ux_vis + 10, uy_vis + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                safe_imshow('YOLO Detection', im_array, wait_ms=1)
                return target_world_pos

            else:
                print("[提示] YOLO未检测到目标。")
                vis_scale = 1.5
                show_img = cv2.resize(color_img, None, fx=vis_scale, fy=vis_scale)
                safe_imshow('YOLO Detection', show_img, wait_ms=1)
                return None

        else:
            print("当前为仿真模式，使用模拟检测结果。")
            self._clear_dynamic_obstacles()
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
        approach_dis=0.08,
        bound_lower=-180,
        bound_upper=180,
        grad=5,
        # viewpoint_joint=np.array([-2.9569019, 0.12503539, -0.20418608, 3.02282313, 1.46630347, 0.03455752]),
        # viewpoint_joint=np.array([-2.98704382,  0.18640117, -0.67127106,  2.92538126,  1.67439902, 0.04036946]),
        viewpoint_joint=np.array([-1.65961608, -1.25787623,  1.45818762,  0.0508589 ,  1.28105425,-3.03720446]),
        continuous_mode=True,
        stop_on_no_detection=True,
        control_frequency=1,
        open_close_retries=3,
        cycle_once: bool = False,
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
            Execute cutting at the target, retreat along the approach path,
            open the gripper/cutter, and return to the viewpoint.
        """
        _task_start_time = time.time()
        self._reset_timing_records()
        robot = self.robot

        def _get_current_joint_values():
            if self.real_robot and self.rm_controller is not None:
                return self.rm_controller.get_joint_values()
            return robot.get_jnt_values()

        start_conf = _get_current_joint_values()
        robot.goto_given_conf(start_conf)

        print("--- 阶段1：移动到观察位 ---")
        _t0 = time.time()
        start_conf = _get_current_joint_values()
        if np.linalg.norm(np.array(start_conf) - np.array(viewpoint_joint)) > 0.03:
            mot_data_view = self.rrtc_planner.plan(
                start_conf=start_conf,
                goal_conf=viewpoint_joint,
                obstacle_list=self._planning_obstacles(),
                ext_dist=.1,
                max_time=300,
            )
            if mot_data_view is not None:
                if self.real_robot and self.rm_controller is not None:
                    self.rm_controller.move_jntspace_path(mot_data_view.jv_list, control_frequency=control_frequency)
                    time.sleep(3.0)
                else:
                    robot.goto_given_conf(viewpoint_joint)
                self._record_timing("phase1_move_view", time.time() - _t0, "moved to viewpoint")
            else:
                self._record_timing("phase1_move_view", time.time() - _t0, "failed")
                print("[错误] 规划到观察位失败。")
                return
        else:
            self._record_timing("phase1_move_view", time.time() - _t0, "already at viewpoint")
            print("当前已经位于观察位。")

        claw_controller = self.claw_controller

        pick_count = 0
        total_success = 0
        no_target_frames = 0
        single_cycle = bool(cycle_once)

        while True:
            _pick_cycle_start = time.time()
            print(f"\n--- 阶段2：检测目标（第 {pick_count + 1} 轮）---")
            _t0 = time.time()
            if self.real_robot:
                tgt_pos = self.scan_and_detect()
            else:
                tgt_pos = np.array([-0.45, -0.1, 0.3])
                print(f"[仿真] 使用模拟目标：{tgt_pos}")
            self._record_timing("phase2_detection", time.time() - _t0, "target detection finished")

            if tgt_pos is None:
                if single_cycle:
                    print("[提示] 单轮采摘模式下未检测到目标，直接结束当前层本轮采摘。")
                    break
                no_target_frames += 1
                print(f"[提示] 当前视角连续无目标帧数：{no_target_frames}/{config.ARM_NO_TARGET_FRAME_LIMIT}")
                if no_target_frames >= config.ARM_NO_TARGET_FRAME_LIMIT:
                    print("[提示] 当前升降台位置视角内无果实，结束该层采摘，返回升降机层。")
                    break
                time.sleep(0.3)
                pick_count += 1
                continue

            no_target_frames = 0

            print(f"--- 阶段3：IK + RRT 规划采摘，目标位置 {tgt_pos} ---")
            _t0 = time.time()
            is_success = False
            phase3_search_time = 0.0
            phase3_ik_time = 0.0
            phase3_plan_time = 0.0
            found_theta = None
            final_grasp_conf = None
            final_approach_conf = None
            final_linear_path = None
            final_approach_path = None

            # Search over several hand orientation angles to find a feasible IK solution.
            # The loop adjusts the hand pose around the target so the arm can find
            # a collision-free grasp posture.
            for theta in range(bound_lower, bound_upper + 1, grad):
                _t_search = time.time()
                # Define a local gripper frame for the grasp:
                #   - hand_z points toward the target
                #   - hand_x / hand_y form the orthogonal grasp frame
                hand_x = np.array([0, 0, -1])
                # Use the same successful approach direction tested in realman_single_arm2.py:
                # hand_z points along world Y+, so the hand TCP approaches the target from Y-.
                hand_z = np.array([0, 1, 0])
                hand_y = np.cross(hand_z, hand_x)
                hand_y = hand_y / np.linalg.norm(hand_y)
                hand_x = np.cross(hand_y, hand_z)
                hand_x = hand_x / np.linalg.norm(hand_x)
                tgt_rotmat_base = np.array([hand_x, hand_y, hand_z]).T
                # Sweep only around the gripper's own Z axis. This keeps the final
                # gripper Z axis aligned with world Y+ while searching wrist roll.
                tgt_rotmat = rm.rotmat_from_axangle(hand_z, np.radians(theta)) @ tgt_rotmat_base
                phase3_search_time += time.time() - _t_search

                _t_ik = time.time()
                grasp_seed = _get_current_joint_values()
                grasp_conf = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=grasp_seed)
                if grasp_conf is None:
                    phase3_ik_time += time.time() - _t_ik
                    print(f"[IK][fail] theta={theta} grasp")
                    continue
                if not self._is_conf_within_joint_margin(grasp_conf):
                    phase3_ik_time += time.time() - _t_ik
                    print(f"[LIMIT][skip] theta={theta} grasp near joint limit")
                    continue

                approach_pos = tgt_pos - hand_z * approach_dis
                approach_conf = robot.ik(tgt_pos=approach_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=grasp_conf)
                if approach_conf is None:
                    phase3_ik_time += time.time() - _t_ik
                    print(f"[IK][fail] theta={theta} approach")
                    continue
                if not self._is_conf_within_joint_margin(approach_conf):
                    phase3_ik_time += time.time() - _t_ik
                    print(f"[LIMIT][skip] theta={theta} approach near joint limit")
                    continue

                robot.goto_given_conf(grasp_conf)
                if robot.is_collided(obstacle_list=self._planning_obstacles()):
                    phase3_ik_time += time.time() - _t_ik
                    print(f"[COLLISION][fail] theta={theta} grasp")
                    continue

                robot.goto_given_conf(approach_conf)
                if robot.is_collided(obstacle_list=self._planning_obstacles()):
                    phase3_ik_time += time.time() - _t_ik
                    print(f"[COLLISION][fail] theta={theta} approach")
                    continue

                _t_plan = time.time()
                mot_data_rrt = self.rrtc_planner.plan(
                    start_conf=start_conf,
                    goal_conf=approach_conf,
                    obstacle_list=self._planning_obstacles(),
                    ext_dist=.1,
                    max_time=300,
                )
                if mot_data_rrt is None:
                    phase3_plan_time += time.time() - _t_plan
                    print(f"[RRT][fail] theta={theta} approach")
                    continue
                if not self._path_within_joint_margin(mot_data_rrt):
                    phase3_plan_time += time.time() - _t_plan
                    print(f"[LIMIT][skip] theta={theta} RRT path near joint limit")
                    continue

                mot_data_linear = self.adp_planner.gen_linear_motion(
                    start_conf=approach_conf,
                    goal_conf=grasp_conf,
                    granularity=0.01,
                    obstacle_list=self._planning_obstacles(),
                )
                phase3_plan_time += time.time() - _t_plan
                if mot_data_linear is None:
                    print(f"[LIN][fail] theta={theta} grasp")
                    continue
                if not self._path_within_joint_margin(mot_data_linear):
                    print(f"[LIMIT][skip] theta={theta} linear path near joint limit")
                    continue

                found_theta = theta
                final_grasp_conf = grasp_conf
                final_approach_conf = approach_conf
                final_approach_path = mot_data_rrt
                final_linear_path = mot_data_linear
                is_success = True
                self.vis_path(mot_data_rrt, rm.const.winter_map)
                self.vis_path(mot_data_linear, rm.const.wistia_map)
                self._record_timing("phase3_search_pose", phase3_search_time, f"theta={theta}")
                self._record_timing("phase3_inverse_ik", phase3_ik_time, f"theta={theta}, success")
                self._record_timing("phase3_path_plan", phase3_plan_time, f"theta={theta}, success")
                self._record_timing("phase3_total", time.time() - _t0, f"theta={theta}, success")
                print(f"角度 {theta}：IK、碰撞检查、RRT 与直线规划全部通过。")
                break

            if not is_success:
                self._record_timing("phase3_search_pose", phase3_ik_time, "failed")
                self._record_timing("phase3_inverse_ik", phase3_ik_time, "failed")
                self._record_timing("phase3_path_plan", 0.0, "failed")
                self._record_timing("phase3_total", time.time() - _t0, "failed")
                print("[失败] 未找到满足约束的直接抓取位姿。")
                if stop_on_no_detection:
                    break
                continue

            print("--- 阶段4：执行剪切、后退并返回观察位 ---")
            _t0 = time.time()
            grasp_start_time = time.time()
            try:
                if self.real_robot and self.rm_controller is not None:
                    # 1) 运动到预剪切位
                    self.rm_controller.move_jntspace_path(final_approach_path.jv_list, control_frequency=control_frequency)
                    time.sleep(1.5)

                    # 2) 靠近目标点
                    self.rm_controller.move_jntspace_path(final_linear_path.jv_list, control_frequency=control_frequency)
                    time.sleep(0.8)

                    # 3) 闭合夹爪/剪刀完成剪切
                    if self.use_gripper and claw_controller is not None:
                        claw_controller.close_gripper()
                        time.sleep(0.8)

                    # 4) 沿原直线路径后退
                    self.rm_controller.move_jntspace_path(final_linear_path.jv_list[::-1], control_frequency=control_frequency)
                    time.sleep(0.8)

                    # 5) 后退后张开夹爪/剪刀
                    if self.use_gripper and claw_controller is not None:
                        claw_controller.open_gripper()
                        time.sleep(0.8)

                    total_success += 1
                    exec_dur = time.time() - _t0
                    self._record_timing("phase4_cut_retreat", exec_dur, f"picked={total_success}")
                    pick_to_return_time = time.time() - _pick_cycle_start
                    self._record_timing("pick_to_return", pick_to_return_time, f"picked={total_success}, before_return")
                    print(f"[成功] 已完成第 {total_success} 个灵芝的剪切与后退，"
                          f"耗时：{time.time() - grasp_start_time:.1f} 秒")
                else:
                    robot.goto_given_conf(final_approach_conf)
                    robot.goto_given_conf(final_linear_path.jv_list[-1])
                    robot.goto_given_conf(final_approach_conf)
                    total_success += 1
                    exec_dur = time.time() - _t0
                    self._record_timing("phase4_cut_retreat", exec_dur, f"picked={total_success}")
                    pick_to_return_time = time.time() - _pick_cycle_start
                    self._record_timing("pick_to_return", pick_to_return_time, f"picked={total_success}, before_return")
                    print(f"[成功] 仿真模式完成第 {total_success} 个灵芝的剪切与后退。")
            except Exception as e:
                print(f"[错误] 执行阶段失败：{e}")
                continue

            print("--- 返回观察位（RRT）---")
            try:
                if self.real_robot and self.rm_controller is not None:
                    cur_conf = _get_current_joint_values()
                    mot_return = self.rrtc_planner.plan(
                        start_conf=cur_conf,
                        goal_conf=viewpoint_joint,
                        obstacle_list=self._planning_obstacles(),
                        ext_dist=.1,
                        max_time=300,
                    )
                    if mot_return is not None:
                        robot.goto_given_conf(viewpoint_joint)
                        self.rm_controller.move_jntspace_path(mot_return.jv_list, control_frequency=0.8)
                        time.sleep(3.0)
                    else:
                        print("[警告] 返回观察位 RRT 规划失败。")
                else:
                    robot.goto_given_conf(viewpoint_joint)
                    time.sleep(1.0)
            except Exception as e:
                print(f"[警告] 返回观察位失败：{e}")

            pick_count += 1
            if single_cycle:
                break
            if not continuous_mode:
                break

        _task_end_time = time.time()
        _duration = _task_end_time - _task_start_time
        self._record_timing("task_total", _duration, f"success={total_success}")
        _result_str = "success" if total_success > 0 else "fail"
        _ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_task_start_time))
        log_line = f"{_ts_str},{_duration:.3f},{_result_str},picked={total_success}\n"
        try:
            with open("pepper_pick_log.txt", "a", encoding="utf-8") as f:
                f.write(log_line)
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
        }



if __name__ == "__main__":
    picker = RealmanPepperPickerFast(real_robot=True, defer_init=False, use_gripper=False)
    picker.initialize()
    picker.pick_place_fast(
        approach_dis=0.08,
        bound_lower=-180,
        bound_upper=180,
        grad=5,
        continuous_mode=True,
        stop_on_no_detection=True,
        control_frequency=1,
        open_close_retries=3,
    )
