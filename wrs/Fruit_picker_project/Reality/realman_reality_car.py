#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the real Realman arm through the pepper-stem observation views from
Simulation/realman_sim_car.py and capture RealSense D405 RGB/depth frames.

Default views:
    01 level_left_30deg
    02 level_left_60deg
    03 level_right_30deg
    04 level_right_60deg
    05 top45_left_30deg
    06 top45_left_60deg
    07 top45_right_30deg
    08 top45_right_60deg
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_THIS_FILE = Path(__file__).resolve()
_WORKSPACE_ROOT = _THIS_FILE.parents[3]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.motion.probabilistic.rrt_connect as rrtc
from wrs.robot_con.realman.realman import RealmanArmController
from wrs.robot_sim.robots.realman.realman_single_arm import Realman


HOME_JNT_VALUES = np.array([0.06897541, 1.55276713, -1.98419496, -0.03705334, 1.97552078, 0.01207768], dtype=np.float64)

PEPPER_BODY_POS = np.array([-0.36, 0.0, 0.22], dtype=np.float64)
PEPPER_BODY_SIZE = np.array([0.035, 0.035, 0.16], dtype=np.float64)
STEM_SIZE = np.array([0.012, 0.012, 0.055], dtype=np.float64)
DEFAULT_STEM_LOOKAT_POS = (
    PEPPER_BODY_POS + np.array([0.0, 0.0, PEPPER_BODY_SIZE[2] / 2.0 + STEM_SIZE[2] / 2.0])
).astype(np.float64)

LEVEL_VIEW_DISTANCE = 0.12
TOP_VIEW_DISTANCE = 0.12
TOP_VIEW_ANGLE_DEG = 45.0
LEVEL_VIEW_ANGLES_DEG = [30, 60, -30, -60]
TOP_VIEW_ANGLES_DEG = [30, 50, -30, -50]
TCP_XY_FLIP_ROLL_DEG = 180.0
ROLL_SEARCH_DEG = [0, 45, -45, 90, -90, 135, -135, 180]
VIEW_DISTANCE_SCALE_SEARCH = [1.0, 0.85, 0.7, 1.15, 0.55, 1.3, 0.45, 1.5, 0.35, 1.8, 2.0]
ANGLE_FALLBACK_STEP_DEG = 5.0
ANGLE_FALLBACK_MIN_ABS_DEG = 30.0

RRT_EXT_DIST = 0.03
RRT_MAX_TIME = 120.0
RRT_SMOOTHING_N_ITER = 10
RRT_COLLISION_GRANULARITY = 0.01

OUTPUT_ROOT = _THIS_FILE.parent / "outputs_real_d405_car"
CAR_STL = _WORKSPACE_ROOT / "wrs" / "LinZhi_picker_project" / "model" / "car.stl"
D405_DEPTH_FAR = 1.0
CAMERA_FRAME_TIMEOUT_MS = 15000
CAMERA_CAPTURE_RETRIES = 3
CAMERA_RETRY_SLEEP = 0.3
DEFAULT_MAX_JNTVEL_DEG = 120.0
DEFAULT_MAX_JNTACC_DEG = 180.0

ROBOT_IP = "192.168.1.18"
ROBOT_PORT = 8080
HAS_GRIPPER = False
USE_CAMERA = True
USE_CAR_OBSTACLE = True
USE_TOPPRA = False
CONTROL_FREQUENCY = 0.3
CANFD_DT = 0.01
MOVE_SETTLE_TIME = 1.0
WAIT_UNTIL_IDLE_AFTER_MOVE = True
IDLE_TIMEOUT = 60.0
CAMERA_WARMUP_FRAMES = 5
CAMERA_DISCARD_FRAMES = 0
CONTINUE_ON_FAIL = True
CONTINUE_ON_CAPTURE_FAIL = True
HOME_JNT_TOLERANCE = 0.03


def safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", name)


def prepare_output_dirs(root: Path):
    for subdir in ("rgb", "depth_raw", "depth_png", "depth_vis", "metadata"):
        (root / subdir).mkdir(parents=True, exist_ok=True)


def intrinsics_to_dict(intrin):
    return {
        "width": int(intrin.width),
        "height": int(intrin.height),
        "ppx": float(intrin.ppx),
        "ppy": float(intrin.ppy),
        "fx": float(intrin.fx),
        "fy": float(intrin.fy),
        "model": str(intrin.model),
        "coeffs": [float(v) for v in intrin.coeffs],
    }


def make_tcp_rotmat_looking_at(tcp_pos: np.ndarray, lookat_pos: np.ndarray):
    tcp_z = rm.unit_vector(lookat_pos - tcp_pos)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    tcp_x = world_up - np.dot(world_up, tcp_z) * tcp_z
    if np.linalg.norm(tcp_x) < 1e-6:
        tcp_x = rm.orthogonal_vector(tcp_z, toggle_unit=True)
    else:
        tcp_x = rm.unit_vector(tcp_x)
    tcp_y = rm.unit_vector(np.cross(tcp_z, tcp_x))
    tcp_rotmat = np.column_stack((tcp_x, tcp_y, tcp_z))
    return tcp_rotmat @ rm.rotmat_from_axangle([0, 0, 1], math.radians(TCP_XY_FLIP_ROLL_DEG))


def roll_tcp_rotmat(tcp_rotmat: np.ndarray, roll_rad: float):
    return tcp_rotmat @ rm.rotmat_from_axangle([0, 0, 1], roll_rad)


def view_angle_name(angle_deg: float):
    side = "left" if angle_deg > 0 else "right"
    abs_angle = abs(float(angle_deg))
    if abs_angle.is_integer():
        angle_label = str(int(abs_angle))
    else:
        angle_label = str(abs_angle).replace(".", "p")
    return f"{side}_{angle_label}deg"


def build_view_targets(
        stem_lookat_pos: np.ndarray,
        level_view_distance: float,
        top_view_distance: float,
        top_view_angle_deg: float,
        level_view_angles_deg: list[float],
        top_view_angles_deg: list[float]):
    targets = []
    front_vec = np.array([level_view_distance, 0.0, 0.0], dtype=np.float64)

    for angle_deg in level_view_angles_deg:
        rot_z = rm.rotmat_from_axangle([0, 0, 1], math.radians(angle_deg))
        tcp_pos = stem_lookat_pos + rot_z @ front_vec
        tcp_rotmat = make_tcp_rotmat_looking_at(tcp_pos, stem_lookat_pos)
        targets.append({
            "name": f"level_{view_angle_name(angle_deg)}",
            "group": "level",
            "angle_deg": float(angle_deg),
            "view_vec": rot_z @ front_vec,
            "tcp_pos": tcp_pos,
            "tcp_rotmat": tcp_rotmat,
            "jnt_values": None,
        })

    top_horizontal_distance = float(top_view_distance)
    top_z_offset = top_horizontal_distance * math.tan(math.radians(top_view_angle_deg))
    top_front_vec = np.array([top_horizontal_distance, 0.0, top_z_offset], dtype=np.float64)

    for angle_deg in top_view_angles_deg:
        rot_z = rm.rotmat_from_axangle([0, 0, 1], math.radians(angle_deg))
        tcp_pos = stem_lookat_pos + rot_z @ top_front_vec
        tcp_rotmat = make_tcp_rotmat_looking_at(tcp_pos, stem_lookat_pos)
        targets.append({
            "name": f"top45_{view_angle_name(angle_deg)}",
            "group": "top45",
            "angle_deg": float(angle_deg),
            "view_vec": rot_z @ top_front_vec,
            "tcp_pos": tcp_pos,
            "tcp_rotmat": tcp_rotmat,
            "jnt_values": None,
        })

    return targets


def make_obstacle_list(enable_car_obstacle: bool):
    if not enable_car_obstacle:
        return []
    if not CAR_STL.is_file():
        print(f"[Warn] car STL not found, planning without car obstacle: {CAR_STL}")
        return []
    robotcar_model = mcm.CollisionModel(
        str(CAR_STL),
        rgb=np.array([0.6, 0.6, 0.6]),
        alpha=0.9,
    )
    robotcar_model.pos = np.array([-0.1, 0.0, -0.08], dtype=np.float64)
    robotcar_model.rotmat = rm.rotmat_from_axangle([0, 1, 0], -math.pi / 2)
    return [robotcar_model]


def validate_path_collision(robot: Realman, path, obstacle_list, granularity: float):
    if path is None or len(path) == 0:
        return False
    for seg_idx in range(len(path) - 1):
        start = np.asarray(path[seg_idx], dtype=np.float64)
        goal = np.asarray(path[seg_idx + 1], dtype=np.float64)
        diff = goal - start
        n_step = max(2, int(np.linalg.norm(diff) / granularity))
        for interp_idx in range(n_step + 1):
            conf = start + diff * (interp_idx / n_step)
            robot.goto_given_conf(conf)
            if robot.is_collided(obstacle_list=obstacle_list):
                print(f"       [Collision] path segment {seg_idx}, interp {interp_idx}/{n_step} hits car STL.")
                return False
    final_conf = np.asarray(path[-1], dtype=np.float64)
    robot.goto_given_conf(final_conf)
    if robot.is_collided(obstacle_list=obstacle_list):
        print("       [Collision] final_conf hits car STL.")
        return False
    return True


def plan_to_conf(
        robot: Realman,
        planner,
        start_conf: np.ndarray,
        goal_conf: np.ndarray,
        obstacle_list,
        granularity: float):
    if np.linalg.norm(start_conf - goal_conf) < 1e-6:
        return [start_conf.copy(), goal_conf.copy()]
    mot_data = planner.plan(
        start_conf=start_conf,
        goal_conf=goal_conf,
        obstacle_list=obstacle_list,
        ext_dist=RRT_EXT_DIST,
        max_time=RRT_MAX_TIME,
        smoothing_n_iter=RRT_SMOOTHING_N_ITER,
    )
    if mot_data is None:
        return None
    path = mot_data.jv_list
    if not validate_path_collision(robot, path, obstacle_list, granularity):
        return None
    return path


def solve_view_ik(
        robot: Realman,
        target: dict,
        seed_jnt_values: np.ndarray,
        stem_lookat_pos: np.ndarray,
        distance_scale_search: list[float],
        roll_search_deg: list[float]):
    if target["jnt_values"] is not None:
        return np.asarray(target["jnt_values"], dtype=np.float64).copy()

    base_view_vec = np.asarray(target["view_vec"], dtype=np.float64)
    for scale in distance_scale_search:
        tcp_pos = stem_lookat_pos + base_view_vec * float(scale)
        base_rotmat = make_tcp_rotmat_looking_at(tcp_pos, stem_lookat_pos)
        for roll_deg in roll_search_deg:
            tcp_rotmat = roll_tcp_rotmat(base_rotmat, math.radians(float(roll_deg)))
            goal_conf = robot.ik(
                tgt_pos=tcp_pos,
                tgt_rotmat=tcp_rotmat,
                seed_jnt_values=seed_jnt_values,
                toggle_dbg=False,
            )
            if goal_conf is not None:
                target["tcp_pos"] = tcp_pos
                target["tcp_rotmat"] = tcp_rotmat
                target["ik_scale"] = float(scale)
                target["ik_roll_deg"] = float(roll_deg)
                target["jnt_values"] = np.asarray(goal_conf, dtype=np.float64).copy()
                return target["jnt_values"].copy()
    return None


class RealSenseCapture:
    def __init__(
            self,
            output_root: Path,
            warmup_frames: int = 15,
            depth_vis_far: float = D405_DEPTH_FAR,
            frame_timeout_ms: int = CAMERA_FRAME_TIMEOUT_MS,
            capture_retries: int = CAMERA_CAPTURE_RETRIES,
            retry_sleep: float = CAMERA_RETRY_SLEEP):
        import pyrealsense2 as rs

        self.rs = rs
        self.output_root = output_root
        self.depth_vis_far = float(depth_vis_far)
        self.frame_timeout_ms = int(frame_timeout_ms)
        self.capture_retries = max(0, int(capture_retries))
        self.retry_sleep = max(0.0, float(retry_sleep))
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        for _ in range(max(0, int(warmup_frames))):
            self._wait_for_frames("warmup", retries=1)
        print("[Setup] RealSense D405 started.")

    def _wait_for_frames(self, label: str, retries: int | None = None):
        retry_count = self.capture_retries if retries is None else max(0, int(retries))
        last_error = None
        for attempt in range(retry_count + 1):
            try:
                return self.pipeline.wait_for_frames(self.frame_timeout_ms)
            except RuntimeError as e:
                last_error = e
                print(
                    f"       [Camera][warn] {label}: wait_for_frames timeout "
                    f"({attempt + 1}/{retry_count + 1}, timeout={self.frame_timeout_ms}ms): {e}"
                )
                if attempt < retry_count and self.retry_sleep > 0:
                    time.sleep(self.retry_sleep)
        raise RuntimeError(f"{label}: RealSense frame did not arrive after {retry_count + 1} attempts") from last_error

    def close(self):
        self.pipeline.stop()
        print("[Cleanup] RealSense D405 stopped.")

    def save_capture(self, cycle_idx: int, idx: int, view_name: str, metadata: dict, discard_frames: int = 3):
        for _ in range(max(0, int(discard_frames))):
            try:
                self._wait_for_frames(f"{view_name}/discard", retries=1)
            except RuntimeError as e:
                print(f"       [Camera][warn] {view_name}: discard frame failed, continuing capture: {e}")
                break
        frames = self._wait_for_frames(view_name)
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError(f"{view_name}: failed to get aligned depth/color frames")

        stem = f"cycle{cycle_idx:02d}_{idx + 1:02d}_{safe_name(view_name)}"
        color_bgr = np.asanyarray(color_frame.get_data())
        depth_z16 = np.asanyarray(depth_frame.get_data())
        depth_m = depth_z16.astype(np.float32) * float(self.depth_scale)

        cv2.imwrite(str(self.output_root / "rgb" / f"{stem}.png"), color_bgr)
        np.save(self.output_root / "depth_raw" / f"{stem}.npy", depth_m)

        depth_mm = np.clip(depth_m * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        cv2.imwrite(str(self.output_root / "depth_png" / f"{stem}.png"), depth_mm)

        valid_depth = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth_vis_src = np.clip(valid_depth, 0.0, self.depth_vis_far)
        depth_vis = cv2.normalize(depth_vis_src, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(self.output_root / "depth_vis" / f"{stem}.png"), depth_vis)

        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        metadata = dict(metadata)
        metadata.update({
            "depth_scale": float(self.depth_scale),
            "depth_intrinsics": intrinsics_to_dict(depth_intrin),
            "color_intrinsics": intrinsics_to_dict(color_intrin),
            "timestamp_ms": float(color_frame.get_timestamp()),
        })
        with open(self.output_root / "metadata" / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"       [Capture] saved RGB/depth: {self.output_root}")


def write_metadata(output_root: Path, cycle_idx: int, idx: int, view_name: str, metadata: dict):
    stem = f"cycle{cycle_idx:02d}_{idx + 1:02d}_{safe_name(view_name)}"
    with open(output_root / "metadata" / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def read_single_key():
    if os.name == "nt":
        import msvcrt

        key = msvcrt.getch()
        try:
            return key.decode("utf-8").lower()
        except UnicodeDecodeError:
            return ""

    if not sys.stdin.isatty():
        line = input("[Idle] Non-TTY input: press Enter/Space+Enter to run, or Q+Enter to quit: ")
        if line.lower().strip() == "q":
            return "q"
        if line == "" or " " in line:
            return " "
        return line[:1].lower()

    import termios
    import tty

    fd = sys.stdin.fileno()
    try:
        old_settings = termios.tcgetattr(fd)
    except termios.error:
        line = input("[Idle] Line input: press Enter/Space+Enter to run, or Q+Enter to quit: ")
        if line.lower().strip() == "q":
            return "q"
        if line == "" or " " in line:
            return " "
        return line[:1].lower()
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
        if key == "\x03":
            raise KeyboardInterrupt
        return key.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class RealityCarRunner:
    def __init__(self):
        prepare_output_dirs(OUTPUT_ROOT)
        self.stem_lookat_pos = np.asarray(DEFAULT_STEM_LOOKAT_POS, dtype=np.float64)
        self.level_view_angles_deg = [float(v) for v in LEVEL_VIEW_ANGLES_DEG]
        self.top_view_angles_deg = [float(v) for v in TOP_VIEW_ANGLES_DEG]
        self.roll_search_deg = [float(v) for v in ROLL_SEARCH_DEG]
        self.distance_scale_search = [float(v) for v in VIEW_DISTANCE_SCALE_SEARCH]

        self.robot_model = Realman(enable_cc=True)
        self.planner = rrtc.RRTConnect(self.robot_model)
        self.obstacle_list = make_obstacle_list(USE_CAR_OBSTACLE)
        self.targets = build_view_targets(
            stem_lookat_pos=self.stem_lookat_pos,
            level_view_distance=LEVEL_VIEW_DISTANCE,
            top_view_distance=TOP_VIEW_DISTANCE,
            top_view_angle_deg=TOP_VIEW_ANGLE_DEG,
            level_view_angles_deg=self.level_view_angles_deg,
            top_view_angles_deg=self.top_view_angles_deg,
        )
        self.controller = RealmanArmController(ip=ROBOT_IP, port=ROBOT_PORT, has_gripper=HAS_GRIPPER)
        self.camera = None
        if USE_CAMERA:
            self.camera = RealSenseCapture(
                OUTPUT_ROOT,
                warmup_frames=CAMERA_WARMUP_FRAMES,
                depth_vis_far=D405_DEPTH_FAR,
                frame_timeout_ms=CAMERA_FRAME_TIMEOUT_MS,
                capture_retries=CAMERA_CAPTURE_RETRIES,
                retry_sleep=CAMERA_RETRY_SLEEP,
            )
        self.cached_cycle_segments = None
        self.cycle_idx = 0
        self._print_setup()

    def _print_setup(self):
        print("\n[Setup] real car observation targets:")
        print(f"  stem_lookat_pos = {np.round(self.stem_lookat_pos, 4)}")
        print(f"  observe/home joints = {np.round(HOME_JNT_VALUES, 6)}")
        print(f"  level_view_angles_deg = {self.level_view_angles_deg}")
        print(f"  top_view_angles_deg = {self.top_view_angles_deg}")
        for idx, target in enumerate(self.targets, start=1):
            print(f"  {idx:02d} {target['name']} tcp={np.round(target['tcp_pos'], 4)}")

    def close(self):
        if self.camera is not None:
            self.camera.close()
        if self.controller is not None:
            self.controller.close_connection()

    def get_current_conf(self):
        return np.asarray(self.controller.get_joint_values(), dtype=np.float64)

    def at_observe_point(self):
        return np.linalg.norm(self.get_current_conf() - HOME_JNT_VALUES) <= HOME_JNT_TOLERANCE

    def execute_path(self, path, name: str):
        path = [np.asarray(conf, dtype=np.float64) for conf in path]
        if len(path) == 0:
            raise RuntimeError(f"{name}: empty path")

        final_conf = path[-1]
        self.robot_model.goto_given_conf(final_conf)
        tcp_pos, tcp_rotmat = self.robot_model.fk(final_conf, update=False)
        print(f"\n[Move] {name}")
        print(f"       path_points = {len(path)}")
        print(f"       tcp_pos = {np.round(tcp_pos, 4)}")
        print(f"       tcp_z_axis = {np.round(tcp_rotmat[:, 2], 4)}")

        if np.linalg.norm(final_conf - self.get_current_conf()) < 1e-6:
            print("       [Skip] already at target.")
            return

        max_jntvel = np.radians([DEFAULT_MAX_JNTVEL_DEG] * 6).tolist()
        max_jntacc = np.radians([DEFAULT_MAX_JNTACC_DEG] * 6).tolist()
        self.controller.move_jntspace_path(
            path,
            max_jntvel=max_jntvel,
            max_jntacc=max_jntacc,
            control_frequency=CONTROL_FREQUENCY,
            use_toppra=USE_TOPPRA,
            canfd_dt=CANFD_DT,
        )
        if WAIT_UNTIL_IDLE_AFTER_MOVE:
            try:
                print("       [Wait] waiting until robot is idle.")
                self.controller.wait_until_idle(timeout=IDLE_TIMEOUT)
            except TimeoutError as e:
                current_after_timeout = self.get_current_conf()
                jnt_err = np.linalg.norm(final_conf - current_after_timeout)
                print(f"       [Warn] wait_until_idle timeout: {e}")
                print(f"       [Warn] joint error to target after timeout = {jnt_err:.6f}")
            except Exception as e:
                print(f"       [Warn] wait_until_idle failed: {e}")
        if MOVE_SETTLE_TIME > 0:
            print(f"       [Wait] settle {MOVE_SETTLE_TIME:.2f}s before capture/next path.")
            time.sleep(MOVE_SETTLE_TIME)

    def plan_from_current_to(self, goal_conf: np.ndarray, name: str):
        start_conf = self.get_current_conf()
        self.robot_model.goto_given_conf(start_conf)
        print(f"\n[RRT] planning {name}")
        print(f"       start_joints(rad) = {np.round(start_conf, 6)}")
        print(f"       goal_joints(rad)  = {np.round(goal_conf, 6)}")
        path = plan_to_conf(
            robot=self.robot_model,
            planner=self.planner,
            start_conf=start_conf,
            goal_conf=np.asarray(goal_conf, dtype=np.float64),
            obstacle_list=self.obstacle_list,
            granularity=RRT_COLLISION_GRANULARITY,
        )
        if path is None:
            raise RuntimeError(f"{name}: [RRT][fail]")
        return [np.asarray(conf, dtype=np.float64) for conf in path]

    def move_to_observe_point(self):
        if self.at_observe_point():
            print("\n[Home] already at observe point.")
            return
        path = self.plan_from_current_to(HOME_JNT_VALUES, "observe_point")
        self.execute_path(path, "observe_point")

    def solve_target_conf(self, target, seed_conf):
        goal_conf = solve_view_ik(
            robot=self.robot_model,
            target=target,
            seed_jnt_values=seed_conf,
            stem_lookat_pos=self.stem_lookat_pos,
            distance_scale_search=self.distance_scale_search,
            roll_search_deg=self.roll_search_deg,
        )
        if goal_conf is None:
            return None
        return np.asarray(goal_conf, dtype=np.float64)

    def capture_view(self, cycle_idx: int, view_idx: int, target: dict, path):
        actual_conf = self.get_current_conf()
        self.robot_model.goto_given_conf(actual_conf)
        actual_tcp_pos, actual_tcp_rotmat = self.robot_model.fk(actual_conf, update=False)
        metadata = {
            "cycle_index": cycle_idx,
            "view_index": view_idx + 1,
            "view_name": target["name"],
            "group": target["group"],
            "angle_deg": target["angle_deg"],
            "stem_lookat_pos": self.stem_lookat_pos.tolist(),
            "target_tcp_pos": target["tcp_pos"].tolist(),
            "target_tcp_rotmat": target["tcp_rotmat"].tolist(),
            "ik_scale": float(target.get("ik_scale", 1.0)),
            "ik_roll_deg": float(target.get("ik_roll_deg", 0.0)),
            "goal_jnt_values_rad": np.asarray(target["jnt_values"], dtype=np.float64).tolist(),
            "actual_jnt_values_rad": actual_conf.tolist(),
            "actual_tcp_pos": actual_tcp_pos.tolist(),
            "actual_tcp_rotmat": actual_tcp_rotmat.tolist(),
            "path_points": len(path),
            "level_view_distance": float(LEVEL_VIEW_DISTANCE),
            "top_view_distance": float(TOP_VIEW_DISTANCE),
            "top_view_angle_deg": float(TOP_VIEW_ANGLE_DEG),
            "level_view_angles_deg": self.level_view_angles_deg,
            "top_view_angles_deg": self.top_view_angles_deg,
        }
        if self.camera is not None:
            try:
                self.camera.save_capture(
                    cycle_idx,
                    view_idx,
                    target["name"],
                    metadata,
                    discard_frames=CAMERA_DISCARD_FRAMES,
                )
                return
            except RuntimeError as e:
                metadata["capture_error"] = str(e)
                print(f"       [Capture][fail] {target['name']}: {e}")
                if not CONTINUE_ON_CAPTURE_FAIL:
                    raise
        write_metadata(OUTPUT_ROOT, cycle_idx, view_idx, target["name"], metadata)

    def build_first_cycle_cache(self, cycle_idx: int):
        print("\n[Cycle] first run: solve IK and plan RRT paths.")
        self.move_to_observe_point()
        segments = []
        for view_idx, target in enumerate(self.targets):
            print(f"\n[View] {view_idx + 1}/{len(self.targets)} {target['name']}")
            seed_conf = self.get_current_conf()
            goal_conf = self.solve_target_conf(target, seed_conf)
            if goal_conf is None:
                msg = "[IK][fail] no solution after roll and distance search"
                print(f"       {msg}")
                segments.append({"kind": "skip", "view_idx": view_idx, "reason": msg})
                if not CONTINUE_ON_FAIL:
                    raise RuntimeError(f"{target['name']}: {msg}")
                continue

            target["jnt_values"] = goal_conf.copy()
            print(f"       target_tcp_pos = {np.round(target['tcp_pos'], 4)}")
            print(f"       ik_scale = {target.get('ik_scale', 1.0):.2f}, ik_roll_deg = {target.get('ik_roll_deg', 0)}")
            try:
                path = self.plan_from_current_to(goal_conf, target["name"])
            except RuntimeError as e:
                print(f"       {e}")
                segments.append({"kind": "skip", "view_idx": view_idx, "reason": str(e)})
                if not CONTINUE_ON_FAIL:
                    raise
                continue

            segments.append({"kind": "view", "view_idx": view_idx, "path": path})
            self.execute_path(path, target["name"])
            self.capture_view(cycle_idx, view_idx, target, path)

        return_home_path = self.plan_from_current_to(HOME_JNT_VALUES, "return_observe_point")
        segments.append({"kind": "home", "path": return_home_path})
        self.execute_path(return_home_path, "return_observe_point")
        self.cached_cycle_segments = segments
        print("\n[Cache] first cycle paths cached. Next cycle will reuse them directly.")

    def run_cached_cycle(self, cycle_idx: int):
        print("\n[Cycle] cached run: reuse first cycle paths, no IK/RRT.")
        if not self.at_observe_point():
            print("[Warn] Current joints are not at observe point; planning back once before using cached cycle paths.")
            self.move_to_observe_point()
        for segment in self.cached_cycle_segments:
            if segment["kind"] == "skip":
                target = self.targets[segment["view_idx"]]
                print(f"\n[Skip] {target['name']} cached skip: {segment['reason']}")
                continue
            if segment["kind"] == "home":
                self.execute_path(segment["path"], "return_observe_point")
                continue
            view_idx = segment["view_idx"]
            target = self.targets[view_idx]
            self.execute_path(segment["path"], target["name"])
            self.capture_view(cycle_idx, view_idx, target, segment["path"])

    def run_cycle(self):
        self.cycle_idx += 1
        cycle_idx = self.cycle_idx
        print("\n" + "=" * 72)
        print(f"[Cycle] start cycle {cycle_idx}")
        if self.cached_cycle_segments is None:
            self.build_first_cycle_cache(cycle_idx)
        else:
            self.run_cached_cycle(cycle_idx)
        print(f"[Cycle] cycle {cycle_idx} finished; robot returned to observe point.")
        print("=" * 72)

    def run_keyboard_loop(self):
        print("\n" + "=" * 72)
        print("  Realman D405 car observe capture")
        print("  Space : run one full observe cycle")
        print("  Enter : run one full observe cycle when stdin is not a TTY")
        print("  Q     : quit")
        print("  First cycle solves IK/RRT and caches paths; later cycles reuse cached paths.")
        print("=" * 72 + "\n")

        while True:
            print("[Idle] Press Space to run a cycle, or Q to quit.")
            key = read_single_key()
            if key == "q":
                print("\n[Exit] user requested quit.")
                return
            if key == " ":
                self.run_cycle()
            else:
                print(f"\n[Idle] ignored key: {repr(key)}")


def main():
    runner = None
    try:
        runner = RealityCarRunner()
        runner.run_keyboard_loop()
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()
