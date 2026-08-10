#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Move the real Realman arm through user-provided joint poses and capture D405 frames.

Fill VIEW_JNT_VALUES with 10 joint configurations in radians. The script does not solve IK:
it plans an RRT path between the current real joint values and each target joint pose, checks
the car collision model, moves the real robot, waits until it is idle, settles for one second,
and captures RGB/depth images. Press Enter repeatedly to alternate between forward
view_01 -> view_10 and reverse view_10 -> view_01 capture cycles using cached paths.
"""

from __future__ import annotations

import json
import math
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


# 在这里填写你的 10 个机械臂关节位姿，单位是弧度。
# 可以继续按 ("view_02", np.array([...], dtype=np.float64)) 的格式往下加。
VIEW_JNT_VALUES = [
    ("view_01", np.array([ 0.18264871 , 1.96112173, -2.19009154 ,-0.16084954 , 1.84809425 ,-0.00324631],
                         dtype=np.float64)),
    ("view_02", np.array([-0.62289054 , 1.96108685 ,-2.14439875 , 0.90024086 , 1.72269238  ,0.17723818],
                         dtype=np.float64)),
    ("view_03", np.array([ 0.8689645 ,  1.9610694,  -1.98632429 ,-1.0563256,   1.65036585,  0.02391101],
                         dtype=np.float64)),
    ("view_04", np.array([ 0.92357588 , 1.39326149 ,-0.28902651 ,-1.38404615 , 1.54231249 , 1.15418619],
                         dtype=np.float64)),
    ("view_05", np.array([-0.83580328 , 1.39378506 ,-0.28824112 ,-1.70155634, -1.53372552 , 1.94708927],
                         dtype=np.float64)),
    ("view_06", np.array([ 0.05323254,  0.80477133, -0.98885122,  0.06571165 , 2.18864291 , 0.03558726],
                         dtype=np.float64)),
    ("view_07", np.array([-0.61859706 , 0.80492839, -0.55311229 , 0.90165453,  1.87940543 ,-0.03028146],
                         dtype=np.float64)),
    ("view_08", np.array([ 0.94139569 , 1.00759601, -0.9129468,  -1.08655474,  1.91317751,  0.17751743],
                         dtype=np.float64)),
    ("view_09", np.array([ 0.98560489 , 1.07880548 ,-0.108926 ,  -1.31128326 , 1.59713339 , 1.05292222],
                         dtype=np.float64)),
    ("view_10", np.array([-0.84732245 , 1.0796258 , -0.10744247, -1.85977048 ,-1.6392132,   2.17523882],
                         dtype=np.float64)),
]
EXPECTED_VIEW_COUNT = 10

ROBOT_IP = "192.168.1.18"
ROBOT_PORT = 8080
HAS_GRIPPER = False

USE_CAMERA = True
USE_CAR_OBSTACLE = True
USE_TOPPRA = False
CONTROL_FREQUENCY = 0.3
CANFD_DT = 0.01
WAIT_UNTIL_IDLE_AFTER_MOVE = True
IDLE_TIMEOUT = 60.0
MOVE_SETTLE_TIME = 1.0

RRT_EXT_DIST = 0.03
RRT_MAX_TIME = 120.0
RRT_SMOOTHING_N_ITER = 10
RRT_COLLISION_GRANULARITY = 0.01

DEFAULT_MAX_JNTVEL_DEG = 120.0
DEFAULT_MAX_JNTACC_DEG = 180.0

OUTPUT_ROOT = _THIS_FILE.parent / "outputs_real_joint_capture"
CAR_STL = _WORKSPACE_ROOT / "wrs" / "LinZhi_picker_project" / "model" / "car.stl"
D405_DEPTH_FAR = 1.0
CAMERA_WARMUP_FRAMES = 5
CAMERA_DISCARD_FRAMES = 0
CAMERA_FRAME_TIMEOUT_MS = 15000
CAMERA_CAPTURE_RETRIES = 3
CAMERA_RETRY_SLEEP = 0.3
CONTINUE_ON_FAIL = True
CONTINUE_ON_CAPTURE_FAIL = True
PATH_START_TOLERANCE = 0.05


def safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", name)


def prepare_output_dirs(root: Path):
    for subdir in ("rgb", "depth_raw", "depth_png", "depth_vis", "metadata"):
        (root / subdir).mkdir(parents=True, exist_ok=True)


def find_last_capture_cycle(root: Path) -> int:
    if not root.exists():
        return 0
    pattern = re.compile(r"^cycle(\d+)_\d+_.+\.(?:png|npy|json)$")
    last_cycle = 0
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        match = pattern.match(file_path.name)
        if match:
            last_cycle = max(last_cycle, int(match.group(1)))
    return last_cycle


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


def normalize_view_joints():
    views = []
    for idx, item in enumerate(VIEW_JNT_VALUES):
        if isinstance(item, tuple) and len(item) == 2:
            name, conf = item
        else:
            name, conf = f"view_{idx + 1:02d}", item
        conf = np.asarray(conf, dtype=np.float64)
        if conf.shape != (6,):
            raise ValueError(f"{name}: expected a 6-DOF joint array, got shape {conf.shape}")
        views.append((str(name), conf.copy()))
    if len(views) == 0:
        raise ValueError("VIEW_JNT_VALUES is empty. Fill it with your joint poses first.")
    if len(views) != EXPECTED_VIEW_COUNT:
        print(f"[Warn] VIEW_JNT_VALUES has {len(views)} poses; expected {EXPECTED_VIEW_COUNT}.")
    return views


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
                print(f"       [Collision] segment {seg_idx}, interp {interp_idx}/{n_step} hits car STL.")
                return False
    final_conf = np.asarray(path[-1], dtype=np.float64)
    robot.goto_given_conf(final_conf)
    if robot.is_collided(obstacle_list=obstacle_list):
        print("       [Collision] final_conf hits car STL.")
        return False
    return True


def plan_to_conf(robot: Realman, planner, start_conf: np.ndarray, goal_conf: np.ndarray, obstacle_list):
    start_conf = np.asarray(start_conf, dtype=np.float64)
    goal_conf = np.asarray(goal_conf, dtype=np.float64)
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
    if not validate_path_collision(robot, path, obstacle_list, RRT_COLLISION_GRANULARITY):
        return None
    return [np.asarray(conf, dtype=np.float64) for conf in path]


class RealSenseCapture:
    def __init__(
            self,
            output_root: Path,
            warmup_frames: int = CAMERA_WARMUP_FRAMES,
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
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
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

    def save_capture(self, cycle_idx: int, idx: int, pose_name: str, metadata: dict,
                     discard_frames: int = CAMERA_DISCARD_FRAMES):
        for _ in range(max(0, int(discard_frames))):
            try:
                self._wait_for_frames(f"{pose_name}/discard", retries=1)
            except RuntimeError as e:
                print(f"       [Camera][warn] {pose_name}: discard frame failed, continuing capture: {e}")
                break
        frames = self._wait_for_frames(pose_name)
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError(f"{pose_name}: failed to get aligned depth/color frames")

        stem = f"cycle{cycle_idx:02d}_{idx + 1:02d}_{safe_name(pose_name)}"
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


def write_metadata(output_root: Path, cycle_idx: int, idx: int, pose_name: str, metadata: dict):
    stem = f"cycle{cycle_idx:02d}_{idx + 1:02d}_{safe_name(pose_name)}"
    with open(output_root / "metadata" / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


class JointPoseCaptureRunner:
    def __init__(self):
        prepare_output_dirs(OUTPUT_ROOT)
        self.views = normalize_view_joints()
        self.robot_model = Realman(enable_cc=True)
        self.planner = rrtc.RRTConnect(self.robot_model)
        self.obstacle_list = make_obstacle_list(USE_CAR_OBSTACLE)
        self.controller = RealmanArmController(ip=ROBOT_IP, port=ROBOT_PORT, has_gripper=HAS_GRIPPER)
        self.camera = RealSenseCapture(OUTPUT_ROOT) if USE_CAMERA else None
        self.cycle_idx = find_last_capture_cycle(OUTPUT_ROOT)
        self.cached_cycle_paths = None
        self.next_direction = "forward"
        print(
            f"[Setup] existing captures up to cycle{self.cycle_idx:02d}; "
            f"next saved cycle will be cycle{self.cycle_idx + 1:02d}."
        )
        print("[Setup] joint pose capture targets:")
        for idx, (name, conf) in enumerate(self.views, start=1):
            print(f"  {idx:02d} {name}: {np.round(conf, 6)}")

    def close(self):
        if self.camera is not None:
            self.camera.close()
        if self.controller is not None:
            self.controller.close_connection()

    def get_current_conf(self):
        return np.asarray(self.controller.get_joint_values(), dtype=np.float64)

    def execute_path(self, path, pose_name: str, settle_after: bool = True):
        path = [np.asarray(conf, dtype=np.float64) for conf in path]
        final_conf = path[-1]
        self.robot_model.goto_given_conf(final_conf)
        tcp_pos, tcp_rotmat = self.robot_model.fk(final_conf, update=False)
        print(f"       path_points = {len(path)}")
        print(f"       target_tcp_pos = {np.round(tcp_pos, 4)}")
        print(f"       target_tcp_z = {np.round(tcp_rotmat[:, 2], 4)}")

        current_conf = self.get_current_conf()
        path_start_err = np.linalg.norm(path[0] - current_conf)
        if path_start_err > PATH_START_TOLERANCE:
            raise RuntimeError(
                f"{pose_name}: current joints are not near cached path start "
                f"(err={path_start_err:.4f}, tol={PATH_START_TOLERANCE:.4f})"
            )

        if np.linalg.norm(final_conf - current_conf) < 1e-6:
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
            print("       [Wait] waiting until robot is idle.")
            try:
                self.controller.wait_until_idle(timeout=IDLE_TIMEOUT)
            except TimeoutError as e:
                current_after_timeout = self.get_current_conf()
                jnt_err = np.linalg.norm(final_conf - current_after_timeout)
                print(f"       [Warn] wait_until_idle timeout: {e}")
                print(f"       [Warn] joint error to target after timeout = {jnt_err:.6f}")
            except Exception as e:
                print(f"       [Warn] wait_until_idle failed: {e}")
        if settle_after and MOVE_SETTLE_TIME > 0:
            print(f"       [Wait] settle {MOVE_SETTLE_TIME:.2f}s before capture.")
            time.sleep(MOVE_SETTLE_TIME)

    def plan_current_to(self, pose_name: str, goal_conf: np.ndarray):
        start_conf = self.get_current_conf()
        self.robot_model.goto_given_conf(start_conf)
        print(f"       start_joints(rad) = {np.round(start_conf, 6)}")
        print(f"       goal_joints(rad)  = {np.round(goal_conf, 6)}")
        return plan_to_conf(
            robot=self.robot_model,
            planner=self.planner,
            start_conf=start_conf,
            goal_conf=goal_conf,
            obstacle_list=self.obstacle_list,
        )

    def plan_between_known_confs(self, pose_name: str, start_conf: np.ndarray, goal_conf: np.ndarray):
        self.robot_model.goto_given_conf(np.asarray(start_conf, dtype=np.float64))
        return plan_to_conf(
            robot=self.robot_model,
            planner=self.planner,
            start_conf=np.asarray(start_conf, dtype=np.float64),
            goal_conf=np.asarray(goal_conf, dtype=np.float64),
            obstacle_list=self.obstacle_list,
        )

    def assert_current_near_view(self, view_idx: int):
        view_name, target_conf = self.views[view_idx]
        current_conf = self.get_current_conf()
        err = np.linalg.norm(current_conf - target_conf)
        if err > PATH_START_TOLERANCE:
            raise RuntimeError(
                f"expected to start at {view_name}, but current joints are too far "
                f"(err={err:.4f}, tol={PATH_START_TOLERANCE:.4f})"
            )
        print(f"       [Start] already at {view_name}, joint_err={err:.6f}")
        return current_conf

    def capture_pose(self, cycle_idx: int, idx: int, pose_name: str, target_conf: np.ndarray, path):
        actual_conf = self.get_current_conf()
        self.robot_model.goto_given_conf(actual_conf)
        actual_tcp_pos, actual_tcp_rotmat = self.robot_model.fk(actual_conf, update=False)
        target_tcp_pos, target_tcp_rotmat = self.robot_model.fk(target_conf, update=False)
        metadata = {
            "cycle_index": cycle_idx,
            "pose_index": idx + 1,
            "pose_name": pose_name,
            "target_jnt_values_rad": np.asarray(target_conf, dtype=np.float64).tolist(),
            "actual_jnt_values_rad": actual_conf.tolist(),
            "target_tcp_pos": target_tcp_pos.tolist(),
            "target_tcp_rotmat": target_tcp_rotmat.tolist(),
            "actual_tcp_pos": actual_tcp_pos.tolist(),
            "actual_tcp_rotmat": actual_tcp_rotmat.tolist(),
            "path_points": len(path),
        }
        if self.camera is not None:
            try:
                self.camera.save_capture(cycle_idx, idx, pose_name, metadata, discard_frames=CAMERA_DISCARD_FRAMES)
                return
            except RuntimeError as e:
                metadata["capture_error"] = str(e)
                print(f"       [Capture][fail] {pose_name}: {e}")
                if not CONTINUE_ON_CAPTURE_FAIL:
                    raise
        write_metadata(OUTPUT_ROOT, cycle_idx, idx, pose_name, metadata)

    def run_first_forward_cycle(self, cycle_idx: int):
        print("\n[Cycle] first forward run: plan RRT paths and cache them.")
        repeat_paths = [None] * len(self.views)
        for idx, (pose_name, goal_conf) in enumerate(self.views):
            print(f"\n[Pose] {idx + 1}/{len(self.views)} {pose_name}")
            path = self.plan_current_to(pose_name, goal_conf)
            if path is None:
                msg = "[RRT][fail] no collision-free path"
                print(f"       {msg}")
                if CONTINUE_ON_FAIL:
                    continue
                raise RuntimeError(f"{pose_name}: {msg}")
            if idx > 0:
                repeat_paths[idx] = path
            self.execute_path(path, pose_name)
            self.capture_pose(cycle_idx, idx, pose_name, goal_conf, path)

        self.cached_cycle_paths = repeat_paths
        self.next_direction = "reverse"
        print("\n[Cache] forward paths cached. The arm is stopped at the last view.")

    def run_cached_forward_cycle(self, cycle_idx: int):
        print("\n[Cycle] cached forward run: reuse RRT paths, no replanning.")
        for idx, (pose_name, goal_conf) in enumerate(self.views):
            print(f"\n[Pose] {idx + 1}/{len(self.views)} {pose_name}")
            if idx == 0:
                current_conf = self.assert_current_near_view(0)
                print("       [SkipMove] already at first view; capture before moving forward.")
                self.capture_pose(cycle_idx, idx, pose_name, goal_conf, [current_conf])
                continue
            path = self.cached_cycle_paths[idx]
            if path is None:
                raise RuntimeError(f"{pose_name}: missing cached forward path")
            self.execute_path(path, pose_name)
            self.capture_pose(cycle_idx, idx, pose_name, goal_conf, path)
        self.next_direction = "reverse"

    def run_cached_reverse_cycle(self, cycle_idx: int):
        print("\n[Cycle] cached reverse run: reuse reversed RRT paths, no replanning.")
        last_idx = len(self.views) - 1
        current_conf = self.assert_current_near_view(last_idx)
        pose_name, goal_conf = self.views[last_idx]
        print(f"\n[Pose] 1/{len(self.views)} {pose_name}")
        print("       [SkipMove] already at last view; capture before moving backward.")
        self.capture_pose(cycle_idx, 0, pose_name, goal_conf, [current_conf])

        capture_idx = 1
        for view_idx in range(last_idx - 1, -1, -1):
            pose_name, goal_conf = self.views[view_idx]
            path = self.cached_cycle_paths[view_idx + 1]
            if path is None:
                from_name = self.views[view_idx + 1][0]
                raise RuntimeError(f"{pose_name}: missing cached forward path for {pose_name} -> {from_name}")
            reverse_path = [np.asarray(conf, dtype=np.float64) for conf in reversed(path)]
            print(f"\n[Pose] {capture_idx + 1}/{len(self.views)} {pose_name}")
            self.execute_path(reverse_path, pose_name)
            self.capture_pose(cycle_idx, capture_idx, pose_name, goal_conf, reverse_path)
            capture_idx += 1
        self.next_direction = "forward"

    def run_cycle(self):
        self.cycle_idx += 1
        cycle_idx = self.cycle_idx
        print("\n" + "=" * 72)
        print(f"[Cycle] start cycle {cycle_idx}, direction={self.next_direction}")
        if self.cached_cycle_paths is None:
            self.run_first_forward_cycle(cycle_idx)
        elif self.next_direction == "reverse":
            self.run_cached_reverse_cycle(cycle_idx)
        else:
            self.run_cached_forward_cycle(cycle_idx)
        print(f"[Cycle] cycle {cycle_idx} finished.")
        print("=" * 72)

    def run_keyboard_loop(self):
        print("\n" + "=" * 72)
        print("  Realman joint-pose capture")
        print("  Enter : run one capture cycle in the current direction")
        print("  Q + Enter : quit")
        print("  1st Enter : plan and capture view_01 -> view_10, then stop at view_10.")
        print("  2nd Enter : capture view_10, then reverse cached paths and capture to view_01.")
        print("  3rd Enter : capture view_01, then reuse cached paths and capture to view_10.")
        print("  Later Enter presses keep alternating these two directions.")
        print("=" * 72 + "\n")
        while True:
            line = input("[Idle] Press Enter to run a cycle, or Q + Enter to quit: ")
            if line.lower().strip() == "q":
                print("\n[Exit] user requested quit.")
                return
            self.run_cycle()


def main():
    runner = None
    try:
        runner = JointPoseCaptureRunner()
        runner.run_keyboard_loop()
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()
