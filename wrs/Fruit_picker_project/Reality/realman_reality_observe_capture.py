#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Move the real Realman arm through the observe views and capture RealSense frames.

Views:
    01 initial_0deg
    02 plus_30deg
    03 minus_30deg
    04 plus_60deg
    05 minus_60deg
    06 top45_0deg
    07 top45_plus_30deg
    08 top45_minus_30deg
    09 top45_plus_45deg
    10 top45_minus_45deg
"""

from __future__ import annotations

import argparse
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


INITIAL_JNT_VALUES = np.array([
    -0.05250703,
    2.1996425,
    -2.328618,
    -0.0,
    1.75946436,
    3.09507254,
], dtype=np.float64)

ORBIT_RADIUS = 0.02
ORBIT_CENTER_OFFSET = np.array([-ORBIT_RADIUS, 0.0, 0.0], dtype=np.float64)
LEVEL_ANGLES_DEG = [30, -30, 60, -60]
TOP_ANGLES_DEG = [30, -30, 45, -45]
TOP_VIEW_ANGLE_DEG = 45.0
TOP_NORMAL_RADIUS = ORBIT_RADIUS
TOP_BIG_RADIUS = 0.02

RRT_EXT_DIST = 0.1
RRT_MAX_TIME = 300.0
RRT_SMOOTHING_N_ITER = 50

CAR_STL = _WORKSPACE_ROOT / "wrs" / "LinZhi_picker_project" / "model" / "car.stl"
OUTPUT_ROOT = _THIS_FILE.parent / "outputs_real_10views"


def safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\-]+", "_", name)


def prepare_output_dirs(root: Path):
    for subdir in ("rgb", "depth_raw", "depth_png", "depth_vis", "metadata"):
        (root / subdir).mkdir(parents=True, exist_ok=True)


def make_tcp_rotmat_looking_at(
        tcp_pos: np.ndarray,
        lookat_pos: np.ndarray,
        reference_rotmat: np.ndarray | None = None):
    tcp_z = rm.unit_vector(lookat_pos - tcp_pos)
    ref_axes = []
    if reference_rotmat is not None:
        ref_axes.extend([reference_rotmat[:, 0], reference_rotmat[:, 1]])
    ref_axes.extend([
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
    ])

    tcp_x = None
    for ref_axis in ref_axes:
        projected_x = ref_axis - np.dot(ref_axis, tcp_z) * tcp_z
        if np.linalg.norm(projected_x) >= 1e-6:
            tcp_x = rm.unit_vector(projected_x)
            break
    if tcp_x is None:
        tcp_x = rm.orthogonal_vector(tcp_z, toggle_unit=True)
    tcp_y = rm.unit_vector(np.cross(tcp_z, tcp_x))
    return np.column_stack((tcp_x, tcp_y, tcp_z))


def build_observe_targets(
        robot: Realman,
        initial_jnt_values: np.ndarray | None = None,
        orbit_radius: float | None = None):
    """Build observation targets, optionally overriding the module defaults."""
    if initial_jnt_values is None:
        initial_jnt_values = INITIAL_JNT_VALUES
    initial_jnt_values = np.asarray(initial_jnt_values, dtype=np.float64)
    if initial_jnt_values.shape != (6,) or not np.all(np.isfinite(initial_jnt_values)):
        raise ValueError("initial_jnt_values must contain 6 finite joint values in radians")

    if orbit_radius is None:
        orbit_radius = ORBIT_RADIUS
    orbit_radius = float(orbit_radius)
    if not np.isfinite(orbit_radius) or orbit_radius <= 0:
        raise ValueError("orbit_radius must be a positive finite value in meters")

    tcp_pos0, tcp_rot0 = robot.fk(initial_jnt_values, update=False)
    orbit_center_offset = np.array([-orbit_radius, 0.0, 0.0], dtype=np.float64)
    circle_center = tcp_pos0 + orbit_center_offset
    radius_vec = tcp_pos0 - circle_center
    top_view_height = np.linalg.norm(radius_vec[:2]) * math.tan(math.radians(TOP_VIEW_ANGLE_DEG))
    top_view_z_offset = np.array([0.0, 0.0, top_view_height], dtype=np.float64)

    targets = [{
        "name": "initial_0deg",
        "group": "level",
        "angle_deg": 0,
        "tcp_pos": tcp_pos0,
        "tcp_rotmat": tcp_rot0,
        "jnt_values": initial_jnt_values.copy(),
    }]
    for angle_deg in LEVEL_ANGLES_DEG:
        rot_z = rm.rotmat_from_axangle([0, 0, 1], math.radians(angle_deg))
        angle_prefix = "plus" if angle_deg > 0 else "minus"
        targets.append({
            "name": f"{angle_prefix}_{abs(angle_deg)}deg",
            "group": "level",
            "angle_deg": angle_deg,
            "tcp_pos": circle_center + rot_z @ radius_vec,
            "tcp_rotmat": rot_z @ tcp_rot0,
            "jnt_values": None,
        })

    top_tcp_pos0 = circle_center + radius_vec + top_view_z_offset
    top_tcp_rot0 = make_tcp_rotmat_looking_at(top_tcp_pos0, circle_center, reference_rotmat=tcp_rot0)
    targets.append({
        "name": "top45_0deg",
        "group": "top45",
        "angle_deg": 0,
        "tcp_pos": top_tcp_pos0,
        "tcp_rotmat": top_tcp_rot0,
        "jnt_values": None,
    })
    for angle_deg in TOP_ANGLES_DEG:
        rot_z = rm.rotmat_from_axangle([0, 0, 1], math.radians(angle_deg))
        angle_prefix = "plus" if angle_deg > 0 else "minus"
        unit_radius_vec = radius_vec / np.linalg.norm(radius_vec)
        scaled_radius_vec = unit_radius_vec * orbit_radius
        tcp_pos = circle_center + rot_z @ scaled_radius_vec + top_view_z_offset
        if abs(angle_deg) == 45:
            tcp_pos[2] -= 0.06
        targets.append({
            "name": f"top45_{angle_prefix}_{abs(angle_deg)}deg",
            "group": "top45",
            "angle_deg": angle_deg,
            "tcp_pos": tcp_pos,
            "tcp_rotmat": make_tcp_rotmat_looking_at(tcp_pos, circle_center, reference_rotmat=rot_z @ tcp_rot0),
            "jnt_values": None,
        })
    return targets, circle_center


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
    robotcar_model.pos = np.array([-0.1, 0.0, -0.08])
    robotcar_model.rotmat = rm.rotmat_from_axangle([0, 1, 0], -math.pi * 90 / 180)
    return [robotcar_model]


class RealSenseCapture:
    def __init__(self, output_root: Path, warmup_frames: int = 15):
        import pyrealsense2 as rs

        self.rs = rs
        self.output_root = output_root
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        for _ in range(max(0, warmup_frames)):
            self.pipeline.wait_for_frames()
        print("[Setup] RealSense started.")

    def close(self):
        self.pipeline.stop()
        print("[Cleanup] RealSense stopped.")

    def save_capture(self, idx: int, pose_name: str, metadata: dict, discard_frames: int = 3):
        for _ in range(max(0, discard_frames)):
            self.pipeline.wait_for_frames()
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError(f"{pose_name}: failed to get aligned depth/color frames")

        stem = f"{idx + 1:02d}_{safe_name(pose_name)}"
        color_bgr = np.asanyarray(color_frame.get_data())
        depth_z16 = np.asanyarray(depth_frame.get_data())
        depth_m = depth_z16.astype(np.float32) * float(self.depth_scale)

        cv2.imwrite(str(self.output_root / "rgb" / f"{stem}.png"), color_bgr)
        np.save(self.output_root / "depth_raw" / f"{stem}.npy", depth_m)

        depth_mm = np.clip(depth_m * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        cv2.imwrite(str(self.output_root / "depth_png" / f"{stem}.png"), depth_mm)

        valid_depth = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth_vis_src = np.clip(valid_depth, 0.0, 3.0)
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

        print(f"[Save] {pose_name}: saved RGB/depth to {self.output_root}")


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


def plan_to_conf(planner, start_conf: np.ndarray, goal_conf: np.ndarray, obstacle_list):
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
    return mot_data.jv_list


def solve_goal_conf(robot: Realman, target: dict, seed_conf: np.ndarray):
    if target["jnt_values"] is not None:
        return np.asarray(target["jnt_values"], dtype=np.float64).copy()
    goal_conf = robot.ik(
        tgt_pos=target["tcp_pos"],
        tgt_rotmat=target["tcp_rotmat"],
        seed_jnt_values=seed_conf,
    )
    if goal_conf is None:
        return None
    target["jnt_values"] = np.asarray(goal_conf, dtype=np.float64).copy()
    return target["jnt_values"].copy()


def execute_observe_sequence(args):
    prepare_output_dirs(args.output_root)

    robot_model = Realman(enable_cc=True)
    planner = rrtc.RRTConnect(robot_model)
    obstacle_list = make_obstacle_list(not args.no_car_obstacle)
    targets, circle_center = build_observe_targets(robot_model)

    print("[Setup] observe targets:")
    for idx, target in enumerate(targets, start=1):
        print(f"  {idx:02d} {target['name']} tcp={np.round(target['tcp_pos'], 4)}")
    print(f"[Setup] circle_center={np.round(circle_center, 4)}")

    if not args.dry_run and not args.yes:
        answer = input("This will move the real robot. Type YES to continue: ").strip()
        if answer != "YES":
            print("[Abort] User did not confirm.")
            return

    controller = None
    camera = None
    try:
        if not args.dry_run:
            controller = RealmanArmController(ip=args.ip, port=args.port, has_gripper=args.has_gripper)
            print("[Setup] Realman controller connected.")
        if not args.skip_camera:
            camera = RealSenseCapture(args.output_root, warmup_frames=args.camera_warmup_frames)

        for idx, target in enumerate(targets):
            pose_name = target["name"]
            print(f"\n[Move] {idx + 1}/{len(targets)} {pose_name}")
            if args.dry_run:
                start_conf = robot_model.get_jnt_values()
            else:
                start_conf = np.asarray(controller.get_joint_values(), dtype=np.float64)
            robot_model.goto_given_conf(start_conf)

            goal_conf = solve_goal_conf(robot_model, target, seed_conf=start_conf)
            if goal_conf is None:
                msg = f"[IK][fail] {pose_name}"
                if args.continue_on_fail:
                    print(msg)
                    continue
                raise RuntimeError(msg)

            print(f"       goal_joints(rad) = {np.round(goal_conf, 6)}")
            path = plan_to_conf(planner, start_conf=start_conf, goal_conf=goal_conf, obstacle_list=obstacle_list)
            if path is None:
                msg = f"[RRT][fail] {pose_name}"
                if args.continue_on_fail:
                    print(msg)
                    continue
                raise RuntimeError(msg)

            print(f"       path_points = {len(path)}")
            if args.dry_run:
                robot_model.goto_given_conf(goal_conf)
                print("       [DryRun] planned only, no real motion.")
            else:
                controller.move_jntspace_path(
                    path,
                    control_frequency=args.control_frequency,
                    use_toppra=not args.no_toppra,
                    canfd_dt=args.canfd_dt,
                )
                try:
                    controller.wait_until_idle(timeout=args.idle_timeout)
                except Exception as e:
                    print(f"       [Warn] wait_until_idle failed: {e}")
                time.sleep(args.settle_time)

            actual_conf = goal_conf if args.dry_run else np.asarray(controller.get_joint_values(), dtype=np.float64)
            actual_tcp_pos, actual_tcp_rotmat = robot_model.fk(actual_conf, update=False)
            metadata = {
                "pose_index": idx + 1,
                "pose_name": pose_name,
                "group": target["group"],
                "angle_deg": target["angle_deg"],
                "target_tcp_pos": target["tcp_pos"].tolist(),
                "target_tcp_rotmat": target["tcp_rotmat"].tolist(),
                "goal_jnt_values_rad": goal_conf.tolist(),
                "actual_jnt_values_rad": actual_conf.tolist(),
                "actual_tcp_pos": actual_tcp_pos.tolist(),
                "actual_tcp_rotmat": actual_tcp_rotmat.tolist(),
                "path_points": len(path),
            }
            if camera is not None:
                camera.save_capture(idx, pose_name, metadata, discard_frames=args.camera_discard_frames)
            else:
                with open(args.output_root / "metadata" / f"{idx + 1:02d}_{safe_name(pose_name)}.json",
                          "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

        print("\n[Done] Real robot observe capture sequence finished.")
    finally:
        if camera is not None:
            camera.close()
        if controller is not None:
            controller.close_connection()


def parse_args():
    parser = argparse.ArgumentParser(description="Move real Realman arm through 8 observe views and capture images.")
    parser.add_argument("--ip", default="192.168.1.18", help="Realman controller IP.")
    parser.add_argument("--port", type=int, default=8080, help="Realman controller port.")
    parser.add_argument("--has-gripper", action="store_true", help="Initialize controller with gripper enabled.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT, help="Output directory.")
    parser.add_argument("--control-frequency", type=float, default=0.3, help="TOPPRA control frequency parameter.")
    parser.add_argument("--canfd-dt", type=float, default=0.02, help="CANFD waypoint interval when not using TOPPRA.")
    parser.add_argument("--settle-time", type=float, default=1.0, help="Seconds to wait before capture after motion.")
    parser.add_argument("--idle-timeout", type=float, default=30.0, help="Timeout for motion idle wait.")
    parser.add_argument("--camera-warmup-frames", type=int, default=15, help="RealSense warmup frame count.")
    parser.add_argument("--camera-discard-frames", type=int, default=3, help="Frames to discard before each capture.")
    parser.add_argument("--no-toppra", action="store_true", help="Disable TOPPRA interpolation in controller.")
    parser.add_argument("--no-car-obstacle", action="store_true", help="Plan without the car STL obstacle.")
    parser.add_argument("--skip-camera", action="store_true", help="Move/plan only, do not initialize RealSense.")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; do not connect to or move the real robot.")
    parser.add_argument("--continue-on-fail", action="store_true", help="Skip failed IK/RRT targets instead of stopping.")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation for real motion.")
    return parser.parse_args()


if __name__ == "__main__":
    execute_observe_sequence(parse_args())
