#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the real Realman arm through the car observation views while making the
RealSense camera center, not the TCP, look at the target point.

This script reuses the motion, RRT, RealSense capture, and keyboard flow from
realman_reality_car.py. The difference is the pose constraint:

    T_world_camera = desired camera-center pose looking at the stem
    T_world_tcp    = T_world_camera @ inv(T_tcp_camera)

The hand-eye matrix is loaded from manual_calibration_realman22.json by default.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

_THIS_FILE = Path(__file__).resolve()
_WORKSPACE_ROOT = _THIS_FILE.parents[3]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

import wrs.basis.robot_math as rm
from wrs.Fruit_picker_project.Reality import realman_reality_car as base


OUTPUT_ROOT = _THIS_FILE.parent / "outputs_real_d405_car_camera_center"
HANDEYE_CALIB_PATH = _WORKSPACE_ROOT / "wrs" / "vision" / "depth_camera" / "manual_calibration_realman22.json"

# RealSense point cloud convention uses +Z as the optical axis.
CAMERA_FORWARD_AXIS = 2


def load_handeye_matrix(calib_path: Path) -> np.ndarray:
    if not calib_path.is_file():
        raise FileNotFoundError(f"hand-eye calibration file not found: {calib_path}")
    with open(calib_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    handeye_mat = np.asarray(data["affine_mat"], dtype=np.float64)
    if handeye_mat.shape != (4, 4):
        raise ValueError(f"hand-eye affine_mat must be 4x4, got {handeye_mat.shape}")
    print(f"[Setup] hand-eye calibration loaded: {calib_path}")
    return handeye_mat


def homomat_from_posrot(pos: np.ndarray, rotmat: np.ndarray) -> np.ndarray:
    homomat = np.eye(4, dtype=np.float64)
    homomat[:3, :3] = np.asarray(rotmat, dtype=np.float64)
    homomat[:3, 3] = np.asarray(pos, dtype=np.float64)
    return homomat


def camera_pose_from_tcp_pose(tcp_pos: np.ndarray, tcp_rotmat: np.ndarray, tcp_to_camera_homomat: np.ndarray):
    world_to_tcp = homomat_from_posrot(tcp_pos, tcp_rotmat)
    world_to_camera = world_to_tcp @ tcp_to_camera_homomat
    return world_to_camera[:3, 3].copy(), world_to_camera[:3, :3].copy()


def tcp_pose_from_camera_pose(camera_pos: np.ndarray, camera_rotmat: np.ndarray, tcp_to_camera_homomat: np.ndarray):
    world_to_camera = homomat_from_posrot(camera_pos, camera_rotmat)
    world_to_tcp = world_to_camera @ np.linalg.inv(tcp_to_camera_homomat)
    return world_to_tcp[:3, 3].copy(), world_to_tcp[:3, :3].copy()


def make_camera_rotmat_looking_at(camera_pos: np.ndarray, lookat_pos: np.ndarray):
    camera_z = rm.unit_vector(lookat_pos - camera_pos)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    camera_x = np.cross(camera_z, world_up)
    if np.linalg.norm(camera_x) < 1e-6:
        camera_x = rm.orthogonal_vector(camera_z, toggle_unit=True)
    else:
        camera_x = rm.unit_vector(camera_x)
    camera_y = rm.unit_vector(np.cross(camera_z, camera_x))
    return np.column_stack((camera_x, camera_y, camera_z))


def roll_camera_rotmat(camera_rotmat: np.ndarray, roll_rad: float):
    return camera_rotmat @ rm.rotmat_from_axangle([0, 0, 1], roll_rad)


def build_camera_center_view_targets(
        stem_lookat_pos: np.ndarray,
        level_view_distance: float,
        top_view_distance: float,
        top_view_angle_deg: float,
        level_view_angles_deg: list[float],
        top_view_angles_deg: list[float],
        tcp_to_camera_homomat: np.ndarray):
    targets = []
    front_vec = np.array([level_view_distance, 0.0, 0.0], dtype=np.float64)

    for angle_deg in level_view_angles_deg:
        rot_z = rm.rotmat_from_axangle([0, 0, 1], math.radians(angle_deg))
        camera_pos = stem_lookat_pos + rot_z @ front_vec
        camera_rotmat = make_camera_rotmat_looking_at(camera_pos, stem_lookat_pos)
        tcp_pos, tcp_rotmat = tcp_pose_from_camera_pose(camera_pos, camera_rotmat, tcp_to_camera_homomat)
        targets.append({
            "name": f"level_{base.view_angle_name(angle_deg)}",
            "group": "level",
            "angle_deg": float(angle_deg),
            "view_vec": rot_z @ front_vec,
            "camera_pos": camera_pos,
            "camera_rotmat": camera_rotmat,
            "tcp_pos": tcp_pos,
            "tcp_rotmat": tcp_rotmat,
            "jnt_values": None,
        })

    top_horizontal_distance = float(top_view_distance)
    top_z_offset = top_horizontal_distance * math.tan(math.radians(top_view_angle_deg))
    top_front_vec = np.array([top_horizontal_distance, 0.0, top_z_offset], dtype=np.float64)

    for angle_deg in top_view_angles_deg:
        rot_z = rm.rotmat_from_axangle([0, 0, 1], math.radians(angle_deg))
        camera_pos = stem_lookat_pos + rot_z @ top_front_vec
        camera_rotmat = make_camera_rotmat_looking_at(camera_pos, stem_lookat_pos)
        tcp_pos, tcp_rotmat = tcp_pose_from_camera_pose(camera_pos, camera_rotmat, tcp_to_camera_homomat)
        targets.append({
            "name": f"top45_{base.view_angle_name(angle_deg)}",
            "group": "top45",
            "angle_deg": float(angle_deg),
            "view_vec": rot_z @ top_front_vec,
            "camera_pos": camera_pos,
            "camera_rotmat": camera_rotmat,
            "tcp_pos": tcp_pos,
            "tcp_rotmat": tcp_rotmat,
            "jnt_values": None,
        })

    return targets


def solve_camera_center_view_ik(
        robot,
        target: dict,
        seed_jnt_values: np.ndarray,
        stem_lookat_pos: np.ndarray,
        tcp_to_camera_homomat: np.ndarray,
        distance_scale_search: list[float],
        roll_search_deg: list[float]):
    if target["jnt_values"] is not None:
        return np.asarray(target["jnt_values"], dtype=np.float64).copy()

    base_view_vec = np.asarray(target["view_vec"], dtype=np.float64)
    for scale in distance_scale_search:
        camera_pos = stem_lookat_pos + base_view_vec * float(scale)
        base_camera_rotmat = make_camera_rotmat_looking_at(camera_pos, stem_lookat_pos)
        for roll_deg in roll_search_deg:
            camera_rotmat = roll_camera_rotmat(base_camera_rotmat, math.radians(float(roll_deg)))
            tcp_pos, tcp_rotmat = tcp_pose_from_camera_pose(camera_pos, camera_rotmat, tcp_to_camera_homomat)
            goal_conf = robot.ik(
                tgt_pos=tcp_pos,
                tgt_rotmat=tcp_rotmat,
                seed_jnt_values=seed_jnt_values,
                toggle_dbg=False,
            )
            if goal_conf is not None:
                target["camera_pos"] = camera_pos
                target["camera_rotmat"] = camera_rotmat
                target["tcp_pos"] = tcp_pos
                target["tcp_rotmat"] = tcp_rotmat
                target["ik_scale"] = float(scale)
                target["ik_roll_deg"] = float(roll_deg)
                target["jnt_values"] = np.asarray(goal_conf, dtype=np.float64).copy()
                return target["jnt_values"].copy()
    return None


class RealityCarCameraCenterRunner(base.RealityCarRunner):
    def __init__(self):
        base.prepare_output_dirs(OUTPUT_ROOT)
        self.stem_lookat_pos = np.asarray(base.DEFAULT_STEM_LOOKAT_POS, dtype=np.float64)
        self.level_view_angles_deg = [float(v) for v in base.LEVEL_VIEW_ANGLES_DEG]
        self.top_view_angles_deg = [float(v) for v in base.TOP_VIEW_ANGLES_DEG]
        self.roll_search_deg = [float(v) for v in base.ROLL_SEARCH_DEG]
        self.distance_scale_search = [float(v) for v in base.VIEW_DISTANCE_SCALE_SEARCH]
        self.handeye_mat = load_handeye_matrix(HANDEYE_CALIB_PATH)

        self.robot_model = base.Realman(enable_cc=True)
        self.planner = base.rrtc.RRTConnect(self.robot_model)
        self.obstacle_list = base.make_obstacle_list(base.USE_CAR_OBSTACLE)
        self.targets = build_camera_center_view_targets(
            stem_lookat_pos=self.stem_lookat_pos,
            level_view_distance=base.LEVEL_VIEW_DISTANCE,
            top_view_distance=base.TOP_VIEW_DISTANCE,
            top_view_angle_deg=base.TOP_VIEW_ANGLE_DEG,
            level_view_angles_deg=self.level_view_angles_deg,
            top_view_angles_deg=self.top_view_angles_deg,
            tcp_to_camera_homomat=self.handeye_mat,
        )
        self.controller = base.RealmanArmController(
            ip=base.ROBOT_IP,
            port=base.ROBOT_PORT,
            has_gripper=base.HAS_GRIPPER,
        )
        self.camera = None
        if base.USE_CAMERA:
            self.camera = base.RealSenseCapture(
                OUTPUT_ROOT,
                warmup_frames=base.CAMERA_WARMUP_FRAMES,
                depth_vis_far=base.D405_DEPTH_FAR,
                frame_timeout_ms=base.CAMERA_FRAME_TIMEOUT_MS,
                capture_retries=base.CAMERA_CAPTURE_RETRIES,
                retry_sleep=base.CAMERA_RETRY_SLEEP,
            )
        self.cached_cycle_segments = None
        self.cycle_idx = 0
        self._print_setup()

    def _print_setup(self):
        print("\n[Setup] real car camera-center observation targets:")
        print(f"  stem_lookat_pos = {np.round(self.stem_lookat_pos, 4)}")
        print(f"  observe/home joints = {np.round(base.HOME_JNT_VALUES, 6)}")
        print(f"  handeye tcp_to_camera translation = {np.round(self.handeye_mat[:3, 3], 4)}")
        print(f"  camera_forward_axis = +Z / column {CAMERA_FORWARD_AXIS}")
        print(f"  level_view_angles_deg = {self.level_view_angles_deg}")
        print(f"  top_view_angles_deg = {self.top_view_angles_deg}")
        for idx, target in enumerate(self.targets, start=1):
            print(
                f"  {idx:02d} {target['name']} "
                f"cam={np.round(target['camera_pos'], 4)} tcp={np.round(target['tcp_pos'], 4)}"
            )

    def solve_target_conf(self, target, seed_conf):
        goal_conf = solve_camera_center_view_ik(
            robot=self.robot_model,
            target=target,
            seed_jnt_values=seed_conf,
            stem_lookat_pos=self.stem_lookat_pos,
            tcp_to_camera_homomat=self.handeye_mat,
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
        actual_camera_pos, actual_camera_rotmat = camera_pose_from_tcp_pose(
            actual_tcp_pos,
            actual_tcp_rotmat,
            self.handeye_mat,
        )
        target_forward = rm.unit_vector(self.stem_lookat_pos - np.asarray(target["camera_pos"], dtype=np.float64))
        actual_forward = actual_camera_rotmat[:, CAMERA_FORWARD_AXIS]
        dot_value = float(np.clip(np.dot(target_forward, actual_forward), -1.0, 1.0))
        camera_lookat_error_deg = math.degrees(math.acos(dot_value))
        metadata = {
            "cycle_index": cycle_idx,
            "view_index": view_idx + 1,
            "view_name": target["name"],
            "group": target["group"],
            "angle_deg": target["angle_deg"],
            "stem_lookat_pos": self.stem_lookat_pos.tolist(),
            "target_camera_pos": target["camera_pos"].tolist(),
            "target_camera_rotmat": target["camera_rotmat"].tolist(),
            "target_tcp_pos": target["tcp_pos"].tolist(),
            "target_tcp_rotmat": target["tcp_rotmat"].tolist(),
            "handeye_tcp_to_camera": self.handeye_mat.tolist(),
            "camera_forward_axis": CAMERA_FORWARD_AXIS,
            "ik_scale": float(target.get("ik_scale", 1.0)),
            "ik_roll_deg": float(target.get("ik_roll_deg", 0.0)),
            "goal_jnt_values_rad": np.asarray(target["jnt_values"], dtype=np.float64).tolist(),
            "actual_jnt_values_rad": actual_conf.tolist(),
            "actual_tcp_pos": actual_tcp_pos.tolist(),
            "actual_tcp_rotmat": actual_tcp_rotmat.tolist(),
            "actual_camera_pos": actual_camera_pos.tolist(),
            "actual_camera_rotmat": actual_camera_rotmat.tolist(),
            "actual_camera_forward": actual_forward.tolist(),
            "camera_lookat_error_deg": camera_lookat_error_deg,
            "path_points": len(path),
            "level_view_distance": float(base.LEVEL_VIEW_DISTANCE),
            "top_view_distance": float(base.TOP_VIEW_DISTANCE),
            "top_view_angle_deg": float(base.TOP_VIEW_ANGLE_DEG),
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
                    discard_frames=base.CAMERA_DISCARD_FRAMES,
                )
                return
            except RuntimeError as e:
                metadata["capture_error"] = str(e)
                print(f"       [Capture][fail] {target['name']}: {e}")
                if not base.CONTINUE_ON_CAPTURE_FAIL:
                    raise
        base.write_metadata(OUTPUT_ROOT, cycle_idx, view_idx, target["name"], metadata)

    def run_keyboard_loop(self):
        print("\n" + "=" * 72)
        print("  Realman D405 camera-center observe capture")
        print("  Space : run one full observe cycle")
        print("  Enter : run one full observe cycle when stdin is not a TTY")
        print("  Q     : quit")
        print("  First cycle solves camera-center IK/RRT and caches paths.")
        print("  Later cycles reuse cached paths.")
        print("=" * 72 + "\n")

        while True:
            print("[Idle] Press Space to run a cycle, or Q to quit.")
            key = base.read_single_key()
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
        runner = RealityCarCameraCenterRunner()
        runner.run_keyboard_loop()
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()
