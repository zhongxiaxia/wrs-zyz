#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulate Realman moving through 10 user-provided joint poses.

This file mirrors the simulation style of realman_sim_car.py, but it does not
solve IK. It uses VIEW_JNT_VALUES directly, plans an RRT path between each
joint pose with the car STL as an obstacle, updates the virtual D405 camera,
and saves RGB/depth images at every reached view.

Keys:
    N / Space : move to the next joint view and capture
    A         : automatically traverse all remaining views
    H / R     : move back to HOME_JNT_VALUES
    C         : capture the current view again
    Q         : quit
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

_THIS_FILE = Path(__file__).resolve()
_WORKSPACE_ROOT = _THIS_FILE.parents[3]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.model_collection as mmc
import wrs.motion.probabilistic.rrt_connect as rrtc
import wrs.visualization.panda.world as wd
from wrs.robot_sim.robots.realman.realman_single_arm import Realman
from wrs.visualization.panda.panda3d_utils import VirtualDepthCamera


VIEW_JNT_VALUES = [
    ("view_01", np.array([0.18264871, 1.96112173, -2.19009154, -0.16084954, 1.84809425, -0.00324631],
                         dtype=np.float64)),
    ("view_02", np.array([-0.62289054, 1.96108685, -2.14439875, 0.90024086, 1.72269238, 0.17723818],
                         dtype=np.float64)),
    ("view_03", np.array([0.8689645, 1.9610694, -1.98632429, -1.0563256, 1.65036585, 0.02391101],
                         dtype=np.float64)),
    ("view_04", np.array([0.92357588, 1.39326149, -0.28902651, -1.38404615, 1.54231249, 1.15418619],
                         dtype=np.float64)),
    ("view_05", np.array([-0.83580328, 1.39378506, -0.28824112, -1.70155634, -1.53372552, 1.94708927],
                         dtype=np.float64)),
    ("view_06", np.array([0.05323254, 0.80477133, -0.98885122, 0.06571165, 2.18864291, 0.03558726],
                         dtype=np.float64)),
    ("view_07", np.array([-0.61859706, 0.80492839, -0.55311229, 0.90165453, 1.87940543, -0.03028146],
                         dtype=np.float64)),
    ("view_08", np.array([0.94139569, 1.00759601, -0.9129468, -1.08655474, 1.91317751, 0.17751743],
                         dtype=np.float64)),
    ("view_09", np.array([0.98560489, 1.07880548, -0.108926, -1.31128326, 1.59713339, 1.05292222],
                         dtype=np.float64)),
    ("view_10", np.array([-0.84732245, 1.0796258, -0.10744247, -1.85977048, -1.6392132, 2.17523882],
                         dtype=np.float64)),
]

HOME_JNT_VALUES = np.array(
    [0.06897541, 1.55276713, -1.98419496, -0.03705334, 1.97552078, 0.01207768],
    dtype=np.float64,
)

# PEPPER_BODY_POS = np.array([-0.36, 0.0, 0.22], dtype=np.float64)
# PEPPER_BODY_SIZE = np.array([0.035, 0.035, 0.16], dtype=np.float64)
# STEM_SIZE = np.array([0.012, 0.012, 0.055], dtype=np.float64)
# STEM_POS = PEPPER_BODY_POS + np.array([0.0, 0.0, PEPPER_BODY_SIZE[2] / 2.0 + STEM_SIZE[2] / 2.0])

RRT_EXT_DIST = 0.03
RRT_MAX_TIME = 120.0
RRT_SMOOTHING_N_ITER = 10
RRT_COLLISION_GRANULARITY = 0.01

OUTPUT_ROOT = _THIS_FILE.parent / "outputs_d405_joint_capture"
CAR_STL = _WORKSPACE_ROOT / "wrs" / "LinZhi_picker_project" / "model" / "car.stl"
SHOW_CAR_OBSTACLE = True

D405_RESOLUTION = (1280, 720)
D405_FOV_DEG = 87.0
D405_DEPTH_FAR = 1.0
CAMERA_FORWARD_DEBUG_LEN = 0.50
D405_BODY_XYZ_LENGTHS = np.array([0.042, 0.023, 0.042], dtype=np.float64)
D405_BODY_CENTER_FROM_OPTICAL = np.array([0.0, -D405_BODY_XYZ_LENGTHS[1] / 2.0, 0.0], dtype=np.float64)
D405_OPTICAL_CENTER_RADIUS = 0.006

# TCP -> RealSense optical camera hand-eye calibration matrix.
# The RealSense optical frame uses +Z as forward, while WRS VirtualDepthCamera
# uses local +Y as forward. LOC_CAM_ROTMAT converts the optical frame into the
# WRS virtual-camera frame without changing the calibrated camera center.
TCP_TO_CAMERA_HOMOMAT = np.array([
    [-0.012102519796519457, 0.9998605053175249, 0.011510817528360709, -0.08799999999999977],
    [-0.9998133994554028, -0.012273682788263483, 0.01491720415436704, 0.010999999999999968],
    [0.01505640340671479, -0.011328133844956416, 0.9998224732918157, -0.0360000000000001],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)
OPTICAL_FROM_WRS_CAMERA_ROTMAT = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)
LOC_CAM_POS = TCP_TO_CAMERA_HOMOMAT[:3, 3].copy()
LOC_CAM_ROTMAT = TCP_TO_CAMERA_HOMOMAT[:3, :3] @ OPTICAL_FROM_WRS_CAMERA_ROTMAT


def safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", name)


def prepare_output_dirs(root: Path):
    for subdir in ("rgb", "depth_raw", "depth_png", "depth_vis", "metadata"):
        (root / subdir).mkdir(parents=True, exist_ok=True)


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
    return views


class RealmanJointPoseSim:
    def __init__(self):
        self.base = wd.World(
            cam_pos=rm.vec(1.25, 1.1, 0.9),
            lookat_pos=rm.vec(-0.25, 0.0, 0.25),
        )
        self.robot = Realman(enable_cc=True)
        self.planner = rrtc.RRTConnect(self.robot)
        self.views = normalize_view_joints()
        self.obstacle_list = []
        self.robotcar_model = None
        self.robot_mesh = None
        self.robot_stick = None
        self.camera_mesh = None
        self.d405 = None
        self.path_meshes = []
        self.current_view_idx = -1

        prepare_output_dirs(OUTPUT_ROOT)
        self._build_scene()
        self._attach_view_debug_models()
        self._init_d405_camera()
        self._bind_keys()
        self.goto_conf(HOME_JNT_VALUES, name="home", use_rrt=False, capture=False)

    def _tcp_to_cam(self, tcp_pos: np.ndarray, tcp_rotmat: np.ndarray):
        cam_pos = tcp_pos + tcp_rotmat @ LOC_CAM_POS
        cam_rotmat = tcp_rotmat @ LOC_CAM_ROTMAT
        return cam_pos, cam_rotmat

    def _update_d405_from_robot(self, jnt_values=None):
        if self.d405 is None:
            return None, None
        if jnt_values is None:
            jnt_values = np.asarray(self.robot.get_jnt_values(), dtype=np.float64)
        tcp_pos, tcp_rotmat = self.robot.fk(np.asarray(jnt_values, dtype=np.float64), update=False)
        cam_pos, cam_rotmat = self._tcp_to_cam(tcp_pos, tcp_rotmat)
        self.d405.cam_pos = cam_pos
        self.d405.cam_rotmat = cam_rotmat
        return cam_pos, cam_rotmat

    def _init_d405_camera(self):
        tcp_pos, tcp_rotmat = self.robot.fk(HOME_JNT_VALUES, update=False)
        cam_pos, cam_rotmat = self._tcp_to_cam(tcp_pos, tcp_rotmat)
        self.d405 = VirtualDepthCamera(
            cam_pos=cam_pos,
            lookat_pos=cam_pos + cam_rotmat[:, 1],
            resolution=D405_RESOLUTION,
            screen_size=np.array((0.05, 0.05)),
            fov=D405_FOV_DEG,
            depth_far=D405_DEPTH_FAR,
        )
        self.d405.cam_rotmat = cam_rotmat
        self.camera_mesh = self._gen_d405_visual_model(cam_pos, cam_rotmat)
        self.camera_mesh.attach_to(self.base)

    def _gen_d405_visual_model(self, cam_pos: np.ndarray, cam_rotmat: np.ndarray):
        m_col = mmc.ModelCollection(name="d405_visual_model")
        body_pos = cam_pos + cam_rotmat @ D405_BODY_CENTER_FROM_OPTICAL
        mgm.gen_box(
            xyz_lengths=D405_BODY_XYZ_LENGTHS,
            pos=body_pos,
            rotmat=cam_rotmat,
            rgb=np.array([0.55, 0.55, 0.55]),
            alpha=0.72,
        ).attach_to(m_col)
        mgm.gen_sphere(
            pos=cam_pos,
            radius=D405_OPTICAL_CENTER_RADIUS,
            rgb=np.array([0.1, 0.35, 1.0]),
            alpha=0.95,
        ).attach_to(m_col)
        mgm.gen_dashed_stick(
            spos=cam_pos,
            epos=cam_pos + cam_rotmat[:, 1] * CAMERA_FORWARD_DEBUG_LEN,
            radius=0.001,
            rgb=np.array([1.0, 0.8, 0.05]),
            alpha=0.75,
        ).attach_to(m_col)
        return m_col

    def _refresh_camera_model(self):
        if self.d405 is None:
            return
        if self.camera_mesh is not None:
            self.camera_mesh.detach()
        self.camera_mesh = self._gen_d405_visual_model(self.d405.cam_pos, self.d405.cam_rotmat)
        self.camera_mesh.attach_to(self.base)

    def _save_d405_capture(self, view_idx: int, view_name: str, jnt_values: np.ndarray, path):
        if self.d405 is None:
            return
        img_rgb = self.d405.get_rgb_image()
        img_depth = self.d405.get_depth_image()
        if img_rgb is None or img_depth is None:
            print(f"       [Capture][warn] {view_name}: no RGB/depth image returned.")
            return

        stem = f"{view_idx + 1:02d}_{safe_name(view_name)}"
        rgb_to_save = img_rgb[:, :, :3]
        if rgb_to_save.dtype != np.uint8:
            rgb_to_save = np.clip(rgb_to_save, 0, 255).astype(np.uint8)
        cv2.imwrite(str(OUTPUT_ROOT / "rgb" / f"{stem}.png"), cv2.cvtColor(rgb_to_save, cv2.COLOR_RGB2BGR))

        depth = np.asarray(img_depth, dtype=np.float32)
        np.save(OUTPUT_ROOT / "depth_raw" / f"{stem}.npy", depth)

        valid_depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_mm = np.clip(valid_depth * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        cv2.imwrite(str(OUTPUT_ROOT / "depth_png" / f"{stem}.png"), depth_mm)

        depth_vis_src = np.clip(valid_depth, 0.0, D405_DEPTH_FAR)
        depth_vis = cv2.normalize(depth_vis_src, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(OUTPUT_ROOT / "depth_vis" / f"{stem}.png"), depth_vis)

        tcp_pos, tcp_rotmat = self.robot.fk(jnt_values, update=False)
        cam_pos, cam_rotmat = self._tcp_to_cam(tcp_pos, tcp_rotmat)
        metadata = {
            "view_index": view_idx + 1,
            "view_name": view_name,
            "jnt_values_rad": np.asarray(jnt_values, dtype=np.float64).tolist(),
            "tcp_pos": tcp_pos.tolist(),
            "tcp_rotmat": tcp_rotmat.tolist(),
            "camera_pos": cam_pos.tolist(),
            "camera_rotmat": cam_rotmat.tolist(),
            "tcp_to_camera_homomat": TCP_TO_CAMERA_HOMOMAT.tolist(),
            "tcp_to_camera_pos": LOC_CAM_POS.tolist(),
            "tcp_to_camera_rotmat": LOC_CAM_ROTMAT.tolist(),
            "optical_from_wrs_camera_rotmat": OPTICAL_FROM_WRS_CAMERA_ROTMAT.tolist(),
            "d405_body_xyz_lengths": D405_BODY_XYZ_LENGTHS.tolist(),
            "d405_body_center_from_optical": D405_BODY_CENTER_FROM_OPTICAL.tolist(),
            "path_points": len(path),
            "collided": bool(self.robot.is_collided(obstacle_list=self.obstacle_list)),
        }
        with open(OUTPUT_ROOT / "metadata" / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"       [Capture] virtual D405 RGB/depth saved: {OUTPUT_ROOT}")

    def _attach_robotcar(self):
        if not SHOW_CAR_OBSTACLE:
            return None
        if not CAR_STL.is_file():
            print(f"[Warn] car STL not found, planning without car obstacle: {CAR_STL}")
            return None
        robotcar_model = mcm.CollisionModel(
            str(CAR_STL),
            rgb=np.array([0.6, 0.6, 0.6]),
            alpha=0.9,
        )
        robotcar_model.pos = np.array([-0.1, 0.0, -0.08], dtype=np.float64)
        robotcar_model.rotmat = rm.rotmat_from_axangle([0, 1, 0], -math.pi / 2)
        robotcar_model.attach_to(self.base)
        self.obstacle_list.append(robotcar_model)
        print(f"[Setup] car model loaded: {CAR_STL}")
        return robotcar_model

    def _build_scene(self):
        mgm.gen_frame(ax_length=0.18).attach_to(self.base)
        self.robotcar_model = self._attach_robotcar()

        ground = mcm.gen_box(
            xyz_lengths=np.array([1.1, 0.9, 0.02]),
            pos=np.array([-0.2, 0.0, -0.02]),
            rotmat=np.eye(3),
            rgb=np.array([0.55, 0.55, 0.55]),
            alpha=0.22,
        )
        ground.attach_to(self.base)

        # pepper_body = mcm.gen_box(
        #     xyz_lengths=PEPPER_BODY_SIZE,
        #     pos=PEPPER_BODY_POS,
        #     rotmat=np.eye(3),
        #     rgb=np.array([0.0, 0.75, 0.12]),
        #     alpha=1.0,
        # )
        # pepper_body.attach_to(self.base)
        #
        # pepper_stem = mcm.gen_box(
        #     xyz_lengths=STEM_SIZE,
        #     pos=STEM_POS,
        #     rotmat=np.eye(3),
        #     rgb=np.array([0.0, 0.45, 0.05]),
        #     alpha=1.0,
        # )
        # pepper_stem.attach_to(self.base)
        #
        # mgm.gen_sphere(
        #     pos=STEM_POS,
        #     radius=0.009,
        #     rgb=np.array([1.0, 0.1, 0.1]),
        #     alpha=0.9,
        # ).attach_to(self.base)

    def _attach_view_debug_models(self):
        for name, conf in self.views:
            tcp_pos, tcp_rotmat = self.robot.fk(conf, update=False)
            mgm.gen_frame(pos=tcp_pos, rotmat=tcp_rotmat, ax_length=0.055).attach_to(self.base)
            cam_pos, cam_rotmat = self._tcp_to_cam(tcp_pos, tcp_rotmat)
            # mgm.gen_dashed_stick(
            #     spos=cam_pos,
            #     epos=cam_pos + cam_rotmat[:, 1] * CAMERA_FORWARD_DEBUG_LEN,
            #     radius=0.001,
            #     rgb=np.array([1.0, 0.8, 0.05]),
            #     alpha=0.45,
            # ).attach_to(self.base)
            # mgm.gen_sphere(pos=cam_pos, radius=0.004, rgb=np.array([0.1, 0.4, 1.0]), alpha=0.75).attach_to(self.base)

    def _bind_keys(self):
        self.base.accept("n", self.next_view)
        self.base.accept("space", self.next_view)
        self.base.accept("a", self.auto_view_all)
        self.base.accept("c", self.capture_current_view)
        self.base.accept("h", lambda: self.goto_conf(HOME_JNT_VALUES, name="home", use_rrt=True, capture=False))
        self.base.accept("r", lambda: self.goto_conf(HOME_JNT_VALUES, name="home", use_rrt=True, capture=False))
        self.base.accept("q", self.base.destroy)

    def _clear_path_meshes(self):
        for mesh in self.path_meshes:
            mesh.detach()
        self.path_meshes.clear()

    def _refresh_robot_model(self):
        if self.robot_mesh is not None:
            self.robot_mesh.detach()
        if self.robot_stick is not None:
            self.robot_stick.detach()
        self.robot_mesh = self.robot.gen_meshmodel(alpha=0.65, toggle_tcp_frame=True)
        self.robot_stick = self.robot.gen_stickmodel(toggle_tcp_frame=False, toggle_jnt_frames=False)
        self.robot_mesh.attach_to(self.base)
        self.robot_stick.attach_to(self.base)

    def _show_path(self, jv_list):
        self._clear_path_meshes()
        if len(jv_list) <= 1:
            return
        color_list = getattr(rm.const, "tab20_list", None)
        for idx, jnt_values in enumerate(jv_list):
            if idx % max(1, len(jv_list) // 12) != 0 and idx != len(jv_list) - 1:
                continue
            self.robot.goto_given_conf(np.asarray(jnt_values, dtype=np.float64))
            mesh = self.robot.gen_meshmodel(alpha=0.18, toggle_tcp_frame=False)
            if color_list is not None:
                mesh.rgb = color_list[idx % len(color_list)]
            mesh.attach_to(self.base)
            self.path_meshes.append(mesh)

    def _validate_path_collision(self, path):
        if path is None or len(path) == 0:
            return False
        for seg_idx in range(len(path) - 1):
            start = np.asarray(path[seg_idx], dtype=np.float64)
            goal = np.asarray(path[seg_idx + 1], dtype=np.float64)
            diff = goal - start
            n_step = max(2, int(np.linalg.norm(diff) / RRT_COLLISION_GRANULARITY))
            for interp_idx in range(n_step + 1):
                conf = start + diff * (interp_idx / n_step)
                self.robot.goto_given_conf(conf)
                if self.robot.is_collided(obstacle_list=self.obstacle_list):
                    print(f"       [Collision] segment {seg_idx}, interp {interp_idx}/{n_step} hits car STL.")
                    return False
        final_conf = np.asarray(path[-1], dtype=np.float64)
        self.robot.goto_given_conf(final_conf)
        if self.robot.is_collided(obstacle_list=self.obstacle_list):
            print("       [Collision] final_conf hits car STL.")
            return False
        return True

    def _plan_path(self, start_conf, goal_conf):
        start_conf = np.asarray(start_conf, dtype=np.float64)
        goal_conf = np.asarray(goal_conf, dtype=np.float64)
        if np.linalg.norm(start_conf - goal_conf) < 1e-6:
            return [start_conf.copy(), goal_conf.copy()]
        mot_data = self.planner.plan(
            start_conf=start_conf,
            goal_conf=goal_conf,
            obstacle_list=self.obstacle_list,
            ext_dist=RRT_EXT_DIST,
            max_time=RRT_MAX_TIME,
            smoothing_n_iter=RRT_SMOOTHING_N_ITER,
        )
        if mot_data is None:
            return None
        path = mot_data.jv_list
        if not self._validate_path_collision(path):
            return None
        return [np.asarray(conf, dtype=np.float64) for conf in path]

    def goto_conf(self, goal_conf, name="target", use_rrt=True, capture=False, view_idx=-1):
        goal_conf = np.asarray(goal_conf, dtype=np.float64)
        start_conf = np.asarray(self.robot.get_jnt_values(), dtype=np.float64)
        self._clear_path_meshes()
        print(f"\n[Move] {name}")
        print(f"       goal_joints = {np.round(goal_conf, 6)}")

        if use_rrt:
            path = self._plan_path(start_conf, goal_conf)
            if path is None:
                print("       [RRT][fail] no collision-free path; move canceled.")
                self.robot.goto_given_conf(start_conf)
                self._update_d405_from_robot(start_conf)
                self._refresh_robot_model()
                self._refresh_camera_model()
                return False
        else:
            path = [start_conf.copy(), goal_conf.copy()]

        self._show_path(path)
        self.robot.goto_given_conf(np.asarray(path[-1], dtype=np.float64))
        self._update_d405_from_robot(np.asarray(path[-1], dtype=np.float64))
        tcp_pos, tcp_rotmat = self.robot.fk(np.asarray(path[-1], dtype=np.float64), update=False)
        cam_pos, cam_rotmat = self._tcp_to_cam(tcp_pos, tcp_rotmat)
        is_collided = self.robot.is_collided(obstacle_list=self.obstacle_list)
        print(f"       path_points = {len(path)}")
        print(f"       tcp_pos = {np.round(tcp_pos, 4)}")
        print(f"       tcp_z_axis = {np.round(tcp_rotmat[:, 2], 4)}")
        print(f"       cam_pos = {np.round(cam_pos, 4)}")
        print(f"       cam_forward = {np.round(cam_rotmat[:, 1], 4)}")
        print(f"       collided = {is_collided}")

        self._refresh_robot_model()
        self._refresh_camera_model()
        if is_collided:
            print("       [Collision] target collides with car STL; rolling back.")
            self.robot.goto_given_conf(start_conf)
            self._update_d405_from_robot(start_conf)
            self._refresh_robot_model()
            self._refresh_camera_model()
            return False
        if capture:
            self._save_d405_capture(view_idx, name, goal_conf, path)
        return True

    def goto_view(self, view_idx: int):
        name, goal_conf = self.views[view_idx]
        print(f"\n[View] {view_idx + 1}/{len(self.views)} {name}")
        success = self.goto_conf(goal_conf, name=name, use_rrt=True, capture=True, view_idx=view_idx)
        if success:
            self.current_view_idx = view_idx
        return success

    def next_view(self):
        for offset in range(1, len(self.views) + 1):
            next_idx = (self.current_view_idx + offset) % len(self.views)
            if self.goto_view(next_idx):
                return
        print("\n[Warn] no view has a collision-free path from the current configuration.")

    def auto_view_all(self):
        start_idx = self.current_view_idx + 1
        for view_idx in range(start_idx, len(self.views)):
            if not self.goto_view(view_idx):
                print(f"\n[Skip] {self.views[view_idx][0]} has no collision-free path; continuing.")
                continue
        print("\n[Done] all requested joint views have been traversed.")

    def capture_current_view(self):
        if self.current_view_idx < 0:
            print("\n[Capture][warn] no current view has been reached yet.")
            return
        name, conf = self.views[self.current_view_idx]
        self._save_d405_capture(self.current_view_idx, name, conf, [conf])

    def run(self):
        print("\n" + "=" * 72)
        print("  Realman joint-pose simulation on car scene")
        print("  N / Space : move to next joint view and capture")
        print("  A         : automatically traverse all remaining joint views")
        print("  C         : capture current view again")
        print("  H / R     : return to home pose")
        print("  Q         : quit")
        print(f"  view count: {len(self.views)}")
        print(f"  output    : {OUTPUT_ROOT}")
        print("=" * 72 + "\n")
        self.base.run()


def main():
    sim = RealmanJointPoseSim()
    sim.run()


if __name__ == "__main__":
    main()
