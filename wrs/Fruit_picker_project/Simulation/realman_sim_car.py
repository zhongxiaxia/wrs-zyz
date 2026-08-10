#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Realman 机械臂观察辣椒果梗仿真
=============================

场景中使用两个绿色长方体模拟一个完整辣椒：
- 下方较长、较粗的绿色长方体表示辣椒主体；
- 上方更细、更短的绿色长方体表示果梗，并连接在主体顶部中间。

机械臂依次观察果梗：
1. 平视：左 30°、左 60°、右 30°、右 60°；
2. 俯视 45°：左 30°、左 60°、右 30°、右 60°。

运行方式：
    python wrs/Fruit_picker_project/Simulation/realman_sim_car.py

按键：
    N / Space  移动到下一个观察视角
    A          自动依次遍历所有观察视角
    H / R      回到 home 位姿
    Q          退出
"""

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
import wrs.motion.probabilistic.rrt_connect as rrtc
import wrs.visualization.panda.world as wd
from wrs.robot_sim.robots.realman.realman_single_arm import Realman
from wrs.visualization.panda.panda3d_utils import VirtualDepthCamera


# HOME_JNT_VALUES = np.array([0.0, -math.pi / 6, math.pi / 2, 0.0, math.pi / 6, 0.0], dtype=np.float64)
HOME_JNT_VALUES = np.array([0.06897541, 1.55276713, -1.98419496, -0.03705334, 1.97552078, 0.01207768], dtype=np.float64)

PEPPER_BODY_POS = np.array([-0.36, 0.0, 0.22], dtype=np.float64)
PEPPER_BODY_SIZE = np.array([0.035, 0.035, 0.16], dtype=np.float64)
STEM_SIZE = np.array([0.012, 0.012, 0.055], dtype=np.float64)
STEM_POS = PEPPER_BODY_POS + np.array([0.0, 0.0, PEPPER_BODY_SIZE[2] / 2.0 + STEM_SIZE[2] / 2.0])
STEM_LOOKAT_POS = STEM_POS.copy()

LEVEL_VIEW_DISTANCE = 0.12
TOP_VIEW_DISTANCE = 0.12
TOP_VIEW_ANGLE_DEG = 45.0
LEVEL_VIEW_ANGLES = [("left_30deg", 30), ("left_60deg", 60), ("right_30deg", -30), ("right_60deg", -60)]
TOP_VIEW_ANGLES = [("left_30deg", 30), ("left_60deg", 50), ("right_30deg", -30), ("right_60deg", -50)]
TCP_XY_FLIP_ROLL_DEG = 180.0
ROLL_SEARCH_DEG = [0, 45, -45, 90, -90, 135, -135, 180]
VIEW_DISTANCE_SCALE_SEARCH = [1.0, 0.85, 0.7, 1.15, 0.55, 1.3, 0.45]

RRT_EXT_DIST = 0.03
RRT_MAX_TIME = 120.0
RRT_SMOOTHING_N_ITER = 10

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs_d405_car"
CAR_STL = _WORKSPACE_ROOT / "wrs" / "LinZhi_picker_project" / "model" / "car.stl"
SHOW_CAR_COLLISION_BOXES = True
D405_RESOLUTION = (640, 480)
D405_FOV_DEG = 87.0
D405_DEPTH_FAR = 1.0
LOC_CAM_POS = np.array([0.04, 0.0, -0.14], dtype=np.float64)
LOC_CAM_ROTMAT = np.column_stack([
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([1.0, 0.0, 0.0]),
]).astype(np.float64)


class RealmanSim:
    """Realman 单臂观察辣椒果梗的 WRS 仿真。"""

    def __init__(self):
        self.base = wd.World(
            cam_pos=rm.vec(1.25, 1.1, 0.9),
            lookat_pos=rm.vec(-0.25, 0.0, 0.25),
        )
        self.robot = Realman(enable_cc=True)
        self.planner = rrtc.RRTConnect(self.robot)
        self.obstacle_list = []
        self.robotcar_model = None
        self.robot_mesh = None
        self.robot_stick = None
        self.camera_mesh = None
        self.d405 = None
        self.path_meshes = []
        self.current_view_idx = -1
        self.view_targets = []

        self._prepare_output_dirs()
        self._build_scene()
        self.view_targets = self._build_view_targets()
        self._attach_view_debug_models()
        self._init_d405_camera()
        self._bind_keys()
        self.goto_conf(HOME_JNT_VALUES, name="home", use_rrt=True)

    @staticmethod
    def _make_tcp_rotmat_looking_at(tcp_pos, lookat_pos):
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

    @staticmethod
    def _roll_tcp_rotmat(tcp_rotmat, roll_rad):
        return tcp_rotmat @ rm.rotmat_from_axangle([0, 0, 1], roll_rad)

    @staticmethod
    def _safe_name(name):
        return re.sub(r"[^0-9A-Za-z_-]+", "_", name)

    def _prepare_output_dirs(self):
        for subdir in ("rgb", "depth_raw", "depth_png", "depth_vis"):
            (OUTPUT_ROOT / subdir).mkdir(parents=True, exist_ok=True)

    def _tcp_to_cam(self, tcp_pos, tcp_rotmat):
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
        self.camera_mesh = self.d405.gen_meshmodel(alpha=0.45)
        self.camera_mesh.attach_to(self.base)

    def _refresh_camera_model(self):
        if self.d405 is None:
            return
        if self.camera_mesh is not None:
            self.camera_mesh.detach()
        self.camera_mesh = self.d405.gen_meshmodel(alpha=0.45)
        self.camera_mesh.attach_to(self.base)

    def _save_d405_capture(self, view_idx, view_name):
        if self.d405 is None:
            return
        img_rgb = self.d405.get_rgb_image()
        img_depth = self.d405.get_depth_image()
        if img_rgb is None or img_depth is None:
            print(f"       [Capture][warn] {view_name}: 未获取到 RGB 或深度图。")
            return

        stem = f"{view_idx + 1:02d}_{self._safe_name(view_name)}"
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
        print(f"       [Capture] D405 RGB/深度图已保存: {OUTPUT_ROOT}")

    def _solve_view_ik(self, target, seed_jnt_values):
        base_view_vec = np.asarray(target["view_vec"], dtype=np.float64)
        for scale in VIEW_DISTANCE_SCALE_SEARCH:
            tcp_pos = STEM_LOOKAT_POS + base_view_vec * scale
            base_rotmat = self._make_tcp_rotmat_looking_at(tcp_pos, STEM_LOOKAT_POS)
            for roll_deg in ROLL_SEARCH_DEG:
                tcp_rotmat = self._roll_tcp_rotmat(base_rotmat, math.radians(roll_deg))
                goal_conf = self.robot.ik(
                    tgt_pos=tcp_pos,
                    tgt_rotmat=tcp_rotmat,
                    seed_jnt_values=seed_jnt_values,
                    toggle_dbg=False,
                )
                if goal_conf is not None:
                    target["tcp_pos"] = tcp_pos
                    target["tcp_rotmat"] = tcp_rotmat
                    target["ik_scale"] = scale
                    target["ik_roll_deg"] = roll_deg
                    return np.asarray(goal_conf, dtype=np.float64)
        return None

    def _attach_robotcar(self):
        if not CAR_STL.is_file():
            print(f"[Warn] 未找到小车模型: {CAR_STL}")
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
        print(f"[Setup] 小车模型已加载: {CAR_STL}")
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

        pepper_body = mcm.gen_box(
            xyz_lengths=PEPPER_BODY_SIZE,
            pos=PEPPER_BODY_POS,
            rotmat=np.eye(3),
            rgb=np.array([0.0, 0.75, 0.12]),
            alpha=1.0,
        )
        pepper_body.attach_to(self.base)

        pepper_stem = mcm.gen_box(
            xyz_lengths=STEM_SIZE,
            pos=STEM_POS,
            rotmat=np.eye(3),
            rgb=np.array([0.0, 0.45, 0.05]),
            alpha=1.0,
        )
        pepper_stem.attach_to(self.base)

        mgm.gen_sphere(
            pos=STEM_LOOKAT_POS,
            radius=0.009,
            rgb=np.array([1.0, 0.1, 0.1]),
            alpha=0.9,
        ).attach_to(self.base)

    def _build_view_targets(self):
        targets = []
        front_vec = np.array([LEVEL_VIEW_DISTANCE, 0.0, 0.0], dtype=np.float64)

        for angle_name, angle_deg in LEVEL_VIEW_ANGLES:
            rot_z = rm.rotmat_from_axangle([0, 0, 1], math.radians(angle_deg))
            tcp_pos = STEM_LOOKAT_POS + rot_z @ front_vec
            tcp_rotmat = self._make_tcp_rotmat_looking_at(tcp_pos, STEM_LOOKAT_POS)
            targets.append({
                "name": f"level_{angle_name}",
                "view_vec": rot_z @ front_vec,
                "tcp_pos": tcp_pos,
                "tcp_rotmat": tcp_rotmat,
                "jnt_values": None,
            })

        top_horizontal_distance = TOP_VIEW_DISTANCE
        top_z_offset = top_horizontal_distance * math.tan(math.radians(TOP_VIEW_ANGLE_DEG))
        top_front_vec = np.array([top_horizontal_distance, 0.0, top_z_offset], dtype=np.float64)

        for angle_name, angle_deg in TOP_VIEW_ANGLES:
            rot_z = rm.rotmat_from_axangle([0, 0, 1], math.radians(angle_deg))
            tcp_pos = STEM_LOOKAT_POS + rot_z @ top_front_vec
            tcp_rotmat = self._make_tcp_rotmat_looking_at(tcp_pos, STEM_LOOKAT_POS)
            targets.append({
                "name": f"top45_{angle_name}",
                "view_vec": rot_z @ top_front_vec,
                "tcp_pos": tcp_pos,
                "tcp_rotmat": tcp_rotmat,
                "jnt_values": None,
            })

        return targets

    def _attach_view_debug_models(self):
        mgm.gen_dashed_stick(
            spos=STEM_LOOKAT_POS - np.array([0.0, 0.0, 0.08]),
            epos=STEM_LOOKAT_POS + np.array([0.0, 0.0, 0.08]),
            radius=0.0015,
            rgb=np.array([1.0, 0.1, 0.1]),
            alpha=0.65,
        ).attach_to(self.base)

        for target in self.view_targets:
            mgm.gen_frame(pos=target["tcp_pos"], rotmat=target["tcp_rotmat"], ax_length=0.055).attach_to(self.base)
            mgm.gen_dashed_stick(
                spos=target["tcp_pos"],
                epos=STEM_LOOKAT_POS,
                radius=0.001,
                rgb=np.array([1.0, 0.8, 0.05]),
                alpha=0.35,
            ).attach_to(self.base)

    def _bind_keys(self):
        self.base.accept("n", self.next_view)
        self.base.accept("space", self.next_view)
        self.base.accept("a", self.auto_view_all)
        self.base.accept("h", lambda: self.goto_conf(HOME_JNT_VALUES, name="home", use_rrt=True))
        self.base.accept("r", lambda: self.goto_conf(HOME_JNT_VALUES, name="home", use_rrt=True))
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

    def _validate_path_collision(self, path, granularity=0.01):
        if path is None or len(path) == 0:
            return False
        for seg_idx in range(len(path) - 1):
            start = np.asarray(path[seg_idx], dtype=np.float64)
            goal = np.asarray(path[seg_idx + 1], dtype=np.float64)
            diff = goal - start
            n_step = max(2, int(np.linalg.norm(diff) / granularity))
            for interp_idx in range(n_step + 1):
                conf = start + diff * (interp_idx / n_step)
                self.robot.goto_given_conf(conf)
                if self.robot.is_collided(obstacle_list=self.obstacle_list):
                    print(f"       [Collision] path segment {seg_idx}, interp {interp_idx}/{n_step} 与小车STL碰撞。")
                    return False
        final_conf = np.asarray(path[-1], dtype=np.float64)
        self.robot.goto_given_conf(final_conf)
        if self.robot.is_collided(obstacle_list=self.obstacle_list):
            print("       [Collision] final_conf 与小车STL碰撞。")
            return False
        return True

    def _plan_path(self, start_conf, goal_conf):
        if np.linalg.norm(start_conf - goal_conf) < 1e-6:
            return [goal_conf]

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
        return path

    def goto_conf(self, goal_conf, name="target", use_rrt=True):
        goal_conf = np.asarray(goal_conf, dtype=np.float64)
        start_conf = np.asarray(self.robot.get_jnt_values(), dtype=np.float64)
        self._clear_path_meshes()
        print(f"\n[Move] {name}")
        print(f"       goal_joints = {np.round(goal_conf, 6)}")

        if use_rrt:
            path = self._plan_path(start_conf, goal_conf)
            if path is None:
                print("       [RRT][fail] 未找到避开小车STL的安全路径，已取消本次移动。")
                self.robot.goto_given_conf(start_conf)
                self._update_d405_from_robot(start_conf)
                self._refresh_robot_model()
                self._refresh_camera_model()
                return False
        else:
            path = [goal_conf]

        self._show_path(path)
        self.robot.goto_given_conf(np.asarray(path[-1], dtype=np.float64))
        self._update_d405_from_robot(np.asarray(path[-1], dtype=np.float64))
        tcp_pos, tcp_rotmat = self.robot.fk(np.asarray(path[-1], dtype=np.float64), update=False)
        is_collided = self.robot.is_collided(obstacle_list=self.obstacle_list)
        print(f"       path_points = {len(path)}")
        print(f"       tcp_pos = {np.round(tcp_pos, 4)}")
        print(f"       tcp_z_axis = {np.round(tcp_rotmat[:, 2], 4)}")
        print(f"       lookat_stem = {np.round(STEM_LOOKAT_POS, 4)}")
        print(f"       collided = {is_collided}")
        self._refresh_robot_model()
        self._refresh_camera_model()
        if is_collided:
            print("       [Collision] 目标位姿与小车STL碰撞，已取消本次移动。")
            self.robot.goto_given_conf(start_conf)
            self._update_d405_from_robot(start_conf)
            self._refresh_robot_model()
            self._refresh_camera_model()
            return False
        return True

    def goto_view(self, view_idx):
        target = self.view_targets[view_idx]
        start_conf = np.asarray(self.robot.get_jnt_values(), dtype=np.float64)
        self._clear_path_meshes()
        print(f"\n[View] {view_idx + 1}/{len(self.view_targets)} {target['name']}")
        print(f"       target_tcp_pos = {np.round(target['tcp_pos'], 4)}")

        if target["jnt_values"] is None:
            goal_conf = self._solve_view_ik(target, start_conf)
            if goal_conf is None:
                print("       [IK][fail] 此观察方向在多roll角和多距离搜索后仍无解。")
                return False
            target["jnt_values"] = np.asarray(goal_conf, dtype=np.float64)
            print(f"       ik_scale = {target.get('ik_scale', 1.0):.2f}, ik_roll_deg = {target.get('ik_roll_deg', 0)}")

        if target["jnt_values"] is None:
            goal_conf = self.robot.ik(
                tgt_pos=target["tcp_pos"],
                tgt_rotmat=target["tcp_rotmat"],
                seed_jnt_values=start_conf,
                toggle_dbg=False,
            )
            if goal_conf is None:
                print("       [IK][fail] 此观察视角无解。")
                return False
            target["jnt_values"] = np.asarray(goal_conf, dtype=np.float64)
        else:
            goal_conf = np.asarray(target["jnt_values"], dtype=np.float64)

        success = self.goto_conf(goal_conf, name=target["name"], use_rrt=True)
        if success:
            self.current_view_idx = view_idx
            self._save_d405_capture(view_idx, target["name"])
        return success

    def next_view(self):

        for offset in range(1, len(self.view_targets) + 1):
            next_idx = (self.current_view_idx + offset) % len(self.view_targets)
            if self.goto_view(next_idx):
                return
        print("\n[Warn] 所有观察视角都没有可用 IK 或路径。")

    def auto_view_all(self):
        start_idx = self.current_view_idx + 1
        for view_idx in range(start_idx, len(self.view_targets)):
            if not self.goto_view(view_idx):
                print(f"\n[Skip] {self.view_targets[view_idx]['name']} 无可用 IK 或路径，继续下一个视角。")
                continue
        else:
            print("\n[Done] 所有果梗观察视角已遍历完成。")

    def run(self):
        print("\n" + "=" * 72)
        print("  Realman 机械臂安装在小车上观察辣椒果梗仿真")
        print("  辣椒模型 : 绿色细长长方体 + 顶部更细短长方体果梗")
        print("  观察顺序 : 平视左30、左60、右30、右60；俯视45度左30、左60、右30、右60")
        print("  N / Space : 移动到下一个观察视角")
        print("  A         : 自动遍历所有观察视角")
        print("  H / R     : 回到 home 位姿")
        print("  Q         : 退出")
        print(f"  观察视角数: {len(self.view_targets)}")
        print("=" * 72 + "\n")
        self.base.run()


def main():
    sim = RealmanSim()
    sim.run()


if __name__ == "__main__":
    main()
