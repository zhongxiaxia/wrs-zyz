#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Realman 机械臂多视角观察仿真
==========================

在 WRS 仿真环境中加载 Realman 机械臂和小车底座模型。机械臂先到达指定
初始关节位姿，再以初始 TCP 点向世界 -X 方向 ORBIT_RADIUS 的点为圆心、
世界 Z 轴为旋转轴，依次通过 IK 和 RRT 到达水平圆弧与 45 度俯视圆弧
观察位姿。每到达一个位姿后，使用安装在末端的虚拟深度相机拍摄并保存：

- 彩色图：outputs/rgb/<序号>_<名称>.png
- 深度图原始值：outputs/depth_raw/<序号>_<名称>.npy
- 深度图 16bit PNG：outputs/depth_png/<序号>_<名称>.png，单位 mm
- 深度伪彩色图：outputs/depth_vis/<序号>_<名称>.png

运行方式：
    python wrs/Fruit_picker_project/Simulation/realman_observe_capture.py

按键：
    N / 空格  移动到下一个观察视角并拍照
    A         自动依次遍历剩余观察视角并拍照
    R         重新拍摄当前位姿
    Q         退出
"""

import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# 允许从本文件直接运行时正确导入 wrs 包
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


# ─── 相机在 TCP 坐标系下的固连关系，沿用 pepper 示例中的眼在手上配置 ───
LOC_CAM_POS = np.array([0.04, 0.0, -0.14], dtype=np.float64)
LOC_CAM_ROTMAT = np.column_stack([
    np.array([0.0, 1.0, 0.0]),   # cam_right   = TCP Y
    np.array([0.0, 0.0, 1.0]),   # cam_forward = TCP Z
    np.array([1.0, 0.0, 0.0]),   # cam_up      = TCP X
]).astype(np.float64)


INITIAL_JNT_VALUES = np.array([-0.02916445, 2.01219011, -2.09998015, -0.04829326, 1.58926189, -0.02731440], dtype=np.float64)
ORBIT_RADIUS = 0.02
ORBIT_CENTER_OFFSET = np.array([-ORBIT_RADIUS, 0.0, 0.0], dtype=np.float64)
# 平视角
LEVEL_ANGLES_DEG = [30,-30,50,-50]
# 俯视角
TOP_ANGLES_DEG = [30,-30,40,-40]
TOP_VIEW_ANGLE_DEG = 30.0
# 俯视角半径配置：0°/±30°用原半径，±60°用更大半径（解决IK无解）
TOP_NORMAL_RADIUS = ORBIT_RADIUS  # 默认半径0.02
TOP_BIG_RADIUS = 0.03             # 专用半径
RRT_EXT_DIST = 0.1
RRT_MAX_TIME = 300.0
RRT_SMOOTHING_N_ITER = 50


OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
CAR_STL = _WORKSPACE_ROOT / "wrs" / "LinZhi_picker_project" / "model" / "car.stl"
PEPPER_OFFSET = np.array([-0.60, 0.0, 0.00], dtype=np.float64)
PEPPER_MODEL_CANDIDATE_DIRS = [
    _WORKSPACE_ROOT / "0000_models" / "pepper",
]


def tcp_to_cam(tcp_pos: np.ndarray, tcp_rotmat: np.ndarray):
    """由 TCP 世界位姿计算相机世界位姿。"""
    cam_pos = tcp_pos + tcp_rotmat @ LOC_CAM_POS
    cam_rotmat = tcp_rotmat @ LOC_CAM_ROTMAT
    return cam_pos, cam_rotmat


def safe_name(name: str) -> str:
    """生成适合保存文件的名称。"""
    return re.sub(r"[^0-9A-Za-z_\-]+", "_", name)


def prepare_output_dirs(root: Path):
    for subdir in ("rgb", "depth_raw", "depth_png", "depth_vis"):
        (root / subdir).mkdir(parents=True, exist_ok=True)


def attach_robotcar(base):
    """加载并附加用户指定的小车底座/障碍物模型。"""
    robotcar_model = mcm.CollisionModel(
        str(CAR_STL),
        rgb=np.array([0.6, 0.6, 0.6]),
        alpha=0.9,
    )
    robotcar_model.pos = np.array([-0.1, 0.0, -0.08])
    robotcar_model.rotmat = rm.rotmat_from_axangle([0, 1, 0], -math.pi * 90 / 180)
    robotcar_model.attach_to(base)
    return robotcar_model


def resolve_pepper_model_dir() -> Path | None:
    """查找 0000_examples/pepper 示例中使用的辣椒 STL 模型目录。"""
    for model_dir in PEPPER_MODEL_CANDIDATE_DIRS:
        if (model_dir / "pepper.stl").is_file():
            return model_dir
    return None


def attach_pepper_models(base):
    """加载 0000_examples/pepper 示例中的辣椒、果柄和背景模型。"""
    model_dir = resolve_pepper_model_dir()
    if model_dir is None:
        print("[Warn] 未找到 pepper.stl，已跳过辣椒模型加载。")
        print(f"       已检查目录: {[str(p) for p in PEPPER_MODEL_CANDIDATE_DIRS]}")
        return []

    pepper_model = mgm.GeometricModel(str(model_dir / "pepper.stl"), rgb=np.array([0.0, 1.0, 0.0]), alpha=1.0)
    stem_model = mgm.GeometricModel(str(model_dir / "stem.stl"), rgb=np.array([0.0, 0.82, 0.0]), alpha=1.0)
    other_model = mgm.GeometricModel(str(model_dir / "other.stl"), rgb=np.array([0.53, 0.81, 0.92]), alpha=1.0)

    models = [pepper_model, stem_model, other_model]
    for model in models:
        model.pos = PEPPER_OFFSET
        model.attach_to(base)
    print(f"[Setup] 辣椒模型已加载: {model_dir}")
    return models


def update_camera_from_robot(robot: Realman, vcam: VirtualDepthCamera, jnt_values: np.ndarray):
    """机械臂到达指定关节值后，同步虚拟相机位姿。"""
    tcp_pos, tcp_rotmat = robot.fk(jnt_values, update=False)
    cam_pos, cam_rotmat = tcp_to_cam(tcp_pos, tcp_rotmat)
    vcam.cam_pos = cam_pos
    vcam.cam_rotmat = cam_rotmat
    return cam_pos, cam_rotmat


def make_tcp_rotmat_looking_at(
        tcp_pos: np.ndarray,
        lookat_pos: np.ndarray,
        reference_rotmat: np.ndarray | None = None):
    """构造 TCP 位姿，使相机前向轴看向指定点。"""
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


def build_orbit_targets(robot: Realman, initial_jnt_values: np.ndarray):
    tcp_pos0, tcp_rot0 = robot.fk(initial_jnt_values, update=False)
    circle_center = tcp_pos0 + ORBIT_CENTER_OFFSET
    radius_vec = tcp_pos0 - circle_center
    top_view_height = np.linalg.norm(radius_vec[:2]) * math.tan(math.radians(TOP_VIEW_ANGLE_DEG))
    top_view_z_offset = np.array([0.0, 0.0, top_view_height], dtype=np.float64)

    targets = [{
        "name": "initial_0deg",
        "angle_deg": 0,
        "group": "level",
        "tcp_pos": tcp_pos0,
        "tcp_rotmat": tcp_rot0,
        "jnt_values": initial_jnt_values.copy(),
    }]
    # 生成平视角（不变）
    for angle_deg in LEVEL_ANGLES_DEG:
        rot_z = rm.rotmat_from_axangle([0, 0, 1], math.radians(angle_deg))
        angle_prefix = "plus" if angle_deg > 0 else "minus"
        targets.append({
            "name": f"{angle_prefix}_{abs(angle_deg)}deg",
            "angle_deg": angle_deg,
            "group": "level",
            "tcp_pos": circle_center + rot_z @ radius_vec,
            "tcp_rotmat": rot_z @ tcp_rot0,
            "jnt_values": None,
        })
    # 生成俯视角0°
    top_tcp_pos0 = circle_center + radius_vec + top_view_z_offset
    top_tcp_rot0 = make_tcp_rotmat_looking_at(top_tcp_pos0, circle_center, reference_rotmat=tcp_rot0)
    targets.append({
        "name": "top45_0deg",
        "angle_deg": 0,
        "group": "top45",
        "tcp_pos": top_tcp_pos0,
        "tcp_rotmat": top_tcp_rot0,
        "jnt_values": None,
    })
    # 生成俯视角 ±30°/±60°（±60°用大半径）
    for angle_deg in TOP_ANGLES_DEG:
        rot_z = rm.rotmat_from_axangle([0, 0, 1], math.radians(angle_deg))
        angle_prefix = "plus" if angle_deg > 0 else "minus"

        # 核心逻辑：专用半径，其他用默认半径
        if abs(angle_deg) == 45:
            use_radius = TOP_BIG_RADIUS
        else:
            use_radius = TOP_NORMAL_RADIUS

        # 计算缩放后的半径向量
        unit_radius_vec = radius_vec / np.linalg.norm(radius_vec)
        scaled_radius_vec = unit_radius_vec * use_radius

        # 计算目标TCP位置
        tcp_pos = circle_center + rot_z @ scaled_radius_vec + top_view_z_offset
        # 👇 新增：仅俯视 ±45° 视角，Z轴向下移动 0.05 米
        if abs(angle_deg) == 40:
            tcp_pos[2] -= 0.1
        targets.append({
            "name": f"top45_{angle_prefix}_{abs(angle_deg)}deg",
            "angle_deg": angle_deg,
            "group": "top45",
            "tcp_pos": tcp_pos,
            "tcp_rotmat": make_tcp_rotmat_looking_at(tcp_pos, circle_center, reference_rotmat=rot_z @ tcp_rot0),
            "jnt_values": None,
        })

    # 必须返回两个值，修复报错的核心！
    return targets, circle_center


def attach_orbit_debug_models(base, targets, circle_center: np.ndarray):
    mgm.gen_sphere(pos=circle_center, radius=0.012, rgb=np.array([1.0, 0.2, 0.1]), alpha=0.9).attach_to(base)
    mgm.gen_dashed_stick(
        spos=circle_center - np.array([0.0, 0.0, 0.2]),
        epos=circle_center + np.array([0.0, 0.0, 0.2]),
        radius=0.002,
        rgb=np.array([1.0, 0.2, 0.1]),
        alpha=0.8,
    ).attach_to(base)
    mgm.gen_dashed_torus(
        axis=np.array([0, 0, 1]),
        center=circle_center,
        major_radius=ORBIT_RADIUS,
        minor_radius=0.0015,
        rgb=np.array([0.1, 0.55, 1.0]),
        alpha=0.45,
    ).attach_to(base)
    top_targets = [target for target in targets if target.get("group") == "top45"]
    if top_targets:
        top_circle_center = circle_center + np.array([0.0, 0.0, top_targets[0]["tcp_pos"][2] - circle_center[2]])
        mgm.gen_dashed_torus(
            axis=np.array([0, 0, 1]),
            center=top_circle_center,
            major_radius=ORBIT_RADIUS,
            minor_radius=0.0015,
            rgb=np.array([0.1, 0.9, 0.35]),
            alpha=0.45,
        ).attach_to(base)
    for target in targets:
        mgm.gen_frame(pos=target["tcp_pos"], rotmat=target["tcp_rotmat"], ax_length=0.05).attach_to(base)


def save_capture(vcam: VirtualDepthCamera, pose_idx: int, pose_name: str, output_root: Path):
    """保存当前虚拟相机的 RGB 图和深度图。"""
    img_rgb = vcam.get_rgb_image()
    img_depth = vcam.get_depth_image()

    if img_rgb is None or img_depth is None:
        print(f"[Warn] {pose_name}: 未能获取 RGB 或深度图，跳过保存。")
        return

    stem = f"{pose_idx + 1:02d}_{safe_name(pose_name)}"

    rgb_to_save = img_rgb[:, :, :3]
    if rgb_to_save.dtype != np.uint8:
        rgb_to_save = np.clip(rgb_to_save, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_root / "rgb" / f"{stem}.png"), cv2.cvtColor(rgb_to_save, cv2.COLOR_RGB2BGR))

    depth = np.asarray(img_depth, dtype=np.float32)
    np.save(output_root / "depth_raw" / f"{stem}.npy", depth)

    valid_depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth_mm = np.clip(valid_depth * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    cv2.imwrite(str(output_root / "depth_png" / f"{stem}.png"), depth_mm)

    depth_vis_src = np.clip(valid_depth, 0.0, 3.0)
    depth_vis = cv2.normalize(depth_vis_src, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_root / "depth_vis" / f"{stem}.png"), depth_vis)

    print(f"[Save] {pose_name}: RGB 和深度图已保存到 {output_root}")


def main():
    prepare_output_dirs(OUTPUT_ROOT)

    base = wd.World(
        cam_pos=rm.vec(1.5, 1.0, 1.2),
        lookat_pos=rm.vec(-0.15, 0.0, 0.15),
    )
    mgm.gen_frame().attach_to(base)
    box_model = attach_robotcar(base)
    pepper_models = attach_pepper_models(base)
    _ = pepper_models

    robot = Realman(enable_cc=True)
    rrtc_planner = rrtc.RRTConnect(robot)
    orbit_targets, circle_center = build_orbit_targets(robot, INITIAL_JNT_VALUES)
    attach_orbit_debug_models(base, orbit_targets, circle_center)
    current_idx = [0]

    robot.goto_given_conf(INITIAL_JNT_VALUES)
    robot_vis_node = [robot.gen_meshmodel(alpha=0.55, toggle_tcp_frame=True)]
    robot_vis_node[0].attach_to(base)

    tcp_pos0, tcp_rot0 = robot.fk(INITIAL_JNT_VALUES, update=False)
    cam_pos0, cam_rot0 = tcp_to_cam(tcp_pos0, tcp_rot0)
    vcam = VirtualDepthCamera(
        cam_pos=cam_pos0,
        lookat_pos=cam_pos0 + cam_rot0[:, 1],
        resolution=(640, 480),
        screen_size=np.array((0.05, 0.05)),
        fov=60,
        depth_far=3.0,
    )
    vcam.cam_rotmat = cam_rot0
    vcam_vis_node = [vcam.gen_meshmodel(alpha=0.45)]
    vcam_vis_node[0].attach_to(base)

    def refresh_scene():
        robot_vis_node[0].detach()
        robot_vis_node[0] = robot.gen_meshmodel(alpha=0.55, toggle_tcp_frame=True)
        robot_vis_node[0].attach_to(base)

        vcam_vis_node[0].detach()
        vcam_vis_node[0] = vcam.gen_meshmodel(alpha=0.45)
        vcam_vis_node[0].attach_to(base)

    def plan_to_conf(start_conf: np.ndarray, goal_conf: np.ndarray):
        if np.linalg.norm(start_conf - goal_conf) < 1e-6:
            return [goal_conf]
        mot_data = rrtc_planner.plan(
            start_conf=start_conf,
            goal_conf=goal_conf,
            obstacle_list=[box_model],
            ext_dist=RRT_EXT_DIST,
            max_time=RRT_MAX_TIME,
            smoothing_n_iter=RRT_SMOOTHING_N_ITER,
        )
        if mot_data is None:
            return None
        return mot_data.jv_list

    def move_to_pose(idx: int, capture: bool = True):
        target = orbit_targets[idx]
        pose_name = target["name"]
        start_conf = np.asarray(robot.get_jnt_values(), dtype=np.float64).copy()
        print(f"\n[Move] {idx + 1}/{len(orbit_targets)}  {pose_name}")
        print(f"       target_tcp = {np.round(target['tcp_pos'], 4)}")

        if target["jnt_values"] is None:
            goal_conf = robot.ik(
                tgt_pos=target["tcp_pos"],
                tgt_rotmat=target["tcp_rotmat"],
                seed_jnt_values=start_conf,
            )
            if goal_conf is None:
                print(f"       [IK][fail] {pose_name}")
                robot.goto_given_conf(start_conf)
                update_camera_from_robot(robot, vcam, start_conf)
                refresh_scene()
                return False
            target["jnt_values"] = np.asarray(goal_conf, dtype=np.float64).copy()
        else:
            goal_conf = np.asarray(target["jnt_values"], dtype=np.float64).copy()

        print(f"       goal_joints = {np.round(goal_conf, 6)}")
        path = plan_to_conf(start_conf=start_conf, goal_conf=goal_conf)
        if path is None:
            print(f"       [RRT][fail] {pose_name}")
            robot.goto_given_conf(start_conf)
            update_camera_from_robot(robot, vcam, start_conf)
            refresh_scene()
            return False

        final_conf = np.asarray(path[-1], dtype=np.float64)
        robot.goto_given_conf(final_conf)
        if robot.is_collided(obstacle_list=[box_model]):
            print("       [Warn] 当前观察位与小车模型发生碰撞，仍继续拍照用于调试。")

        cam_pos, _ = update_camera_from_robot(robot, vcam, final_conf)
        tcp_pos, _ = robot.fk(final_conf, update=False)
        print(f"       path_points = {len(path)}")
        print(f"       reached_tcp = {np.round(tcp_pos, 4)}")
        print(f"       camera = {np.round(cam_pos, 4)}")

        refresh_scene()
        if capture:
            save_capture(vcam, idx, pose_name, OUTPUT_ROOT)
        return True

    def capture_current():
        idx = current_idx[0]
        pose_name = orbit_targets[idx]["name"]
        save_capture(vcam, idx, pose_name, OUTPUT_ROOT)

    def next_pose():
        next_idx = (current_idx[0] + 1) % len(orbit_targets)
        if move_to_pose(next_idx, capture=True):
            current_idx[0] = next_idx

    def auto_capture_all():
        start = current_idx[0] + 1
        all_success = True
        for idx in range(start, len(orbit_targets)):
            if not move_to_pose(idx, capture=True):
                print(f"\n[Stop] Failed at {orbit_targets[idx]['name']}.")
                all_success = False
                break
            current_idx[0] = idx
        if all_success:
            print("\n[Done] 圆弧观察位姿已遍历完成。")

    base.accept("n", next_pose)
    base.accept("space", next_pose)
    base.accept("a", auto_capture_all)
    base.accept("r", capture_current)
    base.accept("q", base.destroy)

    move_to_pose(0, capture=True)

    print("\n" + "=" * 68)
    print("  Realman 多视角观察仿真")
    print("  N / Space : 移动到下一个观察视角并拍照")
    print("  A         : 自动遍历剩余观察视角并拍照")
    print("  R         : 重新拍摄当前位姿")
    print("  Q         : 退出")
    print(f"  视角数量   : {len(orbit_targets)}")
    print(f"  输出目录   : {OUTPUT_ROOT}")
    print("=" * 68 + "\n")

    base.run()


if __name__ == "__main__":
    main()
