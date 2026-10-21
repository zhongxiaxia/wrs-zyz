#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/3/28
"""
辣椒机器人眼在手上 NBV 仿真
============================
复用 v1 的 Realman、小车底座、整株辣椒和 D405 手眼标定，
（眼在手上）。每按一次 M 键，系统：
  1. 用当前帧更新体素占据网格
  2. 以目标辣椒为中心在圆周/圆盘上生成若干候选相机位姿
  3. 计算每个候选视角的信息增益（raycast）
  4. 按增益依次检查 IK/FK、关节限位及运动中的自碰撞、小车/植株碰撞
  5. 移动机械臂到该关节配置，同步更新虚拟相机位姿

窗口说明：
  主窗口   —— Panda3D 3D 场景（小车、机械臂、整株辣椒及 D405 相机）
  左副窗口 —— 点云俯视（当前帧点云，红=辣椒果实，绿=其余）
  右副窗口 —— 体素可视化（红=占据目标区, 蓝=占据背景, 绿=空闲）
  cv2 窗口 —— 虚拟深度相机图像（RGB | 深度伪彩色）

键盘：
  M  — 执行一步 NBV 规划并移动机械臂
  Q  — 退出
"""

import os
import sys

_WORKSPACE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

import numpy as np
import cv2

# ─── WRS 导入 ──────────────────────────────────────────────────────────────
import wrs.visualization.panda.world as wd
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
from wrs.visualization.panda.panda3d_utils import VirtualDepthCamera, ExtraWindow

# ─── 本目录本地模块 ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bbox
import voxel_grid
import policy
from voxel_viz import VoxelGridVisualizer
from pepper_cloud import PointCloudVisualizer
from scene_assets import (
    create_robot_setup, load_plant_scene, gen_d405_model,
    FLOOR_SIZE, FLOOR_CENTER, D405_WIDTH_M, D405_HEIGHT_M,
    load_tcp_from_camera, tcp_to_camera, camera_to_tcp,
)

# ═══════════════════════════════════════════════════════════════════════════
#  全局常量
# ═══════════════════════════════════════════════════════════════════════════

# Share v1's measured TCP-from-camera calibration, including its CV/Panda basis.
_TCP_FROM_CAMERA = load_tcp_from_camera()


def tcp_to_cam(tcp_pos: np.ndarray, tcp_rotmat: np.ndarray):
    """由 TCP 世界位姿计算相机世界位姿。"""
    return tcp_to_camera(tcp_pos, tcp_rotmat, _TCP_FROM_CAMERA)


def cam_to_tcp(cam_pos: np.ndarray, cam_rotmat: np.ndarray):
    """由目标相机世界位姿反推所需 TCP 世界位姿（用于 IK）。"""
    return camera_to_tcp(cam_pos, cam_rotmat, _TCP_FROM_CAMERA)


def look_at_rotmat(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    构造一个相机旋转矩阵（列: [right, forward, up]），使相机朝向 target。
    与 NBVPolicy._look_at 保持相同约定。
    """
    fwd = target - cam_pos
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up)
    return np.column_stack([right, fwd, up]).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # v1's mobile Realman, measured initial joints and eye-in-hand calibration.
    adapter, car, scene_cfg = create_robot_setup()
    robot = adapter.robot
    cur_jnt = [adapter.current_joints.copy()]
    cam_pos0, cam_rot0 = adapter.camera_pose()

    base = wd.World(cam_pos=rm.vec(2.35, 2.10, 1.65),
                    lookat_pos=rm.vec(0.25, 0.0, 0.30), fov=45,
                    w=1280, h=800)
    base.setBackgroundColor(0.92, 0.94, 0.96)
    floor = mgm.gen_box(xyz_lengths=FLOOR_SIZE, pos=FLOOR_CENTER,
                        rgb=np.array([0.72, 0.72, 0.70]), alpha=1.0)
    floor.attach_to(base)
    car.attach_to(base)
    obj1, obj2, plant_models = load_plant_scene(base, scene_cfg.object_offset)
    adapter.set_obstacles([car, *plant_models])
    if not adapter.current_configuration_is_feasible():
        raise RuntimeError(f"Initial robot/plant scene rejected: {adapter.last_check.reason}")
    obj1_min, obj1_max = bbox.world_aabb_from_obj(obj1)
    obj2_min, obj2_max = bbox.world_aabb_from_obj(obj2)
    pepper_center = 0.5 * (obj1_min + obj1_max)
    print(f"[Setup] v1 plant offset = {scene_cfg.object_offset.round(4)}")
    print(f"[Setup] Target fruit AABB: {obj1_min.round(4)} ~ {obj1_max.round(4)}")

    robot_vis_node = [robot.gen_meshmodel(alpha=0.78)]
    robot_vis_node[0].attach_to(base)
    resolution = (scene_cfg.image_width, scene_cfg.image_height)
    vcam = VirtualDepthCamera(
        cam_pos=cam_pos0, lookat_pos=scene_cfg.target_world,
        resolution=resolution, screen_size=np.array([D405_WIDTH_M, D405_HEIGHT_M]),
        fov=scene_cfg.horizontal_fov_deg, depth_far=2.0, w_base=base)
    vcam.cam_rotmat = cam_rot0

    global_min = np.minimum(obj1_min, obj2_min) - 0.02
    global_max = np.maximum(obj1_max, obj2_max) + 0.02
    voxel_size = 0.01
    dims = np.maximum(
        np.ceil((global_max - global_min) / voxel_size).astype(int), 1
    )
    vg = voxel_grid.VoxelGrid3D(
        origin=global_min,
        voxel_size=voxel_size,
        dims=dims,
        p0=0.5,
        p_hit=0.7,
        p_miss=0.4,
    )

    # ── 8. NBV 策略 ────────────────────────────────────────────────────────
    K        = vcam.intrinsics
    img_size = (resolution[0], resolution[1])
    nbv_policy = policy.NBVPolicy(
        vg=vg,
        obj_min=obj1_min,
        obj_max=obj1_max,
        K=K,
        img_size=img_size,
        vcam=vcam,
    )

    # ── 9. 可视化辅助节点 ─────────────────────────────────────────────────
    vcam_vis_node = [gen_d405_model(vcam.cam_pos, vcam.cam_rotmat)]
    vcam_vis_node[0].attach_to(base)

    # 左副窗口：实时点云
    ew_pcd = ExtraWindow(
        base,
        window_title="Point Cloud (eye-in-hand)",
        cam_pos=pepper_center + np.array([0.5, 0.0, 0.5]),
        lookat_pos=pepper_center,
        w=640, h=480,
    )
    ew_pcd.set_origin(np.array([0, 40]))
    pc_viz = PointCloudVisualizer(ew_pcd)

    # 右副窗口：体素网格
    vox_viz = VoxelGridVisualizer(
        base,
        window_origin=(640, 40),
        size=(640, 480),
        cam_pos=pepper_center + np.array([0.5, -0.5, 0.3]),
        lookat=pepper_center,
        voxel_size=vg.voxel_size,
        max_voxels=50_000,
    )

    # ── 10. NBV 步骤逻辑 ──────────────────────────────────────────────────

    def _update_cam_from_robot():
        """根据当前关节配置刷新虚拟相机位姿。"""
        c_pos, c_rot = adapter.camera_pose()
        vcam.cam_pos = c_pos
        vcam.cam_rotmat = c_rot

    def step_nbv():
        """
        执行一步 NBV：
          1) 用当前帧点云更新体素
          2) 生成候选视角（圆周均匀 + 当前位置附近随机）
          3) 计算信息增益，按增益降序依次尝试 IK
          4) 找到第一个可行视角后移动机械臂
        """
        # 1. 更新体素
        nbv_policy._integrate_current_observation(step=8, update_free_space=True)

        # 2. 生成候选视角
        #    a) 以辣椒中心为圆心，半径 0.35 m 均匀圆周取 12 个视角
        cam_poses_circle = nbv_policy._generate_candidate_views(
            center=pepper_center, radius=0.35, n_az=12, n_el=1
        )
        #    b) 在当前相机附近圆盘随机取 8 个视角
        cam_poses_local = nbv_policy._generate_random_views_around_current_cam(
            radius=0.12, n=8
        )
        all_poses = cam_poses_circle + cam_poses_local

        # 3. 计算信息增益并排序
        gains = nbv_policy.nbv_solver.evaluate_views(all_poses)
        sorted_idx = np.argsort(gains)[::-1]
        print(f"\n[NBV] Evaluating {len(all_poses)} candidates, "
              f"max_gain={gains.max():.2f}")

        # 4. 按增益依次尝试 IK
        moved = False
        for rank, idx in enumerate(sorted_idx):
            gain = float(gains[idx])
            if gain <= 0:
                print(f"[NBV] All remaining candidates have gain<=0, stop at rank={rank}.")
                break

            tgt_cam_pos, tgt_cam_rotmat = all_poses[idx]

            # Reuse v1's IK/FK tolerance, joint-limit and sampled-motion checks.
            # External obstacles include the car and all three plant meshes.
            if not adapter.feasible(tgt_cam_pos, tgt_cam_rotmat):
                continue
            adapter.move()
            cur_jnt[0] = adapter.current_joints.copy()
            _update_cam_from_robot()
            tcp_p, _ = robot.fk(cur_jnt[0], update=False)
            print(f"[NBV] rank={rank}, gain={gain:.3f}, "
                  f"cam_pos={vcam.cam_pos.round(3)}, "
                  f"tcp_pos={tcp_p.round(3)}")
            moved = True
            break

        if not moved:
            print("[NBV] No reachable view found this step.")

    base.accept("m", step_nbv)
    base.accept("q", base.destroy)

    # ── 11. 渲染主循环 ────────────────────────────────────────────────────

    def update(task):
        # 刷新机器人 mesh
        robot_vis_node[0].detach()
        robot_vis_node[0] = robot.gen_meshmodel(alpha=0.78)
        robot_vis_node[0].attach_to(base)

        # 刷新相机锥体 mesh
        vcam_vis_node[0].detach()
        vcam_vis_node[0] = gen_d405_model(vcam.cam_pos, vcam.cam_rotmat)
        vcam_vis_node[0].attach_to(base)

        # 获取图像
        img_rgb   = vcam.get_rgb_image()
        img_depth = vcam.get_depth_image()

        # ── 点云 & 体素更新 ─────────────────────────────────────────────
        pcd = vcam.get_point_cloud()
        if len(pcd) > 0:
            step  = 8
            pts   = pcd[::step]

            # obj1 掩码（果实 AABB）
            in_obj1 = (
                (pts[:, 0] >= obj1_min[0]) & (pts[:, 0] <= obj1_max[0]) &
                (pts[:, 1] >= obj1_min[1]) & (pts[:, 1] <= obj1_max[1]) &
                (pts[:, 2] >= obj1_min[2]) & (pts[:, 2] <= obj1_max[2])
            )
            colors_pcd = np.zeros((len(pts), 4), dtype=np.float32)
            colors_pcd[in_obj1]  = [1.0, 0.0, 0.0, 1.0]   # 红 = 果实
            colors_pcd[~in_obj1] = [0.0, 1.0, 0.0, 1.0]   # 绿 = 其余
            pc_viz.update(pts, colors=colors_pcd)

            # 语义 + 占据更新
            sem_probs   = np.ones(len(pts), dtype=np.float32)
            sem_classes = np.where(in_obj1, 0, 1).astype(np.int8)
            vg.update_hits_with_free_space(pts, vcam.cam_pos)
            vg.update_semantics(pts, sem_probs, sem_classes)

            # 体素可视化
            occ  = vg.get_occupancy()
            mask_occupied = occ > 0.8
            mask_free     = occ < 0.2
            idxs = np.column_stack(np.nonzero(mask_occupied | mask_free))

            if len(idxs) > 0:
                centers_v = vg.origin + vg.voxel_size * (idxs + 0.5)
                probs_v   = occ[idxs[:, 0], idxs[:, 1], idxs[:, 2]]

                in_obj1_v = (
                    (centers_v[:, 0] >= obj1_min[0]) & (centers_v[:, 0] <= obj1_max[0]) &
                    (centers_v[:, 1] >= obj1_min[1]) & (centers_v[:, 1] <= obj1_max[1]) &
                    (centers_v[:, 2] >= obj1_min[2]) & (centers_v[:, 2] <= obj1_max[2])
                )
                mask_occ  = probs_v > 0.8
                mask_free_v = probs_v < 0.2

                colors_v = np.zeros((len(centers_v), 4), dtype=np.float32)
                colors_v[mask_occ & in_obj1_v]  = [1.0, 0.0, 0.0, 0.5]  # 红：目标占据
                colors_v[mask_occ & ~in_obj1_v] = [0.0, 0.0, 1.0, 0.5]  # 蓝：背景占据
                colors_v[mask_free_v]            = [0.0, 1.0, 0.0, 0.3]  # 绿：空闲
                vox_viz.update(centers_v, colors_v)
            else:
                vox_viz.clear()

        # ── cv2 图像窗口 ──────────────────────────────────────────────────
        if img_rgb is not None and img_depth is not None:
            depth_clip = np.clip(img_depth, 0.0, 3.0)
            depth_norm = cv2.normalize(
                depth_clip, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

            rgb_show = img_rgb[:, :, :3].copy()
            dist = float(np.linalg.norm(vcam.cam_pos - pepper_center))
            info = (
                f"[M] NBV step  |  cam->pepper: {dist:.3f} m  "
                f"|  jnt: {np.round(cur_jnt[0], 2)}"
            )
            cv2.putText(rgb_show, info, (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)
            combined = np.hstack((rgb_show, depth_color))
            cv2.imshow("Eye-in-Hand NBV  (RGB | Depth)", combined)
            cv2.waitKey(1)

        return task.cont

    base.taskMgr.add(update, "main_loop")

    print()
    print("=" * 62)
    print("  Pepper Robot Eye-in-Hand NBV Simulation")
    print("  M  — 执行一步 NBV 规划（计算增益→IK→移动机械臂）")
    print("  Q  — 退出")
    print("=" * 62)
    print(f"  辣椒中心坐标 : {pepper_center.round(4)}")
    print(f"  初始相机位置 : {vcam.cam_pos.round(4)}")
    print()

    base.run()
