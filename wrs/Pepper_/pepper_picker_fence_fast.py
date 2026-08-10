#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fast pepper picking without RRT/linear planning.

This script follows the user's requested workflow:
- use YOLO/keypoints to detect a target
- solve IK for approach pose and grasp pose
- check electronic fence bounds after IK
- validate robot collision with shelf/box
- execute direct motion with move_jntspace_path([approach_conf, grasp_conf])

Fence parameters are copied from pepper_fence_visualizer.py and can be tuned in one place.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# Keep OpenCV Qt backend quiet when fonts are missing.
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
import wrs.visualization.panda.world as wd
from wrs.Pepper.Grapper.Grapper import ClawController
from wrs.Pepper.vision_io import safe_imshow
from wrs.robot_con.realman.realman import RealmanArmController
from wrs.robot_sim.robots.realman.realman_single_arm import Realman


WRS_ROOT = Path(__file__).resolve().parents[1]
PEPPER_DIR = WRS_ROOT / "Pepper"
PEPPER_NBV_DIR = WRS_ROOT / "Pepper_NBV"

DEFAULT_GRIPPER_COM = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG033GD5-if00-port0"
CLAW_ID = 9
claw_controller = ClawController(CLAW_ID, com_port=DEFAULT_GRIPPER_COM)  # 创建夹爪控制器对象
claw_controller.connect_gripper()  # 连接夹爪
claw_controller.enable_gripper()  # 使能夹爪
# 预通电
claw_controller.close_gripper()
claw_controller.open_gripper()


class FastFenceConfig:
    """Electronic fence copied from pepper_fence_visualizer.py."""

    def __init__(self):
        self.size = np.array([0.9, 1.0, 0.8], dtype=float)
        self.center = np.array([-0.30, 0.00, 0.40], dtype=float)
        self.euler_deg = np.array([0.0, 0.0, 0.0], dtype=float)
        self.rgba = np.array([0.15, 0.75, 1.00, 0.18], dtype=float)
        self.frame_rgba = np.array([0.15, 0.75, 1.00, 0.9], dtype=float)

    @property
    def rotmat(self):
        return rm.rotmat_from_euler(*np.radians(self.euler_deg))

    def in_fence(self, pos: np.ndarray) -> bool:
        """Check whether a point is inside the oriented fence box."""
        pos = np.asarray(pos, dtype=float).reshape(3)
        local = self.rotmat.T @ (pos - self.center)
        half = self.size / 2.0
        return bool(np.all(np.abs(local) <= half + 1e-9))


class RealmanPepperPickerFast:
    def __init__(self, real_robot: bool = True):
        self._reset_timing_records()
        _init_t0 = time.time()
        self.real_robot = real_robot
        self.fence = FastFenceConfig()

        self.base = wd.World(cam_pos=rm.vec(2, 0, 1), lookat_pos=rm.vec(0, 0, 0.5))
        mgm.gen_frame().attach_to(self.base)
        self._record_timing("init_scene", time.time() - _init_t0, "world/frame attached")
        _init_t0 = time.time()

        self.shelf_model = mcm.gen_box(
            xyz_lengths=np.array([0.02, 0.7, 1.55]),
            pos=rm.vec(-1, 0, 0),
            rotmat=np.eye(3),
            rgb=[0.1, 0.1, 0.2],
            alpha=0.8,
        )
        self.shelf_model.attach_to(self.base)
        self._record_timing("init_models", time.time() - _init_t0, "shelf attached")
        _init_t0 = time.time()

        self.box_model = mcm.CollisionModel(
            str(PEPPER_NBV_DIR / "newcar_m.stl"),
            rgb=np.array([0.6, 0.6, 0.6]),
            alpha=0.9,
        )
        self.box_model.pos = np.array([0.13, 0.0, -0.79])
        self.box_model.rotmat = rm.rotmat_from_axangle(np.array([0, 0, 1]), np.pi * 90 / 180)
        self.box_model.attach_to(self.base)

        self.robot = Realman(enable_cc=True)
        self.rrtc_planner = None
        self.adp_planner = adp.ADPlanner(self.robot)
        self.rm_controller = None
        self.rs_pipeline = None
        self.rs_align = None
        self.HANDEYE = self._load_handeye_matrix()
        self.yolo_model = self._init_yolo()
        self._init_hardware()
        self._record_timing("init_robot_planner", time.time() - _init_t0, "robot/adp ready")
        _init_t0 = time.time()
        self._record_timing("init_total", time.time() - _init_t0, "constructor tail")
        self._flush_timing_log(total_success=0)

    def _load_handeye_matrix(self):
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
        import wrs.Pepper.yoloutils.yolo_utils as yoloutils
        return yoloutils.init_yolo(str(PEPPER_DIR / "best.pt"))

    def _record_timing(self, stage, duration, detail=""):
        if not hasattr(self, "_timing_records"):
            self._timing_records = []
        duration = float(duration)
        self._timing_records.append((stage, duration, detail))
        if detail:
            print(f"[计时] {stage}: {duration:.3f} 秒 | {detail}")
        else:
            print(f"[计时] {stage}: {duration:.3f} 秒")

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
            self.rm_controller = RealmanArmController(has_gripper=True)
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

    def get_Keyresult(self, results):
        keypoints_array = []
        target_class_id = 0
        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            for i in range(len(boxes)):
                class_id = int(boxes[i].cls)
                if class_id != target_class_id:
                    continue
                if keypoints is not None and len(keypoints) > i:
                    kpts = keypoints[i].data[0].cpu().numpy()
                    for j, kpt in enumerate(kpts):
                        if j == 0:
                            keypoints_array.append([kpt[0], kpt[1], kpt[2]])
        return keypoints_array

    def scan_and_detect(self):
        """
        获取图像，通过YOLO识别目标（使用关键点检测），并返回目标在世界坐标系下的坐标 (x, y, z)。
        参考 Main_last.py 的处理方法，使用关键点而不是边界框中心。
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
            device = self._yolo_predict_device()
            print(f"[YOLO] 使用设备: {device}")
            _pred_kw = dict(save=False, show_conf=False, device=device)
            results = self.yolo_model.predict(color_img, **_pred_kw)
            keypoints_array = self.get_Keyresult(results)
            target_world_pos = None

            if len(keypoints_array) > 0:
                for keypoint in keypoints_array:
                    if round(keypoint[2]) == 1:
                        ux, uy = int(keypoint[0]), int(keypoint[1])
                        if not (0 <= ux < depth_intrin.width and 0 <= uy < depth_intrin.height):
                            print(f"[警告] 坐标超出范围 ({ux}, {uy})")
                            continue
                        dis = depth_frame.get_distance(ux, uy)
                        print(f"采摘点坐标（像素）：{ux}, {uy}, 深度：{dis}")
                        if dis > 0:
                            camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis)
                            camera_xyz = np.array(camera_xyz)
                            given_conf = self.rm_controller.get_joint_values() if self.rm_controller else None
                            camera_point_cloud = camera_xyz.reshape(1, 3)
                            world_point_cloud = self.transform_point_cloud_handeye(
                                handeye_mat=self.HANDEYE,
                                pcd=camera_point_cloud,
                                given_conf=given_conf,
                            )
                            target_world_pos = world_point_cloud[0]
                            print(f"目标世界坐标：{target_world_pos}")

                            vis_scale = 2.0
                            im_array = results[0].plot()
                            im_array = cv2.resize(im_array, None, fx=vis_scale, fy=vis_scale)
                            ux_vis = int(round(ux * vis_scale))
                            uy_vis = int(round(uy * vis_scale))
                            cv2.circle(im_array, (ux_vis, uy_vis), 5, (0, 255, 0), -1)
                            cv2.putText(im_array, f"Depth: {dis:.3f}m", (ux_vis + 10, uy_vis - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            safe_imshow('YOLO Detection', im_array, wait_ms=1)
                            return target_world_pos
                        else:
                            print(f"[警告] 关键点 ({ux}, {uy}) 深度无效")
                            continue

                if target_world_pos is None:
                    print("[警告] 未找到有效的关键点（可见性为1）")
                    if results and len(results) > 0:
                        im_array = results[0].plot()
                        safe_imshow('YOLO Detection', im_array, wait_ms=1)
                    return None
            else:
                print("[提示] YOLO未检测到目标。")
                vis_scale = 1.5
                show_img = cv2.resize(color_img, None, fx=vis_scale, fy=vis_scale)
                safe_imshow('YOLO Detection', show_img, wait_ms=1)
                return None

        else:
            print("当前为仿真模式，使用模拟检测结果。")
            return np.array([0.45, -0.1, 0.3])

    def transform_point_cloud_handeye(self, handeye_mat, pcd, given_conf=None):
        import wrs.Pepper.yoloutils.yolo_utils as yoloutils
        return yoloutils.transform_point_cloud_handeye(self, handeye_mat, pcd, given_conf=given_conf)

    def vis_path(self, mot_data, colormap):
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
        viewpoint_joint=np.array([-2.9569019, 0.12503539, -0.20418608, 3.02282313, 1.46630347, 0.03455752]),
        place_joint=np.array([-1.62024151, -0.72099554, -0.65041438, 3.06512984, 1.58201875, -0.12065461]),
        continuous_mode=True,
        stop_on_no_detection=True,
        control_frequency=0.8,
        open_close_retries=3,
    ):
        """
        保留原版主脚本的阶段式打印/计时结构，Phase 3 使用 IK + 围栏 + 直线规划，去掉 RRT。
        """
        _task_start_time = time.time()
        self._reset_timing_records()
        robot = self.robot

        if self.real_robot and self.rm_controller is not None:
            start_conf = self.rm_controller.get_joint_values()
        else:
            start_conf = robot.get_jnt_values()
        robot.goto_given_conf(start_conf)

        print("--- 阶段1：移动到观察位 ---")
        _t0 = time.time()
        if np.linalg.norm(np.array(start_conf) - np.array(viewpoint_joint)) > 0.03:
            mot_data_view = None
            try:
                mot_data_view = type("DummyPath", (), {"jv_list": [start_conf, viewpoint_joint]})()
            except Exception:
                mot_data_view = None
            if mot_data_view is not None:
                robot.goto_given_conf(viewpoint_joint)
                if self.real_robot and self.rm_controller is not None:
                    self.rm_controller.move_jntspace_path(mot_data_view.jv_list, control_frequency=control_frequency)
                    time.sleep(3.0)
                self._record_timing("phase1_move_view", time.time() - _t0, "moved to viewpoint")
            else:
                self._record_timing("phase1_move_view", time.time() - _t0, "failed")
                print("[错误] 规划到观察位失败。")
                return
        else:
            self._record_timing("phase1_move_view", time.time() - _t0, "already at viewpoint")
            print("当前已经位于观察位。")

        claw_controller = None
        if self.real_robot:
            try:
                claw_controller = ClawController(CLAW_ID, com_port=DEFAULT_GRIPPER_COM)
                time.sleep(1)
                claw_controller.connect_gripper()
                time.sleep(1)
                claw_controller.enable_gripper()
                claw_controller.close_gripper()
                claw_controller.open_gripper()
            except Exception as e:
                print(f"[错误] 夹爪初始化失败：{e}")
                claw_controller = None

        pick_count = 0
        total_success = 0

        while True:
            _pick_cycle_start = time.time()
            print(f"\n--- 阶段2：检测目标（第 {pick_count + 1} 轮）---")
            _t0 = time.time()
            if self.real_robot:
                tgt_pos = self.scan_and_detect()
            else:
                # start pos
                tgt_pos = np.array([-0.45, -0.1, 0.3])
                print(f"[仿真] 使用模拟目标：{tgt_pos}")
            self._record_timing("phase2_detection", time.time() - _t0, "target detection finished")

            if tgt_pos is None:
                if self.real_robot:
                    print("[提示] 未检测到辣椒，准备再拍两帧确认。")
                    missing_count = 0
                    for retry_idx in range(2):
                        time.sleep(0.5)
                        print(f"[提示] 重新拍摄第 {retry_idx + 1} 帧...")
                        _retry_t0 = time.time()
                        tgt_pos = self.scan_and_detect()
                        self._record_timing("phase2_detection_retry", time.time() - _retry_t0, f"retry={retry_idx + 1}")
                        if tgt_pos is not None:
                            print("[提示] 重试帧中检测到辣椒，继续采摘。")
                            break
                        missing_count += 1
                    if tgt_pos is None and missing_count >= 2:
                        if stop_on_no_detection:
                            print("[提示] 连续两帧都未检测到辣椒，结束采摘。")
                            break
                        print("[提示] 连续两帧都未检测到辣椒，准备重试下一轮。")
                        pick_count += 1
                        continue
                else:
                    if stop_on_no_detection:
                        print("[提示] 未检测到辣椒，结束采摘。")
                        break
                    print("[提示] 未检测到辣椒，准备重试。")
                    time.sleep(1.0)
                    pick_count += 1
                    continue

            print(f"--- 阶段3：直接IK采摘，目标位置 {tgt_pos} ---")
            _t0 = time.time()
            is_success = False
            phase3_search_time = 0.0
            phase3_ik_time = 0.0
            phase3_plan_time = 0.0
            found_theta = None
            final_grasp_conf = None
            final_approach_conf = None
            final_linear_path = None

            # set moduan'pos for serch ik
            for theta in range(bound_lower, bound_upper + 1, grad):
                _t_search = time.time()
                hand_x = np.array([0, 0, -1])
                hand_z = np.array([-1, 0, 0])
                hand_y = np.cross(hand_z, hand_x)
                tgt_rotmat_base = np.array([hand_x, hand_y, hand_z]).T
                tgt_rotmat = rm.rotmat_from_axangle(hand_y, np.radians(theta)) @ tgt_rotmat_base
                phase3_search_time += time.time() - _t_search

                _t_ik = time.time()
                grasp_conf = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=robot.get_jnt_values())
                if grasp_conf is None:
                    phase3_ik_time += time.time() - _t_ik
                    print(f"角度 {theta}：抓取构型IK求解失败。")
                    continue

                approach_pos = tgt_pos - hand_z * approach_dis
                approach_conf = robot.ik(tgt_pos=approach_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=grasp_conf)
                if approach_conf is None:
                    phase3_ik_time += time.time() - _t_ik
                    print(f"角度 {theta}：预抓取构型IK求解失败。")
                    continue

                _t_fence = time.time()
                if not self.fence.in_fence(approach_pos):
                    phase3_plan_time += time.time() - _t_fence
                    print(f"角度 {theta}：预抓取点超出电子围栏。")
                    continue
                if not self.fence.in_fence(tgt_pos):
                    phase3_plan_time += time.time() - _t_fence
                    print(f"角度 {theta}：目标点超出电子围栏。")
                    continue
                phase3_plan_time += time.time() - _t_fence

                robot.goto_given_conf(approach_conf)
                if robot.is_collided(obstacle_list=[self.shelf_model, self.box_model]):
                    phase3_ik_time += time.time() - _t_ik
                    print(f"角度 {theta}：预抓取构型发生碰撞。")
                    continue

                robot.goto_given_conf(grasp_conf)
                if robot.is_collided(obstacle_list=[self.shelf_model, self.box_model]):
                    phase3_ik_time += time.time() - _t_ik
                    print(f"角度 {theta}：抓取构型发生碰撞。")
                    continue

                _t_plan = time.time()
                mot_data_linear = self.adp_planner.gen_linear_motion(
                    start_conf=approach_conf,
                    goal_conf=grasp_conf,
                    granularity=0.01,
                    obstacle_list=[self.shelf_model, self.box_model],
                )
                phase3_plan_time += time.time() - _t_plan
                if mot_data_linear is None:
                    print(f"角度 {theta}：直线运动规划失败。")
                    continue

                found_theta = theta
                final_grasp_conf = grasp_conf
                final_approach_conf = approach_conf
                final_linear_path = mot_data_linear
                is_success = True
                self.vis_path(mot_data_linear, rm.const.wistia_map)
                self._record_timing("phase3_search_pose", phase3_search_time, f"theta={theta}")
                self._record_timing("phase3_inverse_ik", phase3_ik_time, f"theta={theta}, success")
                self._record_timing("phase3_path_plan", phase3_plan_time, f"theta={theta}, success")
                self._record_timing("phase3_total", time.time() - _t0, f"theta={theta}, success")
                print(f"角度 {theta}：IK + 围栏 + 直线规划通过。")
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

            print("--- 阶段4：规划放置路径 ---")
            _t0 = time.time()
            mot_data_place = None
            try:
                mot_data_place = type("DummyPath", (), {"jv_list": [final_grasp_conf, place_joint]})()
                self._record_timing("phase4_place_plan", time.time() - _t0, "success")
            except Exception as e:
                self._record_timing("phase4_place_plan", time.time() - _t0, "failed")
                print(f"[错误] 规划到放置位失败：{e}")
                is_success = False

            if is_success and mot_data_place is not None:
                print("--- 阶段5：执行抓取与放置 ---")
                _t0 = time.time()
                grasp_start_time = time.time()
                try:
                    if self.real_robot and self.rm_controller is not None and claw_controller is not None:
                        claw_controller.open_gripper()

                        self.rm_controller.move_jntspace_path([start_conf, final_approach_conf], control_frequency=control_frequency)
                        # self.rm_controller.wait_until_idle(timeout=20.0)
                        time.sleep(1.5)
                        self.rm_controller.move_jntspace_path(mot_data_linear.jv_list, control_frequency=control_frequency)
                        time.sleep(1)
                        claw_controller.close_gripper()
                        time.sleep(0.8)

                        retreat_jv_list = [final_grasp_conf, final_approach_conf]
                        self.rm_controller.move_jntspace_path(retreat_jv_list, control_frequency=control_frequency)
                        time.sleep(0.8)

                        self.rm_controller.move_jntspace_path(mot_data_place.jv_list, control_frequency=control_frequency)
                        time.sleep(0.8)
                        claw_controller.open_gripper()

                        total_success += 1
                        exec_dur = time.time() - _t0
                        self._record_timing("phase5_exec", exec_dur, f"picked={total_success}")
                        pick_to_return_time = time.time() - _pick_cycle_start
                        self._record_timing("pick_to_return", pick_to_return_time, f"picked={total_success}, before_return")
                        print(f"[成功] 已完成第 {total_success} 个辣椒的抓取与放置，"
                              f"耗时：{time.time() - grasp_start_time:.1f} 秒")
                    else:
                        robot.goto_given_conf(final_approach_conf)
                        robot.goto_given_conf(final_grasp_conf)
                        robot.goto_given_conf(place_joint)
                        total_success += 1
                        exec_dur = time.time() - _t0
                        self._record_timing("phase5_exec", exec_dur, f"picked={total_success}")
                        pick_to_return_time = time.time() - _pick_cycle_start
                        self._record_timing("pick_to_return", pick_to_return_time, f"picked={total_success}, before_return")
                        print(f"[成功] 仿真模式完成第 {total_success} 个辣椒的抓取与放置。")
                except Exception as e:
                    print(f"[错误] 执行阶段失败：{e}")
                    continue

                if continuous_mode:
                    print("--- 返回观察位 ---")
                    try:
                        if self.real_robot and self.rm_controller is not None:
                            mot_return = type("DummyPath", (), {"jv_list": [place_joint, viewpoint_joint]})()
                            robot.goto_given_conf(viewpoint_joint)
                            self.rm_controller.move_jntspace_path(mot_return.jv_list, control_frequency=0.8)
                            time.sleep(3.0)
                        else:
                            robot.goto_given_conf(viewpoint_joint)
                            time.sleep(1.0)
                    except Exception as e:
                        print(f"[警告] 返回观察位失败：{e}")
                else:
                    break

            pick_count += 1
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
        if self.real_robot and self.rs_pipeline is not None:
            self.rs_pipeline.stop()

    def demo_fence(self):
        fence_model = mcm.gen_box(
            xyz_lengths=self.fence.size,
            pos=self.fence.center,
            rotmat=self.fence.rotmat,
            rgb=np.array([0.15, 0.75, 1.0]),
            alpha=0.18,
        )
        fence_model.attach_to(self.base)
        mgm.gen_frame(pos=self.fence.center, rotmat=self.fence.rotmat).attach_to(self.base)
        self.base.run()


if __name__ == "__main__":
    picker = RealmanPepperPickerFast(real_robot=True)
    picker.pick_place_fast(
        approach_dis=0.15,
        bound_lower=-30,
        bound_upper=30,
        grad=3,
        continuous_mode=True,
        stop_on_no_detection=True,
        control_frequency=0.8,
        open_close_retries=3,
    )
