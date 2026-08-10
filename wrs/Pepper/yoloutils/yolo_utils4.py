#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/23 22:04
# @Author : ZhangXi
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/23
# @Author : ZhangXi
"""
YOLO + RealSense 通用工具模块
适用于盘子、方块、苹果等任意检测任务
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
from ultralytics import YOLO
import wrs.basis.robot_math as rm
import matplotlib.pyplot as plt
import pyrealsense2 as rs



def transform_points_by_homomat(homomat: np.ndarray, points: np.ndarray):
    """4x4 齐次矩阵作用于点云"""
    """"利用齐次变换矩阵将一组 3D 点从一个坐标系（例如相机坐标系）转换到另一个坐标系（例如机器人基座坐标系）。"""
    """homomat: 一个 4*4 的齐次变换矩阵。它包含了旋转3*3和平移3*1信息。
    points: 输入的点云数据，形状通常为(N, 3)，即 N个点的 (x, y, z)坐标。"""
    if not isinstance(points, np.ndarray):
        raise ValueError("Points must be np.ndarray!")
    homo_points = np.ones((4, points.shape[0]))
    homo_points[:3, :] = points.T
    return (homomat @ homo_points).T[:, :3]


def _estimate_point_from_neighborhood(target_pixel, pcd_matrix, neighborhood_size=5):
    """用邻域均值估算深度点"""
    h, w = pcd_matrix.shape[:2]
    px, py = target_pixel
    half = neighborhood_size // 2
    x_min, x_max = max(0, px - half), min(w - 1, px + half)
    y_min, y_max = max(0, py - half), min(h - 1, py + half)
    neighborhood = pcd_matrix[y_min:y_max + 1, x_min:x_max + 1].reshape(-1, 3)
    valid = neighborhood[np.any(neighborhood != 0, axis=1)]
    if len(valid) == 0:
        return None
    return np.mean(valid, axis=0)


# def yolo_detect_centers(yolo_model: YOLO, color_img: np.ndarray,
#                         conf_thres=0.3, show=False):
#     """返回每个检测框中心点的像素坐标和类别"""
#     results = yolo_model(color_img, verbose=False)
#     if not results or len(results[0].boxes) == 0:
#         return None
#     res = results[0]
#     boxes = res.boxes.xyxy.cpu().numpy()
#     cls_ids = res.boxes.cls.cpu().numpy()
#     confs = res.boxes.conf.cpu().numpy()
#     centers = []
#     img_disp = color_img.copy()
#
#     for (x1, y1, x2, y2), cls_id, conf in zip(boxes, cls_ids, confs):
#         if conf < conf_thres:
#             continue
#         cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
#         centers.append((cx, cy, int(cls_id)))
#         if show:
#             cv2.rectangle(img_disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             cv2.circle(img_disp, (cx, cy), 5, (0, 0, 255), -1)
#             cv2.putText(img_disp, f"ID:{int(cls_id)}", (cx + 5, cy - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
#
#     if show:
#         cv2.imshow("YOLO Detection", img_disp)
#         cv2.waitKey(1)
#
#     if len(centers) == 0:
#         return None
#     return np.array(centers, dtype=int)


def pick_one_pixel_from_image(img: np.ndarray) -> List[Tuple[int, int]]:
    """
    显示图像并返回用户点选的 ONE 个像素坐标 (x, y) (列, 行)，以列表形式返回。
    """
    # 假设如果 plt 不可用，有一个回退到控制台输入的函数，这里省略该部分。

    print("Close the image window after selecting exactly ONE point (left-click).")

    # 放大绘图，便于点击
    fig, ax = plt.subplots(figsize=(14, 9), dpi=120)
    try:
        # 尝试最大化窗口
        fig.canvas.manager.window.showMaximized()
    except Exception:
        pass

    ax.imshow(img)
    ax.set_title("Click ONE target point, then close the window")
    clicks = []

    # 定义点击处理函数
    def onclick(event):
        # 检查是否为左键点击 (button 1) 且点击发生在图像区域
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            # Matplotlib 返回的是 (xdata, ydata) -> (column, row)
            clicks.append((int(round(event.xdata)), int(round(event.ydata))))
            print(f"Clicked pixel (x, y): ({clicks[0][0]}, {clicks[0][1]})")
            # 自动关闭窗口，因为只需要一个点
            plt.close(fig)

    # 连接处理函数
    fig.canvas.mpl_connect('button_press_event', onclick)

    # 显示绘图并阻塞，直到窗口关闭
    plt.show()

    # 返回结果：一个包含单个 (x, y) 坐标的列表，或空列表
    return [clicks[0]] if clicks else []


def detect_target_point_manual_vision(self) -> Optional[np.ndarray]:
    """
    基于相机画面点选像素，并计算目标的世界坐标。
    模仿用户提供的 detect_pig_injection_point_manual 逻辑。
    """
    if self.camera is None or (self.real_robot and self.piper_controller is None):
        print("[detect] Camera or Controller not available.")
        return None

    try:
        pcd_raw, pcd_color, _depth_img, color_img = self.camera.get_pcd_texture_depth(
            # toggle_pcd_color=False
        )
    except Exception as e:
        print(f"[detect] Camera grab failed: {e}")
        return None

    given_conf = self.piper_controller.get_joint_values() if self.real_robot else self.robot.get_jnt_values()
    print(">>> MANUAL VISION MODE: Please click the target point(s) on the pop-up image.")
    try:
        pixels: List[Tuple[int, int]] = pick_one_pixel_from_image(color_img)
    except Exception as e:
        print(f"[detect] Pixel selection failed: {e}")
        return None

    if not pixels:
        print("[detect] No pixels selected.")
        return None

    # 4. Reshape 点云矩阵 (H x W x 3)
    H, W = color_img.shape[:2]
    try:
        pcd_matrix = pcd_raw.reshape(H, W, 3)
    except Exception as e:
        print(f"Failed to reshape point cloud to ({H},{W},3): {e}")
        return None

    # 5. 估算相机坐标系下的 3D 点 (Cam Frame)
    cam_pts = []
    for p in pixels:
        # 假设 yolo_utils._estimate_point_from_neighborhood 已经实现
        est = _estimate_point_from_neighborhood(
            target_pixel=np.asarray((int(p[0]), int(p[1]))),
            pcd_matrix=pcd_matrix,
            neighborhood_size=5,
        )
        if est is None or not np.isfinite(est).all():
            print(f"[detect] Estimator failed at pixel {p}")
            # 如果单个点估算失败，返回 None
            return None
        cam_pts.append(est)

    cam_pts = np.asarray(cam_pts, dtype=float)
    det_base_frame = self.transform_point_cloud_handeye(
        # handeye_mat=self.cam_to_base_mat,
        handeye_mat=self.HANDEYE,
        pcd=cam_pts,
        given_conf=given_conf,
        toggle_debug=False,
    )
    print(f"Detected candidates (base):\n{det_base_frame}")
    return det_base_frame[0]


def transform_point_cloud_handeye(self,
                                  handeye_mat: np.ndarray,
                                  pcd: np.ndarray,
                                  given_conf: np.ndarray = None,
                                  toggle_debug=False,
                                  ):
    """
    根据当前/给定的关节角计算 TCP 位姿，结合手眼标定矩阵，将点云从 Camera Frame 转换到 World Frame。
    handeye_mat: TCP -> Camera 的齐次变换矩阵 (R2Cam)
    pcd: Camera Frame 下的点云 (N, 3)
    """
    arm = self.robot
    if given_conf is None:
        if self.real_robot and self.piper_controller:
            given_conf = self.piper_controller.get_joint_values()
        else:
            given_conf = arm.get_jnt_values()

    gl_tcp_pos, gl_tcp_rotmat = arm.fk(given_conf, update=False)
    if hasattr(arm, 'end_effector') and arm.end_effector is not None:
        gl_tcp_pos = gl_tcp_pos - gl_tcp_rotmat @ arm.manipulator.loc_tcp_pos
    w2r_mat = rm.homomat_from_posrot(gl_tcp_pos, gl_tcp_rotmat)
    w2cam = w2r_mat.dot(handeye_mat)
    pcd_r = transform_points_by_homomat(w2cam, pcd)
    return pcd_r


def yolo_detect_injection_pix(
        yolo_model: YOLO,
        color_img: np.ndarray,
        toggle_image: bool = False,
        confident_threshold: float = 0.5  # 建议稍微调高阈值
) -> Optional[np.ndarray]:
    """
    使用 YOLO Keypoint 模型检测目标，返回最佳目标的有效关键点像素坐标。
    """
    results = yolo_model(color_img, verbose=False)
    if not results:
        return None

    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    if res.keypoints is None or res.keypoints.xy is None:
        return None

    boxes = res.boxes
    kp = res.keypoints
    top_idx = int(boxes.conf.argmax().item())
    raw_xy = kp.xy[top_idx]
    kp_conf = kp.conf[top_idx] if kp.conf is not None else None
    valid_points = []

    if kp_conf is not None:
        raw_xy_np = raw_xy.cpu().numpy().astype(int)
        kp_conf_np = kp_conf.cpu().numpy()

        for i, (x, y) in enumerate(raw_xy_np):
            conf = kp_conf_np[i]
            if conf > confident_threshold:
                valid_points.append([x, y])
    else:
        valid_points = raw_xy.cpu().numpy().astype(int).tolist()

    detected_pixel_coord = np.array(valid_points)

    if len(detected_pixel_coord) == 0:
        return None
    if toggle_image:
        display_img = color_img.copy()

        x1, y1, x2, y2 = boxes.xyxy[top_idx].cpu().numpy().astype(int)
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        for i, pt in enumerate(detected_pixel_coord):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(display_img, f"kp:{i}", (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 50), 2)

        cv2.imshow("YOLO Debug", display_img)
        cv2.waitKey(1)

    return detected_pixel_coord

def yolo_detect_pick_pos(
    yolo_model: YOLO,
    color_img: np.ndarray,
    pcd_tf: np.ndarray,
    toggle_image: bool = False,
    neighborhood_size: int = 5,
) -> Optional[np.ndarray]:
    """
    主业务接口：
    将 YOLO 检测到的 2D 关键点映射为 3D 世界坐标。

    Workflow:
        1. YOLO 检测 2D 关键点
        2. 点云 reshape 为 (H, W, 3)
        3. 对每个关键点执行邻域 3D 坐标估计

    Notes:
        - numpy 数组索引顺序为 (row=y, col=x)，需特别注意

    Returns:
        np.ndarray, shape = (N, 3)
    """

    pixels_coord = yolo_detect_injection_pix(
        yolo_model, color_img, toggle_image, confident_threshold=0.1
    )
    if pixels_coord is None:
        return None

    pcd_matrix = pcd_tf.reshape(color_img.shape[0], color_img.shape[1], 3)
    detected_points_xyz = []

    for p in pixels_coord:
        estimated_xyz = _estimate_point_from_neighborhood(
            target_pixel=p,
            pcd_matrix=pcd_matrix,
            neighborhood_size=neighborhood_size
        )
        if estimated_xyz is not None:
            detected_points_xyz.append(estimated_xyz)

    return np.asarray(detected_points_xyz) if detected_points_xyz else None


# def yolo_detect_world_positions(yolo_model, color_img, pcd_world,
#                                 show=False, neighborhood_size=5):
#     """检测物体并输出世界坐标"""
#     centers = yolo_detect_centers(yolo_model, color_img, show=show)
#     if centers is None:
#         return None
#
#     pcd_matrix = pcd_world.reshape(color_img.shape[0], color_img.shape[1], 3)
#     world_points = []
#     for cx, cy, cls_id in centers:
#         est_point = _estimate_point_from_neighborhood((cx, cy), pcd_matrix, neighborhood_size)
#         if est_point is not None:
#             world_points.append((cls_id, est_point))
#     return world_points


# ---------- 设备初始化 ----------
class _RSWrapper:
    """轻量级 RealSense 封装，使用 pyrealsense2 pipeline（与 Main_last 类似）。"""

    def __init__(self, serial: Optional[str] = None):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if serial:
            self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.pc = rs.pointcloud()

    def get_pcd_texture_depth(self):
        """返回点云、点云颜色、深度图、彩色图 (pcd_raw, pcd_color, depth_img, color_img)"""
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth = aligned.get_depth_frame()
        color = aligned.get_color_frame()
        if not depth or not color:
            return None, None, None, None

        color_img = np.asanyarray(color.get_data())
        depth_img = np.asanyarray(depth.get_data())

        self.pc.map(color)
        points = self.pc.calculate(depth)
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # (N,3)
        tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32)
        return verts, tex, depth_img, color_img

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


def init_camera(camera_id: str = None):
    """
    使用 pyrealsense2 初始化相机（与 Main_last 方式一致）。
    camera_id 可选传入序列号；None 则默认首个设备。
    """
    try:
        cam = _RSWrapper(serial=camera_id)
        # 预热一次
        cam.get_pcd_texture_depth()
        print("RealSense 相机就绪。")
        return cam
    except Exception as e:
        raise RuntimeError(f"RealSense 初始化失败: {e}")


def init_yolo(weight_path: str):
    """初始化YOLO模型"""
    print(f"加载模型权重: {weight_path}")
    return YOLO(weight_path)
