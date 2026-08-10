#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/12/18
# @Author : cursor
"""
YOLO + RealSense 通用工具模块
适用于辣椒、盘子、方块、苹果等任意检测任务

主要功能：
1. YOLO 模型初始化和检测
2. RealSense 相机初始化和点云获取
3. 点云坐标转换（相机坐标系 -> 世界坐标系）
4. 关键点提取和深度估计
5. 手动像素选择工具
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Union
from ultralytics import YOLO
import pyrealsense2 as rs
import wrs.basis.robot_math as rm
import matplotlib.pyplot as plt


# ==================== 点云变换工具 ====================

def transform_points_by_homomat(homomat: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    使用齐次变换矩阵变换点云
    
    Args:
        homomat: 4x4 齐次变换矩阵
        points: 点云数组 (N, 3) 或单个点 (3,)
    
    Returns:
        变换后的点云 (N, 3) 或单个点 (3,)
    """
    if not isinstance(points, np.ndarray):
        raise ValueError("Points must be np.ndarray!")
    
    # 处理单个点的情况
    if points.ndim == 1:
        points = points.reshape(1, -1)
        single_point = True
    else:
        single_point = False
    
    # 转换为齐次坐标
    homo_points = np.ones((4, points.shape[0]))
    homo_points[:3, :] = points.T
    
    # 应用变换
    transformed = (homomat @ homo_points).T[:, :3]
    
    # 如果是单个点，返回一维数组
    if single_point:
        return transformed[0]
    return transformed


# def _estimate_point_from_neighborhood(target_pixel: Tuple[int, int],
#                                      pcd_matrix: np.ndarray,
#                                      neighborhood_size: int = 5) -> Optional[np.ndarray]:
#     """
#     使用邻域均值估算深度点（当目标像素深度无效时使用）
#
#     Args:
#         target_pixel: 目标像素坐标 (u, v) = (x, y)
#         pcd_matrix: 点云矩阵 (H, W, 3)，相机坐标系下的点云
#         neighborhood_size: 邻域大小（奇数）
#
#     Returns:
#         估算的3D点 (3,)，如果失败返回 None
#     """
#     h, w = pcd_matrix.shape[:2]
#     px, py = target_pixel
#     half = neighborhood_size // 2
#
#     # 计算邻域范围
#     x_min = max(0, px - half)
#     x_max = min(w - 1, px + half)
#     y_min = max(0, py - half)
#     y_max = min(h - 1, py + half)
#
#     # 提取邻域点云
#     neighborhood = pcd_matrix[y_min:y_max + 1, x_min:x_max + 1].reshape(-1, 3)
#
#     # 过滤无效点（全零或NaN）
#     valid = neighborhood[np.any(neighborhood != 0, axis=1)]
#     valid = valid[~np.any(np.isnan(valid), axis=1)]
#
#     if len(valid) == 0:
#         return None
#
#     # 返回邻域均值
#     return np.mean(valid, axis=0)
def _estimate_point_from_neighborhood(
    target_pixel: Tuple[int, int],
    pcd_matrix: np.ndarray,
    neighborhood_size: int = 5,
    mad_k: float = 2.5,
    min_inliers: int = 6,
    use_depth_only: bool = True,
    max_range: Optional[float] = None,   # 比如 2.0 (m)，可选
) -> Optional[np.ndarray]:
    """
    使用邻域鲁棒估计(中位数+MAD剔除离群点)来估算3D点
    """
    h, w = pcd_matrix.shape[:2]
    px, py = target_pixel
    half = neighborhood_size // 2

    x_min = max(0, px - half)
    x_max = min(w - 1, px + half)
    y_min = max(0, py - half)
    y_max = min(h - 1, py + half)

    neighborhood = pcd_matrix[y_min:y_max + 1, x_min:x_max + 1].reshape(-1, 3)

    # 过滤无效点：NaN/Inf/全零/深度<=0
    valid = neighborhood[np.isfinite(neighborhood).all(axis=1)]
    valid = valid[np.linalg.norm(valid, axis=1) > 1e-6]
    valid = valid[valid[:, 2] > 1e-6]  # z(深度)必须>0

    if max_range is not None:
        valid = valid[valid[:, 2] < max_range]

    if len(valid) == 0:
        return None

    # 1) 中位数作为鲁棒中心
    med = np.median(valid, axis=0)

    # 2) 计算偏差（只用深度z 或 用3D距离）
    if use_depth_only:
        dev = np.abs(valid[:, 2] - med[2])
    else:
        dev = np.linalg.norm(valid - med, axis=1)

    # 3) MAD：中位绝对偏差（鲁棒尺度）
    mad = np.median(dev)
    if mad < 1e-9:
        # 邻域非常“干净”，直接返回中位数（比均值更稳）
        return med

    # 把 MAD 转成近似标准差尺度（可选，但常用）
    robust_sigma = 1.4826 * mad
    inliers = valid[dev <= mad_k * robust_sigma]

    # inliers 太少就放宽或回退
    if len(inliers) < min_inliers:
        # 回退策略：直接用中位数，或者用“剪裁均值”
        return med

    # 4) 对内点求均值（更平滑）；你也可以改成 np.median(inliers, axis=0)
    return np.mean(inliers, axis=0)


# ==================== YOLO 检测工具 ====================

def yolo_detect_centers(yolo_model: YOLO,
                        color_img: np.ndarray,
                        conf_thres: float = 0.3,
                        show: bool = False) -> Optional[np.ndarray]:
    """
    使用YOLO检测物体，返回每个检测框中心点的像素坐标和类别
    
    Args:
        yolo_model: YOLO模型实例
        color_img: 彩色图像 (H, W, 3)
        conf_thres: 置信度阈值
        show: 是否显示检测结果
    
    Returns:
        检测结果数组 (N, 3)，每行为 [cx, cy, cls_id]，如果无检测返回 None
    """
    results = yolo_model(color_img, verbose=False)
    
    if not results or len(results[0].boxes) == 0:
        return None
    
    res = results[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    
    centers = []
    img_disp = color_img.copy() if show else None
    
    for (x1, y1, x2, y2), cls_id, conf in zip(boxes, cls_ids, confs):
        if conf < conf_thres:
            continue
        
        # 计算中心点
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        centers.append((cx, cy, int(cls_id)))
        
        # 可视化
        if show:
            cv2.rectangle(img_disp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(img_disp, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(img_disp, f"ID:{int(cls_id)}", (cx + 5, cy - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    if show and img_disp is not None:
        cv2.imshow("YOLO Detection", img_disp)
        cv2.waitKey(1)
    
    if len(centers) == 0:
        return None
    
    return np.array(centers, dtype=int)


def yolo_detect_world_positions(yolo_model: YOLO,
                                color_img: np.ndarray,
                                pcd_world: np.ndarray,
                                show: bool = False,
                                neighborhood_size: int = 5) -> Optional[List[Tuple[int, np.ndarray]]]:
    """
    检测物体并输出世界坐标
    
    Args:
        yolo_model: YOLO模型实例
        color_img: 彩色图像 (H, W, 3)
        pcd_world: 世界坐标系下的点云 (H*W, 3) 或 (H, W, 3)
        show: 是否显示检测结果
        neighborhood_size: 邻域大小
    
    Returns:
        检测结果列表 [(cls_id, world_pos), ...]，如果无检测返回 None
    """
    centers = yolo_detect_centers(yolo_model, color_img, show=show)
    if centers is None:
        return None
    
    # 确保点云是矩阵形式
    if pcd_world.ndim == 2:
        H, W = color_img.shape[:2]
        pcd_matrix = pcd_world.reshape(H, W, 3)
    else:
        pcd_matrix = pcd_world
    
    world_points = []
    for cx, cy, cls_id in centers:
        est_point = _estimate_point_from_neighborhood((cx, cy), pcd_matrix, neighborhood_size)
        if est_point is not None:
            world_points.append((cls_id, est_point))
    
    return world_points if len(world_points) > 0 else None


# ==================== 手动像素选择工具 ====================

def pick_one_pixel_from_image(img: np.ndarray) -> List[Tuple[int, int]]:
    """
    显示图像并返回用户点选的像素坐标
    
    Args:
        img: 图像数组 (H, W, 3) 或 (H, W)
    
    Returns:
        点选的像素坐标列表 [(x, y), ...]，如果未选择返回空列表
    """
    print("Close the image window after selecting exactly ONE point (left-click).")
    
    # 创建大窗口便于点击
    fig, ax = plt.subplots(figsize=(14, 9), dpi=120)
    try:
        fig.canvas.manager.window.showMaximized()
    except Exception:
        pass
    
    ax.imshow(img)
    ax.set_title("Click ONE target point, then close the window")
    clicks = []
    
    def onclick(event):
        """点击事件处理"""
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            clicks.append((x, y))
            print(f"Clicked pixel (x, y): ({x}, {y})")
            plt.close(fig)
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    return clicks


# ==================== 手眼标定转换 ====================

def transform_point_cloud_handeye(self,
                                  handeye_mat: np.ndarray,
                                  pcd: np.ndarray,
                                  given_conf: Optional[np.ndarray] = None,
                                  toggle_debug: bool = False) -> np.ndarray:
    """
    根据当前/给定的关节角计算 TCP 位姿，结合手眼标定矩阵，将点云从 Camera Frame 转换到 World Frame
    
    Args:
        handeye_mat: TCP -> Camera 的齐次变换矩阵 (4x4)
        pcd: Camera Frame 下的点云 (N, 3) 或单个点 (3,)
        given_conf: 给定的关节角（弧度），如果为 None 则从控制器获取
        toggle_debug: 是否打印调试信息
    
    Returns:
        World Frame 下的点云 (N, 3) 或单个点 (3,)
    
    Note:
        此函数需要绑定到类实例使用（通过 __get__ 方法）
    """
    arm = self.robot
    
    # 获取关节角
    if given_conf is None:
        controller = getattr(self, 'rm_controller', None)
        if self.real_robot and controller:
            given_conf = controller.get_joint_values()
        else:
            given_conf = arm.get_jnt_values()
    
    # 计算 TCP 位姿（世界坐标系）
    gl_tcp_pos, gl_tcp_rotmat = arm.fk(given_conf, update=False)
    
    # 如果机器人有末端执行器，需要从 TCP 回退到法兰盘
    # 这是因为手眼标定矩阵通常是相对于法兰盘的
    # if hasattr(arm, 'end_effector') and arm.end_effector is not None:
    #     gl_tcp_pos = gl_tcp_pos - gl_tcp_rotmat @ arm.manipulator.loc_tcp_pos
    
    # 构建 World -> Robot (TCP) 的变换矩阵
    w2r_mat = rm.homomat_from_posrot(gl_tcp_pos, gl_tcp_rotmat)
    
    # 组合变换：World -> Camera
    # w2r_mat: World -> Robot (TCP)
    # handeye_mat: TCP -> Camera
    # w2cam: World -> Camera
    w2cam = w2r_mat.dot(handeye_mat)
    
    # 应用变换（注意：这里使用 w2cam 直接变换，根据实际标定方式可能需要调整）
    pcd_r = transform_points_by_homomat(w2cam, pcd)
    
    if toggle_debug:
        print(f"[Debug] Transformed {pcd.shape[0] if pcd.ndim > 1 else 1} points")
        print(f"[Debug] TCP pose: {gl_tcp_pos}")
    
    return pcd_r


def detect_target_point_manual_vision(self) -> Optional[np.ndarray]:
    """
    基于相机画面点选像素，并计算目标的世界坐标
    
    Returns:
        目标在世界坐标系下的位置 (3,)，如果失败返回 None
    
    Note:
        此函数需要绑定到类实例使用
    """
    controller = getattr(self, 'rm_controller', None)
    
    if self.camera is None or (self.real_robot and controller is None):
        print("[detect] Camera or Controller not available.")
        return None
    
    try:
        pcd_raw, pcd_color, _depth_img, color_img = self.camera.get_pcd_texture_depth()
    except Exception as e:
        print(f"[detect] Camera grab failed: {e}")
        return None
    
    given_conf = controller.get_joint_values() if (self.real_robot and controller) else self.robot.get_jnt_values()
    
    print(">>> MANUAL VISION MODE: Please click the target point(s) on the pop-up image.")
    try:
        pixels: List[Tuple[int, int]] = pick_one_pixel_from_image(color_img)
    except Exception as e:
        print(f"[detect] Pixel selection failed: {e}")
        return None
    
    if not pixels:
        print("[detect] No pixels selected.")
        return None
    
    # 将点云重塑为矩阵形式
    H, W = color_img.shape[:2]
    try:
        pcd_matrix = pcd_raw.reshape(H, W, 3)
    except Exception as e:
        print(f"Failed to reshape point cloud to ({H},{W},3): {e}")
        return None
    
    # 估算相机坐标系下的 3D 点
    cam_pts = []
    for p in pixels:
        est = _estimate_point_from_neighborhood(
            target_pixel=(int(p[0]), int(p[1])),
            pcd_matrix=pcd_matrix,
            neighborhood_size=5,
        )
        if est is None or not np.isfinite(est).all():
            print(f"[detect] Estimator failed at pixel {p}")
            return None
        cam_pts.append(est)
    
    cam_pts = np.asarray(cam_pts, dtype=float)
    
    # 转换到世界坐标系
    det_base_frame = self.transform_point_cloud_handeye(
        handeye_mat=self.HANDEYE,
        pcd=cam_pts,
        given_conf=given_conf,
        toggle_debug=False,
    )
    
    print(f"Detected candidates (base):\n{det_base_frame}")
    return det_base_frame[0]


# ==================== RealSense 相机封装 ====================

class _RSWrapper:
    """
    RealSense 相机轻量级封装
    使用 pyrealsense2 pipeline 方式
    """
    
    def __init__(self, serial: Optional[str] = None):
        """
        初始化 RealSense 相机
        
        Args:
            serial: 相机序列号，如果为 None 则使用第一个可用设备
        """
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        if serial:
            self.config.enable_device(serial)
        
        # 配置深度和彩色流
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 启动 pipeline
        self.profile = self.pipeline.start(self.config)
        
        # 创建对齐器（将深度对齐到彩色）
        self.align = rs.align(rs.stream.color)
        
        # 点云生成器
        self.pc = rs.pointcloud()
    
    def get_pcd_texture_depth(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], 
                                              Optional[np.ndarray], Optional[np.ndarray]]:
        """
        获取点云、点云颜色、深度图、彩色图
        
        Returns:
            (pcd_raw, pcd_color, depth_img, color_img)
            - pcd_raw: 点云 (N, 3)，相机坐标系
            - pcd_color: 点云颜色纹理坐标 (N, 2)
            - depth_img: 深度图 (H, W)
            - color_img: 彩色图 (H, W, 3)
            如果获取失败，返回 (None, None, None, None)
        """
        try:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()
            
            if not depth or not color:
                return None, None, None, None
            
            color_img = np.asanyarray(color.get_data())
            depth_img = np.asanyarray(depth.get_data())
            
            # 生成点云
            self.pc.map_to(color)
            points = self.pc.calculate(depth)
            verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32)
            
            return verts, tex, depth_img, color_img
            
        except Exception as e:
            print(f"[Error] Failed to get frames: {e}")
            return None, None, None, None
    
    def stop(self):
        """停止相机 pipeline"""
        try:
            self.pipeline.stop()
        except Exception:
            pass


# ==================== 设备初始化函数 ====================

def init_camera(camera_id: Optional[str] = None) -> _RSWrapper:
    """
    初始化 RealSense 相机
    
    Args:
        camera_id: 相机序列号，如果为 None 则使用第一个可用设备
    
    Returns:
        _RSWrapper 实例
    
    Raises:
        RuntimeError: 如果初始化失败
    """
    try:
        cam = _RSWrapper(serial=camera_id)
        # 预热一次，确保相机稳定
        cam.get_pcd_texture_depth()
        print("RealSense 相机就绪。")
        return cam
    except Exception as e:
        raise RuntimeError(f"RealSense 初始化失败: {e}")


def init_yolo(weight_path: str) -> YOLO:
    """
    初始化 YOLO 模型
    
    Args:
        weight_path: 模型权重文件路径
    
    Returns:
        YOLO 模型实例
    """
    print(f"加载模型权重: {weight_path}")
    try:
        model = YOLO(weight_path)
        print("YOLO 模型加载成功。")
        return model
    except Exception as e:
        raise RuntimeError(f"YOLO 模型加载失败: {e}")
