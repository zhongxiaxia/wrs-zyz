#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/01/30
# @Author : cursor
"""
YOLO 分割通用工具模块
功能：
1. 分割模型初始化
2. 分割掩码提取
3. 掩码像素点 -> 3D 点
4. 输出与图像对齐的 3D 目标点
"""

from typing import List, Optional, Tuple, Dict, Any
import os
import sys
import importlib
import importlib.abc
import importlib.machinery
import importlib.util

import cv2
import numpy as np
import torch.nn as nn
from ultralytics import YOLO
import pyrealsense2 as rs
import wrs.basis.robot_math as rm

# ── 兼容性补丁 ────────────────────────────────────────────────────────────────
# best.pt 由含自定义分支的 ultralytics（F:\WRS\wrs-virtual\wrs\Pepper\ultralytics）
# 训练，pickle 反序列化时会查找 ultralytics.nn.extra_modules 及其子模块。
# 该自定义分支只提供了编译后的 .pyc 文件（无 .py 源文件），且 extra_modules 依赖
# 自定义的 ultralytics.nn.modules（含 DSConv 等新类）。
# 修复策略：
#   1. 先将自定义 modules/conv.pyc 中的新类（DSConv 等）注入到已安装的标准模块；
#   2. MetaPathFinder 再从 extra_modules/*.pyc 加载真实实现（此时依赖已满足）；
#   3. 任何步骤失败均降级为空壳，确保分割模型（Pepper_NBV/best.pt）不受影响。

_PEPPER_NN_DIR     = r"F:\WRS\wrs-virtual\wrs\Pepper\ultralytics\nn"
_EXTRA_MODULES_PYC = os.path.join(_PEPPER_NN_DIR, "extra_modules", "__pycache__")
_CUSTOM_MODULES_PYC = os.path.join(_PEPPER_NN_DIR, "modules", "__pycache__")
_PY_TAG = f"cpython-{sys.version_info.major}{sys.version_info.minor}"


def _load_pyc_as_temp(pyc_path: str, temp_name: str):
    """用 SourcelessFileLoader 把 .pyc 加载为临时模块，返回模块对象或 None。"""
    if not os.path.isfile(pyc_path):
        return None
    try:
        loader = importlib.machinery.SourcelessFileLoader(temp_name, pyc_path)
        spec   = importlib.util.spec_from_loader(temp_name, loader)
        mod    = importlib.util.module_from_spec(spec)
        # 设置 __package__ 让 .pyc 内的相对 import 能解析到标准包
        mod.__package__ = "ultralytics.nn.modules"
        sys.modules[temp_name] = mod
        loader.exec_module(mod)
        return mod
    except Exception as e:
        sys.modules.pop(temp_name, None)
        print(f"[yolo_seg] 临时加载 {pyc_path} 失败: {e}")
        return None


def _patch_custom_nn_modules():
    """
    将自定义 ultralytics.nn.modules/*.pyc 中的新增类注入到已安装的标准包，
    使 extra_modules 依赖（如 DSConv）得以满足。仅注入标准包中不存在的名称。
    """
    sub_names = ["conv", "block", "head", "transformer", "utils"]
    for sub in sub_names:
        pyc_path = os.path.join(_CUSTOM_MODULES_PYC, f"{sub}.{_PY_TAG}.pyc")
        target_name = f"ultralytics.nn.modules.{sub}"
        target_mod = sys.modules.get(target_name)
        if target_mod is None:
            continue

        custom_mod = _load_pyc_as_temp(pyc_path, f"_pepper_custom_modules_{sub}")
        if custom_mod is None:
            continue

        injected = []
        for name in dir(custom_mod):
            if not name.startswith("_") and not hasattr(target_mod, name):
                try:
                    setattr(target_mod, name, getattr(custom_mod, name))
                    # 同步注入到顶层 ultralytics.nn.modules 命名空间
                    parent = sys.modules.get("ultralytics.nn.modules")
                    if parent is not None:
                        setattr(parent, name, getattr(custom_mod, name))
                    injected.append(name)
                except Exception:
                    pass
        if injected:
            print(f"[yolo_seg] 注入 {sub}: {injected}")


def _find_extra_pyc(module_leaf: str) -> Optional[str]:
    """在自定义 extra_modules/__pycache__ 中搜索对应 .pyc，返回路径或 None。"""
    if not os.path.isdir(_EXTRA_MODULES_PYC):
        return None
    pyc_path = os.path.join(_EXTRA_MODULES_PYC, f"{module_leaf}.{_PY_TAG}.pyc")
    return pyc_path if os.path.isfile(pyc_path) else None


class _ExtraModulesLoader(importlib.abc.Loader):
    """先尝试加载真实 .pyc；找不到时降级为空壳，避免 ImportError。"""

    def __init__(self, pyc_path: Optional[str] = None):
        self._pyc_path = pyc_path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if self._pyc_path and os.path.isfile(self._pyc_path):
            try:
                loader = importlib.machinery.SourcelessFileLoader(
                    module.__name__, self._pyc_path
                )
                loader.exec_module(module)
                return
            except Exception as e:
                print(f"[yolo_seg] 加载 extra_modules .pyc 失败，降级为空壳: "
                      f"{os.path.basename(self._pyc_path)} ({e})")

        # 降级：提供能通过 pickle 反序列化的空壳类
        class _AnyLayer(nn.Module):
            def __init__(self, *a, **k):
                super().__init__()
            def forward(self, x):
                return x if not isinstance(x, (list, tuple)) else x[0]

        module.__getattr__ = lambda name: type(
            name, (_AnyLayer,), {"__module__": module.__name__}
        )


class _ExtraModulesFinder(importlib.abc.MetaPathFinder):
    _PREFIX = "ultralytics.nn.extra_modules"

    def find_spec(self, fullname, path, target=None):
        if fullname != self._PREFIX and not fullname.startswith(self._PREFIX + "."):
            return None
        leaf = fullname.rsplit(".", 1)[-1] if "." in fullname else "__init__"
        if leaf == "extra_modules":
            leaf = "__init__"
        pyc_path = _find_extra_pyc(leaf)
        return importlib.machinery.ModuleSpec(
            fullname, _ExtraModulesLoader(pyc_path), is_package=(leaf == "__init__")
        )


# ① 注册 finder（必须在 ultralytics 已 import 之后执行，保证 nn.modules 在 sys.modules 里）
if not any(isinstance(f, _ExtraModulesFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _ExtraModulesFinder())

# ② 将自定义 nn.modules 新类注入到已安装的标准包（为 extra_modules 提供依赖）
_patch_custom_nn_modules()
# ─────────────────────────────────────────────────────────────────────────────


def init_yolo_seg(weight_path: str) -> YOLO:
    """初始化 YOLO 分割模型（task=segment）"""
    print(f"加载分割模型权重: {weight_path}")
    try:
        model = YOLO(weight_path)
        print(f"YOLO 分割模型加载成功。task={model.task}  names={model.names}")
        return model
    except Exception as e:
        raise RuntimeError(f"YOLO 分割模型加载失败: {e}")


def init_yolo_pose(weight_path: str) -> YOLO:
    """初始化 YOLO 关键点模型（task=pose），用于直接输出采摘点"""
    print(f"加载关键点模型权重: {weight_path}")
    try:
        model = YOLO(weight_path)
        print(f"YOLO 关键点模型加载成功。task={model.task}  names={model.names}")
        return model
    except Exception as e:
        raise RuntimeError(f"YOLO 关键点模型加载失败: {e}")


def init_yolo_auto(weight_path: str) -> YOLO:
    """自动加载模型，根据 task 自动识别分割/关键点类型"""
    print(f"自动加载模型权重: {weight_path}")
    try:
        model = YOLO(weight_path)
        print(f"YOLO 模型加载成功。task={model.task}  names={model.names}")
        return model
    except Exception as e:
        raise RuntimeError(f"YOLO 模型加载失败: {e}")


def yolo_pose_keypoints(
    yolo_model: YOLO,
    color_img: np.ndarray,
    conf_thres: float = 0.3,
    keypoint_idx: int = 0,
    show: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
    """
    关键点模型推理，返回每个检测目标的关键点像素坐标。

    参数
    ----
    yolo_model   : YOLO pose 模型
    color_img    : BGR 图像
    conf_thres   : 检测置信度阈值
    keypoint_idx : 使用第几个关键点作为采摘点（默认 0 = 第一个关键点）
    show         : 是否生成可视化图像

    返回
    ----
    items : 列表，每项 dict 包含:
        cls_id  - 类别 ID
        conf    - 置信度
        kpt_uv  - (u, v) 像素坐标 (int, int)
        kpt_conf- 关键点置信度
        box     - xyxy 边界框
    im_array : 可视化图像（show=True 时有效）
    """
    results = yolo_model(color_img, verbose=False)
    if not results:
        return [], None

    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return [], None
    if res.keypoints is None:
        return [], None

    boxes = res.boxes
    kpts  = res.keypoints  # shape: (N, K, 2) 或 (N, K, 3)

    items: List[Dict[str, Any]] = []
    for i in range(len(boxes)):
        conf = float(boxes.conf[i].item())
        if conf < conf_thres:
            continue
        cls_id = int(boxes.cls[i].item())
        box_xyxy = boxes.xyxy[i].detach().cpu().numpy()

        kpt_data = kpts.data[i]            # (K, 2) 或 (K, 3)
        if keypoint_idx >= kpt_data.shape[0]:
            continue

        kp = kpt_data[keypoint_idx].detach().cpu().numpy()
        u, v = int(kp[0]), int(kp[1])
        kpt_conf = float(kp[2]) if kp.shape[0] > 2 else 1.0

        if u <= 0 and v <= 0:
            continue

        items.append({
            "cls_id":   cls_id,
            "conf":     conf,
            "kpt_uv":   (u, v),
            "kpt_conf": kpt_conf,
            "box":      box_xyxy,
        })

    im_array = res.plot() if show else None
    return items, im_array


def transform_points_by_homomat(homomat: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    使用齐次变换矩阵变换点云
    """
    if not isinstance(points, np.ndarray):
        raise ValueError("Points must be np.ndarray!")

    if points.ndim == 1:
        points = points.reshape(1, -1)
        single_point = True
    else:
        single_point = False

    homo_points = np.ones((4, points.shape[0]))
    homo_points[:3, :] = points.T
    transformed = (homomat @ homo_points).T[:, :3]
    return transformed[0] if single_point else transformed


def _estimate_point_from_neighborhood(
    target_pixel: Tuple[int, int],
    pcd_matrix: np.ndarray,
    neighborhood_size: int = 5,
    mad_k: float = 2.5,
    min_inliers: int = 6,
    use_depth_only: bool = True,
    max_range: Optional[float] = None,
) -> Optional[np.ndarray]:
    """
    使用邻域鲁棒估计(MAD)估算 3D 点
    """
    h, w = pcd_matrix.shape[:2]
    px, py = target_pixel
    half = neighborhood_size // 2

    x_min = max(0, px - half)
    x_max = min(w - 1, px + half)
    y_min = max(0, py - half)
    y_max = min(h - 1, py + half)

    neighborhood = pcd_matrix[y_min:y_max + 1, x_min:x_max + 1].reshape(-1, 3)

    valid = neighborhood[np.isfinite(neighborhood).all(axis=1)]
    valid = valid[np.linalg.norm(valid, axis=1) > 1e-6]
    valid = valid[valid[:, 2] > 1e-6]

    if max_range is not None:
        valid = valid[valid[:, 2] < max_range]

    if len(valid) == 0:
        return None

    med = np.median(valid, axis=0)

    if use_depth_only:
        dev = np.abs(valid[:, 2] - med[2])
    else:
        dev = np.linalg.norm(valid - med, axis=1)

    mad = np.median(dev)
    if mad < 1e-9:
        return med

    robust_sigma = 1.4826 * mad
    inliers = valid[dev <= mad_k * robust_sigma]
    if len(inliers) < min_inliers:
        return med

    return np.mean(inliers, axis=0)


def transform_point_cloud_handeye(self,
                                  handeye_mat: np.ndarray,
                                  pcd: np.ndarray,
                                  given_conf: Optional[np.ndarray] = None,
                                  toggle_debug: bool = False) -> np.ndarray:
    """
    结合手眼标定矩阵，将点从 Camera 坐标系转换到 World 坐标系
    """
    arm = self.robot

    if given_conf is None:
        controller = getattr(self, 'rm_controller', None)
        if getattr(self, "real_robot", False) and controller:
            given_conf = controller.get_joint_values()
        else:
            given_conf = arm.get_jnt_values()

    gl_tcp_pos, gl_tcp_rotmat = arm.fk(given_conf, update=False)
    w2r_mat = rm.homomat_from_posrot(gl_tcp_pos, gl_tcp_rotmat)
    w2cam = w2r_mat.dot(handeye_mat)
    pcd_r = transform_points_by_homomat(w2cam, pcd)

    if toggle_debug:
        print(f"[Debug] Transformed {pcd.shape[0] if pcd.ndim > 1 else 1} points")
        print(f"[Debug] TCP pose: {gl_tcp_pos}")

    return pcd_r


def yolo_seg_masks(yolo_model: YOLO,
                   color_img: np.ndarray,
                   conf_thres: float = 0.3,
                   mask_thres: float = 0.5,
                   show: bool = False) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
    """
    分割推理，返回掩码与检测信息
    返回列表每项包含: cls_id, conf, mask, box
    """
    results = yolo_model(color_img, verbose=False)
    if not results:
        return [], None

    res = results[0]
    if res.masks is None or res.boxes is None or len(res.boxes) == 0:
        return [], None

    masks = res.masks.data
    boxes = res.boxes

    items: List[Dict[str, Any]] = []
    for i in range(len(boxes)):
        conf = float(boxes.conf[i].item())
        if conf < conf_thres:
            continue
        cls_id = int(boxes.cls[i].item())
        mask = masks[i].detach().cpu().numpy() > mask_thres
        box_xyxy = boxes.xyxy[i].detach().cpu().numpy()
        items.append({
            "cls_id": cls_id,
            "conf": conf,
            "mask": mask,
            "box": box_xyxy,
        })

    im_array = res.plot() if show else None
    return items, im_array


def mask_center_pixel(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    计算掩码的像素质心
    """
    if mask is None or mask.size == 0:
        return None
    mask_uint8 = (mask.astype(np.uint8) * 255)
    moments = cv2.moments(mask_uint8)
    if moments["m00"] <= 0:
        return None
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return cx, cy


def _sample_mask_pixels(mask: np.ndarray, sample_n: int = 200) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    if sample_n <= 0 or sample_n >= len(xs):
        idx = np.arange(len(xs))
    else:
        idx = np.random.choice(len(xs), size=sample_n, replace=False)
    return ys[idx], xs[idx]


def mask_to_3d_point(mask: np.ndarray,
                     pcd_matrix: np.ndarray,
                     sample_n: int = 200,
                     method: str = "median") -> Optional[np.ndarray]:
    """
    掩码像素点 -> 3D 点
    pcd_matrix 与 color_img 对齐，形状为 (H, W, 3)
    method: "median" | "mean" | "nearest"
    """
    sampled = _sample_mask_pixels(mask, sample_n=sample_n)
    if sampled is None:
        return None
    ys, xs = sampled
    pts = pcd_matrix[ys, xs]

    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = pts[np.linalg.norm(pts, axis=1) > 1e-6]
    pts = pts[pts[:, 2] > 1e-6]
    if len(pts) == 0:
        return None

    if method == "nearest":
        return pts[np.argmin(pts[:, 2])]
    if method == "mean":
        return np.mean(pts, axis=0)
    return np.median(pts, axis=0)


def yolo_seg_points(yolo_model: YOLO,
                    color_img: np.ndarray,
                    pcd_matrix: np.ndarray,
                    conf_thres: float = 0.3,
                    mask_thres: float = 0.5,
                    sample_n: int = 200,
                    method: str = "median",
                    show: bool = False) -> Tuple[List[Tuple[int, np.ndarray, float]], Optional[np.ndarray]]:
    """
    分割结果 -> 3D 点列表
    返回: [(cls_id, point_xyz, conf), ...] 与可视化图像
    注意：pcd_matrix 是什么坐标系，返回就是什么坐标系
    """
    items, im_array = yolo_seg_masks(
        yolo_model,
        color_img,
        conf_thres=conf_thres,
        mask_thres=mask_thres,
        show=show,
    )

    results: List[Tuple[int, np.ndarray, float]] = []
    for item in items:
        pt = mask_to_3d_point(
            item["mask"],
            pcd_matrix,
            sample_n=sample_n,
            method=method,
        )
        if pt is None:
            continue
        results.append((item["cls_id"], pt, item["conf"]))

    return results, im_array


# ==================== RealSense 封装 ====================

class _RSWrapper:
    """
    RealSense 相机轻量级封装
    """

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

    def get_pcd_texture_depth(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
                                              Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()

            if not depth or not color:
                return None, None, None, None

            color_img = np.asanyarray(color.get_data())
            depth_img = np.asanyarray(depth.get_data())

            self.pc.map_to(color)
            points = self.pc.calculate(depth)
            verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32)

            return verts, tex, depth_img, color_img

        except Exception as e:
            print(f"[Error] Failed to get frames: {e}")
            return None, None, None, None

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


def init_camera(camera_id: Optional[str] = None) -> _RSWrapper:
    """
    初始化 RealSense 相机
    """
    try:
        cam = _RSWrapper(serial=camera_id)
        cam.get_pcd_texture_depth()
        print("RealSense 相机就绪。")
        return cam
    except Exception as e:
        raise RuntimeError(f"RealSense 初始化失败: {e}")
