#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import time
import numpy as np
import cv2
import pyrealsense2 as rs


@dataclass(frozen=True)
class CapturePCDResult:
    color_img: np.ndarray
    pcd_cam: np.ndarray
    depth_intrin: Any
    t_wait: float
    t_align: float
    t_pcd: float


def safe_imshow(window_name: str, img: np.ndarray, wait_ms: int = 1) -> bool:
    try:
        cv2.imshow(window_name, img)
        cv2.waitKey(int(wait_ms))
        return True
    except Exception:
        return False


def yolo_vis_image(results, fallback_bgr: np.ndarray) -> np.ndarray:
    try:
        if results:
            return results[0].plot()
    except Exception:
        pass
    return fallback_bgr


def yolo_predict(yolo_model, color_img: np.ndarray, *, save: bool = False, show_conf: bool = False):
    t0 = time.time()
    results = yolo_model.predict(color_img, save=save, show_conf=show_conf)
    return results, time.time() - t0


def capture_color_and_pcd(
    rs_pipeline,
    rs_align,
    *,
    stabilize_sleep_s: float = 0.0,
) -> Optional[CapturePCDResult]:
    if rs_pipeline is None or rs_align is None:
        return None

    if stabilize_sleep_s and stabilize_sleep_s > 0:
        time.sleep(float(stabilize_sleep_s))

    _t0 = time.time()
    frames = rs_pipeline.wait_for_frames()
    t_wait = time.time() - _t0

    _t0 = time.time()
    aligned = rs_align.process(frames)
    t_align = time.time() - _t0
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    if not depth_frame or not color_frame:
        return None

    color_img = np.asanyarray(color_frame.get_data())
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

    _t0 = time.time()
    pc = rs.pointcloud()
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    t_pcd = time.time() - _t0

    pcd_cam = np.asanyarray(points.get_vertices()).view(np.float32).reshape(
        depth_intrin.height, depth_intrin.width, 3
    )

    return CapturePCDResult(
        color_img=color_img,
        pcd_cam=pcd_cam,
        depth_intrin=depth_intrin,
        t_wait=t_wait,
        t_align=t_align,
        t_pcd=t_pcd,
    )

