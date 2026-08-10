#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""连续拍摄深度图，点击目标并记录 B1 世界坐标。

坐标转换与 ``pick/pepper_picker.py`` 保持一致：

    World->Camera = World->TCP(FK) @ TCP->Camera(手眼矩阵)

程序只读取机械臂状态和相机图像，不会移动机械臂。
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


THIS_FILE = Path(__file__).resolve()
WORKSPACE_ROOT = THIS_FILE.parents[3]
WRS_ROOT = THIS_FILE.parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import wrs.basis.robot_math as rm
from wrs.robot_con.realman.realman import RealmanArmController
from wrs.robot_sim.robots.realman.realman_single_arm import Realman


# ======================== 现场配置 ========================
RECORD_LABEL = "B1"
ROBOT_IP = "192.168.1.18"
ROBOT_PORT = 8080
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_WARMUP_FRAMES = 20
MAX_CAPTURE_MOTION_DEG = 0.10
MAX_VALID_DEPTH_M = 2.0
HANDEYE_PATH = (
    WRS_ROOT / "vision" / "depth_camera" / "manual_calibration_realman22.json"
)
OUTPUT_DIR = THIS_FILE.parent / "B1_records"
WINDOW_NAME = "B1: click color or aligned depth image"
# =========================================================


def load_handeye_matrix() -> np.ndarray:
    with HANDEYE_PATH.open("r", encoding="utf-8") as stream:
        matrix = np.asarray(json.load(stream)["affine_mat"], dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"手眼矩阵必须是 4x4：{HANDEYE_PATH}")
    return matrix


def depth_visualization(depth_m: np.ndarray) -> np.ndarray:
    valid_depth = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    clipped = np.clip(valid_depth, 0.0, MAX_VALID_DEPTH_M)
    gray = np.round(clipped / MAX_VALID_DEPTH_M * 255.0).astype(np.uint8)
    image = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    image[depth_m <= 0.0] = 0
    return image


def choose_pixel(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
) -> tuple[int, int] | None:
    """在图像上单击并返回像素坐标；取消时返回 None。"""
    height, width = color_bgr.shape[:2]
    canvas = np.hstack((color_bgr, depth_visualization(depth_m)))
    state: dict[str, tuple[int, int] | None] = {"pixel": None}

    cv2.line(canvas, (width, 0), (width, height - 1), (255, 255, 255), 1)
    cv2.putText(
        canvas,
        "COLOR",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "ALIGNED DEPTH",
        (width + 12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Click target point | Q/Esc: cancel",
        (12, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    def on_mouse(event, x, y, unused_flags, unused_param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        original_x = x if x < width else x - width
        if 0 <= original_x < width and 0 <= y < height:
            state["pixel"] = (int(original_x), int(y))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, min(1600, width * 2), min(900, height))
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    cv2.imshow(WINDOW_NAME, canvas)

    while state["pixel"] is None:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(WINDOW_NAME)
            return None
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return None

    pixel = state["pixel"]
    cv2.destroyWindow(WINDOW_NAME)
    return pixel


def camera_to_world(
    camera_point_m: np.ndarray,
    joint_values_rad: np.ndarray,
    robot_model: Realman,
    handeye_tcp_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tcp_world_position, tcp_world_rotation = robot_model.fk(
        joint_values_rad,
        update=False,
    )
    world_to_tcp = rm.homomat_from_posrot(
        tcp_world_position,
        tcp_world_rotation,
    )
    world_to_camera = world_to_tcp @ handeye_tcp_to_camera
    camera_point_h = np.append(np.asarray(camera_point_m, dtype=float), 1.0)
    world_point = (world_to_camera @ camera_point_h)[:3]
    return world_point, world_to_camera


def save_result(
    result: dict,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    pixel: tuple[int, int],
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"{RECORD_LABEL}_{timestamp}"

    clicked_image = color_bgr.copy()
    cv2.drawMarker(
        clicked_image,
        pixel,
        (0, 255, 255),
        cv2.MARKER_CROSS,
        30,
        2,
    )
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_clicked.png"), clicked_image)
    cv2.imwrite(
        str(OUTPUT_DIR / f"{stem}_depth.png"),
        depth_visualization(depth_m),
    )
    np.save(OUTPUT_DIR / f"{stem}_depth_m.npy", depth_m)

    output_path = OUTPUT_DIR / f"{stem}.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def capture_one_record(
    controller: RealmanArmController,
    pipeline: rs.pipeline,
    align: rs.align,
    depth_scale: float,
    robot_model: Realman,
    handeye: np.ndarray,
) -> Path | None:
    """采集并保存一条记录；用户取消点选时不保存。"""
    controller.wait_until_idle(timeout=30.0)
    joints_before = np.asarray(
        controller.get_joint_values(),
        dtype=float,
    )
    frames = align.process(pipeline.wait_for_frames())
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        raise RuntimeError("没有获取到对齐后的深度帧和彩色帧")

    color_bgr = np.asanyarray(color_frame.get_data()).copy()
    depth_z16 = np.asanyarray(depth_frame.get_data()).copy()
    depth_m = depth_z16.astype(np.float32) * depth_scale
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    joints_after = np.asarray(
        controller.get_joint_values(),
        dtype=float,
    )

    capture_motion_deg = float(
        np.degrees(np.max(np.abs(joints_after - joints_before)))
    )
    if capture_motion_deg > MAX_CAPTURE_MOTION_DEG:
        raise RuntimeError(
            "拍照期间机械臂发生移动："
            f"{capture_motion_deg:.4f} deg > "
            f"{MAX_CAPTURE_MOTION_DEG:.4f} deg"
        )

    pixel = choose_pixel(color_bgr, depth_m)
    if pixel is None:
        print("已取消本次点选，未保存数据。")
        return None

    depth_value_m = float(depth_frame.get_distance(pixel[0], pixel[1]))
    if not 0.0 < depth_value_m <= MAX_VALID_DEPTH_M:
        raise RuntimeError(
            f"点击位置 {pixel} 的深度无效：{depth_value_m:.6f} m"
        )

    camera_point_m = np.asarray(
        rs.rs2_deproject_pixel_to_point(
            intrinsics,
            [float(pixel[0]), float(pixel[1])],
            depth_value_m,
        ),
        dtype=float,
    )
    world_point_m, world_to_camera = camera_to_world(
        camera_point_m,
        joints_after,
        robot_model,
        handeye,
    )

    result = {
        "label": RECORD_LABEL,
        "recorded_at": time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(),
        ),
        "coordinate_frame": "pepper_picker world",
        "coordinate_unit": "m",
        "b1_world_m": world_point_m.tolist(),
        "pixel": [int(pixel[0]), int(pixel[1])],
        "depth_m": depth_value_m,
        "camera_point_m": camera_point_m.tolist(),
        "joint_values_rad": joints_after.tolist(),
        "joint_values_deg": np.degrees(joints_after).tolist(),
        "capture_motion_deg": capture_motion_deg,
        "handeye_path": str(HANDEYE_PATH.resolve()),
        "handeye_tcp_to_camera": handeye.tolist(),
        "world_to_camera": world_to_camera.tolist(),
    }
    output_path = save_result(
        result,
        color_bgr,
        depth_m,
        pixel,
    )

    print(f"B1 世界坐标（m）： {np.round(world_point_m, 6)}")
    print(f"相机坐标（m）：    {np.round(camera_point_m, 6)}")
    print(f"像素：{pixel}，深度：{depth_value_m:.6f} m")
    print(f"保存位置：{output_path}")
    return output_path


def main() -> int:
    controller = None
    pipeline = None

    try:
        print(f"========== 记录 {RECORD_LABEL} ==========")
        handeye = load_handeye_matrix()
        robot_model = Realman(enable_cc=False)
        controller = RealmanArmController(
            ip=ROBOT_IP,
            port=ROBOT_PORT,
            has_gripper=False,
        )
        controller.wait_until_idle(timeout=30.0)

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.depth,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            rs.format.z16,
            CAMERA_FPS,
        )
        config.enable_stream(
            rs.stream.color,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            rs.format.bgr8,
            CAMERA_FPS,
        )
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        depth_scale = float(
            profile.get_device().first_depth_sensor().get_depth_scale()
        )

        for _ in range(CAMERA_WARMUP_FRAMES):
            pipeline.wait_for_frames()

        record_count = 0
        while True:
            print("\n请让机械臂和相机保持静止。")
            command = input(
                "按 Enter 拍摄并点选；输入 q 后按 Enter 退出："
            ).strip().lower()
            if command in {"q", "quit", "exit"}:
                break

            try:
                output_path = capture_one_record(
                    controller,
                    pipeline,
                    align,
                    depth_scale,
                    robot_model,
                    handeye,
                )
                if output_path is not None:
                    record_count += 1
                    print(f"第 {record_count} 条记录完成。")
            except Exception as exc:
                print(
                    f"[本次记录失败] {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

        print(f"已退出，共成功记录 {record_count} 条数据。")
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\n已停止记录。")
        return 0
    except Exception as exc:
        print(f"[初始化失败] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        if controller is not None:
            try:
                controller.close_connection()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
