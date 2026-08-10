#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Record current Realman joint values together with a RealSense RGB/depth snapshot.

Run this file directly. Press Enter to record the current robot joint angles and
camera frame. Type q then Enter to quit.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_THIS_FILE = Path(__file__).resolve()
_WORKSPACE_ROOT = _THIS_FILE.parents[3]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from wrs.robot_con.realman.realman import RealmanArmController


ROBOT_IP = "192.168.1.18"
ROBOT_PORT = 8080
HAS_GRIPPER = False

OUTPUT_ROOT = _THIS_FILE.parent / "joint_realsense_records"
CAMERA_WARMUP_FRAMES = 15
CAMERA_DISCARD_FRAMES = 0
CAMERA_FRAME_TIMEOUT_MS = 15000
CAMERA_CAPTURE_RETRIES = 3
CAMERA_RETRY_SLEEP = 0.3
DEPTH_VIS_FAR = 1.0
WAIT_UNTIL_IDLE_BEFORE_RECORD = False
IDLE_TIMEOUT = 30.0


def safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", name)


def prepare_output_dirs(root: Path):
    for subdir in ("rgb", "depth_raw", "depth_png", "depth_vis", "metadata"):
        (root / subdir).mkdir(parents=True, exist_ok=True)


def intrinsics_to_dict(intrin):
    return {
        "width": int(intrin.width),
        "height": int(intrin.height),
        "ppx": float(intrin.ppx),
        "ppy": float(intrin.ppy),
        "fx": float(intrin.fx),
        "fy": float(intrin.fy),
        "model": str(intrin.model),
        "coeffs": [float(v) for v in intrin.coeffs],
    }


class RealSenseRecorder:
    def __init__(
            self,
            output_root: Path,
            warmup_frames: int = CAMERA_WARMUP_FRAMES,
            frame_timeout_ms: int = CAMERA_FRAME_TIMEOUT_MS,
            capture_retries: int = CAMERA_CAPTURE_RETRIES,
            retry_sleep: float = CAMERA_RETRY_SLEEP,
            depth_vis_far: float = DEPTH_VIS_FAR):
        import pyrealsense2 as rs

        self.rs = rs
        self.output_root = output_root
        self.frame_timeout_ms = int(frame_timeout_ms)
        self.capture_retries = max(0, int(capture_retries))
        self.retry_sleep = max(0.0, float(retry_sleep))
        self.depth_vis_far = float(depth_vis_far)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        for _ in range(max(0, int(warmup_frames))):
            self._wait_for_frames("warmup", retries=1)
        print("[Setup] RealSense started.")

    def close(self):
        self.pipeline.stop()
        print("[Cleanup] RealSense stopped.")

    def _wait_for_frames(self, label: str, retries: int | None = None):
        retry_count = self.capture_retries if retries is None else max(0, int(retries))
        last_error = None
        for attempt in range(retry_count + 1):
            try:
                return self.pipeline.wait_for_frames(self.frame_timeout_ms)
            except RuntimeError as e:
                last_error = e
                print(
                    f"       [Camera][warn] {label}: wait_for_frames timeout "
                    f"({attempt + 1}/{retry_count + 1}, timeout={self.frame_timeout_ms}ms): {e}"
                )
                if attempt < retry_count and self.retry_sleep > 0:
                    time.sleep(self.retry_sleep)
        raise RuntimeError(f"{label}: RealSense frame did not arrive after {retry_count + 1} attempts") from last_error

    def save_snapshot(self, idx: int, record_name: str, metadata: dict, discard_frames: int = CAMERA_DISCARD_FRAMES):
        for _ in range(max(0, int(discard_frames))):
            try:
                self._wait_for_frames(f"{record_name}/discard", retries=1)
            except RuntimeError as e:
                print(f"       [Camera][warn] {record_name}: discard frame failed, continuing capture: {e}")
                break

        frames = self._wait_for_frames(record_name)
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError(f"{record_name}: failed to get aligned depth/color frames")

        stem = f"{idx:04d}_{safe_name(record_name)}"
        color_bgr = np.asanyarray(color_frame.get_data())
        depth_z16 = np.asanyarray(depth_frame.get_data())
        depth_m = depth_z16.astype(np.float32) * float(self.depth_scale)

        rgb_path = self.output_root / "rgb" / f"{stem}.png"
        depth_raw_path = self.output_root / "depth_raw" / f"{stem}.npy"
        depth_png_path = self.output_root / "depth_png" / f"{stem}.png"
        depth_vis_path = self.output_root / "depth_vis" / f"{stem}.png"
        metadata_path = self.output_root / "metadata" / f"{stem}.json"

        cv2.imwrite(str(rgb_path), color_bgr)
        np.save(depth_raw_path, depth_m)

        depth_mm = np.clip(depth_m * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        cv2.imwrite(str(depth_png_path), depth_mm)

        valid_depth = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth_vis_src = np.clip(valid_depth, 0.0, self.depth_vis_far)
        depth_vis = cv2.normalize(depth_vis_src, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(depth_vis_path), depth_vis)

        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        metadata = dict(metadata)
        metadata.update({
            "depth_scale": float(self.depth_scale),
            "depth_intrinsics": intrinsics_to_dict(depth_intrin),
            "color_intrinsics": intrinsics_to_dict(color_intrin),
            "timestamp_ms": float(color_frame.get_timestamp()),
            "rgb_path": str(rgb_path),
            "depth_raw_path": str(depth_raw_path),
            "depth_png_path": str(depth_png_path),
            "depth_vis_path": str(depth_vis_path),
            "metadata_path": str(metadata_path),
        })
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return metadata


class JointRealSenseRecorder:
    def __init__(self):
        prepare_output_dirs(OUTPUT_ROOT)
        self.records_csv = OUTPUT_ROOT / "records.csv"
        self.records_jsonl = OUTPUT_ROOT / "records.jsonl"
        self.controller = None
        self.camera = None
        try:
            self.controller = RealmanArmController(ip=ROBOT_IP, port=ROBOT_PORT, has_gripper=HAS_GRIPPER)
            self.camera = RealSenseRecorder(OUTPUT_ROOT)
        except Exception:
            self.close()
            raise
        self.next_index = self._load_next_index()
        self._ensure_csv_header()
        print(f"[Setup] output root: {OUTPUT_ROOT}")

    def close(self):
        if self.camera is not None:
            self.camera.close()
        if self.controller is not None:
            self.controller.close_connection()

    def _load_next_index(self) -> int:
        metadata_dir = OUTPUT_ROOT / "metadata"
        max_idx = 0
        for path in metadata_dir.glob("*.json"):
            match = re.match(r"(\d{4})_", path.name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
        return max_idx + 1

    def _ensure_csv_header(self):
        if self.records_csv.is_file():
            return
        with open(self.records_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "index",
                "record_name",
                "timestamp_unix",
                "joint_1_rad",
                "joint_2_rad",
                "joint_3_rad",
                "joint_4_rad",
                "joint_5_rad",
                "joint_6_rad",
                "joint_1_deg",
                "joint_2_deg",
                "joint_3_deg",
                "joint_4_deg",
                "joint_5_deg",
                "joint_6_deg",
                "rgb_path",
                "depth_raw_path",
                "depth_png_path",
                "depth_vis_path",
                "metadata_path",
            ])

    def _append_record_tables(self, metadata: dict):
        joints_rad = metadata["joint_values_rad"]
        joints_deg = metadata["joint_values_deg"]
        with open(self.records_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                metadata["index"],
                metadata["record_name"],
                metadata["timestamp_unix"],
                *joints_rad,
                *joints_deg,
                metadata["rgb_path"],
                metadata["depth_raw_path"],
                metadata["depth_png_path"],
                metadata["depth_vis_path"],
                metadata["metadata_path"],
            ])
        with open(self.records_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    def record_once(self, record_name_prefix: str = "joint_record"):
        idx = self.next_index
        record_name = f"{safe_name(record_name_prefix)}_{idx:04d}"
        if WAIT_UNTIL_IDLE_BEFORE_RECORD:
            print("[Record] waiting until robot is idle.")
            self.controller.wait_until_idle(timeout=IDLE_TIMEOUT)

        joints_rad = np.asarray(self.controller.get_joint_values(), dtype=np.float64)
        try:
            ee_pos, ee_rotmat = self.controller.get_pose()
            ee_pos_list = np.asarray(ee_pos, dtype=float).tolist()
            ee_rotmat_list = np.asarray(ee_rotmat, dtype=float).tolist()
        except Exception as e:
            print(f"[Warn] failed to query current pose: {e}")
            ee_pos_list = None
            ee_rotmat_list = None

        metadata = {
            "index": idx,
            "record_name": record_name,
            "timestamp_unix": time.time(),
            "joint_values_rad": joints_rad.tolist(),
            "joint_values_deg": np.degrees(joints_rad).tolist(),
            "ee_pos": ee_pos_list,
            "ee_rotmat": ee_rotmat_list,
        }
        metadata = self.camera.save_snapshot(idx, record_name, metadata, discard_frames=CAMERA_DISCARD_FRAMES)
        self._append_record_tables(metadata)
        self.next_index += 1
        print(f"[Record] saved {record_name}")
        print(f"         joints(rad) = {np.round(joints_rad, 8)}")

    def run(self):
        print("\n" + "=" * 72)
        print("  Realman joint + RealSense recorder")
        print("  Enter       : record current joints and camera frame")
        print("  name + Enter: record with a custom name prefix")
        print("  q + Enter   : quit")
        print("=" * 72 + "\n")
        while True:
            line = input("[Idle] Press Enter to record, or q + Enter to quit: ").strip()
            if line.lower() == "q":
                print("[Exit] user requested quit.")
                return
            if line:
                self.record_once(record_name_prefix=line)
            else:
                self.record_once()


def main():
    recorder = None
    try:
        recorder = JointRealSenseRecorder()
        recorder.run()
    finally:
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    main()
