#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Save and load complete pepper-picking motion records."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
DEFAULT_RECORD_DIR = Path(__file__).resolve().parent / "path_records"


def _validated_path(path, name: str) -> np.ndarray:
    values = getattr(path, "jv_list", path)
    array = np.asarray(values, dtype=float)
    if (
            array.ndim != 2
            or array.shape[0] < 2
            or array.shape[1] != 6
            or not np.all(np.isfinite(array))):
        raise ValueError(f"path segment {name!r} must be a finite (N, 6) array, N >= 2")
    return array


def _tcp_positions(robot, path: np.ndarray) -> np.ndarray:
    positions = []
    for conf in path:
        tcp_pos, _ = robot.fk(conf)
        positions.append(np.asarray(tcp_pos, dtype=float))
    return np.asarray(positions, dtype=float)


def _join_segments(segment_arrays):
    """Join segment paths while retaining and reporting real discontinuities."""
    joined = []
    join_gaps = []
    previous_end = None
    for index, array in enumerate(segment_arrays):
        start_index = 0
        if previous_end is not None:
            gap = np.asarray(array[0] - previous_end, dtype=float)
            gap_max_rad = float(np.max(np.abs(gap)))
            join_gaps.append({
                "from_segment_index": index - 1,
                "to_segment_index": index,
                "max_abs_rad": gap_max_rad,
                "max_abs_deg": float(np.degrees(gap_max_rad)),
            })
            if np.allclose(previous_end, array[0], atol=1e-8, rtol=0.0):
                start_index = 1
        joined.extend(array[start_index:])
        previous_end = array[-1]
    return np.asarray(joined, dtype=float), join_gaps


def save_pick_path_record(
        *,
        robot,
        segments,
        cycle_number: int,
        run_id: str,
        path_method: str,
        target_pos,
        approach_distance_m: float,
        grasp_angle_deg: float,
        control_period_s: float,
        obstacle,
        real_robot: bool,
        output_dir=None,
        extra_metadata=None,
) -> Path:
    """Persist one successful, closed-loop picking path as portable JSON.

    ``segments`` is an ordered iterable of mappings containing ``name``,
    ``planner`` and ``path``. Paths are joint configurations in radians.
    TCP positions are calculated from the same robot model used for planning.
    """
    output_dir = DEFAULT_RECORD_DIR if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_records = []
    segment_arrays = []
    for segment in segments:
        name = str(segment["name"])
        path = _validated_path(segment["path"], name)
        tcp_xyz = _tcp_positions(robot, path)
        segment_arrays.append(path)
        segment_records.append({
            "name": name,
            "planner": str(segment.get("planner", "unknown")),
            "motion_profile": str(segment.get("motion_profile", "transfer")),
            "joint_values_rad": path.tolist(),
            "joint_values_deg": np.degrees(path).tolist(),
            "tcp_xyz_m": tcp_xyz.tolist(),
            "waypoint_count": int(len(path)),
        })

    full_path, join_gaps = _join_segments(segment_arrays)
    full_tcp = _tcp_positions(robot, full_path)

    obstacle_record = None
    if obstacle is not None:
        obstacle_record = {
            "name": str(obstacle.get("name", "obstacle")),
            "mesh_path": str(obstacle["mesh_path"]),
            "position_m": np.asarray(obstacle["position_m"], dtype=float).tolist(),
            "rotation_matrix": np.asarray(
                obstacle.get("rotation_matrix", np.eye(3)), dtype=float
            ).tolist(),
            "rgb": np.asarray(obstacle.get("rgb", [0.6, 0.6, 0.6]), dtype=float).tolist(),
            "alpha": float(obstacle.get("alpha", 0.9)),
        }

    record = {
        "schema": "wrs.pepper_pick_path",
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "created_at_unix_ns": time.time_ns(),
        "run_id": str(run_id),
        "cycle_number": int(cycle_number),
        "path_method": str(path_method),
        "real_robot": bool(real_robot),
        "joint_unit": "rad",
        "tcp_position_unit": "m",
        "control_period_s": float(control_period_s),
        "target_position_m": np.asarray(target_pos, dtype=float).tolist(),
        "approach_distance_m": float(approach_distance_m),
        "grasp_angle_deg": float(grasp_angle_deg),
        "robot_model": {
            "class": "wrs.robot_sim.robots.realman.realman_single_arm.Realman",
            "manipulator": "RM65B",
            "joint_count": 6,
        },
        "obstacle": obstacle_record,
        "segments": segment_records,
        "segment_join_gaps": join_gaps,
        "full_joint_values_rad": full_path.tolist(),
        "full_joint_values_deg": np.degrees(full_path).tolist(),
        "full_tcp_xyz_m": full_tcp.tolist(),
        "full_waypoint_count": int(len(full_path)),
        "metadata": dict(extra_metadata or {}),
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    filename = f"pepper_pick_{timestamp}_{run_id}_cycle{int(cycle_number):03d}.json"
    destination = output_dir.resolve() / filename
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    return destination


def load_pick_path_record(path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as stream:
        record = json.load(stream)
    if record.get("schema") != "wrs.pepper_pick_path":
        raise ValueError(f"{path} is not a WRS pepper-picking path record")
    if int(record.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported path record schema version: {record.get('schema_version')}"
        )
    _validated_path(record.get("full_joint_values_rad"), "full_joint_values_rad")
    return record


def find_latest_record(record_dir=None) -> Path:
    record_dir = DEFAULT_RECORD_DIR if record_dir is None else Path(record_dir)
    candidates = list(record_dir.glob("pepper_pick_*.json"))
    if not candidates:
        raise FileNotFoundError(f"no pepper-picking path records found in {record_dir.resolve()}")
    return max(candidates, key=lambda item: item.stat().st_mtime_ns).resolve()
