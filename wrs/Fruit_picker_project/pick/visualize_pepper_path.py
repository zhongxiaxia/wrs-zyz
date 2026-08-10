#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize and animate a saved pepper-picking path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
REPOSITORY_ROOT = THIS_FILE.parents[3]
WRS_ROOT = THIS_FILE.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.modeling.geometric_model as mgm
import wrs.visualization.panda.world as wd
from wrs.Fruit_picker_project.pick.pepper_path_io import (
    DEFAULT_RECORD_DIR,
    find_latest_record,
    load_pick_path_record,
)
from wrs.robot_sim.robots.realman.realman_single_arm import Realman


SEGMENT_COLORS = {
    "start_to_view": np.array([0.09, 0.75, 0.81]),
    "view_to_approach": np.array([0.12, 0.47, 0.71]),
    "approach_to_grasp": np.array([1.00, 0.50, 0.05]),
    "grasp_to_approach": np.array([0.84, 0.15, 0.16]),
    "approach_to_place": np.array([0.17, 0.63, 0.17]),
    "place_to_view": np.array([0.58, 0.40, 0.74]),
}
FALLBACK_COLOR = np.array([0.3, 0.3, 0.3])


def _resolve_mesh_path(mesh_path: str) -> Path:
    path = Path(mesh_path)
    if not path.is_absolute():
        path = WRS_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"collision mesh does not exist: {path}")
    return path


def _line_segments(points):
    points = np.asarray(points, dtype=float)
    return [[points[index], points[index + 1]] for index in range(len(points) - 1)]


def _attach_tcp_paths(base, record):
    for segment in record["segments"]:
        points = np.asarray(segment["tcp_xyz_m"], dtype=float)
        color = SEGMENT_COLORS.get(segment["name"], FALLBACK_COLOR)
        if len(points) >= 2:
            mgm.gen_linesegs(
                _line_segments(points),
                thickness=0.004,
                rgb=color,
                alpha=1.0,
            ).attach_to(base)
        mgm.gen_sphere(pos=points[0], radius=0.008, rgb=color, alpha=1.0).attach_to(base)
        mgm.gen_sphere(pos=points[-1], radius=0.008, rgb=color, alpha=1.0).attach_to(base)

    target_pos = np.asarray(record["target_position_m"], dtype=float)
    mgm.gen_sphere(
        pos=target_pos,
        radius=0.018,
        rgb=np.array([1.0, 0.0, 0.0]),
        alpha=1.0,
    ).attach_to(base)


def _attach_ghost_poses(base, robot, full_path, ghost_count):
    if ghost_count <= 0:
        return
    indices = np.unique(np.linspace(
        0,
        len(full_path) - 1,
        min(int(ghost_count), len(full_path)),
        dtype=int,
    ))
    robot.backup_state()
    try:
        for index in indices:
            robot.goto_given_conf(full_path[index])
            robot.gen_meshmodel(alpha=0.10).attach_to(base)
    finally:
        robot.restore_state()


def _attach_obstacle(base, obstacle_record, show_collision_primitives):
    if not obstacle_record:
        return None
    obstacle = mcm.CollisionModel(
        str(_resolve_mesh_path(obstacle_record["mesh_path"])),
        rgb=np.asarray(obstacle_record.get("rgb", [0.6, 0.6, 0.6]), dtype=float),
        alpha=float(obstacle_record.get("alpha", 0.9)),
    )
    obstacle.pos = np.asarray(obstacle_record["position_m"], dtype=float)
    obstacle.rotmat = np.asarray(obstacle_record["rotation_matrix"], dtype=float)
    obstacle.attach_to(base)
    if show_collision_primitives:
        obstacle.show_cdprim()
    return obstacle


class PathAnimation:
    def __init__(self, base, robot, path, *, loop):
        self.base = base
        self.robot = robot
        self.path = np.asarray(path, dtype=float)
        self.loop = bool(loop)
        self.index = 0
        self.model = None

    def update(self, task):
        if self.model is not None:
            self.model.detach()
        self.robot.goto_given_conf(self.path[self.index])
        self.model = self.robot.gen_meshmodel(alpha=1.0)
        self.model.attach_to(self.base)

        self.index += 1
        if self.index >= len(self.path):
            if not self.loop:
                self.index = len(self.path) - 1
                return task.done
            self.index = 0
        return task.again


def visualize_record(
        record_path,
        *,
        interval_s=0.08,
        ghost_count=24,
        show_collision_primitives=True,
        loop=True,
):
    record_path = Path(record_path).resolve()
    record = load_pick_path_record(record_path)
    full_path = np.asarray(record["full_joint_values_rad"], dtype=float)

    base = wd.World(
        cam_pos=rm.vec(1.6, 1.2, 1.1),
        lookat_pos=rm.vec(0.0, 0.0, 0.35),
    )
    mgm.gen_frame(ax_length=0.25).attach_to(base)
    _attach_obstacle(base, record.get("obstacle"), show_collision_primitives)
    _attach_tcp_paths(base, record)

    robot = Realman(enable_cc=True)
    robot.goto_given_conf(full_path[0])
    _attach_ghost_poses(base, robot, full_path, ghost_count)
    robot.goto_given_conf(full_path[0])
    if show_collision_primitives:
        robot.cc.show_cdprim()

    animation = PathAnimation(base, robot, full_path, loop=loop)
    base.taskMgr.doMethodLater(
        float(interval_s),
        animation.update,
        "pepper_path_animation",
    )
    print(f"[路径可视化] 文件：{record_path}")
    print(
        f"[路径可视化] 分段={len(record['segments'])}，"
        f"完整轨迹点={record['full_waypoint_count']}，"
        f"播放间隔={float(interval_s):.3f}s"
    )
    print("[颜色] 青=初始到观察，蓝=观察到接近，橙=接近到抓取，红=退出，绿=放置，紫=返回")
    base.run()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize the robot, collision models, and TCP path from a saved pick."
    )
    parser.add_argument(
        "record",
        nargs="?",
        help="path record JSON; defaults to the newest record",
    )
    parser.add_argument(
        "--record-dir",
        default=str(DEFAULT_RECORD_DIR),
        help="directory searched when no record argument is supplied",
    )
    parser.add_argument("--interval", type=float, default=0.08, help="seconds per waypoint")
    parser.add_argument("--ghost-count", type=int, default=24, help="static ghost robot count")
    parser.add_argument(
        "--hide-collision-primitives",
        action="store_true",
        help="hide collision primitive overlays",
    )
    parser.add_argument("--once", action="store_true", help="stop animation at the final pose")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    if args.ghost_count < 0:
        raise ValueError("--ghost-count must be non-negative")
    record_path = (
        Path(args.record).resolve()
        if args.record
        else find_latest_record(args.record_dir)
    )
    visualize_record(
        record_path,
        interval_s=args.interval,
        ghost_count=args.ghost_count,
        show_collision_primitives=not args.hide_collision_primitives,
        loop=not args.once,
    )


if __name__ == "__main__":
    main()
