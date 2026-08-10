#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验7：快速、10 cm条件下连续采摘十字排列的5个辣椒。

TARGET_STRATEGY 可选择置信度优先或按 pick 点深度最近优先。
每运行一次连续采摘 5 个辣椒，并向 Pepper07_results.csv 追加一行。
"""

from __future__ import annotations

from single_run_experiment_common import run_and_record


# ======================== 本次实验参数（只修改这里） ========================
REAL_ROBOT = True

# 可选："norrt_python"、"rrt_connect_native"
PLANNING_METHOD = "rrt_connect_native"

# 可选：
#   "highest_confidence" = 果梗关键点置信度 + pick 检测框置信度
#   "nearest_depth"     = pick 点相机深度值由小到大
TARGET_STRATEGY = "highest_confidence"

# 实验7固定条件
TARGET_COUNT = 5
APPROACH_DISTANCE_M = 0.10
SPEED_NAME = "fast"
GRIPPER_SPEED = "fastest"
# ========================================================================

ALLOWED_METHODS = {"norrt_python", "rrt_connect_native"}
ALLOWED_STRATEGIES = {"highest_confidence", "nearest_depth"}


def main() -> int:
    if PLANNING_METHOD not in ALLOWED_METHODS:
        raise ValueError(
            f"PLANNING_METHOD={PLANNING_METHOD!r}，"
            f"只能选择 {sorted(ALLOWED_METHODS)}"
        )
    if TARGET_STRATEGY not in ALLOWED_STRATEGIES:
        raise ValueError(
            f"TARGET_STRATEGY={TARGET_STRATEGY!r}，"
            f"只能选择 {sorted(ALLOWED_STRATEGIES)}"
        )

    return run_and_record(
        experiment_id="Pepper07",
        title="实验7：快速、10 cm、十字排列5果连续采摘",
        method_key=PLANNING_METHOD,
        speed_name=SPEED_NAME,
        approach_distance_m=APPROACH_DISTANCE_M,
        target_selection_strategy=TARGET_STRATEGY,
        target_count=TARGET_COUNT,
        real_robot=REAL_ROBOT,
        gripper_speed_label=GRIPPER_SPEED,
    )


if __name__ == "__main__":
    raise SystemExit(main())
