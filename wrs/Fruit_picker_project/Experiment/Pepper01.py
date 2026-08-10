#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验1：比较四种规划/运动学组合的单果采摘时间。

直接修改下方 ``PLANNING_METHOD``，然后运行本文件。每次运行只采摘
一个果实，并向 ``Pepper01_results.csv`` 追加一行结果。

四种方式共用 ``single_run_experiment_common.py`` 中的速度、观察位、
放置位、夹爪和 CSV 记录参数。
"""

from __future__ import annotations

from single_run_experiment_common import run_and_record


# ======================== 当前实验配置（只修改这里） ========================
REAL_ROBOT = True

# 可选择：
# "norrt_python"       = 没有 RRT + Python JLChain
# "rrt_python"         = 传统 RRT + Python JLChain
# "rrt_connect_python" = RRT-Connect + Python JLChain
# "rrt_connect_native" = RRT-Connect + Native JLChain/KDL
PLANNING_METHOD = "rrt_connect_native"

# 七个实验的直线接近距离统一为 10 cm。
APPROACH_DISTANCE_M = 0.10

# 实验1固定使用中速；具体关节速度在 single_run_experiment_common.py
# 的 SPEED_PROFILES["medium"] 中统一配置。
SPEED_NAME = "medium"
GRIPPER_SPEED = "fastest"
# ========================================================================

ALLOWED_METHODS = {
    "norrt_python",
    "rrt_python",
    "rrt_connect_python",
    "rrt_connect_native",
}


def main() -> int:
    if PLANNING_METHOD not in ALLOWED_METHODS:
        raise ValueError(
            f"PLANNING_METHOD={PLANNING_METHOD!r}，"
            f"只能选择 {sorted(ALLOWED_METHODS)}"
        )
    if APPROACH_DISTANCE_M != 0.10:
        raise ValueError("APPROACH_DISTANCE_M 固定为 0.10 m")
    if SPEED_NAME != "medium":
        raise ValueError("实验1的 SPEED_NAME 固定为 medium")

    return run_and_record(
        experiment_id="Pepper01",
        title="实验1：四种规划/运动学组合的单果采摘时间",
        method_key=PLANNING_METHOD,
        speed_name=SPEED_NAME,
        approach_distance_m=APPROACH_DISTANCE_M,
        target_selection_strategy="nearest_tcp",
        target_count=1,
        real_robot=REAL_ROBOT,
        gripper_speed_label=GRIPPER_SPEED,
    )


if __name__ == "__main__":
    raise SystemExit(main())
