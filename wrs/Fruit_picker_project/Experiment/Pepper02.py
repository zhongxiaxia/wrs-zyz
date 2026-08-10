#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验2：固定10 cm直线接近距离下的单果采摘时间。

直接修改下方 PLANNING_METHOD，然后运行本文件。
每运行一次只进行一次采摘，并向 Pepper02_results.csv 追加一行。
"""

from __future__ import annotations

from single_run_experiment_common import run_and_record


# ======================== 本次实验参数（只修改这里） ========================
REAL_ROBOT = True

# 可选："norrt_python"、"rrt_connect_native"
PLANNING_METHOD = "norrt_python"

# 7个实验统一固定为10 cm。
APPROACH_DISTANCE_M = 0.10

# 实验2固定条件
SPEED_NAME = "medium"
GRIPPER_SPEED = "fastest"
# ========================================================================

ALLOWED_METHODS = {"norrt_python", "rrt_connect_native"}


def main() -> int:
    if PLANNING_METHOD not in ALLOWED_METHODS:
        raise ValueError(
            f"PLANNING_METHOD={PLANNING_METHOD!r}，"
            f"只能选择 {sorted(ALLOWED_METHODS)}"
        )
    if APPROACH_DISTANCE_M != 0.10:
        raise ValueError("APPROACH_DISTANCE_M固定为0.10 m")

    return run_and_record(
        experiment_id="Pepper02",
        title="实验2：10 cm直线接近距离下的单果采摘时间",
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
