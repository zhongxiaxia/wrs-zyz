#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验5：果梗部分遮挡时的 0°/+30°/-30° 变视角单果采摘。

正面第一次识别的辣椒边界框置信度低于 0.55 时，按未识别到目标处理，
机械臂依次移动到左侧 30°、右侧 30°视角重新拍照识别；任一视角成功
完成采摘后停止。每运行一次向 CSV 追加一行。
"""

from __future__ import annotations

from single_run_experiment_common import run_and_record


# ======================== 本次实验参数（只修改这里） ========================
REAL_ROBOT = True

# 可选："norrt_python"、"rrt_connect_native"
PLANNING_METHOD = "rrt_connect_native"

# 实验5固定条件
APPROACH_DISTANCE_M = 0.10
SPEED_NAME = "medium"
GRIPPER_SPEED = "fastest"
FIRST_VIEW_MIN_DETECTION_CONFIDENCE = 0.55
# ========================================================================

ALLOWED_METHODS = {"norrt_python", "rrt_connect_native"}


def main() -> int:
    if PLANNING_METHOD not in ALLOWED_METHODS:
        raise ValueError(
            f"PLANNING_METHOD={PLANNING_METHOD!r}，"
            f"只能选择 {sorted(ALLOWED_METHODS)}"
        )

    return run_and_record(
        experiment_id="Pepper05",
        title="实验5：遮挡果梗的正面、左30°、右30°变视角采摘",
        method_key=PLANNING_METHOD,
        speed_name=SPEED_NAME,
        approach_distance_m=APPROACH_DISTANCE_M,
        target_selection_strategy="nearest_tcp",
        target_count=1,
        real_robot=REAL_ROBOT,
        gripper_speed_label=GRIPPER_SPEED,
        occlusion_nbv=True,
        first_view_min_detection_confidence=(
            FIRST_VIEW_MIN_DETECTION_CONFIDENCE
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
