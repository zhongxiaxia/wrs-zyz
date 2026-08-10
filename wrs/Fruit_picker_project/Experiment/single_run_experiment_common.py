#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验1～7共用的单次运行、单行CSV记录工具。

本模块没有命令行入口，也不会批量展开实验条件。各 PepperXX.py 文件
在顶部选择一个条件，然后调用 :func:`run_and_record` 一次。
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


THIS_FILE = Path(__file__).resolve()
WORKSPACE_ROOT = THIS_FILE.parents[3]
DEFAULT_RESULT_DIR = THIS_FILE.parent / "results"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


METHODS = {
    "norrt_python": {
        "label": "没有RRT + Python JLChain",
        "planner": "direct",
        "backend": "python",
        "path_method": "norrt",
    },
    "rrt_python": {
        "label": "传统RRT + Python JLChain",
        "planner": "rrt",
        "backend": "python",
        "path_method": "rrt",
    },
    "rrt_connect_python": {
        "label": "RRT-Connect + Python JLChain",
        "planner": "rrt_connect",
        "backend": "python",
        "path_method": "rrt",
    },
    "rrt_connect_native": {
        "label": "RRT-Connect + Native JLChain/KDL",
        "planner": "rrt_connect",
        "backend": "kdl",
        "path_method": "rrt",
    },
}


# 七个实验的机械臂速度都在这里统一修改。
# 转移段和接近段使用同一组限制；加速度取对应速度的2倍，
# 使各关节约在0.5秒内由静止加速到该档位的最大速度。
SPEED_PROFILES = {
    "slow": {
        "transfer_velocity_deg_s": [100, 100, 150, 150, 150, 150],
        "approach_velocity_deg_s": [100, 100, 150, 150, 150, 150],
        "transfer_acceleration_deg_s2": [200, 200, 300, 300, 300, 300],
        "approach_acceleration_deg_s2": [200, 200, 300, 300, 300, 300],
    },
    "medium": {
        "transfer_velocity_deg_s": [150, 150, 200, 200, 200, 200],
        "approach_velocity_deg_s": [150, 150, 200, 200, 200, 200],
        "transfer_acceleration_deg_s2": [300, 300, 400, 400, 400, 400],
        "approach_acceleration_deg_s2": [300, 300, 400, 400, 400, 400],
    },
    "fast": {
        "transfer_velocity_deg_s": [200, 200, 300, 300, 300, 300],
        "approach_velocity_deg_s": [200, 200, 300, 300, 300, 300],
        "transfer_acceleration_deg_s2": [400, 400, 600, 600, 600, 600],
        "approach_acceleration_deg_s2": [400, 400, 600, 600, 600, 600],
    },
}


VIEWPOINT_JOINT_RAD = [-1.85877565, 1.94225470, -1.88045259, -1.17633450, -1.54990476, 3.06841112]

PLACE_JOINT_RAD = [-1.60015275, 1.78765354, -1.01860905, -3.04907284, -1.97429905, 3.15454307]

GRIPPER_SPEED_NAME = "fastest"
GRIPPER_SPEED_COMMAND = 255
NBV_ORBIT_RADIUS_M = 0.02

CSV_FIELDS = [
    "记录时间",
    "实验编号",
    "运行编号",
    "规划方法",
    "规划器",
    "请求运动学后端",
    "实际运动学后端",
    "直线接近距离_cm",
    "机械臂速度",
    "夹爪速度",
    "夹爪速度指令",
    "目标选择策略",
    "要求采摘数量",
    "实际采摘数量",
    "拍照时间_s",
    "识别时间_s",
    "移动到下一视角时间_s",
    "下一视角拍照时间_s",
    "下一视角识别时间_s",
    "抓取到果实时间_s",
    "投放果实时间_s",
    "单果平均拍照时间_s",
    "单果平均识别时间_s",
    "单果平均抓取时间_s",
    "单果平均投放时间_s",
    "尝试视角",
    "是否成功",
    "状态",
    "详细记录目录",
    "路径记录",
    "错误",
]


def backend_is_native(jlc) -> bool:
    return (
        type(jlc).__module__.startswith("wrs_jlchain_native")
        or getattr(jlc, "_native_backend", None) is not None
    )


def close_picker(picker) -> None:
    pipeline = getattr(picker, "rs_pipeline", None)
    if pipeline is not None:
        try:
            pipeline.stop()
        except Exception:
            pass
    controller = getattr(picker, "rm_controller", None)
    if controller is not None:
        try:
            controller.close_connection()
        except Exception:
            pass


def create_picker(method_key: str, *, real_robot: bool):
    import wrs.motion.probabilistic.rrt as rrt_module
    import wrs.motion.probabilistic.rrt_connect as rrt_connect_module

    method = METHODS[method_key]
    if method["path_method"] == "norrt":
        from wrs.Fruit_picker_project.pick import (
            pepper_picker_norrt as pepper_module,
        )
    else:
        from wrs.Fruit_picker_project.pick import pepper_picker as pepper_module

    picker = pepper_module.RealmanPepperPickerFast(
        real_robot=real_robot,
        defer_init=False,
        use_gripper=real_robot,
    )
    try:
        picker.initialize()
    except Exception:
        close_picker(picker)
        raise
    if real_robot and (
        not picker.real_robot
        or picker.rm_controller is None
        or picker.rs_pipeline is None
        or picker.claw_controller is None
    ):
        close_picker(picker)
        raise RuntimeError("真机、相机或夹爪初始化失败，拒绝使用仿真结果")

    if method["planner"] == "rrt":
        rrt_module.base = SimpleNamespace(toggle_mesh=True)
        picker.rrtc_planner = rrt_module.RRT(picker.robot)
    elif method["planner"] == "rrt_connect":
        picker.rrtc_planner = rrt_connect_module.RRTConnect(picker.robot)

    native_active = backend_is_native(picker.robot.manipulator.jlc)
    backend_active = "native_kdl" if native_active else "python_jlchain"
    if method["backend"] == "kdl" and not native_active:
        close_picker(picker)
        raise RuntimeError(
            "已选择 Native JLChain/KDL，但当前机器人仍在使用 "
            f"{type(picker.robot.manipulator.jlc).__module__}"
        )
    if method["backend"] == "python" and native_active:
        close_picker(picker)
        raise RuntimeError("已选择 Python JLChain，但当前启用了 Native/KDL")
    return picker, backend_active


def stage_sum(records: list[dict], stage: str) -> float | None:
    """Sum every measured occurrence, including no-target/failed attempts."""
    values = [
        float(record["duration_s"])
        for record in records
        if record.get("stage") == stage
    ]
    return sum(values) if values else None


def execute_picker_call(
    picker,
    *,
    method_key: str,
    speed_name: str,
    approach_distance_m: float,
    target_selection_strategy: str,
    target_count: int,
    viewpoint_joint,
    view_name: str,
    run_directory: Path,
    call_index: int,
) -> dict:
    import numpy as np

    method = METHODS[method_key]
    speed = SPEED_PROFILES[speed_name]
    picker._experiment_view_name = view_name
    raw_result = picker.pick_place_fast(
        approach_dis=float(approach_distance_m),
        bound_lower=-30,
        bound_upper=30,
        grad=3,
        viewpoint_joint=np.asarray(viewpoint_joint, dtype=float),
        place_joint=np.asarray(PLACE_JOINT_RAD, dtype=float),
        continuous_mode=target_count > 1,
        stop_on_no_detection=True,
        control_frequency=0.1,
        open_close_retries=3,
        cycle_once=target_count == 1,
        target_selection_strategy=target_selection_strategy,
        path_method=method["path_method"],
        transfer_max_jntvel_deg_s=np.asarray(
            speed["transfer_velocity_deg_s"],
            dtype=float,
        ),
        approach_max_jntvel_deg_s=np.asarray(
            speed["approach_velocity_deg_s"],
            dtype=float,
        ),
        transfer_max_jntacc_deg_s2=np.asarray(
            speed["transfer_acceleration_deg_s2"],
            dtype=float,
        ),
        approach_max_jntacc_deg_s2=np.asarray(
            speed["approach_acceleration_deg_s2"],
            dtype=float,
        ),
        simulation_target_pos=np.asarray([-0.45, -0.1, 0.3], dtype=float),
        target_distance_m=None,
        place_distance_m=None,
        max_successful_picks=int(target_count),
        save_path_record=True,
        path_record_dir=run_directory / "trajectory",
    ) or {"success": False, "picked": 0}

    records = []
    for source in getattr(picker, "_timing_records", []):
        record = dict(source)
        record["call_index"] = int(call_index)
        record["view_name"] = view_name
        records.append(record)
    path_record = raw_result.get("path_record")
    return {
        "picked_count": int(raw_result.get("picked", 0)),
        "records": records,
        "path_records": [] if not path_record else [str(path_record)],
        "raw_picker_result": raw_result,
    }


def build_nbv_joint_map(picker) -> dict:
    import numpy as np
    import wrs.basis.robot_math as rm

    initial = np.asarray(VIEWPOINT_JOINT_RAD, dtype=float)
    tcp_position, tcp_rotation = picker.robot.fk(initial, update=False)
    center = tcp_position + np.array([-NBV_ORBIT_RADIUS_M, 0.0, 0.0])
    radius_vector = tcp_position - center
    result = {"initial_0deg": initial.copy()}
    for name, angle_deg in (("plus_30deg", 30), ("minus_30deg", -30)):
        rotation_z = rm.rotmat_from_axangle(
            [0.0, 0.0, 1.0],
            np.radians(angle_deg),
        )
        goal = picker.robot.ik(
            tgt_pos=center + rotation_z @ radius_vector,
            tgt_rotmat=rotation_z @ tcp_rotation,
            seed_jnt_values=initial,
        )
        if goal is None:
            print(f"[NBV][跳过] {name} 的观察位 IK 无解")
        else:
            result[name] = np.asarray(goal, dtype=float)
    return result


def aggregate_calls(
    calls: list[dict],
    *,
    target_count: int,
    backend_active: str,
    attempted_views: list[str],
) -> dict:
    all_records = [
        record
        for call in calls
        for record in call.get("records", [])
    ]
    picked_count = sum(int(call.get("picked_count", 0)) for call in calls)
    paths = [
        path
        for call in calls
        for path in call.get("path_records", [])
    ]
    raw_results = [
        call.get("raw_picker_result", {})
        for call in calls
    ]
    cycle_totals = [
        record for record in all_records if record.get("stage") == "cycle_total"
    ]
    return {
        "status": (
            "success"
            if picked_count >= target_count
            else (
                cycle_totals[-1].get("status", "incomplete")
                if cycle_totals
                else "incomplete"
            )
        ),
        "success": picked_count >= target_count,
        "picked_count": picked_count,
        "backend_active": backend_active,
        "capture_s": stage_sum(all_records, "experiment_capture"),
        "recognition_s": stage_sum(
            all_records,
            "experiment_recognition",
        ),
        "grasp_s": stage_sum(
            all_records,
            "experiment_grasp_to_close",
        ),
        "place_s": stage_sum(
            all_records,
            "experiment_place_to_open_command",
        ),
        "attempted_views": attempted_views,
        "path_records": paths,
        "events": all_records,
        "raw_picker_results": raw_results,
        "error": "",
    }


def run_standard(
    picker,
    *,
    backend_active: str,
    method_key: str,
    speed_name: str,
    approach_distance_m: float,
    target_selection_strategy: str,
    target_count: int,
    run_directory: Path,
) -> dict:
    call = execute_picker_call(
        picker,
        method_key=method_key,
        speed_name=speed_name,
        approach_distance_m=approach_distance_m,
        target_selection_strategy=target_selection_strategy,
        target_count=target_count,
        viewpoint_joint=VIEWPOINT_JOINT_RAD,
        view_name="default_0deg",
        run_directory=run_directory,
        call_index=1,
    )
    return aggregate_calls(
        [call],
        target_count=target_count,
        backend_active=backend_active,
        attempted_views=["default_0deg"],
    )


def run_occlusion_nbv(
    picker,
    *,
    backend_active: str,
    method_key: str,
    speed_name: str,
    approach_distance_m: float,
    target_selection_strategy: str,
    run_directory: Path,
    first_view_min_detection_confidence: float | None,
) -> dict:
    view_order = ["initial_0deg", "plus_30deg", "minus_30deg"]
    joint_map = build_nbv_joint_map(picker)
    if "plus_30deg" not in joint_map and "minus_30deg" not in joint_map:
        raise RuntimeError("左、右30度观察位均无IK解")

    calls = []
    attempted_views = []
    previous_confidence_threshold = getattr(
        picker,
        "_experiment_min_detection_confidence",
        None,
    )
    try:
        for call_index, view_name in enumerate(view_order, start=1):
            if view_name not in joint_map:
                continue
            print(f"\n[NBV] 尝试观察位：{view_name}")
            attempted_views.append(view_name)
            picker._experiment_min_detection_confidence = (
                first_view_min_detection_confidence
                if view_name == "initial_0deg"
                else None
            )
            call = execute_picker_call(
                picker,
                method_key=method_key,
                speed_name=speed_name,
                approach_distance_m=approach_distance_m,
                target_selection_strategy=target_selection_strategy,
                target_count=1,
                viewpoint_joint=joint_map[view_name],
                view_name=view_name,
                run_directory=run_directory,
                call_index=call_index,
            )
            calls.append(call)
            if call["picked_count"] >= 1:
                break
    finally:
        picker._experiment_min_detection_confidence = (
            previous_confidence_threshold
        )

    result = aggregate_calls(
        calls,
        target_count=1,
        backend_active=backend_active,
        attempted_views=attempted_views,
    )
    all_capture_s = result.get("capture_s")
    all_recognition_s = result.get("recognition_s")
    first_view_records = calls[0].get("records", []) if calls else []
    next_view_records = [
        record
        for call in calls[1:]
        for record in call.get("records", [])
    ]
    result.update(
        {
            # Pepper05 的“拍照/识别”专指第一次 0° 视角。
            "capture_s": stage_sum(first_view_records, "experiment_capture"),
            "recognition_s": stage_sum(
                first_view_records,
                "experiment_recognition",
            ),
            # 若 +30° 仍失败并继续到 -30°，以下三项累计所有后续视角；
            # 每个视角的独立值仍保存在 events/result_details.json 中。
            "next_view_move_s": stage_sum(
                next_view_records,
                "startup_view_total",
            ),
            "next_capture_s": stage_sum(
                next_view_records,
                "experiment_capture",
            ),
            "next_recognition_s": stage_sum(
                next_view_records,
                "experiment_recognition",
            ),
            "all_capture_s": all_capture_s,
            "all_recognition_s": all_recognition_s,
            "first_view_min_detection_confidence": (
                first_view_min_detection_confidence
            ),
        }
    )
    return result


def format_seconds(value) -> str:
    return "" if value is None else f"{float(value):.6f}"


def average_seconds(value, picked_count: int) -> str:
    if value is None or picked_count <= 0:
        return ""
    return f"{float(value) / picked_count:.6f}"


def result_to_row(
    *,
    experiment_id: str,
    run_id: str,
    method_key: str,
    speed_name: str,
    approach_distance_m: float,
    target_selection_strategy: str,
    target_count: int,
    run_directory: Path,
    result: dict,
) -> dict:
    method = METHODS[method_key]
    picked_count = int(result.get("picked_count", 0))
    return {
        "记录时间": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "实验编号": experiment_id,
        "运行编号": run_id,
        "规划方法": method["label"],
        "规划器": method["planner"],
        "请求运动学后端": method["backend"],
        "实际运动学后端": result.get("backend_active", ""),
        "直线接近距离_cm": f"{approach_distance_m * 100.0:.1f}",
        "机械臂速度": speed_name,
        "夹爪速度": GRIPPER_SPEED_NAME,
        "夹爪速度指令": GRIPPER_SPEED_COMMAND,
        "目标选择策略": target_selection_strategy,
        "要求采摘数量": target_count,
        "实际采摘数量": picked_count,
        "拍照时间_s": format_seconds(result.get("capture_s")),
        "识别时间_s": format_seconds(result.get("recognition_s")),
        "移动到下一视角时间_s": format_seconds(
            result.get("next_view_move_s")
        ),
        "下一视角拍照时间_s": format_seconds(
            result.get("next_capture_s")
        ),
        "下一视角识别时间_s": format_seconds(
            result.get("next_recognition_s")
        ),
        "抓取到果实时间_s": format_seconds(result.get("grasp_s")),
        "投放果实时间_s": format_seconds(result.get("place_s")),
        "单果平均拍照时间_s": average_seconds(
            result.get("capture_s"),
            picked_count,
        ),
        "单果平均识别时间_s": average_seconds(
            result.get("recognition_s"),
            picked_count,
        ),
        "单果平均抓取时间_s": average_seconds(
            result.get("grasp_s"),
            picked_count,
        ),
        "单果平均投放时间_s": average_seconds(
            result.get("place_s"),
            picked_count,
        ),
        "尝试视角": json.dumps(
            result.get("attempted_views", []),
            ensure_ascii=False,
        ),
        "是否成功": bool(result.get("success", False)),
        "状态": result.get("status", "error"),
        "详细记录目录": str(run_directory),
        "路径记录": json.dumps(
            result.get("path_records", []),
            ensure_ascii=False,
        ),
        "错误": result.get("error", ""),
    }


def append_one_row(output_csv: Path, row: dict) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists() and output_csv.stat().st_size > 0:
        with output_csv.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = list(reader)
        if existing_fields != CSV_FIELDS:
            temporary_path = output_csv.with_name(
                f".{output_csv.name}.schema_upgrade.tmp"
            )
            with temporary_path.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=CSV_FIELDS,
                    extrasaction="ignore",
                )
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow(
                        {
                            field: existing_row.get(field, "")
                            for field in CSV_FIELDS
                        }
                    )
            os.replace(temporary_path, output_csv)
            print(
                f"[结果表] 已升级CSV表头并保留原有记录：{output_csv}"
            )

    needs_header = not output_csv.exists() or output_csv.stat().st_size == 0
    with output_csv.open("a", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def write_details(
    run_directory: Path,
    *,
    experiment_config: dict,
    result: dict,
) -> None:
    path = run_directory / "result_details.json"
    path.write_text(
        json.dumps(
            {
                "experiment_config": experiment_config,
                "result": result,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def run_and_record(
    *,
    experiment_id: str,
    title: str,
    method_key: str,
    speed_name: str,
    approach_distance_m: float,
    target_selection_strategy: str,
    target_count: int,
    real_robot: bool = True,
    occlusion_nbv: bool = False,
    first_view_min_detection_confidence: float | None = None,
    gripper_speed_label: str = GRIPPER_SPEED_NAME,
    output_csv: Path | None = None,
    run_output_root: Path | None = None,
) -> int:
    if method_key not in METHODS:
        raise ValueError(f"未知规划方法：{method_key}")
    if speed_name not in SPEED_PROFILES:
        raise ValueError(f"未知速度档位：{speed_name}")
    if abs(float(approach_distance_m) - 0.10) > 1e-9:
        raise ValueError("7个实验的直线接近距离统一固定为0.10 m")
    if target_selection_strategy not in {
        "nearest_tcp",
        "nearest_depth",
        "highest_confidence",
    }:
        raise ValueError(f"未知目标选择策略：{target_selection_strategy}")
    if target_count <= 0:
        raise ValueError("target_count必须为正整数")
    if (
        first_view_min_detection_confidence is not None
        and not 0.0 <= float(first_view_min_detection_confidence) <= 1.0
    ):
        raise ValueError("首次视角检测置信度阈值必须在0.0到1.0之间")
    if (
        first_view_min_detection_confidence is not None
        and experiment_id != "Pepper05"
    ):
        raise ValueError(
            "额外的检测框置信度阈值只允许用于Pepper05实验；"
            "其他实验不使用该阈值判断采摘点"
        )
    if first_view_min_detection_confidence is not None and not occlusion_nbv:
        raise ValueError("首次视角检测置信度阈值只适用于occlusion_nbv实验")
    if gripper_speed_label != GRIPPER_SPEED_NAME:
        raise ValueError(
            "当前真机夹爪控制仅允许 fastest；"
            "闭合和张开命令中的速度字节均为255"
        )

    if output_csv is None:
        output_csv = DEFAULT_RESULT_DIR / f"{experiment_id}_results.csv"
    if run_output_root is None:
        run_output_root = DEFAULT_RESULT_DIR / f"{experiment_id}_runs"
    output_csv = Path(output_csv)
    run_output_root = Path(run_output_root)

    method = METHODS[method_key]
    # 后端环境变量必须早于任何WRS运动学模块导入。
    os.environ["WRS_JLCHAIN_BACKEND"] = method["backend"]

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_id = (
        f"{experiment_id}_{timestamp}_{method_key}_"
        f"{time.time_ns() % 1_000_000:06d}"
    )
    run_directory = run_output_root / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    experiment_config = {
        "experiment_id": experiment_id,
        "title": title,
        "method_key": method_key,
        "method": method,
        "speed_name": speed_name,
        "speed_profile": SPEED_PROFILES[speed_name],
        "approach_distance_m": approach_distance_m,
        "gripper_speed_name": GRIPPER_SPEED_NAME,
        "gripper_speed_command": GRIPPER_SPEED_COMMAND,
        "target_selection_strategy": target_selection_strategy,
        "target_count": target_count,
        "occlusion_nbv": occlusion_nbv,
        "first_view_min_detection_confidence": (
            first_view_min_detection_confidence
        ),
    }

    print(f"\n================ {experiment_id} ================")
    print(title)
    print(f"规划方法：{method['label']}")
    print(f"直线接近距离：{approach_distance_m * 100:.0f} cm")
    print(f"机械臂速度：{speed_name}")
    print("夹爪速度：fastest（指令255）")
    print(f"目标选择策略：{target_selection_strategy}")
    print(f"本次要求采摘：{target_count} 个")
    if occlusion_nbv:
        print("识别顺序：正面0° -> 左30° -> 右30°，检测成功后停止切换")
        if first_view_min_detection_confidence is not None:
            print(
                "首次0°视角最低边界框置信度："
                f"{float(first_view_min_detection_confidence):.2f}"
            )
    input("确认机械臂、夹爪、相机和果实均已就绪后按 Enter 开始...")

    picker = None
    try:
        picker, backend_active = create_picker(
            method_key,
            real_robot=real_robot,
        )
        if occlusion_nbv:
            result = run_occlusion_nbv(
                picker,
                backend_active=backend_active,
                method_key=method_key,
                speed_name=speed_name,
                approach_distance_m=approach_distance_m,
                target_selection_strategy=target_selection_strategy,
                run_directory=run_directory,
                first_view_min_detection_confidence=(
                    first_view_min_detection_confidence
                ),
            )
        else:
            result = run_standard(
                picker,
                backend_active=backend_active,
                method_key=method_key,
                speed_name=speed_name,
                approach_distance_m=approach_distance_m,
                target_selection_strategy=target_selection_strategy,
                target_count=target_count,
                run_directory=run_directory,
            )
    except Exception as exc:
        result = {
            "status": "error",
            "success": False,
            "picked_count": 0,
            "backend_active": "",
            "capture_s": None,
            "recognition_s": None,
            "next_view_move_s": None,
            "next_capture_s": None,
            "next_recognition_s": None,
            "grasp_s": None,
            "place_s": None,
            "attempted_views": [],
            "path_records": [],
            "events": [],
            "raw_picker_results": [],
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        if picker is not None:
            close_picker(picker)

    write_details(
        run_directory,
        experiment_config=experiment_config,
        result=result,
    )
    row = result_to_row(
        experiment_id=experiment_id,
        run_id=run_id,
        method_key=method_key,
        speed_name=speed_name,
        approach_distance_m=approach_distance_m,
        target_selection_strategy=target_selection_strategy,
        target_count=target_count,
        run_directory=run_directory,
        result=result,
    )
    append_one_row(output_csv, row)

    print("\n================ 单次结果 ================")
    print(f"拍照时间：       {row['拍照时间_s']} s")
    print(f"识别时间：       {row['识别时间_s']} s")
    if occlusion_nbv:
        print(
            "移动到下一视角时间： "
            f"{row['移动到下一视角时间_s']} s"
        )
        print(
            "下一视角拍照时间：   "
            f"{row['下一视角拍照时间_s']} s"
        )
        print(
            "下一视角识别时间：   "
            f"{row['下一视角识别时间_s']} s"
        )
    print(f"抓取到果实时间： {row['抓取到果实时间_s']} s")
    print(f"投放果实时间：   {row['投放果实时间_s']} s")
    print(f"采摘数量：       {row['实际采摘数量']}/{target_count}")
    print(f"是否成功：       {row['是否成功']}")
    print(f"结果表：         {output_csv}")
    if row["错误"]:
        print(f"错误：           {row['错误']}")
    print("==========================================")
    return 0 if result.get("success", False) else 1
