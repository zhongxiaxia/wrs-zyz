#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测量夹爪在最快速度下从全开到全闭的时间。

测试过程只控制夹爪，不发送任何机械臂运动命令：

1. 连接 RealMan 机械臂并配置末端 Modbus。
2. 激活夹爪，等待激活状态变为 0xF1。
3. 发送全开命令并等待实际位置不大于 open_threshold。
4. 从发送最快闭合命令前开始计时。
5. 实际位置首次达到 close_threshold 时结束计时。
6. 重复多次，输出平均值、标准差、最小值和最大值。

夹爪位置寄存器使用 0~255 的指令量：
- 0 表示全开；
- 255 表示全闭。

注意：本程序采用当前采摘程序相同的速度和力矩指令 255。
运行前必须确保夹爪为空载，人员、辣椒、线缆和其他物体均已离开夹爪。
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
WORKSPACE_ROOT = THIS_FILE.parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from Robotic_Arm.rm_ctypes_wrap import rm_peripheral_read_write_params_t
from wrs.Fruit_picker_project import config
from wrs.robot_con.realman.realman import RealmanArmController


MODBUS_PORT = 1
GRIPPER_DEVICE_ID = 9
GRIPPER_VOLTAGE_TYPE = 3  # 24 V
GRIPPER_BAUDRATE = 115200
GRIPPER_MODBUS_TIMEOUT = 2

OPEN_POSITION_COMMAND = 0
CLOSE_POSITION_COMMAND = 255
MAX_SPEED_COMMAND = 255
MAX_FORCE_COMMAND = 255


def make_params(address: int, num: int = 0) -> rm_peripheral_read_write_params_t:
    params = rm_peripheral_read_write_params_t()
    params.port = MODBUS_PORT
    params.address = address
    params.device = GRIPPER_DEVICE_ID
    params.num = num
    return params


@dataclass(frozen=True)
class TrialResult:
    trial: int
    duration_s: float
    open_position: int
    closed_position: int


class GripperTimer:
    def __init__(self, arm_controller: RealmanArmController, *, verbose: bool = False):
        self.controller = arm_controller
        self.arm = arm_controller.arm
        self.verbose = verbose

    def write_registers(self, address: int, num: int, data: list[int]) -> None:
        result = self.arm.rm_write_registers(make_params(address, num), data)
        if result != 0:
            raise RuntimeError(
                f"写寄存器失败：address={address}, data={data}, result={result}"
            )

    def read_input_register(self, address: int) -> int:
        status, value = self.arm.rm_read_input_registers(make_params(address))
        if status != 0:
            raise RuntimeError(
                f"读输入寄存器失败：address={address}, status={status}, value={value}"
            )
        return int(value) & 0xFFFF

    def read_position(self) -> int:
        # 当前设备通过 RealMan API 返回的寄存器 2001 形如 0x7700；
        # 高字节 0x77 即夹爪实际位置 119。
        raw_value = self.read_input_register(2001)
        return (raw_value >> 8) & 0xFF

    def activate(self, timeout_s: float = 10.0, poll_interval_s: float = 0.05) -> None:
        voltage_result = self.arm.rm_set_tool_voltage(GRIPPER_VOLTAGE_TYPE)
        if voltage_result != 0:
            raise RuntimeError(f"设置末端24V供电失败：result={voltage_result}")

        modbus_result = self.arm.rm_set_modbus_mode(
            MODBUS_PORT,
            GRIPPER_BAUDRATE,
            GRIPPER_MODBUS_TIMEOUT,
        )
        if modbus_result != 0:
            raise RuntimeError(f"设置末端Modbus失败：result={modbus_result}")

        self.write_registers(1000, 1, [0, 0])
        time.sleep(0.2)
        self.write_registers(1000, 1, [0, 1])

        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            activation_status = self.read_input_register(2000)
            if self.verbose:
                print(f"[激活] status=0x{activation_status:04X}")
            if activation_status == 0x00F1:
                print("[激活] 夹爪激活完成，status=0xF1。")
                return
            time.sleep(poll_interval_s)
        raise TimeoutError("夹爪激活超时：未等到状态 0xF1")

    def command_open(self) -> None:
        self.write_registers(
            1000,
            3,
            [
                0,
                9,
                OPEN_POSITION_COMMAND,
                0,
                MAX_SPEED_COMMAND,
                MAX_FORCE_COMMAND,
            ],
        )

    def command_close(self) -> None:
        self.write_registers(
            1000,
            3,
            [
                0,
                9,
                CLOSE_POSITION_COMMAND,
                0,
                MAX_SPEED_COMMAND,
                MAX_FORCE_COMMAND,
            ],
        )

    def wait_for_position(
        self,
        *,
        opening: bool,
        threshold: int,
        timeout_s: float,
        poll_interval_s: float,
        stable_samples: int = 3,
        started_at: float | None = None,
    ) -> tuple[int, float | None]:
        """等待位置达到阈值，并返回最终位置及首次达到阈值的时间。

        连续多次满足阈值用于排除单次异常读数；计时结果仍采用第一次
        达到阈值的时刻，因此不会把稳定性确认所需的额外轮询时间算入。
        """
        deadline = time.perf_counter() + timeout_s
        consecutive = 0
        first_hit_elapsed: float | None = None
        previous_position: int | None = None

        while time.perf_counter() < deadline:
            position = self.read_position()
            sampled_at = time.perf_counter()
            reached = position <= threshold if opening else position >= threshold

            if self.verbose and position != previous_position:
                print(f"[位置] actual_position={position}")
            previous_position = position

            if reached:
                if consecutive == 0 and started_at is not None:
                    first_hit_elapsed = sampled_at - started_at
                consecutive += 1
                if consecutive >= stable_samples:
                    return position, first_hit_elapsed
            else:
                consecutive = 0
                first_hit_elapsed = None

            time.sleep(poll_interval_s)

        direction = "张开" if opening else "闭合"
        raise TimeoutError(
            f"等待夹爪{direction}超时：最后位置={previous_position}, threshold={threshold}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="测量夹爪以速度指令255从全开到全闭的时间。"
    )
    parser.add_argument("--ip", default=config.ARM_IP, help="RealMan机械臂IP")
    parser.add_argument("--port", type=int, default=8080, help="RealMan机械臂端口")
    parser.add_argument("--trials", type=int, default=5, help="测量次数，默认5")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.01,
        help="位置寄存器轮询周期，单位秒，默认0.01",
    )
    parser.add_argument(
        "--open-threshold",
        type=int,
        default=10,
        help="判定全开的最大位置值，默认10",
    )
    parser.add_argument(
        "--close-threshold",
        type=int,
        default=245,
        help="判定全闭的最小位置值，默认245",
    )
    parser.add_argument(
        "--motion-timeout",
        type=float,
        default=5.0,
        help="单次开合超时时间，单位秒，默认5",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.3,
        help="确认全开后的静置时间，单位秒，默认0.3",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV输出路径；默认写入Experiment/results目录",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过运行前的人工安全确认",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印激活状态和位置变化",
    )
    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials 必须大于等于1")
    if args.poll_interval <= 0:
        parser.error("--poll-interval 必须大于0")
    if not 0 <= args.open_threshold < args.close_threshold <= 255:
        parser.error("位置阈值必须满足 0 <= open_threshold < close_threshold <= 255")
    if args.motion_timeout <= 0 or args.settle_time < 0:
        parser.error("超时时间必须大于0，静置时间不能小于0")
    return args


def save_results(
    output_path: Path,
    results: list[TrialResult],
    *,
    args: argparse.Namespace,
) -> None:
    durations = [item.duration_s for item in results]
    mean_s = statistics.fmean(durations)
    std_s = statistics.stdev(durations) if len(durations) > 1 else 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "记录类型",
                "测量序号",
                "全开到全闭时间_s",
                "闭合前位置",
                "闭合后位置",
                "速度指令",
                "力矩指令",
                "轮询周期_s",
                "全开阈值",
                "全闭阈值",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    "trial",
                    item.trial,
                    f"{item.duration_s:.6f}",
                    item.open_position,
                    item.closed_position,
                    MAX_SPEED_COMMAND,
                    MAX_FORCE_COMMAND,
                    f"{args.poll_interval:.6f}",
                    args.open_threshold,
                    args.close_threshold,
                ]
            )
        writer.writerow(
            [
                "mean",
                "",
                f"{mean_s:.6f}",
                "",
                "",
                MAX_SPEED_COMMAND,
                MAX_FORCE_COMMAND,
                f"{args.poll_interval:.6f}",
                args.open_threshold,
                args.close_threshold,
            ]
        )
        writer.writerow(["std", "", f"{std_s:.6f}"])
        writer.writerow(["min", "", f"{min(durations):.6f}"])
        writer.writerow(["max", "", f"{max(durations):.6f}"])


def main() -> int:
    args = parse_args()
    if not args.yes:
        input(
            "安全确认：夹爪将以速度和力矩指令255空载全行程闭合。\n"
            "请移开手、辣椒、线缆及其他物体，确认安全后按 Enter 开始..."
        )

    arm_controller: RealmanArmController | None = None
    timer: GripperTimer | None = None
    results: list[TrialResult] = []

    try:
        arm_controller = RealmanArmController(
            ip=args.ip,
            port=args.port,
            has_gripper=True,
            multiple_thread=True,
        )
        if getattr(arm_controller.handle, "id", -1) < 0:
            raise ConnectionError(
                f"RealMan机械臂连接失败：handle.id={arm_controller.handle.id}"
            )

        timer = GripperTimer(arm_controller, verbose=args.verbose)
        timer.activate()

        for trial in range(1, args.trials + 1):
            print(f"\n--- 第 {trial}/{args.trials} 次测量 ---")
            timer.command_open()
            open_position, _ = timer.wait_for_position(
                opening=True,
                threshold=args.open_threshold,
                timeout_s=args.motion_timeout,
                poll_interval_s=args.poll_interval,
            )
            print(f"[全开] actual_position={open_position}")
            time.sleep(args.settle_time)

            started_at = time.perf_counter()
            timer.command_close()
            closed_position, duration_s = timer.wait_for_position(
                opening=False,
                threshold=args.close_threshold,
                timeout_s=args.motion_timeout,
                poll_interval_s=args.poll_interval,
                started_at=started_at,
            )
            if duration_s is None:
                raise RuntimeError("未能获得闭合计时结果")

            result = TrialResult(
                trial=trial,
                duration_s=duration_s,
                open_position=open_position,
                closed_position=closed_position,
            )
            results.append(result)
            print(
                f"[结果] 第{trial}次：{duration_s:.6f} s，"
                f"position={open_position}->{closed_position}"
            )

        durations = [item.duration_s for item in results]
        mean_s = statistics.fmean(durations)
        std_s = statistics.stdev(durations) if len(durations) > 1 else 0.0
        print("\n================ 测量汇总 ================")
        print(f"测量次数：{len(results)}")
        print(f"平均时间：{mean_s:.6f} s")
        print(f"标准差：  {std_s:.6f} s")
        print(f"最短时间：{min(durations):.6f} s")
        print(f"最长时间：{max(durations):.6f} s")

        output_path = args.output
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = (
                THIS_FILE.parent
                / "results"
                / f"gripper_close_time_{timestamp}.csv"
            )
        save_results(output_path, results, args=args)
        print(f"CSV结果：{output_path.resolve()}")
        return 0
    finally:
        # 无论测量是否成功，退出前都尽量让夹爪保持张开。
        if timer is not None:
            try:
                timer.command_open()
                print("[结束] 已发送夹爪张开命令。")
            except Exception as exc:
                print(f"[结束][警告] 夹爪张开命令失败：{exc}")
        if arm_controller is not None:
            try:
                arm_controller.close_connection()
            except Exception as exc:
                print(f"[结束][警告] 删除机器人连接失败：{exc}")


if __name__ == "__main__":
    raise SystemExit(main())
