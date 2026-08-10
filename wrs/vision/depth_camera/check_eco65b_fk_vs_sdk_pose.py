#!/usr/bin/env python
"""Compare Realman SDK TCP pose against the ECO65B simulation FK.

Run this while the real arm is connected. Move the arm to several static
poses, press Enter at each pose, then inspect whether the sim->sdk delta is
small and stable.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import wrs.basis.robot_math as rm
from wrs.robot_con.realman.realman import RealmanArmController
from wrs.robot_sim.robots.realman.realman_single_arm_ECO65_B import ECO65B


def homomat(pos: np.ndarray, rotmat: np.ndarray) -> np.ndarray:
    return rm.homomat_from_posrot(np.asarray(pos, dtype=float), np.asarray(rotmat, dtype=float))


def rot_error_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rel = np.asarray(rot_a, dtype=float).T @ np.asarray(rot_b, dtype=float)
    cos_theta = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cos_theta)))


def rotmat_to_axangle_deg(rotmat: np.ndarray) -> tuple[np.ndarray, float]:
    ax, angle = rm.axangle_between_rotmat(np.eye(3), rotmat)
    return np.asarray(ax, dtype=float), math.degrees(float(angle))


def make_robot(base_name: str, tcp_mode: str) -> ECO65B:
    if base_name == "identity":
        base_rotmat = np.eye(3)
    elif base_name == "rot_y_neg90":
        base_rotmat = rm.rotmat_from_euler(0, np.radians(-90), 0)
    elif base_name == "rot_y_90":
        base_rotmat = rm.rotmat_from_euler(0, np.radians(90), 0)
    else:
        raise ValueError(f"unknown base candidate: {base_name}")

    robot = ECO65B(rotmat=base_rotmat, enable_cc=False)
    if tcp_mode == "flange":
        robot.manipulator.loc_tcp_pos = np.zeros(3)
        robot.manipulator.loc_tcp_rotmat = np.eye(3)
        robot.update_end_effector()
    elif tcp_mode == "default":
        pass
    else:
        raise ValueError(f"unknown tcp candidate: {tcp_mode}")
    return robot


def compare_one(robot: ECO65B, joints: np.ndarray, sdk_pos: np.ndarray, sdk_rotmat: np.ndarray) -> dict:
    robot.goto_given_conf(joints)
    sim_pos, sim_rotmat = robot.fk(joints, update=False)
    sim_to_sdk = np.linalg.inv(homomat(sim_pos, sim_rotmat)) @ homomat(sdk_pos, sdk_rotmat)
    delta_axis, delta_angle_deg = rotmat_to_axangle_deg(sim_to_sdk[:3, :3])
    return {
        "sim_pos": np.asarray(sim_pos, dtype=float),
        "sim_rotmat": np.asarray(sim_rotmat, dtype=float),
        "pos_err_world": np.asarray(sdk_pos, dtype=float) - np.asarray(sim_pos, dtype=float),
        "rot_err_deg": rot_error_deg(sim_rotmat, sdk_rotmat),
        "sim_to_sdk": sim_to_sdk,
        "delta_axis": delta_axis,
        "delta_angle_deg": delta_angle_deg,
    }


def print_sample(sample_id: int, joints: np.ndarray, sdk_pos: np.ndarray, results: dict[str, dict]) -> None:
    print("\n" + "=" * 88)
    print(f"sample {sample_id}")
    print("joints rad:", np.array2string(joints, precision=6, suppress_small=False))
    print("joints deg:", np.array2string(np.degrees(joints), precision=3, suppress_small=False))
    print("sdk pos   :", np.array2string(sdk_pos, precision=6, suppress_small=False))
    print("-" * 88)
    for name, result in results.items():
        print(f"[{name}]")
        print("  sim pos       :", np.array2string(result["sim_pos"], precision=6, suppress_small=False))
        print("  pos err world :", np.array2string(result["pos_err_world"], precision=6, suppress_small=False))
        print(f"  pos err norm  : {np.linalg.norm(result['pos_err_world']):.6f} m")
        print(f"  rot err       : {result['rot_err_deg']:.4f} deg")
        print("  local delta t :", np.array2string(result["sim_to_sdk"][:3, 3], precision=6, suppress_small=False))
        print(
            "  local delta R : "
            f"axis={np.array2string(result['delta_axis'], precision=4, suppress_small=True)}, "
            f"angle={result['delta_angle_deg']:.4f} deg"
        )


def print_summary(history: dict[str, list[dict]]) -> None:
    if not any(history.values()):
        return
    print("\n" + "=" * 88)
    print("summary: stable local delta means candidate frame is plausible")
    print("-" * 88)
    for name, rows in history.items():
        if not rows:
            continue
        trans = np.asarray([row["sim_to_sdk"][:3, 3] for row in rows])
        rot_err = np.asarray([row["rot_err_deg"] for row in rows])
        pos_norm = np.asarray([np.linalg.norm(row["pos_err_world"]) for row in rows])
        print(f"[{name}]")
        print("  samples          :", len(rows))
        print("  pos err norm mean:", f"{pos_norm.mean():.6f} m", "std:", f"{pos_norm.std():.6f} m")
        print("  rot err mean     :", f"{rot_err.mean():.4f} deg", "std:", f"{rot_err.std():.4f} deg")
        print("  local delta t avg:", np.array2string(trans.mean(axis=0), precision=6, suppress_small=False))
        print("  local delta t std:", np.array2string(trans.std(axis=0), precision=6, suppress_small=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="192.168.1.18")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--samples", type=int, default=0, help="0 means interactive until q")
    parser.add_argument("--no-gripper", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller = RealmanArmController(ip=args.ip, port=args.port, has_gripper=not args.no_gripper)

    candidates = {
        "identity_base/flange_tcp": make_robot("identity", "flange"),
        "identity_base/default_tcp": make_robot("identity", "default"),
        "rot_y_neg90_base/flange_tcp": make_robot("rot_y_neg90", "flange"),
        "rot_y_neg90_base/default_tcp": make_robot("rot_y_neg90", "default"),
        "rot_y_90_base/flange_tcp": make_robot("rot_y_90", "flange"),
        "rot_y_90_base/default_tcp": make_robot("rot_y_90", "default"),
    }
    history: dict[str, list[dict]] = {name: [] for name in candidates}

    print("\nMove the arm to a static pose, wait until it stops, then press Enter.")
    print("Press q + Enter to finish and print the summary.\n")

    sample_id = 0
    while args.samples <= 0 or sample_id < args.samples:
        line = input("[ready] Enter=sample, q=quit: ").strip().lower()
        if line == "q":
            break
        sample_id += 1
        joints = np.asarray(controller.get_joint_values(), dtype=float)
        sdk_pos, sdk_rotmat = controller.get_pose()
        sdk_pos = np.asarray(sdk_pos, dtype=float)
        sdk_rotmat = np.asarray(sdk_rotmat, dtype=float)

        results = {
            name: compare_one(robot, joints, sdk_pos, sdk_rotmat)
            for name, robot in candidates.items()
        }
        for name, result in results.items():
            history[name].append(result)
        print_sample(sample_id, joints, sdk_pos, results)

    print_summary(history)


if __name__ == "__main__":
    main()
