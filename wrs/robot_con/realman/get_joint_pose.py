#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
获取当前真实机械臂关节角度
============================
将机械臂手动移动到目标位置后，运行此脚本，
终端会打印可直接粘贴到代码中的 np.array(...) 格式。

用法：
    python get_joint_pose.py
"""

import numpy as np
from wrs.robot_con.realman.realman import RealmanArmController

# ── 连接机械臂 ────────────────────────────────────────────────
arm = RealmanArmController(ip="192.168.1.18", port=8080, has_gripper=True)

# ── 读取当前关节角（单位：弧度）─────────────────────────────
jnt = arm.get_joint_values()

# ── 打印结果 ─────────────────────────────────────────────────
print("\n当前关节角度（弧度）：")
print(f"  {np.round(jnt, 8).tolist()}")
print()
print("可直接粘贴的代码格式：")
vals = ", ".join(f"{v:.8f}" for v in jnt)
print(f"  joint_conf = np.array([{vals}])")
print()

# ── 也打印度数，方便人工核对 ──────────────────────────────────
print("对应角度（度，仅供参考）：")
print(f"  {np.round(np.degrees(jnt), 4).tolist()}")
