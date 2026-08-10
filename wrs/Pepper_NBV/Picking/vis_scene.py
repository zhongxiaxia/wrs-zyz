#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
场景摆放可视化脚本
运行此脚本查看 shelf / robotcar / 机械臂 的相对位置是否正确，不连接任何硬件。
"""

import math
import numpy as np

import wrs.visualization.panda.world as wd
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
from wrs.robot_sim.robots.realman.realman_single_arm import Realman

# ── 世界 ──────────────────────────────────────────────────────────────────────
base = wd.World(cam_pos=rm.vec(2, 0, 1), lookat_pos=rm.vec(0, 0, 0.5))
mgm.gen_frame(ax_length=0.2).attach_to(base)

# ── 货架（shelf）─────────────────────────────────────────────────────────────
shelf_model = mcm.gen_box(
    xyz_lengths=np.array([0.02, 0.7, 1.55]),
    pos=rm.vec(-0.8, 0, 0),
    rotmat=np.eye(3),
    rgb=[0.1, 0.1, 0.2],
    alpha=0.8,
)
shelf_model.attach_to(base)

# ── 机器人小车底座（robotcar）────────────────────────────────────────────────
robotcar_model = mcm.CollisionModel(
    r"F:\WRS\wrs-virtual\wrs\Pepper_NBV\newcar_m.stl",
    rgb=np.array([0.6, 0.6, 0.6]),
    alpha=0.9,
)
robotcar_model.pos    = np.array([0.13, 0.0, -0.79])
robotcar_model.rotmat = rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi * 90 / 180)
robotcar_model.attach_to(base)

# ── 机械臂（Realman）─────────────────────────────────────────────────────────
robot = Realman(enable_cc=True)
robot.gen_meshmodel(alpha=0.7, toggle_tcp_frame=True).attach_to(base)

# ── 运行 ─────────────────────────────────────────────────────────────────────
print("=== 场景可视化 ===")
print("  绿色坐标轴 = 世界原点 (机械臂底座)")
print("  灰色模型   = 机器人小车")
print("  深色模型   = 货架")
print("关闭窗口退出。")
base.run()
