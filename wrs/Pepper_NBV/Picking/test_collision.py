#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
碰撞测试脚本
在仿真中扫描一批关节配置，检测机械臂是否与小车/货架发生碰撞，
并把碰撞位姿（红色）和安全位姿（绿色透明）都可视化出来。

运行：
    python test_collision.py
"""

import math
import numpy as np

import wrs.visualization.panda.world as wd
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
from wrs.robot_sim.robots.realman.realman_single_arm import Realman

# ── 世界 ──────────────────────────────────────────────────────────────────────
base = wd.World(cam_pos=rm.vec(2, 0.5, 1.5), lookat_pos=rm.vec(0, 0, 0.5))
mgm.gen_frame(ax_length=0.2).attach_to(base)

# ── 障碍物（与 pepper_pick.py 保持完全一致）─────────────────────────────────
shelf_model = mcm.gen_box(
    xyz_lengths=np.array([0.02, 0.7, 1.55]),
    pos=rm.vec(-0.8, 0, 0),
    rotmat=np.eye(3),
    rgb=[0.1, 0.1, 0.2],
    alpha=0.8,
)
shelf_model.attach_to(base)

robotcar_model = mcm.CollisionModel(
    r"F:\WRS\wrs-virtual\wrs\Pepper_NBV\newcar_m.stl",
    rgb=np.array([0.6, 0.6, 0.6]),
    alpha=0.9,
)
robotcar_model.pos    = np.array([0.13, 0.0, -0.79])
robotcar_model.rotmat = rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi * 90 / 180)
robotcar_model.attach_to(base)

OBSTACLES = [shelf_model, robotcar_model]

# ── 机械臂 ────────────────────────────────────────────────────────────────────
robot = Realman(enable_cc=True)

# ── 待测关节配置 ──────────────────────────────────────────────────────────────
# 1. 来自 pepper_pick.py __main__ 的实际工作位姿
VIEW_POINT  = np.array([-0.03005457, -1.20970511,  1.26979681,
                         0.01317724,  1.52136859, -0.00029671])
PLACE_POINT = np.array([ 1.57254163e+00,  6.96909976e-02,  1.92076978e+00,
                          1.71216801e-02,  7.77579056e-01,  2.96705989e-04])
HOME_CONF   = robot.home_conf

NAMED_CONFS = {
    "home":       HOME_CONF,
    "view_point": VIEW_POINT,
    "place_point": PLACE_POINT,
}

# 2. 关节空间网格扫描（粗粒度，快速覆盖整个工作空间）
jnt_limits = robot.manipulator.jlc.jnts  # 各关节限位
SCAN_CONFS = []
rng = np.random.default_rng(42)
for _ in range(300):    # 随机采样 300 个配置
    conf = np.array([
        rng.uniform(j.motion_range[0], j.motion_range[1])
        for j in jnt_limits
    ])
    SCAN_CONFS.append(conf)

# ── 测试函数 ───────────────────────────────────────────────────────────────────
def test_conf(name, conf, show_safe=True):
    robot.goto_given_conf(conf)
    collided = robot.is_collided(obstacle_list=OBSTACLES)
    if collided:
        # 红色：发生碰撞
        mm = robot.gen_meshmodel(alpha=0.7, rgb=rm.const.red)
        mm.attach_to(base)
        print(f"  [COLLISION] {name}")
    elif show_safe:
        # 绿色半透明：安全
        mm = robot.gen_meshmodel(alpha=0.15, rgb=rm.const.green)
        mm.attach_to(base)

    return collided


# ── 执行测试 ───────────────────────────────────────────────────────────────────
print("=" * 55)
print("  碰撞测试开始")
print("=" * 55)

collision_count = 0

# --- 命名关键位姿 ---
print("\n[1] 关键工作位姿检测：")
for name, conf in NAMED_CONFS.items():
    if test_conf(name, conf, show_safe=True):
        collision_count += 1

# --- 随机扫描 ---
print(f"\n[2] 随机关节空间扫描（{len(SCAN_CONFS)} 个配置）：")
for i, conf in enumerate(SCAN_CONFS):
    if test_conf(f"random_{i:03d}", conf, show_safe=False):   # 安全的不显示，太多
        collision_count += 1

print("\n" + "=" * 55)
print(f"  测试完成：{len(NAMED_CONFS) + len(SCAN_CONFS)} 个配置")
print(f"  碰撞数量：{collision_count}")
print(f"  碰撞位姿 = 红色  |  安全位姿 = 绿色（仅关键位姿）")
print("=" * 55)

# 恢复到观测点
robot.goto_given_conf(VIEW_POINT)
robot.gen_meshmodel(alpha=0.8).attach_to(base)

base.run()
