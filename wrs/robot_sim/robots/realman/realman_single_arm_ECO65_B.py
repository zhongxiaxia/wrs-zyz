#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECO65-B 单臂整合类

仿照 Realman 单臂的封装方式，将 ECO65-B 本体与通用手爪组合在一起，
提供统一的前/逆向运动学、TCP 设置和自碰撞检测接口。
"""

import math
import numpy as np

import wrs.robot_sim.robots.single_arm_robot_interface as sari
# 请根据你的实际文件路径调整导入，此处假设操纵器接口文件位于该路径
from wrs.robot_sim.manipulators.realman_ECO65.ECO65_B import ECO65B as ECO65BManipulator
import wrs.robot_sim.end_effectors.grippers.hand.scissors as hand


class ECO65B(sari.SglArmRobotInterface):
    """ECO65-B 机械臂 + Hand 夹爪的一体化接口。"""

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3),
                 name: str = "eco65b_arm", enable_cc: bool = True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)

        # 机械臂本体
        # 默认 home 位姿，可根据实际零位需求修改
        home_conf = np.zeros(6)
        home_conf[1] = -math.pi / 6
        home_conf[2] = math.pi / 2
        home_conf[4] = math.pi / 6
        self.manipulator = ECO65BManipulator(pos=self.pos,
                                            rotmat=self.rotmat,
                                            name=name + "_arm",
                                            enable_cc=False)
        self.manipulator.home_conf = home_conf

        # 末端执行器（通用 hand 夹爪）
        self.end_effector = hand.Hand(pos=self.manipulator.gl_flange_pos,
                                      rotmat=self.manipulator.gl_flange_rotmat,
                                      name=name + "_hand")

        # 设置 TCP 为夹爪作用中心（指尖位置）
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat
        # 若使用法兰盘中心作为 TCP，取消下方注释
        # self.manipulator.loc_tcp_pos = np.zeros(3)
        # self.manipulator.loc_tcp_rotmat = np.eye(3)

        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        """配置自碰撞检测（含末端执行器底座）。"""
        mlb = self.cc.add_cce(self.manipulator.jlc.anchor.lnk_list[0])
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
        # 仅加入 hand 基座碰撞体，指尖可按需补充
        mlee = self.cc.add_cce(self.end_effector.jlc.anchor.lnk_list[0])

        from_list = [ml3, ml4, ml5, mlee]
        into_list = [mlb, ml0]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        self.cc.enable_extcd_by_id_list(id_list=[ml1, ml2, ml3, ml4, ml5, mlee], type="from")
        self.cc.dynamic_into_list = [mlb, ml0, ml1, ml2, ml3]
        self.cc.dynamic_ext_list = []

    def fix_to(self, pos, rotmat):
        """固定基座到指定世界位姿，并同步更新末端执行器。"""
        self._pos = pos
        self._rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        self.update_end_effector()

    def get_jaw_width(self):
        """读取手爪开度。"""
        return self.end_effector.get_jaw_width()

    def change_jaw_width(self, jaw_width):
        """改变手爪开度。"""
        self.end_effector.change_jaw_width(jaw_width=jaw_width)


if __name__ == '__main__':
    import wrs.basis.robot_math as rm
    import wrs.modeling.collision_model as mcm
    import wrs.visualization.panda.world as wd

    SIM_BASE_ROTMAT = np.eye(3)
    IK_HAND_DIRECTIONS = (
        ("Y+", np.array([0.0, 1.0, 0.0])),
        ("Y-", np.array([0.0, -1.0, 0.0])),
        ("X+", np.array([1.0, 0.0, 0.0])),
        ("X-", np.array([-1.0, 0.0, 0.0])),
        ("Z+", np.array([0.0, 0.0, 1.0])),
        ("Z-", np.array([0.0, 0.0, -1.0])),
        ("diag_xy_pp", np.array([1.0, 1.0, 0.0])),
        ("diag_xy_np", np.array([-1.0, 1.0, 0.0])),
        ("diag_xy_pn", np.array([1.0, -1.0, 0.0])),
        ("diag_xy_nn", np.array([-1.0, -1.0, 0.0])),
        ("diag_xz_pn", np.array([1.0, 0.0, -1.0])),
        ("diag_yz_pn", np.array([0.0, 1.0, -1.0])),
    )

    def normalized(vec):
        vec = np.asarray(vec, dtype=float)
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return None
        return vec / norm

    def make_grasp_rotmat(hand_z, theta_deg):
        hand_z = normalized(hand_z)
        hand_x = np.array([0.0, 0.0, -1.0])
        if abs(np.dot(hand_z, hand_x)) > 0.95:
            hand_x = np.array([1.0, 0.0, 0.0])
        hand_y = np.cross(hand_z, hand_x)
        hand_y = hand_y / np.linalg.norm(hand_y)
        hand_x = np.cross(hand_y, hand_z)
        hand_x = hand_x / np.linalg.norm(hand_x)
        tgt_rotmat_base = np.array([hand_x, hand_y, hand_z]).T
        return rm.rotmat_from_axangle(hand_z, np.radians(theta_deg)) @ tgt_rotmat_base

    def build_seed_list(robot, tgt_pos):
        base_seeds = [
            robot.get_jnt_values(),
            np.zeros(6),
            np.array([0.0, -math.pi / 6, math.pi / 2, 0.0, math.pi / 6, 0.0]),
            np.array([-0.6, -0.8, 1.6, 0.0, 0.8, 0.0]),
            np.array([0.6, -0.8, 1.6, 0.0, 0.8, 0.0]),
            np.array([-1.2, -0.5, 1.4, 0.0, 0.6, 0.0]),
            np.array([1.2, -0.5, 1.4, 0.0, 0.6, 0.0]),
            np.array([-1.8, 0.2, 1.2, 0.0, 0.8, 0.0]),
            np.array([1.8, 0.2, 1.2, 0.0, 0.8, 0.0]),
            np.array([math.atan2(tgt_pos[1], tgt_pos[0]), -0.8, 1.6, 0.0, 0.8, 0.0]),
        ]
        seed_list = []
        for seed in base_seeds:
            clipped_seed = np.asarray(seed, dtype=float).copy()
            for jnt_id, jnt_range in enumerate(robot.jnt_ranges):
                clipped_seed[jnt_id] = np.clip(clipped_seed[jnt_id], jnt_range[0] + 1e-3, jnt_range[1] - 1e-3)
            if not any(np.allclose(clipped_seed, old_seed, atol=1e-6) for old_seed in seed_list):
                seed_list.append(clipped_seed)
        return seed_list

    def build_hand_direction_list(tgt_pos):
        direction_list = list(IK_HAND_DIRECTIONS)
        radial_in = normalized(-tgt_pos)
        radial_out = normalized(tgt_pos)
        horizontal_in = normalized(np.array([-tgt_pos[0], -tgt_pos[1], 0.0]))
        horizontal_out = normalized(np.array([tgt_pos[0], tgt_pos[1], 0.0]))
        for name, direction in (
            ("target_radial_in", radial_in),
            ("target_radial_out", radial_out),
            ("target_horizontal_in", horizontal_in),
            ("target_horizontal_out", horizontal_out),
        ):
            if direction is not None:
                direction_list.append((name, direction))
        return direction_list

    def search_target_ik(robot, tgt_pos, seed_list, hand_direction_list, theta_step_deg):
        best_grasp = None
        best_approach = None
        best_score = float("inf")
        for hand_dir_name, hand_z in hand_direction_list:
            hand_z = normalized(hand_z)
            print(f"\n[IK][search] try hand_z={hand_dir_name}, vector={hand_z}")
            for seed_id, seed_conf in enumerate(seed_list):
                for theta in range(bound_lower, bound_upper + 1, theta_step_deg):
                    tgt_rotmat = make_grasp_rotmat(hand_z, theta)
                    grasp_conf = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=seed_conf)
                    if grasp_conf is None:
                        continue
                    score = np.linalg.norm(grasp_conf - seed_conf)
                    candidate = (grasp_conf, tgt_rotmat, hand_dir_name, theta, hand_z, seed_id)
                    if score < best_score:
                        best_score = score
                        best_grasp = candidate
                    approach_pos = tgt_pos - hand_z * approach_dis
                    approach_conf = robot.ik(tgt_pos=approach_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=grasp_conf)
                    if approach_conf is not None and best_approach is None:
                        best_approach = (candidate, approach_conf)
                    if approach_conf is not None:
                        return candidate, approach_conf
        return best_grasp, None if best_approach is None else best_approach[1]

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mcm.mgm.gen_frame().attach_to(base)
    robot = ECO65B(rotmat=SIM_BASE_ROTMAT, enable_cc=True)

    # ========== 示例障碍物（可按需修改或删除） ==========

    # 机器人小车底座模型（STL 单位为 mm，scale=0.001 转换为 m）
    # 车体高 489mm=0.489m，pos_z=-0.489 使车顶与机械臂底座 z=0 对齐
    robotcar_model = mcm.CollisionModel(
        r"F:\WRS\wrs-virtual_20260523\wrs-virtual\wrs\LinZhi_picker_project\model\car.stl",
        rgb=np.array([0.6, 0.6, 0.6]),
        alpha=0.9,
    )
    # robotcar_model.set_scale(np.array([0.1, 0.1, 0.1]))
    # robotcar_model.pos = np.array([0.13, 0.0, -0.79])
    robotcar_model.pos = np.array([0.0, 0.0, 0.0])
    # robotcar_model.rotmat = rm.rotmat_from_axangle([0, -1, 0], math.pi * 90 / 180)  # 绕 Z 轴旋转，修改角度(°)即可
    robotcar_model.attach_to(base)

    # ========== 目标点与 IK 搜索测试 ==========
    tgt_pos = np.array([ -0.06849,0.60431, 0.3698])

    approach_dis = 0.08
    bound_lower = -180
    bound_upper = 180
    grad = 5
    start_conf = robot.get_jnt_values()
    seed_list = build_seed_list(robot, tgt_pos)
    hand_direction_list = build_hand_direction_list(tgt_pos)
    cur_tcp_pos, _ = robot.fk(start_conf, update=False)
    print(f"[IK][target] pos={tgt_pos}, norm={np.linalg.norm(tgt_pos):.6f}")
    print(f"[IK][debug] start_tcp={cur_tcp_pos}, target_dist={np.linalg.norm(tgt_pos - cur_tcp_pos):.6f}")
    print(f"[IK][debug] seeds={len(seed_list)}, directions={len(hand_direction_list)}, theta_step={grad}")

    final_candidate, final_approach_conf = search_target_ik(robot, tgt_pos, seed_list, hand_direction_list, grad)
    final_grasp_conf = None
    final_tgt_rotmat = None
    final_hand_dir_name = None
    final_theta = None
    final_hand_z = None
    final_seed_id = None

    if final_candidate is not None:
        final_grasp_conf, final_tgt_rotmat, final_hand_dir_name, final_theta, final_hand_z, final_seed_id = final_candidate
        if final_approach_conf is None:
            print("[IK][partial] 目标点抓取 IK 已解出，但 approach 位姿未解出，先显示抓取位姿。")
        else:
            robot.goto_given_conf(final_grasp_conf)
            grasp_collided = robot.is_collided(obstacle_list=[robotcar_model])
            robot.goto_given_conf(final_approach_conf)
            approach_collided = robot.is_collided(obstacle_list=[robotcar_model])
            if grasp_collided or approach_collided:
                print(f"[COLLISION][warn] IK 已解出，但与小车模型碰撞: grasp={grasp_collided}, approach={approach_collided}")
            else:
                print(f"[IK][success] dir={final_hand_dir_name}, seed={final_seed_id}, theta={final_theta}")

    if final_grasp_conf is not None:
        mcm.mgm.gen_frame(pos=tgt_pos, rotmat=final_tgt_rotmat).attach_to(base)
        approach_pos = tgt_pos - final_hand_z * approach_dis
        mcm.mgm.gen_frame(pos=approach_pos, rotmat=final_tgt_rotmat, ax_length=.08).attach_to(base)
        robot.goto_given_conf(jnt_values=final_grasp_conf)
        print("final_hand_dir:", final_hand_dir_name)
        print("final_theta:", final_theta)
        print("final_seed_id:", final_seed_id)
        print("final_grasp_conf:", final_grasp_conf)
        print("final_approach_conf:", final_approach_conf)
        robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
        robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
    else:
        print("[IK][failed] 未找到目标点抓取 IK。请检查目标是否为相机坐标/小车坐标，或是否仍需要 SIM_BASE_ROTMAT 坐标变换。")

    base.run()
