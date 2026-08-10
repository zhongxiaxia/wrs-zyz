#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Realman 单臂整合类

仿照 PiperSglArm 的封装方式，将 Realman 本体与通用手爪组合在一起，
提供统一的前/逆向运动学、TCP 设置和自碰撞检测接口。
"""

import math
import numpy as np

import wrs.robot_sim.robots.single_arm_robot_interface as sari
from wrs.robot_sim.manipulators.realman_arm.RM65_B import RM65B
import wrs.robot_sim.end_effectors.grippers.hand.hand as hand


class Realman(sari.SglArmRobotInterface):
    """Realman 机械臂 + Hand 夹爪的一体化接口。"""

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3),
                 name: str = "realman_arm2", enable_cc: bool = True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        

        # 机械臂
        home_conf = np.zeros(6)
        home_conf[1] = -math.pi / 6
        home_conf[2] = math.pi / 2
        home_conf[4] = math.pi / 6
        self.manipulator = RM65B(pos=self.pos,
                                        rotmat=self.rotmat,
                                        name=name + "_arm",
                                        enable_cc=False)
        self.manipulator.home_conf = home_conf

        # 末端执行器（使用通用 hand 夹爪）
        self.end_effector = hand.Hand(pos=self.manipulator.gl_flange_pos,
                                      rotmat=self.manipulator.gl_flange_rotmat,
                                      name=name + "_hand")

        # 设置 TCP 为作用中心位置（夹爪指尖位置）
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat
        # 如果使用法兰盘位置作为 TCP，取消下面的注释
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
        # 仅加入 hand 的基座碰撞体（指尖可按需补充）
        mlee = self.cc.add_cce(self.end_effector.jlc.anchor.lnk_list[0])

        from_list = [ml3, ml4, ml5, mlee]
        into_list = [mlb, ml0]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        self.cc.dynamic_into_list = [mlb, ml0, ml1, ml2, ml3]
        self.cc.dynamic_ext_list = []

    def fix_to(self, pos, rotmat):
        """固定基座到指定世界位姿，并同步末端执行器。"""
        self._pos = pos
        self._rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        self.update_end_effector()

    def get_jaw_width(self):
        """方便外部直接读取手爪开度。"""
        return self.end_effector.get_jaw_width()

    def change_jaw_width(self, jaw_width):
        """改变手爪开度。"""
        self.end_effector.change_jaw_width(jaw_width=jaw_width)


if __name__ == '__main__':
    import wrs.basis.robot_math as rm
    import wrs.modeling.collision_model as mcm
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mcm.mgm.gen_frame().attach_to(base)
    robot = Realman(enable_cc=True)

    # 创建并附加 box 模型
    # shelf_model = mcm.gen_box(xyz_lengths=np.array([0.02, 0.7, 1.55]),  # 例如 60cm x 30cm x 80cm 的货架
    #                           pos=rm.vec(-0.8, 0, 0),
    #                           rotmat=np.eye(3),
    #                           rgb=[0.1, 0.1, 0.2],
    #                           alpha=0.8)
    # shelf_model.attach_to(base)

    # 创建并附加 box 模型（铝型材底座，已替换为 robotcar 模型，保留备用）
    # box_model = mcm.gen_box(xyz_lengths=np.array([0.5, 0.5, 0.66]),  # 例如 60cm x 30cm x 80cm 的货架
    #                           pos=rm.vec(0, 0, -0.33),
    #                           rotmat=np.eye(3),
    #                           rgb=[0.1, 0.1, 0.2],
    #                           alpha=0.8)
    # box_model.attach_to(base)

    # 机器人小车底座模型（STL 单位为 mm，scale=0.001 转换为 m）
    # 车体高 489mm=0.489m，pos_z=-0.489 使车顶与机械臂底座 z=0 对齐
    import wrs.modeling.geometric_model as mgm_local
    robotcar_model = mgm_local.GeometricModel(
        r"F:\WRS\wrs-virtual_20260523\wrs-virtual\wrs\LinZhi_picker_project\model\car.stl",
        rgb=np.array([0.6, 0.6, 0.6]),
        alpha=0.9,
    )
    # robotcar_model.set_scale(np.array([0.1, 0.1, 0.1]))
    # robotcar_model.pos = np.array([0.13, 0.0, -0.79])
    robotcar_model.pos = np.array([-0.1, 0.0, -0.08])
    robotcar_model.rotmat = rm.rotmat_from_axangle([0, 1, 0], -math.pi * 90 / 180)
    # robotcar_model.rotmat = rm.rotmat_from_axangle([0, 0, 1], math.pi * 90 / 180)  # 绕 Z 轴旋转，修改角度(°)即可
    robotcar_model.attach_to(base)



    tgt_pos = np.array([-.1, .02, .1])
    # tgt_pos = np.array([ -0.0061308  ,   0.40546   ,  0.58425])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    mcm.mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

    jnt_values = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, toggle_dbg=False)
    print(jnt_values)
    if jnt_values is not None:
        robot.goto_given_conf(jnt_values=jnt_values)
        robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
        robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)

    base.run()