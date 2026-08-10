#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/30 15:55
# @Author : ZhangXi
from Robotic_Arm.rm_robot_interface import *
from wrs.drivers.devices.omnipicker.omnipicker import OmniPickerCANController  # 引入夹爪控制器
import time

from wrs.robot_con.realman.demo import move_dual_arm


class RealmanArmController:
    def __init__(self, ip: str = "192.168.1.18", port: int = 8080, has_gripper=False, multiple_thread=False):
        """  机器人的 IP 地址
        :param port: 机器人的端口号
        :param has_gripper: 是否安装夹爪
        """
        # 初始化 Realman 机器人控制类
        if multiple_thread:
            self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        else:
            self.arm = RoboticArm()
        self.handle = self.arm.rm_create_robot_arm(ip, port)  # 创建机器人连接
        print(f"连接ID: {self.handle.id}")  # 打印连接 ID

        self._has_gripper = has_gripper
        if has_gripper:
            # 如果安装了夹爪，初始化夹爪控制器
            self.gripper_controller = OmniPickerCANController(channel='can0', bitrate=5000000, can_node_id=0x02)

    def move_j(self, joint_angles: list, speed: int = 10, accel: int = 0, decel: int = 0):
        """
        控制机器人运动到指定的关节角度位置
        :param joint_angles: 目标关节角度，六个角度值，单位为度
        :param speed: 运动速度，单位为度/秒
        :param accel: 加速度
        :param decel: 减速度
        """
        result = self.arm.rm_movej(joint_angles, speed, accel, decel, 1)
        if result == 0:
            print(f"成功移动到关节位置: {joint_angles}")
        else:
            print("关节运动失败")

    def control_gripper(self, pos_cmd, force_cmd, vel_cmd, acc_cmd, dec_cmd):
        """
        控制夹爪
        :param pos_cmd: 夹爪目标位置
        :param force_cmd: 夹爪目标力矩
        :param vel_cmd: 夹爪目标速度
        :param acc_cmd: 夹爪目标加速度
        :param dec_cmd: 夹爪目标减速度
        """
        if self._has_gripper:
            self.gripper_controller.set_gripper_command(pos_cmd, force_cmd, vel_cmd, acc_cmd, dec_cmd)
            print(
                f"夹爪命令已发送：位置={pos_cmd}, 力矩={force_cmd}, 速度={vel_cmd}, 加速度={acc_cmd}, 减速度={dec_cmd}")
        else:
            print("未安装夹爪，无法发送夹爪命令")

    def close_connection(self):
        """
        删除机器人连接，释放资源
        """
        self.arm.rm_delete_robot_arm()
        print("机器人连接已删除")
        if self._has_gripper:
            del self.gripper_controller  # 释放夹爪控制器资源
            print("夹爪控制器已释放")

    def move_dual_arm(j1, j2, arm1, arm2):
        arm1.move_j(j1)
        arm2.move_j(j2)


if __name__ == "__main__":
    # 创建两个 RealMan 机器人控制器实例，分别连接不同的 IP
    realman_controller_1 = RealmanArmController(ip="192.168.0.18", port=8080, has_gripper=True, multiple_thread=True)
    realman_controller_2 = RealmanArmController(ip="192.168.3.18", port=8080, has_gripper=False)

    # 移动机械臂1、2到初始关节角度
    move_dual_arm(j2=[0, -80, 0, 0, 0, 90],
                  j1=[175, 80, 0, 0, 0, 90],
                  arm1=realman_controller_1,
                  arm2=realman_controller_2)

    # 等待机械臂完成初始位置移动后，继续移动到末尾角度
    print("\n=== 初始位置移动完成，接下来移动到末尾关节角度 ===")

    # 移动机械臂1、2到末尾位置
    move_dual_arm(j2=[26, -95, 1, 85, -95, 0],
                  j1=[152, 90, 1, -93, 93, 0],
                  arm1=realman_controller_1,
                  arm2=realman_controller_2)

    # 控制夹爪（例如：位置=100，力矩=200，速度=50，加速度=50，减速度=50）
    print("\n=== 控制夹爪 ===")
    realman_controller_1.control_gripper(pos_cmd=80, force_cmd=100, vel_cmd=255, acc_cmd=255, dec_cmd=255)
    time.sleep(1.8)
    realman_controller_1.control_gripper(pos_cmd=255, force_cmd=100, vel_cmd=255, acc_cmd=255, dec_cmd=255)
    time.sleep(1.8)
    realman_controller_1.control_gripper(pos_cmd=80, force_cmd=100, vel_cmd=255, acc_cmd=255, dec_cmd=255)
    time.sleep(1.8)
    realman_controller_1.control_gripper(pos_cmd=255, force_cmd=100, vel_cmd=255, acc_cmd=255, dec_cmd=255)
    time.sleep(1.8)
    realman_controller_1.control_gripper(pos_cmd=80, force_cmd=100, vel_cmd=255, acc_cmd=255, dec_cmd=255)
    time.sleep(1.8)
    realman_controller_1.control_gripper(pos_cmd=255, force_cmd=100, vel_cmd=255, acc_cmd=255, dec_cmd=255)
    time.sleep(1.8)

    # 移动机械臂1、2到初始关节角度
    move_dual_arm(j2=[0, -80, 0, 0, 0, 90],
                  j1=[175, 80, 0, 0, 0, 90],
                  arm1=realman_controller_1,
                  arm2=realman_controller_2)

    # 关闭连接
    realman_controller_1.close_connection()
    realman_controller_2.close_connection()
