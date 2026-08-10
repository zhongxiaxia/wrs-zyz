from Robotic_Arm.rm_robot_interface import *

# 实例化RoboticArm类
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
# 创建机械臂连接，打印连接id
handle = arm.rm_create_robot_arm("192.168.1.18", 8080)
print(handle.id)

electronic_fence_enable = rm_electronic_fence_enable_t(True, 0, 0)
print(arm.rm_set_electronic_fence_enable(electronic_fence_enable))

arm.rm_delete_robot_arm()