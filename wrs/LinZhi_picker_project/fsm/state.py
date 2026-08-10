# fsm/state.py
from __future__ import annotations


class State:
    """状态基类"""

    def __init__(self, robot_context):
        self.robot = robot_context

    def enter(self):
        pass

    def execute(self):
        return None

    def exit(self):
        pass


class InitState(State):
    def enter(self):
        print("[状态: INIT] 正在检查各硬件状态...")

    def execute(self):
        self.robot.reset_workflow()
        self.robot.lift_api.reset_to_bottom()
        self.robot.arm_api.reset_to_safe_pose()
        self.robot.reset_nested_states()
        return "MOVE"


class MoveState(State):
    """外层：移动嵌套"""

    def enter(self):
        print("[状态: MOVE] 进入移动嵌套")
        self.robot.on_enter_move_layer()

    def execute(self):
        return self.robot.run_move_layer()


class LiftState(State):
    """中层：升降机嵌套"""

    def enter(self):
        print("[状态: LIFT] 进入升降机嵌套")
        if not self.robot.lift_inited:
            self.robot.reset_lift_layer()
        else:
            print("[状态: LIFT] 继续执行中层嵌套，不重复初始化。")

    def execute(self):
        return self.robot.run_lift_layer()


class ArmState(State):
    """下层：机械臂采摘嵌套"""

    def enter(self):
        print("[状态: ARM] 进入机械臂采摘嵌套")
        self.robot.enter_arm_cycle()

    def execute(self):
        return self.robot.run_arm_layer()


class IdleState(State):
    def enter(self):
        print("[状态: IDLE] 机器人处于待机状态，等待任务下发...")

    def execute(self):
        return None


class ErrorState(State):
    def enter(self):
        print(f"\n[!!! 致命错误 !!!] 机器人进入故障模式！")
        print(f"错误原因: {self.robot.error_info}")
        print("停止所有电机动作，等待人工介入...")
        self.robot.chassis_api.set_diagonal_mode(False)

    def execute(self):
        return None


class ShutdownState(State):
    def enter(self):
        print("[状态: SHUTDOWN] 执行安全关机程序...")
        self.robot.chassis_api.set_diagonal_mode(False)
        self.robot.arm_api.reset_to_safe_pose()
        self.robot.lift_api.reset_to_bottom()
        print("[状态: SHUTDOWN] 所有机构已归位，电源即将切断。")

    def execute(self):
        return "ERROR_STOP"
