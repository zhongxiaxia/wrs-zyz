from jodellSdk.jodellSdkDemo import ClawHepgTool
import time

class ClawController:
    def __init__(self, claw_id, com_port="COM3", baud_rate=115200):
        """
        初始化夹爪控制器
        :param claw_id: 夹爪的ID
        :param com_port: 串口，默认为COM3
        :param baud_rate: 波特率，默认为115200
        """
        self.clawTool = ClawHepgTool()  # 创建操作对象
        self.claw_id = claw_id
        self.com_port = com_port
        self.baud_rate = baud_rate

        # 搜索可用的串口
        self.com_list = self.clawTool.searchCom()
        print("可用串口列表:", self.com_list)

    def connect_gripper(self):
        """
        连接夹爪
        """
        flag = self.clawTool.serialOperation(self.com_port, self.baud_rate, True)  # True表示连接
        if flag == 1:
            print("夹爪连接成功")
        else:
            print(f"夹爪连接失败，错误代码: {flag}")
            exit()

    def disconnect_gripper(self):
        """
        断开夹爪连接
        """
        flag = self.clawTool.serialOperation(self.com_port, self.baud_rate, False)  # False表示断开连接
        if flag == 1:
            print("夹爪断开连接成功")
        else:
            print(f"夹爪断开连接失败，错误代码: {flag}")
            exit()

    def enable_gripper(self):
        """
        使能夹爪
        """
        flag = self.clawTool.clawEnable(self.claw_id, True)  # True表示使能
        if flag == 1:
            print("夹爪使能成功")
        else:
            print(f"夹爪使能失败，错误代码: {flag}")
            exit()

    def close_gripper(self):
        """
        夹爪全速闭合
        """
        flag = self.clawTool.runWithParam(self.claw_id, torque=255, speed=255, pos=245)
        if flag == 1:
            print("夹爪全速闭合成功")
        else:
            print(f"夹爪全速闭合失败，错误代码: {flag}")

    def open_gripper(self):
        """
        夹爪全速张开
        """
        flag = self.clawTool.runWithParam(self.claw_id, torque=255, speed=255, pos=0)
        if flag == 1:
            print("夹爪全速张开成功")
        else:
            print(f"夹爪全速张开失败，错误代码: {flag}")

    def get_claw_current_torque(self, claw_id):
        """
        获取夹爪当前力矩
        参数说明：
            slave_id: 夹爪站点id
        """
        # 调用 ClawHepgTool 的方法获取当前力矩
        torque = self.clawTool.getClawCurrentTorque(claw_id)
        print(f"夹爪站点 {claw_id} 当前力矩: {torque}")

    def run_sequence(self):
        """
        运行夹爪操作序列
        """
        self.connect_gripper()  # 连接夹爪
        self.enable_gripper()   # 使能夹爪
        self.open_gripper()     # 打开夹爪
        self.close_gripper()    # 关闭夹爪
        self.disconnect_gripper()  # 断开夹爪连接

if __name__ == "__main__":
    claw_id = 9  # 夹爪ID
    com_port = 'COM6'
    claw_controller = ClawController(claw_id,com_port=com_port)  # 创建夹爪控制器对象
    time.sleep(1)
    claw_controller.connect_gripper()  # 连接夹爪
    time.sleep(1)

    claw_controller.enable_gripper()   # 使能夹爪
    claw_controller.open_gripper()     # 打开夹爪
    time.sleep(3)
    claw_controller.close_gripper()    # 关闭夹爪
    time.sleep(3)
    claw_controller.open_gripper()     # 打开夹爪
    claw_controller.get_claw_current_torque(claw_id)  # 获取当前力矩
    claw_controller.disconnect_gripper()  # 断开夹爪连接
