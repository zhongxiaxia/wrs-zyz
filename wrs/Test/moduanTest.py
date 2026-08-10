from __future__ import annotations

import time
from typing import Optional

from Robotic_Arm.rm_ctypes_wrap import rm_peripheral_read_write_params_t
from Robotic_Arm.rm_robot_interface import *
import keyboard



class RealManGripperController:
    def __init__(self, robot_ip: str, robot_port: int = 8080, device_id: int = 9, modbus_port: int = 1):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.device_id = device_id
        self.modbus_port = modbus_port

        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(self.robot_ip, self.robot_port)
        print(f"[RealMan] connected, handle id={self.handle.id}")

    def close(self):
        try:
            self.arm.rm_delete_robot_arm()
        except Exception as exc:
            print(f"[RealMan] close warning: {exc}")

    def _make_params(self, address: int, num: int = 0):
        params = rm_peripheral_read_write_params_t()
        params.port = self.modbus_port
        params.address = address
        params.num = num
        params.device = self.device_id
        return params

    def set_tool_voltage_24v(self):
        result = self.arm.rm_set_tool_voltage(3)
        print(f"[RealMan] set_tool_voltage result={result}")
        return result

    def set_modbus_mode(self, baudrate: int = 115200, timeout: int = 2):
        result = self.arm.rm_set_modbus_mode(self.modbus_port, baudrate, timeout)
        print(f"[RealMan] set_modbus_mode result={result}")
        return result

    def write_registers(self, address: int, num: int, data: list[int]):
        params = self._make_params(address=address, num=num)
        result = self.arm.rm_write_registers(params, data)
        print(f"[RealMan] write_registers address={address}, num={num}, data={data}, result={result}")
        return result

    def read_input_registers(self, address: int):
        params = self._make_params(address=address)
        result = self.arm.rm_read_input_registers(params)
        print(f"[RealMan] read_input_registers address={address}, result={result}")
        return result

    def read_holding_registers(self, address: int, num: int):
        params = self._make_params(address=address, num=num)
        result = self.arm.rm_read_multiple_holding_registers(params)
        print(f"[RealMan] read_holding_registers address={address}, num={num}, result={result}")
        return result

    def wait_until_activated(self, timeout_s: float = 10.0, poll_interval_s: float = 0.2):
        deadline = time.time() + timeout_s
        last_status = None
        last_value = None
        while time.time() < deadline:
            last_status, last_value = self.read_input_registers(2000)
            print(f"[RealMan] activation status={last_status}, value={last_value}")
            # 不同固件/SDK 可能返回不同数值。
            # 你当前设备已经从 209 变化到 241 并保持稳定，说明激活流程已经到达可用状态。
            if last_status == 0 and last_value in (0x31, 49, 0xD1, 209, 0xF1, 241):
                return last_status, last_value
            time.sleep(poll_interval_s)
        raise TimeoutError(f"Gripper activation timeout, last_status={last_status}, last_value={last_value}")

    def activate_gripper(self):
        self.write_registers(address=1000, num=1, data=[0, 0])
        time.sleep(0.2)
        self.write_registers(address=1000, num=1, data=[0, 1])
        return self.wait_until_activated()

    def close_gripper(self):
        self.write_registers(address=1000, num=3, data=[0, 9, 255, 0, 255, 255])

    def open_gripper(self):
        self.write_registers(address=1000, num=3, data=[0, 9, 100, 0, 255, 255])

    def wait_motion_done(self, timeout_s: float = 10.0, poll_interval_s: float = 0.2):
        deadline = time.time() + timeout_s
        last_result = None
        while time.time() < deadline:
            last_result = self.read_holding_registers(2000, 6)
            print(f"[RealMan] motion status={last_result}")
            time.sleep(poll_interval_s)
        return last_result

    def get_gripper_torque(self):
        status, data = self.read_holding_registers(2000, 6)
        print(f"[RealMan] gripper status={status}, data={data}")
        if isinstance(data, (list, tuple)) and len(data) >= 6:
            return data[4]
        return data

    def run_demo(self):
        print("[RealMan] demo start")

        # self.set_tool_voltage_24v()
        # time.sleep(1.0)

        # self.set_modbus_mode()
        # time.sleep(1.0)
        #
        # self.activate_gripper()
        # time.sleep(1.0)

        # print("[RealMan] open gripper")
        # self.open_gripper()
        # time.sleep(5.02)

        # print("[RealMan] close gripper")
        self.close_gripper()
        # self.wait_motion_done(timeout_s=4.83)
        time.sleep(3)
        print("close")
        # time.sleep(0.5)
        self.open_gripper()
        # time.sleep(7.43)
        # self.close_gripper()
        # time.sleep(4.08)
        # self.open_gripper()




        # print("[RealMan] reopen gripper")
        # self.open_gripper()
        # self.wait_motion_done(timeout_s=3.0)
        #
        # torque = self.get_gripper_torque()
        # print(f"[RealMan] torque={torque}")
        # print("[RealMan] demo done")


if __name__ == "__main__":
    controller = RealManGripperController(
        robot_ip="192.168.1.18",
        robot_port=8080,
        device_id=9,
        modbus_port=1,
    )
    try:
        controller.run_demo()
        # while True:
        #     if keyboard.is_pressed("o"):
        #         controller.close_gripper()
        #     elif keyboard.is_pressed("p"):
        #         controller.open_gripper()




    finally:
        controller.close()
