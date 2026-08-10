# single_cut.py
import os
import time

try:
    from .main import SCISSOR_PORT_ENV, STM32ScissorController, list_serial_ports
except ImportError:
    from main import SCISSOR_PORT_ENV, STM32ScissorController, list_serial_ports


def resolve_scissor_port():
    env_port = os.environ.get(SCISSOR_PORT_ENV, "").strip()
    if env_port:
        return env_port

    available_ports = list_serial_ports()
    if available_ports:
        if len(available_ports) > 1:
            print("发现多个串口，默认使用排序后的第一个：")
            for index, port in enumerate(available_ports):
                print(f"  [{index}] {port}")
        return available_ports[0]

    raise RuntimeError(
        f"未找到串口，请设置 {SCISSOR_PORT_ENV}，例如 /dev/serial/by-id/..."
    )


def single_cut_example():
    """单次唤醒示例"""
    controller = STM32ScissorController(resolve_scissor_port())

    try:
        # 连接
        if not controller.connect():
            return

        # 检查并唤醒
        if controller.checkstate() == 0:
            print("唤醒设备...")
            if not controller.shearawake():
                print("唤醒失败")
                return



    finally:
        controller.cleanup()


if __name__ == "__main__":
    single_cut_example()
