from __future__ import annotations

import glob
import os
import platform
import time
from typing import List, Optional

from jodellSdk.jodellSdkDemo import ClawHepgTool


class ClawController:
    def __init__(self, claw_id, com_port: Optional[str] = None, baud_rate: int = 115200, auto_detect: bool = True):
        """
        Args:
            claw_id: gripper id
            com_port: explicit serial port. Example: 'COM6' on Windows, '/dev/ttyUSB0' on Linux.
            baud_rate: serial baud rate
            auto_detect: when True, fallback to auto-selected port if com_port is missing/invalid
        """
        self.clawTool = ClawHepgTool()
        self.claw_id = claw_id
        self.baud_rate = baud_rate

        try:
            self.com_list = self.clawTool.searchCom() or []
        except Exception:
            self.com_list = []

        self.com_port = self._resolve_com_port(com_port=com_port, auto_detect=auto_detect)
        print(f"[Claw] available ports from SDK: {self.com_list}")
        print(f"[Claw] selected port: {self.com_port}")

    @staticmethod
    def _is_windows() -> bool:
        return platform.system().lower().startswith("win")

    @staticmethod
    def _linux_candidates() -> List[str]:
        patterns = ["/dev/ttyUSB*", "/dev/ttyACM*", "/dev/ttyTHS*", "/dev/ttyS*"]
        ports: List[str] = []
        for p in patterns:
            ports.extend(glob.glob(p))
        return sorted(set(ports))

    def _is_port_usable(self, port: Optional[str]) -> bool:
        if not port:
            return False
        if self._is_windows():
            return True
        # Linux/macOS: require device path exists
        if port.startswith("/dev/"):
            return os.path.exists(port)
        # COMx is invalid on Linux/macOS
        if port.upper().startswith("COM"):
            return False
        return os.path.exists(port)

    def _rank_port(self, port: str) -> int:
        # smaller score = higher priority
        if port.startswith("/dev/ttyUSB"):
            return 0
        if port.startswith("/dev/ttyACM"):
            return 1
        if port.startswith("COM"):
            return 2
        if port.startswith("/dev/ttyTHS"):
            return 3
        if port.startswith("/dev/ttyS"):
            return 4
        return 10

    def _resolve_com_port(self, com_port: Optional[str], auto_detect: bool) -> Optional[str]:
        # 1) user-provided valid port wins
        if self._is_port_usable(com_port):
            return com_port

        if not auto_detect:
            return com_port

        # 2) fallback to SDK discovered ports
        sdk_ports = [p for p in self.com_list if isinstance(p, str) and p]
        sdk_ports = sorted(sdk_ports, key=self._rank_port)
        for p in sdk_ports:
            if self._is_port_usable(p):
                return p

        # 3) fallback to OS scan on Linux/macOS
        if not self._is_windows():
            local_ports = self._linux_candidates()
            if local_ports:
                return local_ports[0]

        # 4) last resort: keep user input (for environments where exists-check is restricted)
        return com_port

    def connect_gripper(self):
        if not self.com_port:
            raise RuntimeError(
                "No serial port selected for gripper. "
                "Set com_port explicitly, or check /dev/ttyUSB* /dev/ttyACM*."
            )

        flag = self.clawTool.serialOperation(self.com_port, self.baud_rate, True)
        if flag == 1:
            print("[Claw] connected")
        else:
            raise RuntimeError(f"[Claw] connect failed, code={flag}, port={self.com_port}")

    def disconnect_gripper(self):
        flag = self.clawTool.serialOperation(self.com_port, self.baud_rate, False)
        if flag == 1:
            print("[Claw] disconnected")
        else:
            raise RuntimeError(f"[Claw] disconnect failed, code={flag}")

    def enable_gripper(self):
        flag = self.clawTool.clawEnable(self.claw_id, True)
        if flag == 1:
            print("[Claw] enabled")
        else:
            raise RuntimeError(f"[Claw] enable failed, code={flag}")

    def close_gripper(self, retries: int = 3, retry_delay_s: float = 0.2):
        last_flag = None
        for attempt in range(1, max(1, int(retries)) + 1):
            last_flag = self.clawTool.runWithParam(self.claw_id, torque=255, speed=255, pos=250)
            if last_flag == 1:
                print("[Claw] closed")
                return True
            print(f"[Claw] close failed, code={last_flag}, attempt={attempt}/{retries}")
            if attempt < retries and retry_delay_s > 0:
                time.sleep(float(retry_delay_s))
        return False

    def open_gripper(self, retries: int = 3, retry_delay_s: float = 0.2):
        last_flag = None
        for attempt in range(1, max(1, int(retries)) + 1):
            last_flag = self.clawTool.runWithParam(self.claw_id, torque=255, speed=255, pos=0)
            if last_flag == 1:
                print("[Claw] opened")
                return True
            print(f"[Claw] open failed, code={last_flag}, attempt={attempt}/{retries}")
            if attempt < retries and retry_delay_s > 0:
                time.sleep(float(retry_delay_s))
        return False

    def get_claw_current_torque(self, claw_id):
        torque = self.clawTool.getClawCurrentTorque(claw_id)
        print(f"[Claw] id={claw_id}, torque={torque}")

    def run_sequence(self):
        self.connect_gripper()
        self.enable_gripper()
        self.open_gripper()
        self.close_gripper()
        self.disconnect_gripper()


if __name__ == "__main__":
    claw_id = 9
    # Linux: set None for auto-detect, or set '/dev/ttyUSB0' explicitly
    # Windows: set 'COM6' etc.
    claw_controller = ClawController(claw_id, com_port="/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG033GD5-if00-port0")
    time.sleep(1)
    claw_controller.connect_gripper()
    time.sleep(1)

    claw_controller.enable_gripper()
    claw_controller.open_gripper()
    time.sleep(3)
    claw_controller.close_gripper()
    time.sleep(3)
    claw_controller.open_gripper()
    claw_controller.get_claw_current_torque(claw_id)
    claw_controller.disconnect_gripper()



