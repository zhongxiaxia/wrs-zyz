# hardware/lift_api.py
from __future__ import annotations

import base64
import json
import time
from typing import Any
from urllib import error, request


class LiftAPI:
    """升降机构底层 API。

    对齐 lift_test.py 的通信方式：
    - POST /woosh/ros/CallAction 发送动作
    - POST /woosh/ros/Feedbacks 轮询反馈
    - 动作字段使用 lift_control3
    - height 单位为米
    """

    ACTION_NAME = "woosh.ros.action.LiftControl3"

    def __init__(
        self,
        host: str = "169.254.128.2",
        port: int = 5480,
        username: str | None = None,
        password: str | None = None,
        timeout: float = 10.0,
        feedback_timeout: float = 120.0,
        feedback_interval: float = 1.0,
        debug: bool = True,
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.timeout = timeout
        self.feedback_timeout = feedback_timeout
        self.feedback_interval = feedback_interval
        self.debug = debug
        self.current_height = 0.0
        self.action_endpoint = f"http://{self.host}:{self.port}/woosh/ros/CallAction"
        self.feedback_endpoint = f"http://{self.host}:{self.port}/woosh/ros/Feedbacks"
        print(f"[硬件模块] 升降平台初始化完毕 (endpoint: {self.action_endpoint})")

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
        }
        if self.username and self.password:
            token = base64.b64encode(f"{self.username}:{self.password}".encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {token}"
        return headers

    def _post(self, url: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(url, data=body, headers=self._build_headers(), method="POST")
        try:
            with request.urlopen(req, timeout=timeout or self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
                try:
                    data = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    data = {"raw": raw}
                return {"status": resp.status, "reason": getattr(resp, "reason", ""), "data": data, "raw": raw}
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"升降机构请求失败: HTTP {exc.code} - {exc.reason}\n响应体: {raw}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"升降机构请求失败: {exc.reason}") from exc

    def _send_lift_action(self, height: float) -> dict[str, Any]:
        payload = {
            "lift_control3": {
                "execute_mode": 1,
                "height": height,
                "speed": 0.03,
                "action": 1,
            }
        }
        if self.debug:
            print(f"[升降 API] 发送动作: {json.dumps(payload, ensure_ascii=False)}")
        result = self._post(self.action_endpoint, payload, timeout=self.timeout)
        if self.debug:
            print(f"[升降 API] 动作响应: {json.dumps(result, ensure_ascii=False)}")
        msg = result.get("data", {}).get("msg")
        if msg and msg != "Request succeed":
            raise RuntimeError(f"升降机构动作被拒绝: {msg}")
        return result

    def _wait_feedback(self, target_height: float) -> dict[str, Any]:
        start_time = time.time()
        consecutive_missing = 0

        while time.time() - start_time < self.feedback_timeout:
            try:
                feedback = self._post(self.feedback_endpoint, {}, timeout=self.timeout)
                feedback_data = feedback.get("data", {})
            except Exception as exc:
                consecutive_missing += 1
                if self.debug:
                    print(f"[升降 API] 获取反馈失败: {exc}")
                time.sleep(self.feedback_interval)
                continue

            fbs = feedback_data.get("body", {}).get("fbs", [])
            lift_action_found = False

            for fb in fbs:
                if fb.get("action") == self.ACTION_NAME:
                    lift_action_found = True
                    state = fb.get("state")

                    if state == 1:
                        if self.debug:
                            print(f"[升降 API] 升降完成，目标高度: {target_height}m")
                        self.current_height = target_height
                        return {"status": 0, "data": feedback_data, "raw": feedback.get("raw", "")}

                    if state == -1:
                        raise RuntimeError(f"升降动作失败: {fb.get('msg', '未知原因')}")

                    if state == 2 and self.debug:
                        print(f"[升降 API] 执行中，目标高度: {target_height}m")

            if not lift_action_found:
                consecutive_missing += 1
                if self.debug:
                    print(f"[升降 API] 未找到升降动作反馈，连续次数: {consecutive_missing}")
                if consecutive_missing >= 3:
                    if self.debug:
                        print("[升降 API] 连续未找到升降反馈，认为动作窗口已结束")
                    return {"status": -1, "data": feedback_data, "raw": feedback.get("raw", "")}

            time.sleep(self.feedback_interval)

        raise RuntimeError(f"升降动作超时（{self.feedback_timeout}秒）")

    def move_level_3(self, height: float = 0.5) -> dict[str, Any]:
        """移动到指定高度（单位：米）。"""
        if self.debug:
            print(f"[升降 API] 请求升降机构3，height={height}...")
        self._send_lift_action(height)
        return self._wait_feedback(height)

    def reset_to_bottom(self):
        """回到底部，状态机内用于复位。"""
        print("[升降 API] 升降平台正在降至最低点...")
        try:
            self.move_level_3(0.05)
        finally:
            self.current_height = 0.05
            print("[升降 API] 已回到底部。")
