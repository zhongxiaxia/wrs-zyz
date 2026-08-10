from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

# Keep all changes inside FLASK and make the original project importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WRS_ROOT = PROJECT_ROOT.parents[1]
PEPPER_ROOT = WRS_ROOT / "Pepper_"
for path in (str(PROJECT_ROOT), str(PEPPER_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from FLASK.worker import run_robot_worker  # noqa: E402


class RobotProcessManager:
    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._process: mp.Process | None = None
        self._control_queue = self._ctx.Queue()
        self._status_queue = self._ctx.Queue(maxsize=1)
        self._last_error: str | None = None
        self._last_action: str | None = None
        self._last_transition_at: float | None = None
        self._cached_status: dict[str, Any] = self._make_idle_status()

    def _make_idle_status(self) -> dict[str, Any]:
        return {
            "running": False,
            "stop_requested": False,
            "current_state": None,
            "task": "待机",
            "detail": "机器人服务未运行",
            "last_action": self._last_action,
            "last_error": self._last_error,
            "last_transition_at": self._last_transition_at,
            "lift": {},
            "arm": {},
            "move": {},
        }

    def _process_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def _drain_status_queue(self) -> None:
        latest = None
        while True:
            try:
                latest = self._status_queue.get_nowait()
            except Exception:
                break
        if isinstance(latest, dict):
            self._cached_status = latest
            self._last_error = latest.get("last_error", self._last_error)
            self._last_action = latest.get("last_action", self._last_action)
            self._last_transition_at = latest.get("last_transition_at", self._last_transition_at)

    def _prepare_queues(self) -> None:
        try:
            self._control_queue.close()
        except Exception:
            pass
        try:
            self._control_queue.join_thread()
        except Exception:
            pass
        self._control_queue = self._ctx.Queue()
        self._status_queue = self._ctx.Queue(maxsize=1)

    def start(self) -> dict[str, Any]:
        if self._process_alive():
            self._drain_status_queue()
            return {"success": True, "message": "Robot service already running.", "data": self.status()}

        self.stop(force=True)
        self._last_error = None
        self._last_action = "start"
        self._last_transition_at = time.time()
        self._cached_status = self._make_idle_status()
        self._cached_status.update(
            {
                "running": False,
                "stop_requested": False,
                "task": "初始化",
                "detail": "准备启动机器人",
                "last_action": "start",
                "last_transition_at": self._last_transition_at,
            }
        )

        self._prepare_queues()
        self._process = self._ctx.Process(
            target=run_robot_worker,
            args=(self._control_queue, self._status_queue),
            daemon=True,
        )
        self._process.start()
        time.sleep(0.1)
        self._drain_status_queue()
        return {"success": True, "message": "Robot service started.", "data": self.status()}

    def stop(self, force: bool = False) -> dict[str, Any]:
        self._last_action = "stop"
        self._last_transition_at = time.time()
        proc = self._process
        if proc is None:
            self._cached_status = self._make_idle_status()
            self._cached_status.update(
                {
                    "stop_requested": True,
                    "task": "关机",
                    "detail": "机器人资源已释放",
                    "last_action": "stop",
                    "last_transition_at": self._last_transition_at,
                }
            )
            return {"success": True, "message": "Robot process not running.", "data": self.status()}

        try:
            if proc.is_alive():
                self._control_queue.put({"type": "stop"})
                if not force:
                    proc.join(timeout=12.0)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5.0)
        finally:
            self._process = None
            self._prepare_queues()
            self._cached_status = self._make_idle_status()
            self._cached_status.update(
                {
                    "stop_requested": True,
                    "task": "关机",
                    "detail": "机器人资源已释放",
                    "last_action": "stop",
                    "last_transition_at": self._last_transition_at,
                }
            )
        return {"success": True, "message": "Robot service stopped.", "data": self.status()}

    def reset(self) -> dict[str, Any]:
        stop_result = self.stop()
        if not stop_result.get("success"):
            return stop_result
        time.sleep(0.2)
        return self.start()

    def step(self) -> dict[str, Any]:
        if not self._process_alive():
            return {"success": False, "message": "Robot is not initialized."}
        self._last_action = "step"
        self._last_transition_at = time.time()
        self._control_queue.put({"type": "step"})
        time.sleep(0.2)
        self._drain_status_queue()
        return {"success": True, "message": "Robot updated once.", "data": self.status()}

    def trigger_state(self, state_name: str) -> dict[str, Any]:
        if not self._process_alive():
            return {"success": False, "message": "Robot is not initialized."}
        state_name = (state_name or "").strip().upper()
        if not state_name:
            return {"success": False, "message": "Robot state is empty."}
        self._last_action = state_name
        self._last_transition_at = time.time()
        self._control_queue.put({"type": "state", "state": state_name})
        time.sleep(0.2)
        self._drain_status_queue()
        return {"success": True, "message": f"State changed to {state_name}.", "data": self.status()}

    def action(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = (payload.get("action") or "").strip().lower()
        if action in {"start", "init"}:
            return self.start()
        if action in {"stop", "shutdown"}:
            return self.stop()
        if action == "reset":
            return self.reset()
        if action == "step":
            return self.step()
        if action == "state":
            state_name = (payload.get("state") or payload.get("state_name") or "").strip()
            return self.trigger_state(state_name)
        return {"success": False, "message": f"Unsupported action: {action}"}

    def status(self) -> dict[str, Any]:
        self._drain_status_queue()
        data = dict(self._cached_status)
        running = self._process_alive() or bool(data.get("running"))
        data["running"] = running
        data.setdefault("stop_requested", not running)
        data.setdefault("current_state", None)
        data.setdefault("task", "待机" if not running else None)
        data.setdefault("detail", "机器人服务未运行" if not running else None)
        data.setdefault("lift", {})
        data.setdefault("arm", {})
        data.setdefault("move", {})
        data["last_action"] = self._last_action
        data["last_error"] = self._last_error
        data["last_transition_at"] = self._last_transition_at
        return data


manager = RobotProcessManager()
app = Flask(__name__)


@app.get("/health")
def health() -> Any:
    return jsonify({"success": True, "message": "ok"})


@app.get("/status")
def status() -> Any:
    return jsonify(manager.status())


@app.post("/robot/start")
def robot_start() -> Any:
    return jsonify(manager.start())


@app.post("/robot/stop")
def robot_stop() -> Any:
    return jsonify(manager.stop())


@app.post("/robot/state/<state_name>")
def robot_state(state_name: str) -> Any:
    return jsonify(manager.trigger_state(state_name))


@app.post("/robot/step")
def robot_step() -> Any:
    return jsonify(manager.step())


@app.post("/robot/reset")
def robot_reset() -> Any:
    return jsonify(manager.reset())


@app.post("/robot/action")
def robot_action() -> Any:
    payload = request.get_json(silent=True) or {}
    result = manager.action(payload)
    status_code = 400 if not result.get("success") else 200
    return jsonify(result), status_code


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
