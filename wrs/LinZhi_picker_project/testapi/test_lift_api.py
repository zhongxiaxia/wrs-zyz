# test_lift_api.py
from __future__ import annotations

import argparse
import json
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
import sys
import time

# 允许直接运行本文件时，正确找到项目根目录下的 hardware 包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.lift_api import LiftAPI


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="测试升降机构 API 可用性")
    parser.add_argument("--host", default="169.254.128.2", help="机器人 IP")
    parser.add_argument("--port", default=5480, type=int, help="机器人端口")
    parser.add_argument("--user", default="admin", help="Basic Auth 用户名")
    parser.add_argument("--password", default="123456", help="Basic Auth 密码")
    parser.add_argument("--timeout", default=10.0, type=float, help="请求超时时间(秒)")
    parser.add_argument("--sleep", default=1.0, type=float, help="动作之间等待时间(秒)")
    parser.add_argument("--only", choices=["level3", "reset"], help="只测试单个动作")
    parser.add_argument("--debug-file", action="store_true", help="仅保存日志到文件，不在控制台重复输出")
    return parser.parse_args()


def log_result(title: str, result):
    print(f"\n===== {title} =====")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    args = parse_args()

    logs_dir = Path(__file__).with_name("logs")
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"lift_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    with log_file.open("w", encoding="utf-8") as f:
        tee_out = Tee(sys.stdout, f) if not args.debug_file else Tee(f)
        tee_err = Tee(sys.stderr, f) if not args.debug_file else Tee(f)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"[日志] 本次输出已保存到: {log_file}")
            print(f"[日志] 参数: {args}")

            api = LiftAPI(
                host=args.host,
                port=args.port,
                username=args.user,
                password=args.password,
            )

            tests = {
                "level3": ("move_level_3", lambda: api.move_level_3(height=0.1)),
                "reset": ("reset_to_bottom", api.reset_to_bottom),
            }

            if args.only:
                run_list = [tests[args.only]]
            else:
                run_list = [tests[k] for k in ["level3", "reset"]]

            for name, func in run_list:
                try:
                    print(f"\n[TEST] 开始测试: {name}")
                    result = func()
                    if result is not None:
                        log_result(name, result)
                    else:
                        print(f"[TEST] {name} 执行完成，无返回值")
                    print(f"[TEST] 当前记录高度: {api.current_height}")
                except Exception as exc:
                    print(f"[TEST] {name} 失败: {repr(exc)}")

                if args.sleep > 0:
                    time.sleep(args.sleep)

            print("[TEST] 升降机构 API 测试结束")


if __name__ == "__main__":
    main()
