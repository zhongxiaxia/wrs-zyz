from __future__ import annotations

import config
from hardware.arm_api import ArmAPI
from hardware.camera_api import CameraAPI
from hardware.chassis_api import ChassisAPI
from hardware.lift_api import LiftAPI
from fsm.state import ArmState, ErrorState, IdleState, InitState, LiftState, MoveState, ShutdownState


class FruitPickerRobot:
    """机器人上下文类：负责状态切换和三层嵌套状态机框架"""

    def __init__(self):
        self.chassis_api = ChassisAPI(config.CHASSIS_IP)
        self.arm_api = ArmAPI(config.ARM_IP)
        self.camera_api = CameraAPI(
            config.YOLO_MODEL_PATH,
            device=config.CAMERA_DEVICE,
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
        )
        self.lift_api = LiftAPI(host=config.LIFT_HOST, port=config.LIFT_PORT)

        self.error_info = ""
        self.move_repeat_count = 0
        self.move_repeat_limit = 3
        self.lift_step_index = 0
        self.lift_min_height = config.LIFT_MIN_HEIGHT_MM
        self.lift_max_height = config.LIFT_MAX_HEIGHT_MM
        self.lift_step_mm = config.LIFT_STEP_MM
        self.lift_current_target = self.lift_min_height
        self.lift_return_required = False
        self.lift_advance_required = False
        self.lift_inited = False

        self.arm_state = "OBSERVE"
        self.arm_no_target_frames = 0
        self.arm_target_queue: list = []
        self.arm_active_target = None
        self.arm_pick_done = False
        self.arm_retry_count = 0
        self.arm_layer_no_target_frames = 0
        self.arm_layer_no_target_frames = 0

        self.real_picker = None
        self.states = {
            "INIT": InitState(self),
            "MOVE": MoveState(self),
            "LIFT": LiftState(self),
            "ARM": ArmState(self),
            "IDLE": IdleState(self),
            "ERROR": ErrorState(self),
            "SHUTDOWN": ShutdownState(self),
        }

        self.current_state_name = None
        self.current_state = None

    def configure_lift_layer(self, step_mm: int | None = None, min_height_mm: int | None = None, max_height_mm: int | None = None):
        if step_mm is not None:
            self.lift_step_mm = step_mm
        if min_height_mm is not None:
            self.lift_min_height = min_height_mm
        if max_height_mm is not None:
            self.lift_max_height = max_height_mm
        self.lift_current_target = self.lift_min_height
        print(
            f"[配置] 升降机中层参数：step={self.lift_step_mm}mm, "
            f"min_height={self.lift_min_height}mm, max_height={self.lift_max_height}mm"
        )

    def change_state(self, new_state_name):
        if self.current_state:
            self.current_state.exit()
        self.current_state_name = new_state_name
        self.current_state = self.states[new_state_name]
        self.current_state.enter()

    def update(self):
        if self.current_state:
            next_state_name = self.current_state.execute()
            if next_state_name:
                self.change_state(next_state_name)

    def reset_workflow(self):
        self.move_repeat_count = 0
        self.reset_nested_states()

    def reset_nested_states(self):
        self.lift_step_index = 0
        self.lift_current_target = self.lift_min_height
        self.lift_return_required = False
        self.lift_advance_required = False
        self.lift_inited = False
        self.arm_state = "OBSERVE"
        self.arm_no_target_frames = 0
        self.arm_layer_no_target_frames = 0
        self.arm_target_queue = []
        self.arm_active_target = None
        self.arm_pick_done = False
        self.arm_retry_count = 0

    def on_enter_move_layer(self):
        print("[MOVE] 外层移动逻辑：当前仅保留框架，等待后续实现。")

    def run_move_layer(self):
        if self.move_repeat_count < self.move_repeat_limit:
            self.move_repeat_count += 1
            print(f"[MOVE] 外层第 {self.move_repeat_count}/{self.move_repeat_limit} 次循环")
            return "LIFT"
        print("[MOVE] 外层循环已完成 3 次，准备退出到 IDLE。")
        return "IDLE"

    def reset_lift_layer(self):
        self.lift_step_index = 0
        self.lift_current_target = self.lift_min_height
        self.lift_return_required = False
        self.lift_inited = False
        self.lift_api.reset_to_bottom()
        print("[LIFT] 进入中层前，升降机已回到底部。")

    def run_lift_layer(self):
        if not self.lift_inited:
            self.lift_api.reset_to_bottom()
            self.lift_current_target = self.lift_min_height
            self.lift_inited = True
            print(f"[LIFT] 中层初始化完成，当前目标高度: {self.lift_current_target}")

        if self.lift_return_required:
            print("[LIFT] 采摘结束，直接回到底部并退出中层。")
            self.lift_api.reset_to_bottom()
            self.reset_nested_states()
            return "MOVE"

        if self.lift_advance_required:
            self.lift_advance_required = False
            if self.lift_current_target + self.lift_step_mm > self.lift_max_height:
                print("[LIFT] 下一次上升将超过最高高度，切换到回底流程。")
                self.lift_return_required = True
                return None
            self.lift_current_target += self.lift_step_mm
            print(f"[LIFT] 当前层采摘完成，升到下一层: {self.lift_current_target}")
            self.lift_api.move_level_3(height=self.lift_current_target / 1000.0)
            self.lift_step_index = 1
            return "ARM"

        if self.lift_step_index == 0:
            if self.lift_current_target == self.lift_min_height:
                print(f"[LIFT] 初始位置也执行一次采摘，当前高度: {self.lift_current_target}")
            else:
                next_height = self.lift_current_target + self.lift_step_mm
                if next_height > self.lift_max_height:
                    print("[LIFT] 下一次上升将超过最高高度，切换到回底流程。")
                    self.lift_return_required = True
                    return None
                self.lift_current_target = next_height
                print(f"[LIFT] 升降机移动到固定层位: {self.lift_current_target}")
                self.lift_api.move_level_3(height=self.lift_current_target / 1000.0)

            self.lift_step_index = 1
            return "ARM"

        self.lift_step_index = 0
        return None

    def enter_arm_cycle(self):
        self.arm_state = "OBSERVE"
        self.arm_no_target_frames = 0
        self.arm_layer_no_target_frames = 0
        self.arm_target_queue = []
        self.arm_active_target = None
        self.arm_pick_done = False
        self.arm_retry_count = 0
        print("[ARM] 机械臂进入采摘循环，移动到初始观察点。")
        if self.real_picker is not None:
            print("[ARM] 已接入 RealmanPepperPickerFast，优先走真实采摘链路。")
        else:
            self.arm_api.move_to_observe_pose()
        self.arm_state = "OBSERVE"

    def _arm_exit_to_lift(self):
        print("[ARM] 当前层连续5帧无果，准备切换到下一升降位。")
        self.arm_state = "IDLE"
        self.lift_advance_required = True
        self.arm_layer_no_target_frames = 0
        self.arm_no_target_frames = 0
        self.lift_step_index = 0
        return "LIFT"

    def run_arm_layer(self):
        if self.arm_state == "IDLE":
            return None

        if self.real_picker is not None:
            try:
                result = self.real_picker.pick_place_fast(
                    continuous_mode=False,
                    stop_on_no_detection=True,
                    cycle_once=True,
                )
                if result and result.get("success"):
                    print(f"[ARM] 真实采摘器本轮成功：picked={result.get('picked')}, duration={result.get('duration'):.3f}s")
                    self.arm_pick_done = True
                    self.arm_retry_count = 0
                    self.arm_layer_no_target_frames = 0
                    self.arm_no_target_frames = 0
                    self.arm_state = "OBSERVE"
                    return None
                self.arm_layer_no_target_frames += 1
                self.arm_no_target_frames = self.arm_layer_no_target_frames
                print(f"[ARM] 真实采摘器本轮未成功，连续无目标/失败帧数: {self.arm_no_target_frames}")
                if self.arm_no_target_frames >= config.ARM_NO_TARGET_FRAME_LIMIT:
                    return self._arm_exit_to_lift()
                self.arm_state = "OBSERVE"
                return None
            except Exception as e:
                self.error_info = f"RealmanPepperPickerFast 采摘失败: {e}"
                print(f"[ARM] {self.error_info}")
                self.real_picker = None
                self.arm_state = "OBSERVE"

        if self.arm_state == "OBSERVE":
            targets = self.camera_api.detect_targets()
            if not targets:
                self.arm_layer_no_target_frames += 1
                self.arm_no_target_frames = self.arm_layer_no_target_frames
                print(f"[ARM] 未识别到目标点，连续无目标帧数: {self.arm_no_target_frames}")
                if self.arm_no_target_frames >= config.ARM_NO_TARGET_FRAME_LIMIT:
                    return self._arm_exit_to_lift()
                return None

            self.arm_layer_no_target_frames = 0
            self.arm_no_target_frames = 0
            self.arm_target_queue = targets
            self.arm_active_target = self.arm_target_queue[0]
            print(f"[ARM] 识别到目标点: {self.arm_active_target.xyz}, confidence={self.arm_active_target.confidence}")
            self.arm_state = "TARGET_SELECT"
            return None

        if self.arm_state == "TARGET_SELECT":
            if not self.arm_active_target:
                self.arm_state = "OBSERVE"
                return None
            if not self.arm_api.pre_grasp_check(self.arm_active_target.xyz):
                self.arm_state = "OBSERVE"
                return None
            self.arm_state = "APPROACH"
            return None

        if self.arm_state == "APPROACH":
            if not self.arm_api.approach_target(self.arm_active_target.xyz):
                self.arm_state = "OBSERVE"
                return None
            self.arm_state = "GRASP"
            return None

        if self.arm_state == "GRASP":
            if self.arm_active_target is None:
                self.arm_state = "OBSERVE"
                return None
            if self.arm_api.grab(self.arm_active_target.xyz):
                self.arm_state = "RETREAT"
                self.arm_pick_done = True
                self.arm_retry_count = 0
            else:
                self.arm_retry_count += 1
                print(f"[ARM] 抓取失败，重试次数: {self.arm_retry_count}")
                if self.arm_retry_count <= config.ARM_PICK_RETRY_LIMIT:
                    self.arm_state = "APPROACH"
                else:
                    self.arm_state = "OBSERVE"
                    self.arm_active_target = None
            return None

        if self.arm_state == "RETREAT":
            if self.arm_api.retreat_after_grab():
                self.arm_state = "DROP"
            else:
                self.arm_state = "OBSERVE"
            return None

        if self.arm_state == "DROP":
            if self.arm_api.put_in_basket():
                self.arm_state = "RESCAN"
            else:
                self.arm_state = "OBSERVE"
            return None

        if self.arm_state == "RESCAN":
            print("[ARM] 采摘完成，重新启动相机继续观察。")
            self.arm_state = "OBSERVE"
            return None

        return None
