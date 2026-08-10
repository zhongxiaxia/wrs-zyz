# config.py
# --- 硬件网络配置 ---
CHASSIS_IP = "192.168.1.100"
ARM_IP = "192.168.1.18"
LIFT_HOST = "169.254.128.2"
LIFT_PORT = 5480
# --- 视觉与算法配置 ---
YOLO_MODEL_PATH = "/home/nvidia/wrs_main/wrs-virtual/wrs/Pepper/best.pt"
CAMERA_DEVICE = 0
CONFIDENCE_THRESHOLD = 0.8  # 置信度阈值
ARM_NO_TARGET_FRAME_LIMIT = 5
ARM_PICK_RETRY_LIMIT = 1
ARM_MOVE_RETRY_LIMIT = 1
ARM_GRIPPER_OPEN_WAIT_S = 0.5
ARM_GRIPPER_CLOSE_WAIT_S = 0.5
ARM_APPROACH_OFFSET_M = 0.12
ARM_RETREAT_OFFSET_M = 0.08
# --- 运动参数配置 ---
# 中层升降机嵌套参数（单位：mm）
# 状态机专用范围：0.05m ~ 0.5m
LIFT_MIN_HEIGHT_MM = 50     # 最低位置 0.05m
LIFT_MAX_HEIGHT_MM = 500    # 最高位置 0.5m
LIFT_STEP_MM = 150          # 每层固定上升/下降高度，默认 20mm
# --- 机械臂采摘参数 ---
ARM_OBSERVE_POSE_NAME = "observe"
ARM_SAFE_POSE_NAME = "safe"
ARM_BASKET_POSE_NAME = "basket_drop"