"""Shared car, camera, floor and initial-joint scene parameters."""
import os
import numpy as np
from .paths import WORKSPACE_ROOT


D405_WIDTH_M = 0.042
D405_HEIGHT_M = 0.042
D405_DEPTH_M = 0.023

# Match Fruit_picker_project/Simulation/realman_sim_car.py exactly: RM65 stays
# at the mobile-base origin, while car.stl is offset and rotated below/behind it.
CAR_MODEL_PATH = os.path.join(
    WORKSPACE_ROOT, "wrs", "LinZhi_picker_project", "model", "car.stl")
CAR_FROM_ROBOT_BASE = np.array([-0.10, 0.0, -0.08])
CAR_ROTATION = np.array([[np.cos(-np.pi / 2), 0., np.sin(-np.pi / 2)],
                         [0., 1., 0.],
                         [-np.sin(-np.pi / 2), 0., np.cos(-np.pi / 2)]])
ROBOT_BASE_Z = 0.0
CAR_LOCAL_MIN = np.array([-0.62436676, -0.34500006, -0.49075317])
CAR_LOCAL_MAX = np.array([1.37558246, 0.34500000, 0.0])
FLOOR_SIZE = np.array([3.2, 2.2, 0.02])


def transformed_car_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Return the car AABB relative to the mobile RM65 base."""
    corners = np.array([
        [x, y, z]
        for x in (CAR_LOCAL_MIN[0], CAR_LOCAL_MAX[0])
        for y in (CAR_LOCAL_MIN[1], CAR_LOCAL_MAX[1])
        for z in (CAR_LOCAL_MIN[2], CAR_LOCAL_MAX[2])
    ])
    transformed = (CAR_ROTATION @ corners.T).T + CAR_FROM_ROBOT_BASE
    return transformed.min(axis=0), transformed.max(axis=0)


CAR_RELATIVE_MIN, CAR_RELATIVE_MAX = transformed_car_bounds()
# Put the floor's upper surface exactly at the rotated car mesh's lowest point.
# The car/robot mounting transform remains identical to realman_sim_car.py.
FLOOR_TOP_Z = float(CAR_RELATIVE_MIN[2])
FLOOR_CENTER = np.array(
    [-0.20, 0.0, FLOOR_TOP_Z - FLOOR_SIZE[2] / 2.0])

# Measured initial observation configuration from the physical RM65-B.
# INITIAL_VIEWPOINT_JOINT_RAD = np.array(
#     [-1.85877565, 1.94225470, -1.88045259,
#      -1.17633450, -1.54990476, 3.06841112],
#     dtype=float,
# )
INITIAL_VIEWPOINT_JOINT_RAD = np.array(
    [0.06897541, 1.55276713, -1.98419496,
     -0.03705334, 1.97552078, 0.01207768],
    dtype=float,
)
