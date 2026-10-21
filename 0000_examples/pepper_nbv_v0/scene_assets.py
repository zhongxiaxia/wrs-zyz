"""Use v1 scene assets while retaining v0's single-fruit NBV region.

The legacy pepper/stem meshes are *unrendered* ROI references, not duplicate
scene geometry. The visible plant uses v1's three complete semantic meshes.
"""
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
V1_ROOT = PROJECT_ROOT.parent / "pepper_nbv_v1"
for root in (WORKSPACE_ROOT, V1_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from pepper_nbv.paths import MODEL_DIR
from pepper_nbv.config import NBVConfig
from pepper_nbv.robot_adapter import RealmanAdapter, load_tcp_from_camera, tcp_to_camera, camera_to_tcp
from pepper_nbv.scene_config import (
    CAR_MODEL_PATH, CAR_FROM_ROBOT_BASE, CAR_ROTATION, ROBOT_BASE_Z,
    INITIAL_VIEWPOINT_JOINT_RAD, FLOOR_SIZE, FLOOR_CENTER,
    D405_WIDTH_M, D405_HEIGHT_M,
)
from pepper_nbv.scene_models import gen_d405_model


def load_plant_scene(base, offset=None):
    """Attach the v1 plant; return unrendered fruit/stem ROIs and scene models."""
    import wrs.modeling.geometric_model as mgm
    import wrs.modeling.collision_model as mcm

    offset = np.zeros(3) if offset is None else np.asarray(offset, dtype=float)
    specs = (
        ("peppers_all.stl", [0.92, 0.12, 0.04], 1.0),
        ("stems_all.stl", [0.20, 0.82, 0.08], 1.0),
        ("plant_other_all.stl", [0.12, 0.42, 0.16], 0.72),
    )
    paths = [MODEL_DIR / filename for filename, _, _ in specs]
    paths.extend(MODEL_DIR / filename for filename in ("pepper.stl", "stem.stl"))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required v1 plant asset is missing: {path}")
    models = []
    for filename, rgb, alpha in specs:
        model = mcm.CollisionModel(str(MODEL_DIR / filename), rgb=np.array(rgb), alpha=alpha)
        model.pos = offset.copy()
        model.attach_to(base)
        models.append(model)
    references = []
    for filename in ("pepper.stl", "stem.stl"):
        model = mgm.GeometricModel(str(MODEL_DIR / filename))
        model.pos = offset.copy()
        references.append(model)
    return *references, models


def create_robot_setup():
    """Match v1 scene_viewer's car, initial joints, calibration and plant offset."""
    import wrs.modeling.collision_model as mcm

    robot_base = np.array([0.0, 0.0, ROBOT_BASE_Z])
    car = mcm.CollisionModel(CAR_MODEL_PATH, rgb=np.array([0.60, 0.60, 0.60]), alpha=0.92)
    car.pos = robot_base + CAR_FROM_ROBOT_BASE
    car.rotmat = CAR_ROTATION
    adapter = RealmanAdapter(use_tracik=False, base_pos=robot_base,
                             initial_joints=INITIAL_VIEWPOINT_JOINT_RAD,
                             obstacle_list=[car])
    adapter.set_joint_configuration(INITIAL_VIEWPOINT_JOINT_RAD)
    if not adapter.current_configuration_is_feasible():
        raise RuntimeError(f"v1 initial robot configuration rejected: {adapter.last_check.reason}")
    cam_pos, cam_rot = adapter.camera_pose()
    cfg = NBVConfig()
    target_local = 0.5 * (cfg.top_local + cfg.pick_local)
    cfg.object_offset = cam_pos + cfg.camera_radius * cam_rot[:, 1] - target_local
    cfg.object_offset[2] += 0.05  # scene_viewer's default --plant-lift
    return adapter, car, cfg
