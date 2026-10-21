"""Display geometry shared by the scene viewer and dataset replay."""
import numpy as np
import wrs.modeling.geometric_model as mgm
import wrs.modeling.model_collection as mmc

from .scene_config import D405_DEPTH_M, D405_HEIGHT_M, D405_WIDTH_M


def gen_d405_model(cam_pos: np.ndarray, cam_rotmat: np.ndarray) -> mmc.ModelCollection:
    """Generate a compact 42 x 42 x 23 mm D405 body at the optical frame."""
    collection = mmc.ModelCollection(name="realsense_d405")
    right, forward, up = cam_rotmat[:, 0], cam_rotmat[:, 1], cam_rotmat[:, 2]
    # The optical origin is on the front plane; the housing extends backward.
    body_center = cam_pos - forward * (D405_DEPTH_M / 2.0)
    mgm.gen_box(
        xyz_lengths=np.array([D405_WIDTH_M, D405_DEPTH_M, D405_HEIGHT_M]),
        pos=body_center,
        rotmat=cam_rotmat,
        rgb=np.array([0.10, 0.10, 0.11]),
        alpha=1.0,
    ).attach_to(collection)
    for offset in (-0.011, 0.011):
        mgm.gen_sphere(
            pos=cam_pos + right * offset - forward * 0.0005,
            radius=0.004,
            rgb=np.array([0.12, 0.28, 0.48]),
            alpha=1.0,
        ).attach_to(collection)
    return collection
