"""Configuration for the pepper peduncle NBV simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


from pepper_nbv.paths import MODEL_DIR, WORKSPACE_ROOT as ROOT


@dataclass
class NBVConfig:
    # These two points are initial estimates in the STL local frame. Calibrate them
    # once against the mesh before using the simulation for quantitative claims.
    top_local: np.ndarray = field(
        default_factory=lambda: np.array([0.002, 0.069, 0.503], dtype=float)
    )
    pick_local: np.ndarray = field(
        default_factory=lambda: np.array([-0.002, 0.055, 0.535], dtype=float)
    )
    object_offset: np.ndarray = field(
        default_factory=lambda: np.array([-0.45, 0.0, 0.05], dtype=float)
    )

    image_width: int = 640
    image_height: int = 480
    horizontal_fov_deg: float = 60.0
    camera_radius: float = 0.30
    # At 30 cm standoff, +/-57.3 degrees is +/-30 cm of arc travel from
    # the initial view. This keeps the camera on the paper's fixed-radius sphere.
    surface_motion_limit_m: float = 0.30
    action_step_m: float = 0.03
    max_steps: int = 15

    peduncle_samples: int = 17
    success_visible_fraction: float = 0.75
    success_min_projected_length_px: float = 12.0
    success_requires_endpoints: bool = True

    leaf_width_range: tuple[float, float] = (0.080, 0.140)
    leaf_height_range: tuple[float, float] = (0.060, 0.110)
    leaf_depth_fraction_range: tuple[float, float] = (0.35, 0.70)
    leaf_lateral_jitter: float = 0.012
    leaf_roll_range_deg: tuple[float, float] = (-35.0, 35.0)

    move_cost: float = 0.01
    invalid_action_penalty: float = 0.10
    success_reward: float = 1.0
    quality_gain_weight: float = 0.5

    @property
    def top_world(self) -> np.ndarray:
        return self.top_local + self.object_offset

    @property
    def pick_world(self) -> np.ndarray:
        return self.pick_local + self.object_offset

    @property
    def target_world(self) -> np.ndarray:
        return 0.5 * (self.top_world + self.pick_world)

    @property
    def focal_px(self) -> float:
        return (self.image_width / 2.0) / np.tan(np.deg2rad(self.horizontal_fov_deg) / 2.0)

    @property
    def angular_limit_deg(self) -> float:
        return float(np.rad2deg(self.surface_motion_limit_m / self.camera_radius))

    @property
    def action_step_deg(self) -> float:
        return float(np.rad2deg(self.action_step_m / self.camera_radius))

    @property
    def azimuth_limit_deg(self) -> tuple[float, float]:
        return (-self.angular_limit_deg, self.angular_limit_deg)

    @property
    def elevation_limit_deg(self) -> tuple[float, float]:
        return (-self.angular_limit_deg, self.angular_limit_deg)

    @property
    def azimuth_step_deg(self) -> float:
        return self.action_step_deg

    @property
    def elevation_step_deg(self) -> float:
        return self.action_step_deg
