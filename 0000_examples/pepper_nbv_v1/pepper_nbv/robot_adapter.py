"""Realman eye-in-hand adapter for constraining geometric NBV actions by IK."""

from __future__ import annotations

import numpy as np
import json
from dataclasses import asdict, dataclass
from pathlib import Path


from pepper_nbv.paths import DEFAULT_CALIBRATION


@dataclass
class ReachabilityResult:
    feasible: bool
    reason: str
    position_error_m: float | None = None
    rotation_error_rad: float | None = None
    checked_samples: int = 0
    failed_fraction: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def load_tcp_from_camera(path: Path = DEFAULT_CALIBRATION) -> np.ndarray:
    """Load the camera pose expressed in the scissor-tip TCP frame.

    Matrices follow ``p_parent = T_parent_child @ p_child``. The supplied
    calibration is already ``T_tcp_camera``: the real perception pipeline
    applies it directly after the robot TCP pose. ``Realman`` also defines the
    TCP at +180 mm on flange Z through the ``Hand`` model, so converting from
    the flange here would shift the camera backward by 180 mm a second time.
    """
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    tcp_from_camera = np.asarray(payload["affine_mat"], dtype=float)
    if (tcp_from_camera.shape != (4, 4) or not np.all(np.isfinite(tcp_from_camera)) or
            not np.allclose(tcp_from_camera[3], [0, 0, 0, 1], atol=1e-8)):
        raise ValueError("affine_mat must be a finite rigid 4x4 TCP-from-camera transform")
    # The saved manual calibration has small rounding/non-orthogonality errors.
    # Project only a near-rotation to SO(3); reject scaling/shear or reflection.
    rotation = tcp_from_camera[:3, :3]
    if (np.linalg.det(rotation) <= 0 or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)):
        raise ValueError("affine_mat rotation is not close to a proper rotation")
    u, _, vt = np.linalg.svd(rotation)
    tcp_from_camera[:3, :3] = u @ vt
    return tcp_from_camera


def camera_to_tcp(cam_pos: np.ndarray, cam_rotmat: np.ndarray,
                  tcp_from_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # VirtualDepthCamera uses [right, forward, up]. Calibration uses OpenCV
    # [right, down, forward], hence this explicit basis conversion.
    world_from_camera = np.eye(4)
    world_from_camera[:3, :3] = np.column_stack(
        [cam_rotmat[:, 0], -cam_rotmat[:, 2], cam_rotmat[:, 1]])
    world_from_camera[:3, 3] = cam_pos
    world_from_tcp = world_from_camera @ np.linalg.inv(tcp_from_camera)
    return world_from_tcp[:3, 3], world_from_tcp[:3, :3]


def tcp_to_camera(tcp_pos: np.ndarray, tcp_rotmat: np.ndarray,
                  tcp_from_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return optical-center pose using Panda columns [right, forward, up]."""
    world_from_tcp = np.eye(4)
    world_from_tcp[:3, :3] = tcp_rotmat
    world_from_tcp[:3, 3] = tcp_pos
    world_from_camera_cv = world_from_tcp @ tcp_from_camera
    cv_rot = world_from_camera_cv[:3, :3]
    panda_rot = np.column_stack([cv_rot[:, 0], cv_rot[:, 2], -cv_rot[:, 1]])
    return world_from_camera_cv[:3, 3], panda_rot


class RealmanAdapter:
    def __init__(self, calibration_path: Path = DEFAULT_CALIBRATION,
                 use_tracik: bool = False,
                 base_pos: np.ndarray | None = None,
                 base_rotmat: np.ndarray | None = None,
                 initial_joints: np.ndarray | None = None,
                 obstacle_list: list | None = None,
                 motion_step_rad: float = 0.05,
                 position_tolerance_m: float = 0.002,
                 rotation_tolerance_rad: float = 0.02):
        from wrs.robot_sim.robots.realman.realman_single_arm import Realman
        for value in (motion_step_rad, position_tolerance_m, rotation_tolerance_rad):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("reachability steps and tolerances must be finite and positive")
        base_pos = np.zeros(3) if base_pos is None else np.asarray(base_pos, dtype=float)
        base_rotmat = (np.eye(3) if base_rotmat is None
                       else np.asarray(base_rotmat, dtype=float))
        self.robot = Realman(pos=base_pos, rotmat=base_rotmat, enable_cc=True)
        # The Realman wrapper configures self-pairs but leaves external masks off.
        self.robot.cc.enable_extcd_by_id_list(list(self.robot.cc.cce_dict), type="from")
        # The native Windows TRAC-IK binding can access-violate after Panda3D
        # opens its graphics pipe. WRS's numerical IK is slower but safe for an
        # interactive viewer. Batch tools may opt in before creating a window.
        if not use_tracik:
            self.robot.manipulator._ik_solver = None
        self.tcp_from_camera = load_tcp_from_camera(calibration_path)
        self.current_joints = (self.robot.home_conf.copy() if initial_joints is None
                               else np.asarray(initial_joints, dtype=float).copy())
        if not self._joints_valid(self.current_joints):
            raise ValueError("initial_joints must be finite and within the robot joint limits")
        self.robot.goto_given_conf(self.current_joints)
        self.obstacle_list = [] if obstacle_list is None else list(obstacle_list)
        self._candidate_joints = None
        self._has_valid_view_solution = False
        self._use_tracik = use_tracik
        self.motion_step_rad = float(motion_step_rad)
        self.position_tolerance_m = float(position_tolerance_m)
        self.rotation_tolerance_rad = float(rotation_tolerance_rad)
        self.last_check = ReachabilityResult(False, "not_checked")
        self._camera_body = None

    def set_joint_configuration(self, joints: np.ndarray,
                                valid_view_solution: bool = True) -> None:
        """Set a known joint configuration, such as a measured real pose."""
        joints = np.asarray(joints, dtype=float)
        if not self._joints_valid(joints):
            raise ValueError("joints must be finite and within the robot joint limits")
        self.current_joints = joints.copy()
        self.robot.goto_given_conf(self.current_joints)
        self._candidate_joints = None
        self._has_valid_view_solution = bool(valid_view_solution)

    def set_base_pose(self, pos: np.ndarray, rotmat: np.ndarray | None = None) -> None:
        """Move the robot base while keeping its current joint configuration.

        This is used by the mobile platform viewer: the car, arm and eye-in-hand
        camera move as one rigid assembly. Cached target IK is invalidated because
        a solution computed for the old car pose cannot be reused at the new pose.
        """
        pos = np.asarray(pos, dtype=float)
        rotmat = self.robot.rotmat if rotmat is None else np.asarray(rotmat, dtype=float)
        if pos.shape != (3,) or rotmat.shape != (3, 3):
            raise ValueError("base pose must contain a 3-vector and a 3x3 rotation")
        self.robot.fix_to(pos=pos, rotmat=rotmat)
        self.robot.goto_given_conf(self.current_joints)
        self._candidate_joints = None

    def set_obstacles(self, obstacle_list: list) -> None:
        """Set fixed/moving scene obstacles used by every IK feasibility check."""
        self.obstacle_list = list(obstacle_list)
        self._candidate_joints = None

    def _joints_valid(self, joints: np.ndarray) -> bool:
        limits = self.robot.jnt_ranges
        return (joints.shape == (len(limits),) and np.all(np.isfinite(joints)) and
                np.all(joints >= limits[:, 0]) and np.all(joints <= limits[:, 1]))

    def _configuration_reason(self, joints: np.ndarray) -> str:
        if not self._joints_valid(joints):
            return "joint_limits"
        self.robot.goto_given_conf(joints)
        if self.robot.is_collided():
            return "self_collision"
        # Primitive broad phase followed by mesh surface intersection avoids
        # treating empty space inside the large car/plant AABB as solid matter.
        if self.obstacle_list and self.robot.is_collided(obstacle_list=self.obstacle_list):
            for cce in self.robot.cc.cce_dict.values():
                if cce.lnk.cmodel.is_mcdwith(self.obstacle_list):
                    return "obstacle_collision"
        if self.obstacle_list:
            import wrs.modeling.collision_model as mcm
            if self._camera_body is None:
                self._camera_body = mcm.gen_box(
                    xyz_lengths=np.array([0.042, 0.023, 0.042]))
            pos, rot = self.camera_pose_for(joints)
            self._camera_body.pos = pos - rot[:, 1] * (0.023 / 2)
            self._camera_body.rotmat = rot
            if (self._camera_body.is_pcdwith(self.obstacle_list) and
                    self._camera_body.is_mcdwith(self.obstacle_list)):
                return "camera_obstacle_collision"
        return "ok"

    def check_joint_motion(self, target_joints: np.ndarray,
                           start_joints: np.ndarray | None = None,
                           check_path: bool = True) -> ReachabilityResult:
        """Check a joint-linear transition, restoring robot state on every exit.

        Samples include both ends; this is a discrete check, not continuous
        collision certification or a search for a detour around obstacles.
        """
        old = self.current_joints.copy()
        self._candidate_joints = None
        start = old if start_joints is None else np.asarray(start_joints, dtype=float)
        target = np.asarray(target_joints, dtype=float)
        try:
            if not self._joints_valid(start) or not self._joints_valid(target):
                self.last_check = ReachabilityResult(False, "joint_limits")
                return self.last_check
            intervals = max(1, int(np.ceil(np.max(np.abs(target - start)) / self.motion_step_rad)))
            fractions = np.linspace(0, 1, intervals + 1) if check_path else [1.0]
            for count, fraction in enumerate(fractions, 1):
                reason = self._configuration_reason(start + fraction * (target - start))
                if reason != "ok":
                    self.last_check = ReachabilityResult(
                        False, reason, checked_samples=count, failed_fraction=float(fraction))
                    return self.last_check
            self.last_check = ReachabilityResult(True, "ok", checked_samples=len(fractions))
            return self.last_check
        finally:
            self.robot.goto_given_conf(old)

    def current_configuration_is_feasible(self) -> bool:
        """Check both robot self-collision and configured scene obstacles."""
        return self.check_joint_motion(self.current_joints, check_path=False).feasible

    def feasible(self, cam_pos: np.ndarray, cam_rotmat: np.ndarray,
                 candidate_joints: np.ndarray | None = None,
                 check_path: bool = True) -> bool:
        """Validate camera pose, IK, FK error and the joint-linear transition.

        Cached IK can be supplied, but it goes through the same checks.
        ``check_path=False`` is for initial placement, never an NBV action.
        """
        self._candidate_joints = None
        cam_pos, cam_rotmat = np.asarray(cam_pos, float), np.asarray(cam_rotmat, float)
        if (cam_pos.shape != (3,) or cam_rotmat.shape != (3, 3) or
                not np.all(np.isfinite(cam_pos)) or not np.all(np.isfinite(cam_rotmat)) or
                not np.allclose(cam_rotmat.T @ cam_rotmat, np.eye(3), atol=1e-5) or
                not np.isclose(np.linalg.det(cam_rotmat), 1.0, atol=1e-5)):
            self.last_check = ReachabilityResult(False, "invalid_camera_pose")
            return False
        tcp_pos, tcp_rotmat = camera_to_tcp(cam_pos, cam_rotmat, self.tcp_from_camera)
        seeds = [self.current_joints]
        # Numerical IK is seed-sensitive. Use deterministic robot-space
        # multi-start only for the first camera pose; neighboring NBV poses then
        # use the previous valid solution and remain responsive.
        if candidate_joints is None and not self._use_tracik and not self._has_valid_view_solution:
            rng_state = np.random.get_state()
            np.random.seed(20260919)
            try:
                seeds.extend(self.robot.rand_conf() for _ in range(48))
            finally:
                np.random.set_state(rng_state)
        old = self.current_joints.copy()
        self.last_check = ReachabilityResult(False, "ik_failed")
        try:
            for seed in ([None] if candidate_joints is not None else seeds):
                joints = (np.asarray(candidate_joints, float) if candidate_joints is not None else
                          self.robot.ik(tgt_pos=tcp_pos, tgt_rotmat=tcp_rotmat,
                                        seed_jnt_values=seed))
                if joints is None:
                    continue
                joints = np.asarray(joints, float)
                if not self._joints_valid(joints):
                    self.last_check = ReachabilityResult(False, "joint_limits")
                    continue
                actual_pos, actual_rot = self.camera_pose_for(joints)
                pos_error = float(np.linalg.norm(actual_pos - cam_pos))
                rot_error = float(np.arccos(np.clip(
                    (np.trace(actual_rot.T @ cam_rotmat) - 1) / 2, -1, 1)))
                if pos_error > self.position_tolerance_m or rot_error > self.rotation_tolerance_rad:
                    self.last_check = ReachabilityResult(False, "pose_error", pos_error, rot_error)
                    continue
                result = self.check_joint_motion(joints, check_path=check_path)
                result.position_error_m, result.rotation_error_rad = pos_error, rot_error
                if result.feasible:
                    self._candidate_joints = joints.copy()
                    return True
            return False
        finally:
            self.robot.goto_given_conf(old)

    def move(self, check_path: bool = True) -> None:
        if self._candidate_joints is None:
            raise RuntimeError("call feasible() successfully before move()")
        target = self._candidate_joints.copy()
        # Obstacles may have moved since the candidate was selected.
        if not self.check_joint_motion(target, check_path=check_path).feasible:
            raise RuntimeError(f"view motion rejected: {self.last_check.reason}")
        self.current_joints = target
        self.robot.goto_given_conf(self.current_joints)
        self._has_valid_view_solution = True
        self._candidate_joints = None

    def camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the displayed camera pose from robot FK and calibration."""
        return self.camera_pose_for(self.current_joints)

    def camera_pose_for(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tcp_pos, tcp_rotmat = self.robot.fk(joints, update=False)
        return tcp_to_camera(tcp_pos, tcp_rotmat, self.tcp_from_camera)
