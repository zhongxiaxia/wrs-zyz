#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize the Realman arm with an adjustable electronic fence.

This script is meant to help tune the fence size and its relative pose
with respect to the robot base / workspace.

What it does:
- Creates a Panda3D world
- Loads the Realman simulated arm
- Draws an adjustable fence box
- Optionally calls the real robot fence toggle API if a controller is connected
- Lets you change the fence parameters in one place

Notes:
- The exact low-level fence geometry/limits depend on your hardware SDK.
- Here we mainly visualize a fence volume in the scene so you can tune it
  before applying the same parameters to your controller.
"""

from __future__ import annotations

import numpy as np

import wrs.visualization.panda.world as wd
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm

from wrs.robot_sim.robots.realman.realman_single_arm import Realman

try:
    from wrs.robot_con.realman.realman import RealmanArmController
except Exception:
    RealmanArmController = None


class FenceConfig:
    """Editable fence configuration."""

    def __init__(self):
        # Fence size in meters
        self.size = np.array([0.90, 0.80, 0.90], dtype=float)  # x, y, z
        # Fence center relative to the robot base frame
        self.center = np.array([0.35, 0.00, 0.35], dtype=float)
        # Fence orientation relative to base frame (degrees)
        self.euler_deg = np.array([0.0, 0.0, 0.0], dtype=float)
        # Visual appearance
        self.rgba = np.array([0.15, 0.75, 1.00, 0.18], dtype=float)
        self.frame_rgba = np.array([0.15, 0.75, 1.00, 0.9], dtype=float)


class PepperFenceVisualizer:
    def __init__(self, real_robot: bool = False):
        self.real_robot = real_robot
        self.cfg = FenceConfig()

        self.base = wd.World(cam_pos=rm.vec(1.8, -1.2, 1.1), lookat_pos=rm.vec(0.2, 0.0, 0.35))
        mgm.gen_frame().attach_to(self.base)

        self.robot = Realman(enable_cc=True)
        self.robot.goto_home_conf()

        self.controller = None
        if self.real_robot and RealmanArmController is not None:
            try:
                self.controller = RealmanArmController(has_gripper=True)
            except Exception as e:
                print(f"[Warning] Failed to connect real robot controller: {e}")
                self.controller = None

        self.fence_model = None
        self.fence_frame = None
        self.robot_model = None
        self._draw_scene()

        if self.controller is not None:
            self._apply_fence_to_controller(enable=True)

    def _apply_fence_to_controller(self, enable: bool = True):
        """Enable/disable fence mode on the real controller if available."""
        if self.controller is None:
            return
        if not hasattr(self.controller.arm, "set_fence_mode"):
            print("[Info] Current controller does not expose set_fence_mode().")
            return
        try:
            self.controller.arm.set_fence_mode(bool(enable))
            print(f"[Info] Real robot fence mode set to {enable}.")
        except Exception as e:
            print(f"[Warning] Failed to set fence mode: {e}")

    def _draw_scene(self):
        if self.fence_model is not None:
            self.fence_model.detach()
        if self.fence_frame is not None:
            self.fence_frame.detach()
        if self.robot_model is not None:
            self.robot_model.detach()

        self.robot_model = self.robot.gen_meshmodel(
            alpha=0.95,
            toggle_tcp_frame=True,
            toggle_jnt_frames=False,
            toggle_flange_frame=False,
            toggle_cdprim=False,
            toggle_cdmesh=False,
        )
        self.robot_model.attach_to(self.base)

        rotmat = rm.rotmat_from_euler(*np.radians(self.cfg.euler_deg))
        self.fence_model = mcm.gen_box(
            xyz_lengths=self.cfg.size,
            pos=self.cfg.center,
            rotmat=rotmat,
            rgb=self.cfg.rgba[:3],
            alpha=float(self.cfg.rgba[3]),
        )
        self.fence_model.attach_to(self.base)

        self.fence_frame = mgm.gen_frame(
            pos=self.cfg.center,
            rotmat=rotmat,
        )
        self.fence_frame.attach_to(self.base)

    def update_fence(
        self,
        size: np.ndarray | None = None,
        center: np.ndarray | None = None,
        euler_deg: np.ndarray | None = None,
    ):
        if size is not None:
            self.cfg.size = np.asarray(size, dtype=float)
        if center is not None:
            self.cfg.center = np.asarray(center, dtype=float)
        if euler_deg is not None:
            self.cfg.euler_deg = np.asarray(euler_deg, dtype=float)
        self._draw_scene()

    def print_current_params(self):
        print("\n========== Fence Params ==========")
        print(f"size      = {self.cfg.size.tolist()} m")
        print(f"center    = {self.cfg.center.tolist()} m")
        print(f"euler_deg  = {self.cfg.euler_deg.tolist()} deg")
        print("==================================\n")

    def show_joint_values(self):
        jv = self.robot.get_jnt_values()
        print("Current robot 6-DOF joint values (rad):", jv)
        print("Current robot 6-DOF joint values (deg):", np.degrees(jv))
        return jv

    def interactive_demo(self):
        """Simple keyboard-free demo.

        Edit the parameters here, save, and rerun the script to tune the fence.
        """
        self.print_current_params()
        self.show_joint_values()
        print("[Info] Use update_fence(...) in code to adjust the fence and rerun.")
        print("[Info] Close the Panda window to exit.")
        self.base.run()


if __name__ == "__main__":
    visualizer = PepperFenceVisualizer(real_robot=False)

    # Example parameter tweak:
    visualizer.update_fence(
        size=np.array([0.9, 1.0, 0.8]),
        center=np.array([-0.30, 0.00, 0.40]),
        euler_deg=np.array([0.0, 0.0, 0.0]),
    )

    visualizer.interactive_demo()
