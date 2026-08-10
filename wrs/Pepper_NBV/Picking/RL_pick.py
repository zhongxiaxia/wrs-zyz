#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/02/03
"""
Segmentation-based pepper picking with left/center/right view selection.
This is based on Pepper_NBV/Picking/pepper_pick.py and uses the NBV RL_looking.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import wrs.visualization.panda.world as wd
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.motion.probabilistic.rrt_connect as rrtc
import wrs.motion.primitives.approach_depart_planner as adp
from wrs.robot_sim.robots.realman.realman_single_arm import Realman
from wrs.robot_con.realman.realman import RealmanArmController
from wrs.Pepper.Grapper.Grapper import ClawController

from wrs.Pepper_NBV.yoloutils import yolo_seg
from wrs.Pepper_NBV.viewpint_planning.RL_looking import RLLooking


base = wd.World(cam_pos=rm.vec(2, 0, 1), lookat_pos=rm.vec(0, 0, 0.5))
mgm.gen_frame().attach_to(base)

DEFAULT_GRIPPER_COM = "COM6"
claw_id = 9
claw_controller = ClawController(claw_id, com_port=DEFAULT_GRIPPER_COM)
claw_controller.connect_gripper()
claw_controller.enable_gripper()


def _load_handeye_from_json(path: str | Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "affine_mat" not in data:
        raise ValueError(f"missing 'affine_mat' in {path}")
    mat = np.array(data["affine_mat"], dtype=float)
    if mat.shape != (4, 4):
        raise ValueError(f"invalid handeye matrix shape {mat.shape} in {path}")
    return mat


class RealmanPepperPickerSegRL:
    def __init__(self,
                 real_robot: bool = True,
                 weight_path: str | Path = r"F:\WRS\wrs-virtual\wrs\Pepper_NBV\weights\best.pt",
                 handeye_path: str | Path = r"F:\WRS\wrs-virtual\wrs\Pepper_NBV\calibration\manual_calibration_realman.json",
                 stem_class_name: Optional[str] = "stem",
                 stem_class_id: int = 1,
                 yaw_offset_deg: float = 20.0,
                 target_distance_thresh: float = 0.08):
        self.real_robot = real_robot
        self.rm_controller: Optional[RealmanArmController] = None

        # Bind hand-eye transform from yolo_seg
        self.transform_point_cloud_handeye = yolo_seg.transform_point_cloud_handeye.__get__(self)

        self.shelf_model = mcm.gen_box(
            xyz_lengths=np.array([0.02, 0.7, 1.55]),
            pos=rm.vec(-0.8, 0, 0),
            rotmat=np.eye(3),
            rgb=[0.1, 0.1, 0.2],
            alpha=0.8,
        )
        self.shelf_model.attach_to(base)

        self.box_model = mcm.gen_box(
            xyz_lengths=np.array([0.5, 0.5, 0.66]),
            pos=rm.vec(0, 0, -0.33),
            rotmat=np.eye(3),
            rgb=[0.1, 0.1, 0.2],
            alpha=0.8,
        )
        self.box_model.attach_to(base)

        self.robot = Realman(enable_cc=True)
        self.rrtc_planner = rrtc.RRTConnect(self.robot)
        self.adp_planner = adp.ADPlanner(self.robot)

        self.HANDEYE = _load_handeye_from_json(handeye_path)

        self.yolo_model = yolo_seg.init_yolo_seg(str(weight_path))
        self.stem_class_name = stem_class_name
        self.stem_class_id = stem_class_id

        self.camera = None
        if self.real_robot:
            try:
                self.rm_controller = RealmanArmController(has_gripper=True)
                print("Realman Arm Connected.")
            except Exception as e:
                print(f"[Error] Failed to connect Realman: {e}")
                print("[Info] Fallback to simulation.")
                self.real_robot = False

        if self.real_robot:
            try:
                self.camera = yolo_seg.init_camera()
            except Exception as e:
                print(f"[Error] Failed to init RealSense: {e}")
                print("[Info] Fallback to simulation.")
                self.real_robot = False
                self.camera = None

        self.rl_looking = RLLooking(self,
                                    target_distance_thresh=target_distance_thresh,
                                    yaw_offset_deg=yaw_offset_deg)

    def _resolve_stem_class_id(self) -> Optional[int]:
        if self.stem_class_name:
            names = getattr(self.yolo_model, "names", None)
            if isinstance(names, dict):
                for k, v in names.items():
                    if v == self.stem_class_name:
                        return int(k)
            elif isinstance(names, list):
                if self.stem_class_name in names:
                    return int(names.index(self.stem_class_name))
        return self.stem_class_id

    def _get_pcd_and_color(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.camera is None:
            return None, None
        pcd_raw, _pcd_color, _depth_img, color_img = self.camera.get_pcd_texture_depth()
        if pcd_raw is None or color_img is None:
            return None, None
        h, w = color_img.shape[:2]
        pcd_matrix = pcd_raw.reshape(h, w, 3)
        return pcd_matrix, color_img

    def scan_and_segment_one(self, conf_thresh: float = 0.3, mask_thres: float = 0.5):
        """
        Segment stems and return [(world_xyz, conf)], plus visualization image.
        """
        if not self.real_robot:
            return [(np.array([-0.45, -0.1, 0.3]), 1.0)], None

        pcd_matrix, color_img = self._get_pcd_and_color()
        if pcd_matrix is None or color_img is None:
            print("[Error] Camera grab failed.")
            return [], None

        items, im_array = yolo_seg.yolo_seg_masks(
            self.yolo_model,
            color_img,
            conf_thres=conf_thresh,
            mask_thres=mask_thres,
            show=True,
        )
        if not items:
            print("[Info] No segmentation result.")
            return [], im_array

        stem_id = self._resolve_stem_class_id()
        if stem_id is None:
            print("[Error] Stem class id not resolved.")
            return [], im_array

        stem_items = [it for it in items if it["cls_id"] == stem_id]
        if not stem_items:
            print("[Info] No stem detected.")
            return [], im_array

        #1 按置信度排序
        # stem_items.sort(key=lambda x: x["conf"], reverse=True)
        # best = stem_items[0]
        # mask = best["mask"]
        # conf = float(best["conf"])

        # 按面积排序
        stem_items.sort(key=lambda x: np.sum(x["mask"]), reverse=True)
        best = stem_items[0]
        mask = best["mask"]
        area = int(np.sum(mask))

        center = yolo_seg.mask_center_pixel(mask)
        if center is None:
            print("[Warn] Stem mask centroid not found.")
            return [], im_array

        u, v = center
        camera_xyz = yolo_seg._estimate_point_from_neighborhood(
            target_pixel=(u, v),
            pcd_matrix=pcd_matrix,
            neighborhood_size=5,
        )
        if camera_xyz is None or np.any(np.isnan(camera_xyz)):
            print("[Warn] Failed to estimate 3D point from centroid.")
            return [], im_array

        given_conf = self.rm_controller.get_joint_values() if self.rm_controller else None
        world_point = self.transform_point_cloud_handeye(
            handeye_mat=self.HANDEYE,
            pcd=np.asarray(camera_xyz).reshape(1, 3),
            given_conf=given_conf,
        )[0]

        if im_array is not None:
            try:
                cv2.circle(im_array, (int(u), int(v)), 5, (0, 255, 0), -1)
                #2
                # text = f"W:{world_point[0]:.3f},{world_point[1]:.3f},{world_point[2]:.3f} conf:{conf:.2f}"
                text = f"W:{world_point[0]:.3f},{world_point[1]:.3f},{world_point[2]:.3f} area:{area}"
                cv2.putText(im_array, text, (int(u) + 10, int(v) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            except Exception:
                pass

        if im_array is not None:
            try:
                cv2.imshow("YOLO Segmentation", im_array)
                cv2.waitKey(1)
            except Exception:
                pass

        #3
        # return [(world_point, conf)], im_array
        return [(world_point, area)], im_array

    def pick_place_pepper(self,
                          approach_dis=0.2,
                          bound_lower=-40,
                          bound_upper=40,
                          grad=50,
                          viewpoint_joint=np.array([-0.16688839, -1.59552764, 1.95066723, 0.09162979, 1.25496153, -0.07829547]),
                          place_joint=np.array([1.57254163e+00, 6.96909976e-02, 1.92076978e+00, 1.71216801e-02, 7.77579056e-01, 2.96705989e-04]),
                          continuous_mode=True,
                          max_targets_per_round=1,
                          stop_on_no_detection=True):
        _task_start_time = time.time()
        robot = self.robot
        rrtc_planner = self.rrtc_planner

        if self.real_robot and self.rm_controller is not None:
            start_conf = self.rm_controller.get_joint_values()
        else:
            start_conf = robot.get_jnt_values()

        robot.goto_given_conf(start_conf)

        print("--- Phase 1: Moving to Viewpoint ---")
        if np.linalg.norm(np.array(start_conf) - np.array(viewpoint_joint)) > 0.03:
            mot_data_view = rrtc_planner.plan(start_conf=start_conf,
                                              goal_conf=viewpoint_joint,
                                              ext_dist=.1,
                                              max_time=300)
            if mot_data_view is not None:
                robot.goto_given_conf(viewpoint_joint)
                if self.real_robot:
                    self.rm_controller.move_jntspace_path(mot_data_view.jv_list, control_frequency=0.8)
                    time.sleep(3.0)
            else:
                print("[Error] Failed to plan to viewpoint.")
                return
        else:
            print("Already at viewpoint.")

        pick_count = 0
        total_success = 0

        while continuous_mode:
            print(f"\n--- Phase 2: Segmenting Target (Round {pick_count + 1}) ---")
            detections, _ = self.scan_and_segment_one(conf_thresh=0.3, mask_thres=0.5)

            if len(detections) == 0:
                if stop_on_no_detection:
                    print("[Info] No stem detected. Ending continuous picking.")
                    break
                print("[Info] No stem detected. Continuing to next round...")
                time.sleep(1.0)
                continue

            for tgt_pos, conf in detections[:max_targets_per_round]:
                print(f"\n--- Picking target with confidence {conf:.3f} at {tgt_pos} ---")
                print(f"--- Phase 3: Searching Grasp Pose ---")
                start_conf = robot.get_jnt_values()

                is_success = False
                final_approach_path = None
                final_linear_path = None
                chosen_approach_conf = None
                chosen_grasp_conf = None

                ik_total_time = 0.0
                for theta in range(bound_lower, bound_upper + 1, grad):
                    hand_x = np.array([0, 0, -1])
                    hand_z = np.array([-1, 0, 0])
                    hand_y = np.cross(hand_z, hand_x)
                    tgt_rotmat_base = np.array([hand_x, hand_y, hand_z]).T
                    tgt_rotmat = rm.rotmat_from_axangle(hand_y, np.radians(theta)) @ tgt_rotmat_base

                    ik_start_time = time.time()
                    grasp_conf = robot.ik(tgt_pos=tgt_pos,
                                          tgt_rotmat=tgt_rotmat,
                                          seed_jnt_values=start_conf)
                    ik_time = time.time() - ik_start_time
                    ik_total_time += ik_time
                    if grasp_conf is not None:
                        robot.goto_given_conf(grasp_conf)
                        if not robot.is_collided():
                            tgt_pos_approach = tgt_pos - hand_z * approach_dis
                            approach_conf = robot.ik(tgt_pos_approach, tgt_rotmat, seed_jnt_values=grasp_conf)

                            if approach_conf is not None:
                                robot.goto_given_conf(approach_conf)
                                if not robot.is_collided():
                                    mot_data_rrt = rrtc_planner.plan(start_conf=start_conf,
                                                                     goal_conf=approach_conf,
                                                                     obstacle_list=[self.shelf_model, self.box_model],
                                                                     ext_dist=.1,
                                                                     max_time=300)

                                    if mot_data_rrt is not None:
                                        mot_data_linear = self.adp_planner.gen_linear_motion(
                                            start_conf=approach_conf,
                                            goal_conf=grasp_conf,
                                            granularity=0.01,
                                            obstacle_list=[self.shelf_model, self.box_model]
                                        )

                                        if mot_data_linear is not None:
                                            final_approach_path = mot_data_rrt
                                            final_linear_path = mot_data_linear
                                            chosen_approach_conf = approach_conf.copy()
                                            chosen_grasp_conf = grasp_conf.copy()
                                            self.vis_path(mot_data_rrt, rm.const.winter_map)
                                            self.vis_path(mot_data_linear, rm.const.wistia_map)
                                            is_success = True
                                            break

                print(f"[Timing] IK solving total time: {ik_total_time:.3f}s")

                mot_data_place = None
                if is_success:
                    print("--- Phase 4: Planning Placement Route ---")
                    if hasattr(final_approach_path, "jv_list"):
                        current_conf_for_place_planning = final_approach_path.jv_list[-1]
                    else:
                        current_conf_for_place_planning = final_approach_path[-1]
                    robot.goto_given_conf(current_conf_for_place_planning)

                    mot_data_place = rrtc_planner.plan(start_conf=current_conf_for_place_planning,
                                                       goal_conf=place_joint,
                                                       obstacle_list=[self.shelf_model, self.box_model],
                                                       ext_dist=.1,
                                                       max_time=300)

                    if mot_data_place is not None:
                        self.vis_path(mot_data_place, rm.const.copper_map)
                    else:
                        print("[Error] Failed to plan to place position.")
                        is_success = False

                if is_success and mot_data_place is not None and self.real_robot:
                    print("--- Executing Grasp and Place ---")
                    grasp_start_time = time.time()

                    claw_controller.open_gripper()
                    time.sleep(0.5)

                    self.rm_controller.move_jntspace_path(final_approach_path, control_frequency=0.8)
                    time.sleep(2)

                    linear_path_to_use = final_linear_path
                    grasp_conf_for_insert = chosen_grasp_conf
                    if self.rl_looking is not None:
                        path_candidate, grasp_conf_after_view = self.rl_looking.capture_view_confidences(
                            robot=robot,
                            tgt_pos=tgt_pos,
                            chosen_approach_conf=chosen_approach_conf,
                            chosen_grasp_conf=chosen_grasp_conf,
                            final_linear_path=final_linear_path
                        )
                        selected_view_idx = getattr(self.rl_looking, "last_best_view_idx", 0)
                        if grasp_conf_after_view is not None:
                            grasp_conf_for_insert = grasp_conf_after_view
                        if selected_view_idx in (1, 2) and path_candidate is None:
                            print("[Error] Selected side view but failed to replan insertion; skip this target.")
                            pick_count += 1
                            continue
                        if path_candidate is not None:
                            linear_path_to_use = path_candidate

                    self.rm_controller.move_jntspace_path(linear_path_to_use.jv_list, control_frequency=0.8)
                    time.sleep(2)

                    claw_controller.close_gripper()
                    time.sleep(1.0)

                    self.rm_controller.move_jntspace_path(linear_path_to_use.jv_list[::-1], control_frequency=0.8)
                    time.sleep(2)

                    self.rm_controller.move_jntspace_path(mot_data_place.jv_list, control_frequency=0.8)
                    time.sleep(2)

                    claw_controller.open_gripper()
                    time.sleep(0.5)

                    total_success += 1
                    grasp_time = time.time() - grasp_start_time
                    print(f"[Success] Picked and placed pepper {total_success}")
                    print(f"[Timing] Grasp execution time: {grasp_time:.3f}s")

                    print("--- Returning to Viewpoint ---")
                    mot_data_return = rrtc_planner.plan(start_conf=place_joint,
                                                        goal_conf=viewpoint_joint,
                                                        obstacle_list=[self.shelf_model, self.box_model],
                                                        ext_dist=.1,
                                                        max_time=300)
                    if mot_data_return is not None:
                        robot.goto_given_conf(viewpoint_joint)
                        if self.real_robot:
                            self.rm_controller.move_jntspace_path(mot_data_return.jv_list, control_frequency=0.8)
                            time.sleep(3.0)
                elif is_success and not self.real_robot:
                    print("Simulation Success.")
                    robot.goto_given_conf(place_joint)
                    robot.goto_given_conf(viewpoint_joint)
                    time.sleep(2)
                else:
                    print(f"[Failed] Failed to pick target at {tgt_pos}")

                pick_count += 1

        _task_end_time = time.time()
        _duration = _task_end_time - _task_start_time
        _result_str = "success" if total_success > 0 else "fail"
        _ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_task_start_time))
        log_line = f"{_ts_str},{_duration:.3f},{_result_str},picked={total_success}\n"
        try:
            with open("pepper_pick_log.txt", "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception as e:
            print(f"[Warn] Failed to write pick log: {e}")
        print(f"[Log] Pick task {_result_str}, duration: {_duration:.3f}s, picked: {total_success}")

    def vis_path(self, mot_data, colormap):
        if mot_data:
            n = len(mot_data.mesh_list)
            for i, model in enumerate(mot_data.mesh_list):
                model.rgb = colormap(i / n)
                model.alpha = 0.3
                model.attach_to(base)


if __name__ == '__main__':
    picker = RealmanPepperPickerSegRL(real_robot=True)
    robot = Realman()
    robot.gen_meshmodel(toggle_jnt_frames=True).attach_to(base)

    view_point = np.array([-0.03005457, -1.20970511, 1.26979681, 0.01317724, 1.52136859, -0.00029671])
    place_point = np.array([1.57254163e+00, 6.96909976e-02, 1.92076978e+00, 1.71216801e-02, 7.77579056e-01, 2.96705989e-04])

    picker.pick_place_pepper(
        approach_dis=0.25,
        bound_lower=0,
        bound_upper=20,
        grad=3,
        viewpoint_joint=view_point,
        place_joint=place_point,
        continuous_mode=True,
        max_targets_per_round=1,
        stop_on_no_detection=True,
    )

    base.run()
