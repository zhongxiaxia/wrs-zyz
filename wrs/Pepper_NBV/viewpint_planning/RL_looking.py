import time
from typing import Optional, Tuple

import cv2
import numpy as np

import wrs.basis.robot_math as rm
from wrs.Pepper_NBV.yoloutils import yolo_seg


def safe_imshow(window_name: str, img: np.ndarray, wait_ms: int = 1) -> bool:
    try:
        cv2.imshow(window_name, img)
        cv2.waitKey(int(wait_ms))
        return True
    except Exception:
        return False


def wait_until_reached(rm_controller,
                       target_joints_rad,
                       tol_rad=np.deg2rad(0.8),
                       timeout_s=3.0,
                       poll_dt=0.05,
                       settle_count=2):
    """Poll joint states until they stay within tol_rad for settle_count samples or timeout."""
    target = np.asarray(target_joints_rad, dtype=float).reshape(-1)
    ok_streak = 0
    t0 = time.time()
    while True:
        cur = np.asarray(rm_controller.get_joint_values(), dtype=float).reshape(-1)
        err = np.max(np.abs(cur - target))
        if err <= tol_rad:
            ok_streak += 1
            if ok_streak >= settle_count:
                return True
        else:
            ok_streak = 0
        if time.time() - t0 >= timeout_s:
            return False
        time.sleep(poll_dt)


class RLLooking:
    """Collect multi-view confidences (YOLO seg) and pick the best observation pose."""

    def __init__(self, picker, target_distance_thresh=0.08, yaw_offset_deg=20.0):
        self.picker = picker
        # skip detections that are too far from the target point (in meters)
        self.target_distance_thresh = target_distance_thresh
        # left/right viewpoint yaw offset (degrees) around target point in world frame
        self.yaw_offset_deg = yaw_offset_deg
        # 0=center/approach, 1=left, 2=right (best view selected last time)
        self.last_best_view_idx = 0

    def _get_pcd_and_color(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if hasattr(self.picker, "_get_pcd_and_color"):
            return self.picker._get_pcd_and_color()
        cam = getattr(self.picker, "camera", None)
        if cam is None:
            return None, None
        pcd_raw, _pcd_color, _depth_img, color_img = cam.get_pcd_texture_depth()
        if pcd_raw is None or color_img is None:
            return None, None
        h, w = color_img.shape[:2]
        return pcd_raw.reshape(h, w, 3), color_img

    def _resolve_stem_class_id(self) -> Optional[int]:
        if hasattr(self.picker, "_resolve_stem_class_id"):
            return self.picker._resolve_stem_class_id()
        if hasattr(self.picker, "stem_class_id"):
            return int(self.picker.stem_class_id)
        return None

    def _best_conf_from_seg(self,
                            tgt_pos: np.ndarray,
                            conf_thres: float = 0.3,
                            mask_thres: float = 0.5,
                            show_window_name: Optional[str] = None) -> float:
        pcd_matrix, color_img = self._get_pcd_and_color()
        if pcd_matrix is None or color_img is None:
            return 0.0

        items, im_array = yolo_seg.yolo_seg_masks(
            self.picker.yolo_model,
            color_img,
            conf_thres=conf_thres,
            mask_thres=mask_thres,
            show=True,
        )
        if show_window_name and im_array is not None:
            safe_imshow(show_window_name, im_array, wait_ms=1)

        if not items:
            return 0.0

        stem_id = self._resolve_stem_class_id()
        if stem_id is None:
            return 0.0

        stem_items = [it for it in items if it.get("cls_id") == stem_id]
        if not stem_items:
            return 0.0

        #4 按置信度比较
        # best_conf = 0.0
        # min_distance = float("inf")
        # for it in stem_items:
        #     conf = float(it.get("conf", 0.0))
        #     if conf < conf_thres:
        #         continue
        #     mask = it.get("mask")
        #     center = yolo_seg.mask_center_pixel(mask)
        #     if center is None:
        #         continue
        #     u, v = center
        #     camera_xyz = yolo_seg._estimate_point_from_neighborhood(
        #         target_pixel=(u, v),
        #         pcd_matrix=pcd_matrix,
        #         neighborhood_size=5,
        #     )
        #     if camera_xyz is None or np.any(np.isnan(camera_xyz)):
        #         continue

        #     given_conf = None
        #     if getattr(self.picker, "rm_controller", None) is not None:
        #         given_conf = self.picker.rm_controller.get_joint_values()
        #     world_point = self.picker.transform_point_cloud_handeye(
        #         handeye_mat=self.picker.HANDEYE,
        #         pcd=np.asarray(camera_xyz).reshape(1, 3),
        #         given_conf=given_conf,
        #     )[0]

        #     distance = np.linalg.norm(world_point - tgt_pos)
        #     if distance < min_distance and distance <= self.target_distance_thresh:
        #         min_distance = distance
        #         best_conf = conf

        # return best_conf
        # 按面积比较
        best_area = 0
        min_distance = float("inf")
        for it in stem_items:
            mask = it.get("mask")
            if mask is None:
                continue
            area = int(np.sum(mask))
            center = yolo_seg.mask_center_pixel(mask)
            if center is None:
                continue
            u, v = center
            camera_xyz = yolo_seg._estimate_point_from_neighborhood(
                target_pixel=(u, v),
                pcd_matrix=pcd_matrix,
                neighborhood_size=5,
            )
            if camera_xyz is None or np.any(np.isnan(camera_xyz)):
                continue

            given_conf = None
            if getattr(self.picker, "rm_controller", None) is not None:
                given_conf = self.picker.rm_controller.get_joint_values()
            world_point = self.picker.transform_point_cloud_handeye(
                handeye_mat=self.picker.HANDEYE,
                pcd=np.asarray(camera_xyz).reshape(1, 3),
                given_conf=given_conf,
            )[0]

            distance = np.linalg.norm(world_point - tgt_pos)
            if distance < min_distance and distance <= self.target_distance_thresh:
                min_distance = distance
                best_area = area

        return float(best_area)

    def capture_view_confidences(self, robot, tgt_pos, chosen_approach_conf, chosen_grasp_conf, final_linear_path):
        """Take images at the current approach pose and +/- 20-degree viewpoints, then move to the highest-confidence view.

        Returns (linear_path_to_use, grasp_conf_used) for insertion (may be replanned from the selected view).
        """
        def _jnts_within_limits(jnts_rad: np.ndarray, margin: float = 1e-3) -> bool:
            jnts_rad = np.asarray(jnts_rad, dtype=float).reshape(-1)
            if jnts_rad.size != 6:
                return False
            try:
                motion_ranges = np.asarray(
                    [robot.manipulator.jlc.jnts[i].motion_range for i in range(6)],
                    dtype=float
                )
            except Exception:
                return True
            lower = motion_ranges[:, 0] + margin
            upper = motion_ranges[:, 1] - margin
            return bool(np.all(jnts_rad >= lower) and np.all(jnts_rad <= upper))

        conf_select = np.zeros(3)
        joint_select = np.zeros((6, 3))
        tcp_select = np.zeros((3, 3))

        rm_controller = self.picker.rm_controller

        time.sleep(3.0)
        # Center/approach view
        if getattr(self.picker, "camera", None) is not None:
            try:
                time.sleep(0.3)
                t0_center = time.time()
                best_conf = self._best_conf_from_seg(
                    tgt_pos=tgt_pos,
                    conf_thres=0.3,
                    mask_thres=0.5,
                    show_window_name="YOLO Seg (approach view)",
                )
                t_center_total = time.time() - t0_center
                print(f"[Timing][View][Center] total={t_center_total:.3f}s")

                if best_conf > 0.0:
                    print(f"[Info] Target confidence at approach point: {best_conf:.3f}")
                    conf_select[0] = best_conf

                    if rm_controller:
                        current_joints = rm_controller.get_joint_values()
                        joint_select[:, 0] = np.degrees(current_joints)
                        current_tcp_pos, _ = robot.fk(current_joints, update=False)
                        tcp_select[:, 0] = current_tcp_pos
                else:
                    print("[Info] Original target not found at approach point")
            except Exception as e:
                print(f"[Error] Approach view failed: {e}")
        else:
            print("[Warning] Camera not available at approach view")

        time.sleep(1.0)
        print(f"--- Rotating {self.yaw_offset_deg:.1f}deg left around target Z-axis ---")
        try:
            current_joints = rm_controller.get_joint_values()
            current_tcp_pos, current_tcp_rotmat = robot.fk(current_joints, update=False)

            rot_world_z = rm.rotmat_from_axangle(np.array([0, 0, 1]), np.radians(self.yaw_offset_deg))
            target_to_tcp_world = current_tcp_pos - tgt_pos
            rotated_offset = rot_world_z @ target_to_tcp_world

            new_tcp_pos = tgt_pos + rotated_offset
            new_tcp_pos[2] = tgt_pos[2] + target_to_tcp_world[2]
            new_tcp_rotmat = rot_world_z @ current_tcp_rotmat

            _t0 = time.time()
            new_joint_conf = robot.ik(tgt_pos=new_tcp_pos, tgt_rotmat=new_tcp_rotmat)
            t_ik_left = time.time() - _t0
            print(f"[Timing][View][Left] IK={t_ik_left:.3f}s")
            if new_joint_conf is None:
                print("[Warning] Cannot reach rotated position")
            else:
                if not _jnts_within_limits(new_joint_conf):
                    print(f"[Warning] Rotated-left IK out of joint limits (rad): {new_joint_conf}")
                result = rm_controller.move_j(new_joint_conf, speed=75, is_block=True) if _jnts_within_limits(new_joint_conf) else 1
                if result is None or result == 0:
                    print("[Info] Successfully moved to rotated position")
                    wait_until_reached(rm_controller, new_joint_conf, timeout_s=3.0)

                    if getattr(self.picker, "camera", None) is not None:
                        time.sleep(0.3)
                        t0_left = time.time()
                        best_conf_rotated = self._best_conf_from_seg(
                            tgt_pos=tgt_pos,
                            conf_thres=0.2,
                            mask_thres=0.5,
                            show_window_name="YOLO Seg (left view)",
                        )
                        t_left_total = time.time() - t0_left
                        print(f"[Timing][View][Left] total={t_left_total:.3f}s")

                        if best_conf_rotated > 0.0:
                            print(f"[Info] Target confidence at rotated position: {best_conf_rotated:.3f}")
                            conf_select[1] = best_conf_rotated
                            current_joints_rotated = rm_controller.get_joint_values()
                            joint_select[:, 1] = np.degrees(current_joints_rotated)
                            current_tcp_pos_rotated, _ = robot.fk(current_joints_rotated, update=False)
                            tcp_select[:, 1] = current_tcp_pos_rotated
                        else:
                            print("[Info] Original target not found at rotated position")
                    else:
                        print("[Warning] Camera not available at rotated position")
                else:
                    print(f"[Warning] Failed to move to rotated position, error code: {result}")

            time.sleep(1.0)

            print(f"--- Rotating {self.yaw_offset_deg:.1f}deg right around target Z-axis ---")
            rot_world_z_right = rm.rotmat_from_axangle(np.array([0, 0, 1]), np.radians(-self.yaw_offset_deg))
            rotated_offset_right = rot_world_z_right @ target_to_tcp_world
            new_tcp_pos_right = tgt_pos + rotated_offset_right
            new_tcp_pos_right[2] = tgt_pos[2] + target_to_tcp_world[2]
            new_tcp_rotmat_right = rot_world_z_right @ current_tcp_rotmat

            _t0 = time.time()
            new_joint_conf_right = robot.ik(tgt_pos=new_tcp_pos_right, tgt_rotmat=new_tcp_rotmat_right)
            t_ik_right = time.time() - _t0
            print(f"[Timing][View][Right] IK={t_ik_right:.3f}s")
            if new_joint_conf_right is None:
                print("[Warning] Cannot reach rotated-right position")
            else:
                if not _jnts_within_limits(new_joint_conf_right):
                    print(f"[Warning] Rotated-right IK out of joint limits (rad): {new_joint_conf_right}")
                result_right = rm_controller.move_j(new_joint_conf_right, speed=80, is_block=True) if _jnts_within_limits(new_joint_conf_right) else 1
                if result_right is None or result_right == 0:
                    print("[Info] Successfully moved to rotated-right position")
                    wait_until_reached(rm_controller, new_joint_conf_right, timeout_s=3.0)

                    if getattr(self.picker, "camera", None) is not None:
                        t0_right = time.time()
                        best_conf_rotated_r = self._best_conf_from_seg(
                            tgt_pos=tgt_pos,
                            conf_thres=0.3,
                            mask_thres=0.5,
                            show_window_name="YOLO Seg (right view)",
                        )
                        t_right_total = time.time() - t0_right
                        print(f"[Timing][View][Right] total={t_right_total:.3f}s")

                        if best_conf_rotated_r > 0.0:
                            print(f"[Info] Target confidence at rotated-right position: {best_conf_rotated_r:.3f}")
                            conf_select[2] = best_conf_rotated_r
                            current_joints_rotated_r = rm_controller.get_joint_values()
                            joint_select[:, 2] = np.degrees(current_joints_rotated_r)
                            current_tcp_pos_rotated_r, _ = robot.fk(current_joints_rotated_r, update=False)
                            tcp_select[:, 2] = current_tcp_pos_rotated_r
                        else:
                            print("[Info] Original target not found at rotated-right position")
                    else:
                        print("[Warning] Camera not available at rotated-right position")
                else:
                    print(f"[Warning] Failed to move to rotated-right position, error code: {result_right}")
        except Exception as e:
            print(f"[Error] Failed to rotate and capture: {e}")

        time.sleep(1.0)

        print("\n=== Confidence and Position Records ===")
        print(f"Confidence array: {conf_select}")
        print(f"Joint positions array (6x3):\n{joint_select}")
        print(f"TCP positions array (3x3):\n{tcp_select}")

        best_idx = int(np.argmax(conf_select))
        start_conf_linear = chosen_approach_conf
        if conf_select[best_idx] <= 0:
            print("[Info] No valid detection in any of the three views; staying at current pose.")
        else:
            view_names = {0: "center/approach", 1: "left (+20deg)", 2: "right (-20deg)"}
            print(f"[Info] Selecting best view: {view_names.get(best_idx, best_idx)} "
                  f"with confidence {conf_select[best_idx]:.3f}")

            target_joints_rad = np.radians(joint_select[:, best_idx])
            try:
                if not _jnts_within_limits(target_joints_rad):
                    print(f"[Warning] Selected-view joints out of limits (rad): {target_joints_rad}")
                move_ret = rm_controller.move_j(target_joints_rad, speed=50, is_block=True) if _jnts_within_limits(target_joints_rad) else 1
                if move_ret is None or move_ret == 0:
                    print("[Info] Moved to best-confidence view for picking.")
                    start_conf_linear = rm_controller.get_joint_values()
                else:
                    print(f"[Warning] Failed to move to best view, error code: {move_ret}")
            except Exception as e:
                print(f"[Error] Move to best view failed: {e}")

        linear_path_to_use = final_linear_path
        try:
            if rm_controller is not None:
                start_conf_linear = rm_controller.get_joint_values()
        except Exception:
            pass

        disallow_side_fallback = best_idx in (1, 2)
        if disallow_side_fallback:
            linear_path_to_use = None

        goal_conf_for_insert = np.asarray(chosen_grasp_conf, dtype=float) if chosen_grasp_conf is not None else None

        if best_idx in (1, 2) and goal_conf_for_insert is not None:
            try:
                yaw_sign = 1.0 if best_idx == 1 else -1.0
                rot_world_z = rm.rotmat_from_axangle(np.array([0, 0, 1]), np.radians(self.yaw_offset_deg * yaw_sign))
                goal_tcp_pos, goal_tcp_rotmat = robot.fk(goal_conf_for_insert, update=False)
                new_goal_rotmat = rot_world_z @ goal_tcp_rotmat
                goal_conf_yawed = robot.ik(
                    tgt_pos=goal_tcp_pos + np.array([-0.06, (2 * best_idx - 3) * 0.07, 0.04]),
                    tgt_rotmat=new_goal_rotmat,
                    seed_jnt_values=goal_conf_for_insert
                )
                if goal_conf_yawed is not None:
                    goal_conf_for_insert = goal_conf_yawed
                    print(f"[Info] Updated grasp orientation with yaw {self.yaw_offset_deg * yaw_sign:+.1f}deg for side view.")
                else:
                    print("[Warning] Failed to update grasp orientation for side view; fallback to original grasp pose.")
            except Exception as e:
                print(f"[Warning] Exception while updating grasp orientation for side view: {e}")

        if chosen_grasp_conf is not None and start_conf_linear is not None:
            try:
                replanned = self.picker.adp_planner.gen_linear_motion(
                    start_conf=np.asarray(start_conf_linear, dtype=float),
                    goal_conf=goal_conf_for_insert if goal_conf_for_insert is not None else np.asarray(chosen_grasp_conf, dtype=float),
                    granularity=0.01,
                    obstacle_list=[self.picker.shelf_model]
                )
                if replanned is not None:
                    linear_path_to_use = replanned
                    print("[Info] Replanned linear path from best view to grasp pose.")
                else:
                    if disallow_side_fallback:
                        print("[Error] Failed to replan linear path from selected side view; abort insertion to avoid center-view fallback.")
                        linear_path_to_use = None
                    else:
                        print("[Warning] Failed to replan linear path from best view; fallback to original path.")
            except Exception as e:
                print(f"[Error] Replan linear path failed: {e}")
                if disallow_side_fallback:
                    linear_path_to_use = None

        try:
            self.last_best_view_idx = int(best_idx)
        except Exception:
            self.last_best_view_idx = 0

        return linear_path_to_use, goal_conf_for_insert
