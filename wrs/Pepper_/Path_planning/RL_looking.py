import numpy as np
import time
import wrs.basis.robot_math as rm
from wrs.Pepper.vision_io import capture_color_and_pcd, safe_imshow, yolo_predict, yolo_vis_image


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
    """Collect multi-view confidences and pick the best observation pose."""

    def __init__(self, picker, target_distance_thresh=0.08, yaw_offset_deg=20.0):
        self.picker = picker
        # skip detections that are too far from the target point (in meters)
        self.target_distance_thresh = target_distance_thresh
        # left/right viewpoint yaw offset (degrees) around target point in world frame
        self.yaw_offset_deg = yaw_offset_deg
        # 0=center/approach, 1=left, 2=right (best view selected last time)
        self.last_best_view_idx = 0

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

        time.sleep(3.0)  # 等待到位
        # 在预接近点获取置信度
        if self.picker.rs_pipeline is not None and self.picker.rs_align is not None:
            try:
                time.sleep(0.3)  # 等待相机稳定
                t0_center = time.time()

                cap = capture_color_and_pcd(self.picker.rs_pipeline, self.picker.rs_align)
                if cap is not None:
                    color_img = cap.color_img
                    depth_intrin = cap.depth_intrin
                    pcd_cam = cap.pcd_cam
                    t_center_wait = cap.t_wait
                    t_center_align = cap.t_align
                    t_center_pcd = cap.t_pcd

                    results, t_center_yolo = yolo_predict(self.picker.yolo_model, color_img, save=False, show_conf=False)
                    # 可视化检测结果（预接近/预观察点视角）
                    try:
                        if results:
                            vis_img = yolo_vis_image(results, color_img)
                        else:
                            vis_img = color_img
                        safe_imshow("YOLO (approach view)", vis_img, wait_ms=1)
                    except Exception as e:
                        print(f"[Warn] Failed to visualize YOLO at approach view: {e}")

                    t_center_total = time.time() - t0_center
                    print(f"[计时][观察][中] 取帧={t_center_wait:.3f}秒 对齐={t_center_align:.3f}秒 "
                          f"点云={t_center_pcd:.3f}秒 YOLO={t_center_yolo:.3f}秒 IK=0.000秒 总计={t_center_total:.3f}秒")

                    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                        res = results[0]
                        boxes = res.boxes
                        kpts_all = res.keypoints

                        best_conf = 0.0
                        min_distance = float('inf')

                        for i in range(len(boxes)):
                            conf = float(boxes.conf[i].item())
                            if conf < 0.3:  # 置信度太低跳过
                                continue

                            if kpts_all is not None and len(kpts_all.data) > i:
                                # 获取关键点
                                kpts_raw = kpts_all.data[i].cpu().numpy()
                                if kpts_raw.ndim == 3:
                                    kpts = kpts_raw[0]
                                else:
                                    kpts = kpts_raw

                                if kpts.shape[0] > 0:
                                    u, v, vis = kpts[0]
                                    if vis > 0.1:  # 关键点可见
                                        u, v = int(u), int(v)
                                        if (0 <= u < depth_intrin.width and
                                            0 <= v < depth_intrin.height):

                                            # 获取相机坐标
                                            camera_xyz = pcd_cam[v, u]
                                            if np.linalg.norm(camera_xyz) > 1e-6:
                                                # 转换为世界坐标
                                                given_conf = rm_controller.get_joint_values() if rm_controller else None
                                                world_point_cloud = self.picker.transform_point_cloud_handeye(
                                                    handeye_mat=self.picker.HANDEYE,
                                                    pcd=camera_xyz.reshape(1, 3),
                                                    given_conf=given_conf
                                                )
                                                detected_pos = world_point_cloud[0]

                                                # 计算与tgt_pos的距离
                                                distance = np.linalg.norm(detected_pos - tgt_pos)
                                                if distance < min_distance and distance <= self.target_distance_thresh:
                                                    min_distance = distance
                                                    best_conf = conf

                        if best_conf > 0.0:
                            print(f"[Info] Target confidence at approach point: {best_conf:.3f} (distance: {min_distance:.3f}m)")

                            # 记录置信度和位置信息
                            conf_select[0] = best_conf

                            # 记录当前关节位置（6个关节）
                            if rm_controller:
                                current_joints = rm_controller.get_joint_values()
                                # 存储所有6个关节的角度在第一列（转换为角度制）
                                joint_select[:, 0] = np.degrees(current_joints)
                                print(f"[Debug] Recorded joint positions (6 joints): {joint_select[:, 0]}")

                                # 记录当前末端位置
                                current_tcp_pos, _ = robot.fk(current_joints, update=False)
                                # 存储[x,y,z]坐标在第一列
                                tcp_select[:, 0] = current_tcp_pos
                                print(f"[Debug] Recorded TCP position: {tcp_select[:, 0]}")
                        else:
                            print("[Info] Original target not found at approach point")
                    else:
                        print("[Info] No targets detected at approach point")
                else:
                    t_center_total = time.time() - t0_center
                    print("[警告] 观测点(中)获取彩色/深度失败")
                    print(f"[计时][观察][中] 取帧={t_center_wait:.3f}秒 对齐={t_center_align:.3f}秒 总计={t_center_total:.3f}秒")
            except Exception as e:
                try:
                    t_center_total = time.time() - t0_center
                    print(f"[错误] 观测点(中)识别失败: {e}")
                    print(f"[计时][观察][中] 总计={t_center_total:.3f}秒")
                except Exception:
                    print(f"[错误] 观测点(中)识别失败: {e}")
        else:
            print("[警告] 相机不可用，无法在观测点(中)统计识别耗时")

        time.sleep(1.0)
        # 已移动到预接近点后，再围绕目标点Z轴±20°拍照
        print(f"--- Rotating {self.yaw_offset_deg:.1f}° left around target Z-axis ---")
        try:
            # Get current TCP pose in world frame (robot.fk already returns global/world TCP).
            current_joints = rm_controller.get_joint_values()
            current_tcp_pos, current_tcp_rotmat = robot.fk(current_joints, update=False)

            # 在世界坐标系下绕目标点的Z轴左转20°
            rot_world_z = rm.rotmat_from_axangle(np.array([0, 0, 1]), np.radians(self.yaw_offset_deg))
            target_to_tcp_world = current_tcp_pos - tgt_pos
            rotated_offset = rot_world_z @ target_to_tcp_world

            # 计算新的TCP位置：绕世界Z轴旋转，保持相对Z高度
            new_tcp_pos = tgt_pos + rotated_offset
            new_tcp_pos[2] = tgt_pos[2] + target_to_tcp_world[2]

            # 姿态：仅在世界Z轴偏转20°，保持相同夹爪不倾斜
            new_tcp_rotmat = rot_world_z @ current_tcp_rotmat

            # 计算到达新位置和朝向的关节角度
            print(f"[Debug] Attempting IK for rotated position: {new_tcp_pos}")
            print(f"[Debug] Rotation matrix:\n{new_tcp_rotmat}")

            _t0 = time.time()
            new_joint_conf = robot.ik(tgt_pos=new_tcp_pos, tgt_rotmat=new_tcp_rotmat)
            t_ik_left = time.time() - _t0
            print(f"[计时][观察][左] IK求解={t_ik_left:.3f}秒")
            if new_joint_conf is None:
                print("[Warning] Cannot reach rotated position")
            else:
                # 移动到旋转后的位姿（部分驱动返回 None 也视为成功）
                if not _jnts_within_limits(new_joint_conf):
                    print(f"[Warning] Rotated-left IK out of joint limits (rad): {new_joint_conf}")
                result = rm_controller.move_j(new_joint_conf, speed=75, is_block=True) if _jnts_within_limits(new_joint_conf) else 1
                if result is None or result == 0:
                    print("[Info] Successfully moved to rotated position")
                    wait_until_reached(rm_controller, new_joint_conf, timeout_s=3.0)

                    # 在新位置获取置信度
                    if self.picker.rs_pipeline is not None and self.picker.rs_align is not None:
                        time.sleep(0.3)  # 等待相机稳定
                        t0_left = time.time()

                        cap = capture_color_and_pcd(self.picker.rs_pipeline, self.picker.rs_align)
                        if cap is not None:
                            color_img = cap.color_img
                            depth_intrin = cap.depth_intrin
                            pcd_cam = cap.pcd_cam
                            t_left_wait = cap.t_wait
                            t_left_align = cap.t_align
                            t_left_pcd = cap.t_pcd

                            results, t_left_yolo = yolo_predict(self.picker.yolo_model, color_img, save=False, show_conf=False)
                            # 可视化检测结果（左视角）
                            try:
                                vis_img = yolo_vis_image(results, color_img)
                                safe_imshow("YOLO (left view)", vis_img, wait_ms=1)
                            except Exception as e:
                                print(f"[Warn] Failed to visualize YOLO at left view: {e}")

                            t_left_total = time.time() - t0_left
                            print(f"[计时][观察][左] 取帧={t_left_wait:.3f}秒 对齐={t_left_align:.3f}秒 "
                                  f"点云={t_left_pcd:.3f}秒 YOLO={t_left_yolo:.3f}秒 总计={t_left_total:.3f}秒")

                            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                                res = results[0]
                                boxes = res.boxes
                                kpts_all = res.keypoints
                                print(f"[Debug] Left view detections: {len(boxes)} boxes, has_keypoints={kpts_all is not None}")

                                # 寻找最接近tgt_pos的目标
                                best_conf_rotated = 0.0
                                min_distance_rotated = float('inf')

                                for i in range(len(boxes)):
                                    conf = float(boxes.conf[i].item())
                                    if conf < 0.2:  # 略放宽旋转视角阈值
                                        continue

                                    if kpts_all is not None and len(kpts_all.data) > i:
                                        kpts_raw = kpts_all.data[i].cpu().numpy()
                                        if kpts_raw.ndim == 3:
                                            kpts = kpts_raw[0]
                                        else:
                                            kpts = kpts_raw

                                        if kpts.shape[0] > 0:
                                            u, v, vis = kpts[0]
                                            if vis > 0.1:
                                                u, v = int(u), int(v)
                                                if (0 <= u < depth_intrin.width and
                                                    0 <= v < depth_intrin.height):

                                                    camera_xyz = pcd_cam[v, u]
                                                    if np.linalg.norm(camera_xyz) > 1e-6:
                                                        given_conf_rotated = rm_controller.get_joint_values()
                                                        world_point_cloud = self.picker.transform_point_cloud_handeye(
                                                            handeye_mat=self.picker.HANDEYE,
                                                            pcd=camera_xyz.reshape(1, 3),
                                                            given_conf=given_conf_rotated
                                                        )
                                                        detected_pos_rotated = world_point_cloud[0]

                                                        distance_rotated = np.linalg.norm(detected_pos_rotated - tgt_pos)
                                                        if distance_rotated < min_distance_rotated and distance_rotated <= self.target_distance_thresh:
                                                            min_distance_rotated = distance_rotated
                                                            best_conf_rotated = conf
                                    # 如果没有关键点，仍然用置信度最高的框作为参考
                                    elif best_conf_rotated == 0.0:
                                        best_conf_rotated = conf

                                if best_conf_rotated > 0.0:
                                    print(f"[Info] Target confidence at rotated position: {best_conf_rotated:.3f} (distance: {min_distance_rotated:.3f}m)")

                                    # 记录到数组第二列
                                    conf_select[1] = best_conf_rotated

                                    # 记录关节位置到第二列
                                    current_joints_rotated = rm_controller.get_joint_values()
                                    joint_select[:, 1] = np.degrees(current_joints_rotated)
                                    print(f"[Debug] Recorded rotated joint positions: {joint_select[:, 1]}")

                                    # 记录TCP位置到第二列
                                    current_tcp_pos_rotated, _ = robot.fk(current_joints_rotated, update=False)
                                    tcp_select[:, 1] = current_tcp_pos_rotated
                                    print(f"[Debug] Recorded rotated TCP position: {tcp_select[:, 1]}")

                                else:
                                    print("[Info] Original target not found at rotated position")
                            else:
                                print("[Info] No targets detected at rotated position")
                        else:
                            print("[警告] 观测点(左)获取图像失败")
                            try:
                                t_left_total = time.time() - t0_left
                                print(f"[计时][观察][左] 取帧={t_left_wait:.3f}秒 对齐={t_left_align:.3f}秒 总计={t_left_total:.3f}秒")
                            except Exception:
                                pass
                    else:
                        print("[Warning] Camera not available at rotated position")
                else:
                    print(f"[Warning] Failed to move to rotated position, error code: {result}")

            time.sleep(1.0)

            # 在世界坐标系下绕目标点的Z轴右转20°
            print(f"--- Rotating {self.yaw_offset_deg:.1f}° right around target Z-axis ---")
            rot_world_z_right = rm.rotmat_from_axangle(np.array([0, 0, 1]), np.radians(-self.yaw_offset_deg))
            rotated_offset_right = rot_world_z_right @ target_to_tcp_world

            # 计算新的TCP位置：右旋，保持相对Z高度
            new_tcp_pos_right = tgt_pos + rotated_offset_right
            new_tcp_pos_right[2] = tgt_pos[2] + target_to_tcp_world[2]

            # 姿态：仅在世界Z轴偏转20°，保持相同夹爪不倾斜
            new_tcp_rotmat_right = rot_world_z_right @ current_tcp_rotmat

            # 计算到达新位置和朝向的关节角度
            print(f"[Debug] Attempting IK for rotated-right position: {new_tcp_pos_right}")
            print(f"[Debug] Rotation matrix (right):\n{new_tcp_rotmat_right}")

            _t0 = time.time()
            new_joint_conf_right = robot.ik(tgt_pos=new_tcp_pos_right, tgt_rotmat=new_tcp_rotmat_right)
            t_ik_right = time.time() - _t0
            print(f"[计时][观察][右] IK求解={t_ik_right:.3f}秒")
            if new_joint_conf_right is None:
                print("[Warning] Cannot reach rotated-right position")
            else:
                # 移动到右旋后的位姿（部分驱动返回 None 也视为成功）
                if not _jnts_within_limits(new_joint_conf_right):
                    print(f"[Warning] Rotated-right IK out of joint limits (rad): {new_joint_conf_right}")
                result_right = rm_controller.move_j(new_joint_conf_right, speed=80, is_block=True) if _jnts_within_limits(new_joint_conf_right) else 1
                if result_right is None or result_right == 0:
                    print("[Info] Successfully moved to rotated-right position")
                    wait_until_reached(rm_controller, new_joint_conf_right, timeout_s=3.0)

                    # 在右旋位置获取置信度
                    if self.picker.rs_pipeline is not None and self.picker.rs_align is not None:
                        t0_right = time.time()

                        cap = capture_color_and_pcd(self.picker.rs_pipeline, self.picker.rs_align)
                        if cap is not None:
                            color_img = cap.color_img
                            depth_intrin = cap.depth_intrin
                            pcd_cam = cap.pcd_cam
                            t_right_wait = cap.t_wait
                            t_right_align = cap.t_align
                            t_right_pcd = cap.t_pcd

                            results, t_right_yolo = yolo_predict(self.picker.yolo_model, color_img, save=False, show_conf=False)
                            # 可视化检测结果（右视角）
                            try:
                                vis_img = yolo_vis_image(results, color_img)
                                safe_imshow("YOLO (right view)", vis_img, wait_ms=1)
                            except Exception as e:
                                print(f"[Warn] Failed to visualize YOLO at right view: {e}")

                            t_right_total = time.time() - t0_right
                            print(f"[计时][观察][右] 取帧={t_right_wait:.3f}秒 对齐={t_right_align:.3f}秒 "
                                  f"点云={t_right_pcd:.3f}秒 YOLO={t_right_yolo:.3f}秒 总计={t_right_total:.3f}秒")

                            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                                res = results[0]
                                boxes = res.boxes
                                kpts_all = res.keypoints

                                # 寻找最接近tgt_pos的目标
                                best_conf_rotated_r = 0.0
                                min_distance_rotated_r = float('inf')

                                for i in range(len(boxes)):
                                    conf = float(boxes.conf[i].item())
                                    if conf < 0.3:
                                        continue

                                    if kpts_all is not None and len(kpts_all.data) > i:
                                        kpts_raw = kpts_all.data[i].cpu().numpy()
                                        if kpts_raw.ndim == 3:
                                            kpts = kpts_raw[0]
                                        else:
                                            kpts = kpts_raw

                                        if kpts.shape[0] > 0:
                                            u, v, vis = kpts[0]
                                            if vis > 0.1:
                                                u, v = int(u), int(v)
                                                if (0 <= u < depth_intrin.width and
                                                    0 <= v < depth_intrin.height):

                                                    camera_xyz = pcd_cam[v, u]
                                                    if np.linalg.norm(camera_xyz) > 1e-6:
                                                        given_conf_rotated_r = rm_controller.get_joint_values()
                                                        world_point_cloud = self.picker.transform_point_cloud_handeye(
                                                            handeye_mat=self.picker.HANDEYE,
                                                            pcd=camera_xyz.reshape(1, 3),
                                                            given_conf=given_conf_rotated_r
                                                        )
                                                        detected_pos_rotated_r = world_point_cloud[0]

                                                        distance_rotated_r = np.linalg.norm(detected_pos_rotated_r - tgt_pos)
                                                        if distance_rotated_r < min_distance_rotated_r and distance_rotated_r <= self.target_distance_thresh:
                                                            min_distance_rotated_r = distance_rotated_r
                                                            best_conf_rotated_r = conf

                                if best_conf_rotated_r > 0.0:
                                    print(f"[Info] Target confidence at rotated-right position: {best_conf_rotated_r:.3f} (distance: {min_distance_rotated_r:.3f}m)")

                                    # 记录到数组第三列
                                    conf_select[2] = best_conf_rotated_r

                                    # 记录关节位置到第三列
                                    current_joints_rotated_r = rm_controller.get_joint_values()
                                    joint_select[:, 2] = np.degrees(current_joints_rotated_r)
                                    print(f"[Debug] Recorded rotated-right joint positions: {joint_select[:, 2]}")

                                    # 记录TCP位置到第三列
                                    current_tcp_pos_rotated_r, _ = robot.fk(current_joints_rotated_r, update=False)
                                    tcp_select[:, 2] = current_tcp_pos_rotated_r
                                    print(f"[Debug] Recorded rotated-right TCP position: {tcp_select[:, 2]}")

                                else:
                                    print("[Info] Original target not found at rotated-right position")
                            else:
                                print("[Info] No targets detected at rotated-right position")
                        else:
                            print("[警告] 观测点(右)获取图像失败")
                            try:
                                t_right_total = time.time() - t0_right
                                print(f"[计时][观察][右] 取帧={t_right_wait:.3f}秒 对齐={t_right_align:.3f}秒 总计={t_right_total:.3f}秒")
                            except Exception:
                                pass
                    else:
                        print("[Warning] Camera not available at rotated-right position")
                else:
                    print(f"[Warning] Failed to move to rotated-right position, error code: {result_right}")
        except Exception as e:
            print(f"[Error] Failed to rotate and capture: {e}")

        time.sleep(1.0)

        # 输出预设点置信度和位置记录
        print("\n=== Confidence and Position Records ===")
        print(f"Confidence array: {conf_select}")
        print(f"Joint positions array (6x3):\n{joint_select}")
        print(f"TCP positions array (3x3):\n{tcp_select}")

        # 根据置信度选择最佳视角（0=原视角/预接近点，1=左偏转20°，2=右偏转20°）
        best_idx = int(np.argmax(conf_select))
        start_conf_linear = chosen_approach_conf  # 默认用预接近姿态
        if conf_select[best_idx] <= 0:
            print("[Info] No valid detection in any of the three views; staying at current pose.")
        else:
            view_names = {0: "center/approach", 1: "left (+20°)", 2: "right (-20°)"}
            print(f"[Info] Selecting best view: {view_names.get(best_idx, best_idx)} "
                  f"with confidence {conf_select[best_idx]:.3f}")

            # joint_select 存的是角度制，执行前转换为弧度
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

        # 视角切换后，基于当前最佳视角重新规划线性插入到抓取姿态
        linear_path_to_use = final_linear_path
        # Always prefer the robot's current joints as the insertion start (handles controller return-code mismatches).
        try:
            if rm_controller is not None:
                start_conf_linear = rm_controller.get_joint_values()
        except Exception:
            pass

        # 0=center: allow fallback to the original (approach-view) linear path
        # 1/2=left/right: do NOT fallback to approach-view insertion if replanning fails
        disallow_side_fallback = best_idx in (1, 2)
        if disallow_side_fallback:
            linear_path_to_use = None

        goal_conf_for_insert = np.asarray(chosen_grasp_conf, dtype=float) if chosen_grasp_conf is not None else None

        # If side view selected, align grasp orientation yaw with the selected view before replanning
        if best_idx in (1, 2) and goal_conf_for_insert is not None:
            try:
                yaw_sign = 1.0 if best_idx == 1 else -1.0
                rot_world_z = rm.rotmat_from_axangle(np.array([0, 0, 1]), np.radians(self.yaw_offset_deg * yaw_sign))
                goal_tcp_pos, goal_tcp_rotmat = robot.fk(goal_conf_for_insert, update=False)
                new_goal_rotmat = rot_world_z @ goal_tcp_rotmat
                goal_conf_yawed = robot.ik(
                    # tgt_pos=goal_tcp_pos,
                    tgt_pos=goal_tcp_pos + np.array([-0.06, (2*best_idx-3)*0.07, 0.04]),
                    tgt_rotmat=new_goal_rotmat,
                    seed_jnt_values=goal_conf_for_insert
                )
                if goal_conf_yawed is not None:
                    goal_conf_for_insert = goal_conf_yawed
                    print(f"[Info] Updated grasp orientation with yaw {self.yaw_offset_deg * yaw_sign:+.1f}° for side view.")
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
