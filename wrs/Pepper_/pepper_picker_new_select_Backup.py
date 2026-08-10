#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/12/18
# @Author : cursor
"""
整合版辣椒采摘代码
- 修复了角度/弧度转换问题
- 修复了坐标转换矩阵方向问题
- 使用点云矩阵而非 get_distance
- 支持连续采摘（按置信度排序）
"""

import numpy as np
import time
import cv2
import wrs.visualization.panda.world as wd
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.motion.probabilistic.rrt_connect as rrtc
import wrs.motion.primitives.approach_depart_planner as adp
from wrs.robot_sim.robots.realman.realman_single_arm import Realman
from wrs.robot_con.realman.realman import RealmanArmController
import pyrealsense2 as rs
import wrs.Pepper.yoloutils.yolo_utils as yoloutils
from wrs.Pepper.Grapper.Grapper import ClawController
from wrs.Pepper.Path_planning.RL_looking import RLLooking
from wrs.Pepper.vision_io import capture_color_and_pcd, safe_imshow, yolo_predict, yolo_vis_image

base = wd.World(cam_pos=rm.vec(2, 0, 1), lookat_pos=rm.vec(0, 0, 0.5))
mgm.gen_frame().attach_to(base)


DEFAULT_GRIPPER_COM = "COM6"  # 如现场端口不同，改这里
claw_id = 9  # 夹爪ID
claw_controller = ClawController(claw_id, com_port=DEFAULT_GRIPPER_COM)
claw_controller.connect_gripper()
claw_controller.enable_gripper()

class RealmanPepperPicker():
    def __init__(self, real_robot=True, yaw_offset_deg=30.0, target_distance_thresh=0.08):
        """
        初始化辣椒采摘系统
        - real_robot: True=真机模式, False=仿真模式
        """
        # 绑定yolo_utils的手眼变换函数
        self.transform_point_cloud_handeye = yoloutils.transform_point_cloud_handeye.__get__(self)

        # 货架模型（碰撞检测用）
        self.shelf_model = mcm.gen_box(xyz_lengths=np.array([0.02, 0.7, 1.55]),
                                       pos=rm.vec(-0.8, 0, 0),
                                       rotmat=np.eye(3),
                                       rgb=[0.1, 0.1, 0.2],
                                       alpha=0.8)
        self.shelf_model.attach_to(base)

        # 旅行材模型（碰撞检测用）
        box_model = mcm.gen_box(xyz_lengths=np.array([0.5, 0.5, 0.66]),  # 例如 60cm x 30cm x 80cm 的货架
                                pos=rm.vec(0, 0, -0.33),
                                rotmat=np.eye(3),
                                rgb=[0.1, 0.1, 0.2],
                                alpha=0.8)
        box_model.attach_to(base)

        # 机器人模型和规划器
        self.robot = Realman(enable_cc=True)
        self.rrtc_planner = rrtc.RRTConnect(self.robot)
        self.adp_planner = adp.ADPlanner(self.robot)

        self.real_robot = real_robot
        self.rm_controller = None

        # 加载 YOLO 模型
        self.yolo_model = yoloutils.init_yolo(r"F:\WRS\wrs-virtual\wrs\Pepper\best.pt")

        # RealSense 相机
        self.rs_pipeline = None
        self.rs_align = None

        # 手眼标定矩阵（TCP -> Camera）
        # self.HANDEYE = np.array([[0.008677594379434769, 0.9977978977644354, -0.0657575438438099, -0.07999999999999977],
        #                          [-0.9995990128019849, 0.010428221287874299, 0.026326143010957117,0.011999999999999969],
        #                          [0.026953904371157977, 0.06550272831994479, 0.9974882854557063, -0.0330000000000001],
        #                          [0.0, 0.0, 0.0, 1.0]])
        self.HANDEYE = np.array([[0.1336507447770641, 0.9901388643593643, 0.04198223082863791, -0.04999999999999974], [-0.9909988680911875, 0.13385433834108337, -0.002063867481476148, 0.01299999999999997], [-0.007663019133927183, -0.04132850580510994, 0.9991162258442235, -0.04300000000000011], [0.0, 0.0, 0.0, 1.0]])

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
                    self.rs_pipeline = rs.pipeline()
                    self.rs_config = rs.config()
                    self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                    self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                    self.rs_profile = self.rs_pipeline.start(self.rs_config)
                    self.rs_align = rs.align(rs.stream.color)

                    # warm-up: 丢弃前30帧，让曝光/深度稳定
                    for _ in range(30):
                        self.rs_pipeline.wait_for_frames()
                    print("RealSense 相机已启动（pipeline）。")
                except Exception as e:
                    print(f"[Error] Failed to init RealSense: {e}")
                    print("[Info] Fallback to simulation.")
                    self.real_robot = False
                    self.rs_pipeline = None
                    self.rs_align = None

        # 可配置左右偏航角和匹配距离阈值
        self.rl_looking = RLLooking(self,
                                    target_distance_thresh=target_distance_thresh,
                                    yaw_offset_deg=yaw_offset_deg)

    # def get_aligned_images(pipeline, align):
    #     """获取对齐的RGB-D图像"""
    #     frames = pipeline.wait_for_frames()
    #     aligned_frames = align.process(frames)

    #     aligned_depth_frame = aligned_frames.get_depth_frame()
    #     aligned_color_frame = aligned_frames.get_color_frame()

    #     depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    #     color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics

    #     img_color = np.asanyarray(aligned_color_frame.get_data())
    #     img_depth = np.asanyarray(aligned_depth_frame.get_data())

    #     return depth_intrin, img_color, aligned_depth_frame

    # def get_Keyresult(self, results):
    #     """
    #     从YOLO检测结果中提取关键点信息
    #     返回关键点数组，每个元素为 [x, y, confidence]
    #     """
    #     keypoints_array = []
    #     target_class_id = 0  # 目标类别ID，根据实际情况调整

    #     for result in results:
    #         boxes = result.boxes
    #         keypoints = result.keypoints

    #         for i in range(len(boxes)):
    #             class_id = int(boxes[i].cls)
    #             if class_id == target_class_id:
    #                 box_confidence = boxes.conf[i].item()
    #                 print(f"目标框的置信度：{box_confidence:.3f}")

    #                 if keypoints is not None and len(keypoints) > i:
    #                     kpts = keypoints[i].data[0].cpu().numpy()
    #                     for j, kpt in enumerate(kpts):
    #                         if j == 0:  # 只取第一个关键点
    #                             keypoints_array.append([kpt[0], kpt[1], kpt[2]])

    #     return keypoints_array

    def scan_and_detect_multi(self, max_targets=5, conf_thresh=0.3):
        """
        一次识别多个目标，按置信度从高到低返回 [(world_pos, conf), ...]
        - max_targets: 最多返回的目标数量
        - conf_thresh: 置信度阈值
        """
        if not self.real_robot:
            return [(np.array([-0.45, -0.1, 0.3]), 1.0)], None

        if self.rs_pipeline is None or self.rs_align is None:
            print("[Error] Camera pipeline not initialized.")
            return [], None

        t_total0 = time.time()
        t_wait = 0.0
        t_align = 0.0
        t_pc = 0.0
        t_yolo = 0.0

        try:
            cap = capture_color_and_pcd(self.rs_pipeline, self.rs_align)
            if cap is None:
                print("[Error] Failed to get depth/color frames.")
                return [], None

            color_img = cap.color_img
            depth_intrin = cap.depth_intrin
            pcd_cam = cap.pcd_cam
            t_wait = cap.t_wait
            t_align = cap.t_align
            t_pc = cap.t_pcd
        except Exception as e:
            print(f"[Error] Camera grab failed: {e}")
            return [], None

        results, yolo_time = yolo_predict(self.yolo_model, color_img, save=False, show_conf=False)
        t_yolo = yolo_time
        print(f"[Timing] YOLO detection time: {yolo_time:.3f}s")
        t_total = time.time() - t_total0
        print(f"[Timing][Detect] grab={t_wait:.3f}s align={t_align:.3f}s pcd={t_pc:.3f}s "
              f"yolo={t_yolo:.3f}s total={t_total:.3f}s")
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            print("[Info] No detection.")
            # 返回空列表和None
            try:
                safe_imshow("YOLO Detection", color_img, wait_ms=1)
            except Exception as e:
                print(f"[Warn] Failed to show empty detection window: {e}")
            return [], None

        # 绘制YOLO检测结果
        im_array = yolo_vis_image(results, color_img)

        detections = []
        res = results[0]

        boxes = res.boxes
        kpts_all = res.keypoints

        # 遍历所有检测
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].item())
            if conf < conf_thresh:
                continue

            if kpts_all is None or kpts_all.data is None or len(kpts_all.data) <= i:
                continue

            # 获取关键点（兼容不同形状）
            kpts_raw = kpts_all.data[i].cpu().numpy()
            if kpts_raw.ndim == 3:
                kpts = kpts_raw[0]
            else:
                kpts = kpts_raw

            if kpts.shape[0] == 0:
                continue

            u, v, vis = kpts[0]
            if vis < 0.1:  # 关键点可见性阈值
                continue

            u, v = int(u), int(v)
            if not (0 <= u < depth_intrin.width and 0 <= v < depth_intrin.height):
                continue

            # 从点云矩阵中获取3D坐标（优先使用点云，而不是 get_distance）
            camera_xyz = pcd_cam[v, u]  # numpy 索引是 [row, col] = [v, u]

            # 如果点云无效，尝试邻域估计
            if np.linalg.norm(camera_xyz) < 1e-6:
                camera_xyz = yoloutils._estimate_point_from_neighborhood(
                    target_pixel=(u, v),
                    pcd_matrix=pcd_cam,
                    neighborhood_size=2
                )
                if camera_xyz is None or np.any(np.isnan(camera_xyz)):
                    continue

            camera_xyz = np.asarray(camera_xyz)

            # 相机 -> 世界坐标（手眼变换）
            given_conf = self.rm_controller.get_joint_values() if self.rm_controller else None
            world_point_cloud = self.transform_point_cloud_handeye(
                handeye_mat=self.HANDEYE,
                pcd=camera_xyz.reshape(1, 3),
                given_conf=given_conf
            )
            world_xyz = world_point_cloud[0]
            detections.append((world_xyz, conf))

            # 在可视化图像上标记本次选中的关键点和世界坐标/置信度
            try:
                cv2.circle(im_array, (u, v), 5, (0, 255, 0), -1)
                text = f"W:{world_xyz[0]:.3f},{world_xyz[1]:.3f},{world_xyz[2]:.3f} conf:{conf:.2f}"
                cv2.putText(
                    im_array,
                    text,
                    (u + 10, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
            except Exception as e:
                print(f"[Warn] Failed to draw keypoint: {e}")

        # 按置信度降序排序，限制数量
        detections.sort(key=lambda x: x[1], reverse=True)
        if len(detections) > max_targets:
            detections = detections[:max_targets]

        # 显示带有所有采摘点标记的YOLO结果图像
        if not safe_imshow("YOLO Detection", im_array, wait_ms=1):
            print("[Warn] Failed to show YOLO window.")

        print(f"[Info] Detected {len(detections)} targets (sorted by conf).")
        return detections, im_array

    def pick_place_pepper(self,
                          approach_dis=0.2,
                          bound_lower=-40,
                          bound_upper=40,
                          grad=50,
                          viewpoint_joint=np.array(
                              [-0.16688839, -1.59552764, 1.95066723, 0.09162979, 1.25496153, -0.07829547]),
                          place_joint=np.array(
                              [1.57254163e+00, 6.96909976e-02, 1.92076978e+00, 1.71216801e-02, 7.77579056e-01,
                               2.96705989e-04]),
                          continuous_mode=True,
                          max_targets_per_round=5,
                          detect_retry_frames=5,
                          detect_retry_sleep=0.05,
                          near_target_replan_enabled=True,
                          near_target_replan_distance=0.1,
                          stop_on_no_detection=True):
        """
        执行完整的采摘任务（支持连续采摘）
        - approach_dis: 预接近距离
        - bound_lower/upper: 抓取姿态搜索角度范围
        - grad: 角度搜索步长
        - viewpoint_joint: 观测点关节角（弧度）
        - place_joint: 放置点关节角（弧度）
        - continuous_mode: 是否连续采摘模式
        - max_targets_per_round: 每轮最多采摘的目标数
        - detect_retry_frames: 单轮检测最多重试帧数（避免单帧深度空洞导致漏检）
        - detect_retry_sleep: 检测重试间隔（秒）
        - near_target_replan_enabled: 左/右观察点时是否在近距离分段插入
        - near_target_replan_distance: 分段点距离目标TCP的距离（米），如0.05
        - stop_on_no_detection: 当视角中没有检测到辣椒时是否停止采摘
        """
        def _ts(t=None):
            if t is None:
                t = time.time()
            return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))

        _task_start_time = time.time()
        print(f"[Timing][Task] start={_ts(_task_start_time)} ts={_task_start_time:.3f}")
        robot = self.robot
        rrtc_planner = self.rrtc_planner

        timing_sum = {
            "detect_total": 0.0,
            "ik_total": 0.0,
            "looking_total": 0.0,
            "grasp_total": 0.0,
        }

        # 同步真机状态到仿真
        if self.real_robot and self.rm_controller is not None:
            start_conf = self.rm_controller.get_joint_values()
        else:
            start_conf = robot.get_jnt_values()

        robot.goto_given_conf(start_conf)

        # Phase 1: 移动到观测点
        print("--- Phase 1: Moving to Viewpoint ---")
        if np.linalg.norm(np.array(start_conf) - np.array(viewpoint_joint)) > 0.03:
            mot_data_view = rrtc_planner.plan(start_conf=start_conf,
                                              goal_conf=viewpoint_joint,  # 统一使用弧度
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

        # 连续采摘循环
        pick_count = 0
        total_success = 0

        while continuous_mode:
            # Phase 2: 检测目标
            print(f"\n--- Phase 2: Detecting Target (Round {pick_count + 1}) ---")
            _t0 = time.time()
            detections = []
            for attempt in range(max(1, int(detect_retry_frames))):
                detections, _ = self.scan_and_detect_multi(max_targets=max_targets_per_round, conf_thresh=0.3)
                if len(detections) > 0:
                    break
                if detect_retry_sleep and detect_retry_sleep > 0:
                    time.sleep(detect_retry_sleep)
            t_detect_total = time.time() - _t0
            timing_sum["detect_total"] += t_detect_total
            print(f"[Timing][Round {pick_count + 1}] detect_total={t_detect_total:.3f}s")

            if len(detections) == 0:
                if stop_on_no_detection:
                    print("[Info] No more peppers detected in current view. Ending continuous picking.")
                    break
                else:
                    print("[Info] No peppers detected in current view. Continuing to next round...")
                    time.sleep(1.0)  # 短暂等待后继续
                    continue

            # 按置信度从高到低依次采摘
            for tgt_pos, conf in detections:
                print(f"\n--- Picking target with confidence {conf:.3f} at {tgt_pos} ---")
                # tgt_pos = tgt_pos + np.array([0.0, -0.02, 0.0])

                # Phase 3: 搜索抓取姿态并规划路径
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
                            # tgt_pos_approach_ultimate = tgt_pos - hand_z * approach_dis
                            tgt_pos_approach = tgt_pos - hand_z * approach_dis + np.array([0, 0, -0.04])  # 预接近点向下偏移8cm
                            approach_conf = robot.ik(tgt_pos_approach, tgt_rotmat, seed_jnt_values=grasp_conf)

                            if approach_conf is not None:
                                robot.goto_given_conf(approach_conf)
                                if not robot.is_collided():

                                    jv_list_fast = self.fast_joint_interp_path(
                                        start_conf=start_conf,
                                        goal_conf=approach_conf,
                                        step=0.03,  # 你可以试 0.02~0.05，越小越稳
                                        obstacle_list=[self.shelf_model]
                                    )

                                    if jv_list_fast is not None:
                                        mot_data_linear = self.adp_planner.gen_linear_motion(
                                            start_conf=approach_conf,
                                            goal_conf=grasp_conf,
                                            granularity=0.01,
                                            obstacle_list=[self.shelf_model]
                                        )

                                        if mot_data_linear is not None:
                                            final_approach_path = jv_list_fast  # 注意：这里不再是 mot_data_rrt 对象
                                            final_linear_path = mot_data_linear
                                            chosen_approach_conf = approach_conf.copy()
                                            chosen_grasp_conf = grasp_conf.copy()
                                            is_success = True
                                            break

                print(f"[Timing] IK solving total time: {ik_total_time:.3f}s")
                timing_sum["ik_total"] += ik_total_time

                # Phase 4: 规划放置路径
                mot_data_place = None
                if is_success:
                    print("--- Phase 4: Planning Placement Route ---")
                    # current_conf_for_place_planning = final_approach_path.jv_list[-1]
                    current_conf_for_place_planning = final_approach_path[-1]
                    robot.goto_given_conf(current_conf_for_place_planning)

                    mot_data_place = rrtc_planner.plan(start_conf=current_conf_for_place_planning,
                                                       goal_conf=place_joint,
                                                       obstacle_list=[self.shelf_model],
                                                       ext_dist=.1,
                                                       max_time=300)

                    if mot_data_place is not None:
                        self.vis_path(mot_data_place, rm.const.copper_map)
                    else:
                        print("[Error] Failed to plan to place position.")
                        is_success = False

                # Phase 5: 执行抓取和放置
                if is_success and mot_data_place is not None and self.real_robot:
                    print("--- Executing Grasp and Place ---")
                    grasp_start_time = time.time()

                    # 打开夹爪
                    _t_open0 = time.time()
                    claw_controller.open_gripper()
                    time.sleep(0.5)
                    t_open = time.time() - _t_open0

                    # 移动到预接近点（已向下偏移8/4cm）
                    # self.rm_controller.move_jntspace_path(final_approach_path.jv_list, control_frequency=0.8)
                    _t_approach0 = time.time()
                    self.rm_controller.move_jntspace_path(final_approach_path, control_frequency=0.8)
                    time.sleep(2)
                    t_approach = time.time() - _t_approach0

                    linear_path_to_use = final_linear_path
                    grasp_conf_for_insert = chosen_grasp_conf
                    if self.rl_looking is not None:
                        _t_looking0 = time.time()
                        path_candidate, grasp_conf_after_view = self.rl_looking.capture_view_confidences(
                            robot=robot,
                            tgt_pos=tgt_pos,
                            chosen_approach_conf=chosen_approach_conf,
                            chosen_grasp_conf=chosen_grasp_conf,
                            final_linear_path=final_linear_path
                        )
                        t_looking = time.time() - _t_looking0
                        timing_sum["looking_total"] += t_looking
                        print(f"[Timing][Looking] total={t_looking:.3f}s")
                        selected_view_idx_after_looking = getattr(self.rl_looking, "last_best_view_idx", 0)
                        if grasp_conf_after_view is not None:
                            grasp_conf_for_insert = grasp_conf_after_view
                        if selected_view_idx_after_looking in (1, 2) and path_candidate is None:
                            print("[Error] Selected side view but failed to replan insertion from that view; skip this target to avoid center-view insertion.")
                            try:
                                cur_conf = self.rm_controller.get_joint_values()
                                robot.goto_given_conf(cur_conf)
                                mot_back = rrtc_planner.plan(
                                    start_conf=cur_conf,
                                    goal_conf=viewpoint_joint,
                                    obstacle_list=[self.shelf_model],
                                    ext_dist=.1,
                                    max_time=300
                                )
                                if mot_back is not None:
                                    self.rm_controller.move_jntspace_path(mot_back.jv_list, control_frequency=0.8)
                                    time.sleep(2.0)
                                    robot.goto_given_conf(viewpoint_joint)
                            except Exception as e:
                                print(f"[Warn] Failed to return to viewpoint after side-view replan failure: {e}")

                            pick_count += 1
                            continue
                        if path_candidate is not None:
                            linear_path_to_use = path_candidate
                    else:
                        t_looking = 0.0
                        grasp_conf_for_insert = chosen_grasp_conf

                    # Execute linear insertion
                    _t_insert0 = time.time()
                    insert_jv_list = None
                    insert_mot_1 = None
                    insert_mot_2 = None

                    selected_view_idx = getattr(self.rl_looking, "last_best_view_idx", 0) if self.rl_looking is not None else 0
                    use_near_target_replan = (
                        bool(near_target_replan_enabled)
                        and self.rl_looking is not None
                        and selected_view_idx in (1, 2)
                        and near_target_replan_distance is not None
                        and float(near_target_replan_distance) > 1e-6
                    )

                    if use_near_target_replan:
                        try:
                            start_conf_insert = np.asarray(linear_path_to_use.jv_list[0], dtype=float)
                            goal_conf_insert = np.asarray(grasp_conf_for_insert, dtype=float)

                            robot.goto_given_conf(start_conf_insert)
                            start_tcp_pos, _ = robot.fk(start_conf_insert, update=False)
                            goal_tcp_pos, goal_tcp_rotmat = robot.fk(goal_conf_insert, update=False)

                            direction = goal_tcp_pos - start_tcp_pos
                            dist = float(np.linalg.norm(direction))
                            stop_d = float(near_target_replan_distance)
                            print(f"[Info] Near-target replan enabled (view={selected_view_idx}), dist={dist:.3f} stop_d={stop_d:.3f}")

                            if dist > stop_d + 1e-3:
                                direction = direction / dist
                                mid_tcp_pos = goal_tcp_pos - direction * stop_d
                                mid_conf = robot.ik(
                                    tgt_pos=mid_tcp_pos,
                                    tgt_rotmat=goal_tcp_rotmat,
                                    seed_jnt_values=start_conf_insert
                                )
                                if mid_conf is None:
                                    print("[Warn] Near-target replan: IK for mid pose failed; fallback to one-shot insertion.")
                                else:
                                    insert_mot_1 = self.adp_planner.gen_linear_motion(
                                        start_conf=start_conf_insert,
                                        goal_conf=mid_conf,
                                        granularity=0.01,
                                        obstacle_list=[self.shelf_model]
                                    )
                                    if insert_mot_1 is None:
                                        print("[Warn] Near-target replan: failed to plan first segment; fallback to one-shot insertion.")
                                    else:
                                        insert_mot_2 = self.adp_planner.gen_linear_motion(
                                            start_conf=mid_conf,
                                            goal_conf=goal_conf_insert,
                                            granularity=0.01,
                                            obstacle_list=[self.shelf_model]
                                        )
                                        if insert_mot_2 is None:
                                            print("[Warn] Near-target replan: failed to plan second segment; fallback to one-shot insertion.")
                        except Exception as e:
                            print(f"[Warn] Near-target replan exception; fallback to one-shot insertion: {e}")

                    if insert_mot_1 is not None and insert_mot_2 is not None:
                        self.rm_controller.move_jntspace_path(insert_mot_1.jv_list, control_frequency=0.8)
                        time.sleep(0.2)
                        self.rm_controller.move_jntspace_path(insert_mot_2.jv_list, control_frequency=0.8)
                        insert_jv_list = insert_mot_1.jv_list + insert_mot_2.jv_list
                    else:
                        insert_jv_list = linear_path_to_use.jv_list
                        self.rm_controller.move_jntspace_path(insert_jv_list, control_frequency=0.8)
                    time.sleep(2)
                    t_insert = time.time() - _t_insert0

                    # 闭合夹爪
                    _t_close0 = time.time()
                    claw_controller.close_gripper()
                    time.sleep(1.0)
                    t_close = time.time() - _t_close0

                    # 直线撤退
                    _t_retreat0 = time.time()
                    if insert_jv_list is None:
                        insert_jv_list = final_linear_path.jv_list
                    self.rm_controller.move_jntspace_path(insert_jv_list[::-1], control_frequency=0.8)
                    time.sleep(2)
                    t_retreat = time.time() - _t_retreat0

                    # 移动到放置点
                    _t_to_place0 = time.time()
                    start_conf_to_place = None
                    try:
                        start_conf_to_place = self.rm_controller.get_joint_values()
                    except Exception as e:
                        print(f"[Warn] Failed to read current joints before place planning: {e}")
                    if start_conf_to_place is None:
                        start_conf_to_place = robot.get_jnt_values()

                    robot.goto_given_conf(start_conf_to_place)
                    mot_data_place_rt = rrtc_planner.plan(start_conf=start_conf_to_place,
                                                       goal_conf=place_joint,
                                                       obstacle_list=[self.shelf_model],
                                                       ext_dist=.1,
                                                       max_time=300)
                    if mot_data_place_rt is None:
                        raise RuntimeError("Failed to plan to place position from current pose.")

                    self.rm_controller.move_jntspace_path(mot_data_place_rt.jv_list, control_frequency=0.8)
                    time.sleep(2)
                    t_to_place = time.time() - _t_to_place0

                    # 放下
                    _t_release0 = time.time()
                    claw_controller.open_gripper()
                    time.sleep(0.5)
                    t_release = time.time() - _t_release0

                    total_success += 1
                    grasp_time = time.time() - grasp_start_time
                    print(f"[Success] Picked and placed pepper_nbv_v0 {total_success}")
                    timing_sum["grasp_total"] += grasp_time
                    print(f"[Timing] Grasp execution time: {grasp_time:.3f}s "
                          f"(open={t_open:.3f}s approach={t_approach:.3f}s looking={t_looking:.3f}s "
                          f"insert={t_insert:.3f}s close={t_close:.3f}s retreat={t_retreat:.3f}s "
                          f"to_place={t_to_place:.3f}s release={t_release:.3f}s)")

                    # 返回观测点，准备下一轮
                    print("--- Returning to Viewpoint ---")
                    start_conf_return = place_joint
                    try:
                        start_conf_return = self.rm_controller.get_joint_values()
                    except Exception as e:
                        print(f"[Warn] Failed to read current joints for return planning: {e}")

                    mot_data_return = rrtc_planner.plan(start_conf=start_conf_return,
                                                        goal_conf=viewpoint_joint,
                                                        obstacle_list=[self.shelf_model],
                                                        ext_dist=.1,
                                                        max_time=300)
                    if mot_data_return is not None:
                        robot.goto_given_conf(start_conf_return)
                        if self.real_robot:
                            self.rm_controller.move_jntspace_path(mot_data_return.jv_list, control_frequency=0.8)
                            time.sleep(3.0)
                        robot.goto_given_conf(viewpoint_joint)
                elif is_success and not self.real_robot:
                    print("Simulation Success.")
                    robot.goto_given_conf(place_joint)
                    robot.goto_given_conf(viewpoint_joint)
                    time.sleep(2)
                else:
                    print(f"[Failed] Failed to pick target at {tgt_pos}")

                pick_count += 1

        # 时间统计与日志记录
        _task_end_time = time.time()
        _duration = _task_end_time - _task_start_time
        print(f"[Timing][Task] end={_ts(_task_end_time)} ts={_task_end_time:.3f}")
        print(f"[Timing][Summary] total={_duration:.3f}s detect_total={timing_sum['detect_total']:.3f}s "
              f"ik_total={timing_sum['ik_total']:.3f}s looking_total={timing_sum['looking_total']:.3f}s "
              f"grasp_total={timing_sum['grasp_total']:.3f}s")
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
        """可视化轨迹"""
        if mot_data:
            n = len(mot_data.mesh_list)
            for i, model in enumerate(mot_data.mesh_list):
                model.rgb = colormap(i / n)
                model.alpha = 0.3
                model.attach_to(base)

    def fast_joint_interp_path(self, start_conf, goal_conf, step=0.03, obstacle_list=None):
        """
        快速关节空间直连插补（代替RRT）
        - step: 每步关节最大变化（弧度），越小越安全但点更多
        返回：jv_list（list[np.ndarray]）或 None（有碰撞/不可达）
        """
        robot = self.robot
        start_conf = np.asarray(start_conf, dtype=float)
        goal_conf = np.asarray(goal_conf, dtype=float)

        diff = goal_conf - start_conf
        max_delta = float(np.max(np.abs(diff)))
        n = max(2, int(np.ceil(max_delta / step)) + 1)

        jv_list = []
        for a in np.linspace(0.0, 1.0, n):
            conf = start_conf + a * diff
            robot.goto_given_conf(conf)
            if robot.is_collided():  # 这里会用你 enable_cc=True 的碰撞检测
                return None
            jv_list.append(conf.copy())
        return jv_list


if __name__ == '__main__':
    picker = RealmanPepperPicker(real_robot=True)
    robot = Realman()
    robot.gen_meshmodel(toggle_jnt_frames=True).attach_to(base)

    view_point = np.array([-0.03005457, -1.20970511, 1.26979681, 0.01317724, 1.52136859, -0.00029671])
    place_point = np.array(
        [1.57254163e+00, 6.96909976e-02, 1.92076978e+00, 1.71216801e-02, 7.77579056e-01, 2.96705989e-04])

    picker.pick_place_pepper(
        approach_dis=0.25,
        bound_lower=0,
        bound_upper=20,
        grad=2,
        viewpoint_joint=view_point,
        place_joint=place_point,
        continuous_mode=True,  # 连续采摘模式
        max_targets_per_round=5,
        stop_on_no_detection=True  # 当视角中没有辣椒时停止采摘
    )

    base.run()
