#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/12/18
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

base = wd.World(cam_pos=rm.vec(2, 0, 1), lookat_pos=rm.vec(0, 0, 0.5))
mgm.gen_frame().attach_to(base)

DEFAULT_GRIPPER_COM = "COM6"  # 如现场端口不同，改这里
claw_id = 9  # 夹爪ID
claw_controller = ClawController(claw_id, com_port=DEFAULT_GRIPPER_COM)
claw_controller.connect_gripper()
claw_controller.enable_gripper()

class RealmanPepperPicker():
    def __init__(self, real_robot=True):
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
        self.box_model = mcm.gen_box(xyz_lengths=np.array([0.5, 0.5, 0.66]),  # 例如 60cm x 30cm x 80cm 的货架
                                pos=rm.vec(0, 0, -0.33),
                                rotmat=np.eye(3),
                                rgb=[0.1, 0.1, 0.2],
                                alpha=0.8)
        self.box_model.attach_to(base)
        
        # 机器人模型和规划器
        self.robot = Realman(enable_cc=True)
        self.rrtc_planner = rrtc.RRTConnect(self.robot)
        self.adp_planner = adp.ADPlanner(self.robot)
        
        self.real_robot = real_robot
        self.rm_controller = None
        
        # 加载 YOLO 模型
        self.yolo_model = yoloutils.init_yolo(r"F:\WRS\wrs-virtual\wrs\Pepper\SLPepper.pt")
        
        # RealSense 相机
        self.rs_pipeline = None
        self.rs_align = None
        
        # 手眼标定矩阵（TCP -> Camera）
        # self.HANDEYE = np.array([[0.008677594379434769, 0.9977978977644354, -0.0657575438438099, -0.07999999999999977], [-0.9995990128019849, 0.010428221287874299, 0.026326143010957117, 0.011999999999999969], [0.026953904371157977, 0.06550272831994479, 0.9974882854557063, -0.0330000000000001], [0.0, 0.0, 0.0, 1.0]])
        self.HANDEYE = np.array([[-0.012102519796519457, 0.9998605053175249, 0.011510817528360709, -0.04899999999999974], [-0.9998133994554028, -0.012273682788263483, 0.01491720415436704, 0.007999999999999965], [0.01505640340671479, -0.011328133844956416, 0.9998224732918157, -0.04200000000000011], [0.0, 0.0, 0.0, 1.0]])
        # self.HANDEYE = np.array([[-0.03213569371611176, 0.9989935824218366, -0.03129088475875401, -0.03799999999999973], [-0.9989666556119892, -0.031096632420903828, 0.03314544353513603, 0.0029999999999999654], [0.03213904423669037, 0.0323237023201315, 0.9989605898649643, -0.04700000000000011], [0.0, 0.0, 0.0, 1.0]])

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

    def get_aligned_images(pipeline, align):
        """获取对齐的RGB-D图像"""
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics

        img_color = np.asanyarray(aligned_color_frame.get_data())
        img_depth = np.asanyarray(aligned_depth_frame.get_data())

        return depth_intrin, img_color, aligned_depth_frame

    def get_Keyresult(self, results):
        """
        从YOLO检测结果中提取关键点信息
        返回关键点数组，每个元素为 [x, y, confidence]
        """
        keypoints_array = []
        target_class_id = 0  # 目标类别ID，根据实际情况调整
        
        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            
            for i in range(len(boxes)):
                class_id = int(boxes[i].cls)
                if class_id == target_class_id:
                    box_confidence = boxes.conf[i].item()
                    print(f"目标框的置信度：{box_confidence:.3f}")

                    if keypoints is not None and len(keypoints) > i:
                        kpts = keypoints[i].data[0].cpu().numpy()
                        for j, kpt in enumerate(kpts):
                            if j == 0:  # 只取第一个关键点
                                keypoints_array.append([kpt[0], kpt[1], kpt[2]])
        
        return keypoints_array

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

        try:
            # 获取一帧
            frames = self.rs_pipeline.wait_for_frames()
            aligned = self.rs_align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            
            if not depth_frame or not color_frame:
                print("[Error] Failed to get depth/color frames.")
                return [], None
            
            color_img = np.asanyarray(color_frame.get_data())
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            
            # 生成点云矩阵 (H, W, 3)
            pc = rs.pointcloud()
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)
            pcd_cam = np.asanyarray(points.get_vertices()).view(np.float32).reshape(
                depth_intrin.height, depth_intrin.width, 3
            )
            
        except Exception as e:
            print(f"[Error] Camera grab failed: {e}")
            return [], None

        # YOLO 推理
        yolo_start_time = time.time()
        results = self.yolo_model.predict(color_img, save=False, show_conf=False)
        yolo_time = time.time() - yolo_start_time
        print(f"[Timing] YOLO detection time: {yolo_time:.3f}s")
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            print("[Info] No detection.")
            # 返回空列表和None
            try:
                cv2.imshow("YOLO Detection", color_img)
                cv2.waitKey(1)
            except Exception as e:
                print(f"[Warn] Failed to show empty detection window: {e}")
            return [], None

        # 绘制YOLO检测结果
        im_array = results[0].plot()

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
                    neighborhood_size=1
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
        try:
            cv2.imshow("YOLO Detection", im_array)
            cv2.waitKey(1)  # 非阻塞显示
        except Exception as e:
            print(f"[Warn] Failed to show YOLO window: {e}")

        print(f"[Info] Detected {len(detections)} targets (sorted by conf).")
        return detections, im_array

    def pick_place_pepper(self,
                          approach_dis=0.3,
                          bound_lower=-40,
                          bound_upper=40,
                          grad=50,
                          viewpoint_joint=np.array([-0.16688839, -1.59552764, 1.95066723, 0.09162979, 1.25496153, -0.07829547]),
                          place_joint=np.array([1.57254163e+00, 6.96909976e-02, 1.92076978e+00, 1.71216801e-02, 7.77579056e-01, 2.96705989e-04]),
                          continuous_mode=True,
                          max_targets_per_round=5,
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
        - stop_on_no_detection: 当视角中没有检测到辣椒时是否停止采摘
        """
        _task_start_time = time.time()
        robot = self.robot
        rrtc_planner = self.rrtc_planner
        
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
            detections, _ = self.scan_and_detect_multi(max_targets=max_targets_per_round, conf_thresh=0.3)

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
                tgt_pos = tgt_pos + np.array([-0.0, -0.0, -0.03])
                
                # Phase 3: 搜索抓取姿态并规划路径
                print(f"--- Phase 3: Searching Grasp Pose ---")
                start_conf = robot.get_jnt_values()

                is_success = False
                final_approach_path = None
                final_linear_path = None

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
                                    # mot_data_rrt = rrtc_planner.plan(start_conf=start_conf,
                                    #                                  goal_conf=approach_conf,
                                    #                                  obstacle_list=[self.shelf_model, self.box_model],
                                    #                                  ext_dist=.1,
                                    #                                  max_time=300)
                                    #
                                    # if mot_data_rrt is not None:
                                    #     mot_data_linear = self.adp_planner.gen_linear_motion(
                                    #         start_conf=approach_conf,
                                    #         goal_conf=grasp_conf,
                                    #         granularity=0.01,
                                    #         obstacle_list=[self.shelf_model, self.box_model]
                                    #     )
                                    #
                                    #     if mot_data_linear is not None:
                                    #         final_approach_path = mot_data_rrt
                                    #         final_linear_path = mot_data_linear
                                    #         self.vis_path(mot_data_rrt, rm.const.winter_map)
                                    #         self.vis_path(mot_data_linear, rm.const.wistia_map)
                                    #         is_success = True
                                    #         break
                            #         用快速插补代替RRT：start_conf -> approach_conf
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
                                            is_success = True
                                            break

                print(f"[Timing] IK solving total time: {ik_total_time:.3f}s")

                # Phase 4: 规划放置路径
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
                
                # Phase 5: 执行抓取和放置
                if is_success and mot_data_place is not None and self.real_robot:
                    print("--- Executing Grasp and Place ---")
                    grasp_start_time = time.time()
                    
                    # 打开夹爪
                    claw_controller.open_gripper()
                    time.sleep(0.5)
                    
                    # 移动到预接近点
                    # self.rm_controller.move_jntspace_path(final_approach_path.jv_list, control_frequency=0.8)
                    self.rm_controller.move_jntspace_path(final_approach_path, control_frequency=0.8)
                    time.sleep(2)
                    
                    # 直线插入
                    print(f"[Debug] final_linear_path points: {len(final_linear_path.jv_list) if final_linear_path and hasattr(final_linear_path, 'jv_list') else 'None'}")
                    cart_linear_jv = self.cartesian_linear_ik_path(
                        start_conf=final_linear_path.jv_list[0],
                        goal_conf=final_linear_path.jv_list[-1],
                        step=0.005
                    )
                    if cart_linear_jv is not None:
                        print(f"[Debug] cartesian_linear_ik_path points: {len(cart_linear_jv)}")
                        self.rm_controller.move_jntspace_path(cart_linear_jv, control_frequency=0.8)
                    else:
                        print("[Warn] cartesian_linear_ik_path failed, fallback to adp jv_list")
                        self.rm_controller.move_jntspace_path(final_linear_path.jv_list, control_frequency=0.8)
                    time.sleep(2)
                    
                    # 闭合夹爪
                    claw_controller.close_gripper()
                    time.sleep(1.0)
                    
                    # 直线撤退
                    print(f"[Debug] final_linear_path points (retreat): {len(final_linear_path.jv_list) if final_linear_path and hasattr(final_linear_path, 'jv_list') else 'None'}")
                    if cart_linear_jv is not None:
                        self.rm_controller.move_jntspace_path(cart_linear_jv[::-1], control_frequency=0.8)
                    else:
                        self.rm_controller.move_jntspace_path(final_linear_path.jv_list[::-1], control_frequency=0.8)
                    time.sleep(2)
                    
                    # 移动到放置点
                    self.rm_controller.move_jntspace_path(mot_data_place.jv_list, control_frequency=0.8)
                    time.sleep(2)
                    
                    # 放下
                    claw_controller.open_gripper()
                    time.sleep(0.5)
                    
                    total_success += 1
                    grasp_time = time.time() - grasp_start_time
                    print(f"[Success] Picked and placed pepper_nbv_v0 {total_success}")
                    print(f"[Timing] Grasp execution time: {grasp_time:.3f}s")

                    # 返回观测点，准备下一轮
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
        
        # 时间统计与日志记录
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

    def cartesian_linear_ik_path(self, start_conf, goal_conf, step=0.005):
        """
        笛卡尔直线采样 + IK，尽量贴近直线。
        返回关节路径列表；失败返回 None。
        """
        robot = self.robot
        start_conf = np.asarray(start_conf, dtype=float)
        goal_conf = np.asarray(goal_conf, dtype=float)

        start_pos, start_rot = robot.fk(start_conf, update=False)
        goal_pos, goal_rot = robot.fk(goal_conf, update=False)

        dist = float(np.linalg.norm(goal_pos - start_pos))
        n = max(2, int(np.ceil(dist / step)) + 1)

        path = []
        seed = start_conf.copy()
        for a in np.linspace(0.0, 1.0, n):
            pos = start_pos + a * (goal_pos - start_pos)
            # 保持抓取姿态（使用目标姿态）
            conf = robot.ik(tgt_pos=pos, tgt_rotmat=goal_rot, seed_jnt_values=seed)
            if conf is None:
                return None
            robot.goto_given_conf(conf)
            if robot.is_collided():
                return None
            path.append(conf.copy())
            seed = conf
        return path

    # def retreat_to_pick_pose(self, pick_joint, obstacle_list=None, control_frequency=0.8):
    #     """没检测到辣椒时，收回到采摘点位姿"""
    #     if obstacle_list is None:
    #         obstacle_list = [self.shelf_model]
    #
    #     robot = self.robot
    #     if self.real_robot and self.rm_controller is not None:
    #         start_conf = self.rm_controller.get_joint_values()
    #     else:
    #         start_conf = robot.get_jnt_values()
    #
    #     mot = self.rrtc_planner.plan(start_conf=start_conf,
    #                                  goal_conf=pick_joint,
    #                                  obstacle_list=obstacle_list,
    #                                  ext_dist=.1,
    #                                  max_time=300)
    #
    #     if mot is None:
    #         print("[Error] Failed to plan retreat to pick pose.")
    #         if not self.real_robot:
    #             robot.goto_given_conf(pick_joint)
    #         return False
    #
    #     robot.goto_given_conf(pick_joint)
    #     if self.real_robot and self.rm_controller is not None:
    #         self.rm_controller.move_jntspace_path(mot.jv_list, control_frequency=control_frequency)
    #         time.sleep(1.0)
    #
    #     print("[Info] Retreated to pick pose.")
    #     return True


if __name__ == '__main__':
    picker = RealmanPepperPicker(real_robot=True)
    robot = Realman()
    robot.gen_meshmodel(toggle_jnt_frames=True).attach_to(base)
    
    view_point = np.array([-0.03005457, -1.20970511,  1.26979681,  0.01317724,  1.52136859, -0.00029671])
    place_point = np.array([1.57254163e+00, 6.96909976e-02, 1.92076978e+00, 1.71216801e-02, 7.77579056e-01, 2.96705989e-04])

    picker.pick_place_pepper(
        approach_dis=0.25,
        bound_lower=0,
        bound_upper=20,
        grad=3,
        viewpoint_joint=view_point,
        place_joint=place_point,
        continuous_mode=True,  # 连续采摘模式
        max_targets_per_round=5,
        stop_on_no_detection=True  # 当视角中没有辣椒时停止采摘
    )
    
    base.run()
