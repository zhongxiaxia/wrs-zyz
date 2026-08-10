#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/12/05
# @Author : Zhang Xi

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
from wrs.Pepper.Grapper.Grapper import ClawController


base = wd.World(cam_pos=rm.vec(2, 0, 1), lookat_pos=rm.vec(0, 0, 0.5))
mgm.gen_frame().attach_to(base)


DEFAULT_GRIPPER_COM = "COM3"  # 如现场端口不同，改这里
claw_id = 9  # 夹爪ID
claw_controller = ClawController(claw_id, com_port=DEFAULT_GRIPPER_COM)  # 创建夹爪控制器对象
claw_controller.connect_gripper()  # 连接夹爪
claw_controller.enable_gripper()  # 使能夹爪


class RealmanPepperPicker():
    def __init__(self, real_robot=True):
        # 绑定yolo_utils的手眼变换函数，让它成为类方法
        import wrs.Pepper.yoloutils.yolo_utils as yoloutils
        self.transform_point_cloud_handeye = yoloutils.transform_point_cloud_handeye.__get__(self)
        self.shelf_model = mcm.gen_box(xyz_lengths=np.array([0.02, 0.7, 1.55]),
                                       pos=rm.vec(-0.8, 0, 0),
                                       rotmat=np.eye(3),
                                       rgb=[0.1, 0.1, 0.2],
                                       alpha=0.8)
        self.shelf_model.attach_to(base)
        self.robot = Realman(enable_cc=True)
        self.rrtc_planner = rrtc.RRTConnect(self.robot)
        self.adp_planner = adp.ADPlanner(self.robot)
        self.real_robot = real_robot
        self.rm_controller = None
        # 加载 YOLO 权重（请确保路径存在）
        self.yolo_model = yoloutils.init_yolo(r"F:\WRS\wrs-virtual\wrs\Pepper\best.pt")
        # RealSense pipeline（真机）或 None（仿真）
        self.rs_pipeline = None
        self.rs_align = None
        self.HANDEYE = np.array([[-0.013634785865167112, 0.9993155344887175, 0.03438829951851894, -0.10199999999999979], [-0.9990437250270998, -0.012186134398893706, -0.041989684593803374, 0.028999999999999984], [-0.04154188366317311, -0.03492793520631615, 0.9985260693860681, -0.029000000000000095], [0.0, 0.0, 0.0, 1.0]])

        if self.real_robot:
            try:
                # RealmanArmController 不支持 can_name/enable，使用默认参数
                self.rm_controller = RealmanArmController(has_gripper=True)
                print("Realman Arm Connected.")
            except Exception as e:
                print(f"[Error] Failed to connect Realman: {e}")
                print("[Info] Fallback to simulation.")
                self.real_robot = False

            # RealSense 初始化（与 Main_last.py 相同，直接启动 pipeline）
            if self.real_robot:
                try:
                    self.rs_pipeline = rs.pipeline()
                    self.rs_config = rs.config()
                    self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                    self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                    self.rs_profile = self.rs_pipeline.start(self.rs_config)
                    self.rs_align = rs.align(rs.stream.color)
                    print("RealSense 相机已启动（pipeline）。")
                except Exception as e:
                    print(f"[Error] Failed to init RealSense: {e}")
                    print("[Info] Fallback to simulation.")
                    self.real_robot = False
                    self.rs_pipeline = None
                    self.rs_align = None

            # 此处已在 __init__ 顶部加载权重，无需重复加载

    def get_Keyresult(self, results):
        """
        从YOLO检测结果中提取关键点信息（参考 Main_last.py）
        返回关键点数组，每个元素为 [x, y, confidence]
        """
        keypoints_array = []
        target_class_id = 0  # 目标类别ID，根据实际情况调整

        for result in results:
            # 获取所有检测框的类别编号（整数形式）
            class_ids = result.boxes.cls.tolist()  # 返回浮点数列表
            class_ids = [int(x) for x in class_ids]  # 转换为整数

            boxes = result.boxes
            keypoints = result.keypoints  # 关键点数据

            # 遍历每个检测到的对象
            for i in range(len(boxes)):
                class_id = int(boxes[i].cls)
                if class_id == target_class_id:
                    # 获取目标框的置信度
                    box_confidence = boxes.conf[i].item()  # 获取单个目标框的置信度
                    print(f"目标框的置信度：{box_confidence}")

                    # 获取关键点坐标（形状为 [num_keypoints, 3]）
                    # 每行格式: [x, y, confidence]
                    if keypoints is not None and len(keypoints) > i:
                        kpts = keypoints[i].data[0].cpu().numpy()
                        for j, kpt in enumerate(kpts):
                            print(f"关键点 {j}: X={kpt[0]:.2f}, Y={kpt[1]:.2f}, 可见={kpt[2]:.4f}")
                            if j == 0:  # 只取第一个关键点（可根据需要调整）
                                # 将关键点数据添加到数组中
                                keypoints_array.append([kpt[0], kpt[1], kpt[2]])

        return keypoints_array

    # def visualize_pointcloud_once(self, downsample_step: int = 10, rgba=np.array([0, 0, 1, 0.4])):
    #     """
    #     从 RealSense 获取一帧 RGB-D，生成点云，做手眼变换后在 Panda3D 场景中可视化。
    #     - downsample_step: 每隔多少个点取一个，避免点数过大导致卡顿
    #     - rgba: 点云颜色 (r,g,b,a)
    #     """
    #     if not self.real_robot:
    #         print("[PCD] real_robot=False，当前函数仅支持真机模式。")
    #         return
    #     if self.rs_pipeline is None or self.rs_align is None:
    #         print("[PCD] Camera pipeline not initialized.")
    #         return

    #     try:
    #         frames = self.rs_pipeline.wait_for_frames()
    #         aligned = self.rs_align.process(frames)
    #         depth_frame = aligned.get_depth_frame()
    #         color_frame = aligned.get_color_frame()
    #         if not depth_frame or not color_frame:
    #             print("[PCD] Failed to get depth/color frames.")
    #             return
    #     except Exception as e:
    #         print(f"[PCD] Camera grab failed: {e}")
    #         return

    #     # 生成相机坐标系下的点云
    #     pc = rs.pointcloud()
    #     pc.map_to(color_frame)
    #     points = pc.calculate(depth_frame)
    #     pcd_cam = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

    #     # 变换到世界坐标系
    #     given_conf = self.rm_controller.get_joint_values() if self.rm_controller else None
    #     try:
    #         pcd_world = self.transform_point_cloud_handeye(
    #             handeye_mat=self.HANDEYE,
    #             pcd=pcd_cam,
    #             given_conf=given_conf
    #         )
    #     except Exception as e:
    #         print(f"[PCD] Hand-eye transform failed: {e}")
    #         return

    #     # 简单下采样，减少点数
    #     if downsample_step > 1:
    #         pcd_world = pcd_world[::downsample_step]

    #     print(f"[PCD] Visualizing pointcloud, num_points={pcd_world.shape[0]}")
    #     # 使用几何模型生成点云，并挂到全局 world 上
    #     try:
    #         pcd_model = mgm.gen_pointcloud(pcd_world, rgba=rgba)
    #         pcd_model.attach_to(base)
    #     except Exception as e:
    #         print(f"[PCD] Failed to create pointcloud model: {e}")

    # def scan_and_detect(self):
    #     """
    #     获取图像并识别目标的世界坐标。
    #     **此方法已完全依赖 yolo_utils.py 中的函数**
    #     """
    #     # 真机：用 pyrealsense2 pipeline 取帧；仿真：返回 None 交由上层使用 mock 点
    #     if self.real_robot:
    #         if self.rs_pipeline is None or self.rs_align is None:
    #             print("Camera pipeline not initialized.")
    #             return None
    #         try:
    #             frames = self.rs_pipeline.wait_for_frames()
    #             aligned = self.rs_align.process(frames)
    #             depth = aligned.get_depth_frame()
    #             color = aligned.get_color_frame()
    #             if not depth or not color:
    #                 print("Failed to get depth/color frames.")
    #                 return None
    #             color_img = np.asanyarray(color.get_data())
    #             depth_img = np.asanyarray(depth.get_data())
    #         except Exception as e:
    #             print(f"Camera grab failed: {e}")
    #             return None
    #
    #         # 点云（如需）
    #         pc = rs.pointcloud()
    #         pc.map_to(color)
    #         points = pc.calculate(depth)
    #         pcd_raw = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    #
    #         given_conf = self.rm_controller.get_joint_values() if self.rm_controller else None
    #         pcd_world = yoloutils.transform_point_cloud_handeye(
    #             self, self.HANDEYE, pcd_raw, given_conf=given_conf
    #         )
    #
    #         # 这里调用 detect_target_point_manual_vision 仍依赖 self.camera，改为直接返回点云/图像，交由上层或自定义检测
    #         # 若仍想复用手动点击，可将 self.camera 替换为当前 pipeline 的数据源；此处简单返回 None 表示未检测
    #         print("Frames captured; please插入你的检测逻辑（YOLO或手动点选）。")
    #         return None
    #     else:
    #         print("Simulation mode: skip camera capture.")
    #         return None
    def scan_and_detect(self):
        """
        获取图像，通过YOLO识别目标（使用关键点检测），并返回目标在世界坐标系下的坐标 (x, y, z)。
        参考 Main_last.py 的处理方法，使用关键点而不是边界框中心。
        """
        if self.real_robot:
            # 1. 检查相机流
            if self.rs_pipeline is None or self.rs_align is None:
                print("[Error] Camera pipeline not initialized.")
                return None

            try:
                # 2. 获取对齐的帧
                frames = self.rs_pipeline.wait_for_frames()
                aligned = self.rs_align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()

                if not depth_frame or not color_frame:
                    print("[Error] Failed to get depth/color frames.")
                    return None

                color_img = np.asanyarray(color_frame.get_data())
                # depth_img = np.asanyarray(depth_frame.get_data()) # 暂时没用到，可注释

                height, width, _ = color_img.shape  # 获取图像宽高，用于计算索引

                # 获取深度内参（用于坐标转换）
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            except Exception as e:
                print(f"[Error] Camera grab failed: {e}")
                return None

            # 3. YOLO 推理
            print("--- Running YOLO Inference ---")
            # 使用 predict 方法，与 Main_last.py 保持一致
            results = self.yolo_model.predict(color_img, save=False, show_conf=False)

            # 4. 获取关键点结果
            keypoints_array = self.get_Keyresult(results)

            target_world_pos = None

            # 5. 处理关键点，转换为世界坐标
            if len(keypoints_array) > 0:
                # 选择第一个有效关键点（可见性为1）
                for keypoint in keypoints_array:
                    if round(keypoint[2]) == 1:  # 关键点可见性为1
                        ux, uy = int(keypoint[0]), int(keypoint[1])

                        # 添加边界检查
                        if not (0 <= ux < depth_intrin.width and 0 <= uy < depth_intrin.height):
                            print(f"[Warning] 坐标超出范围 ({ux}, {uy})")
                            continue  # 跳过无效坐标

                        # 获取深度值
                        dis = depth_frame.get_distance(ux, uy)
                        print(f"采摘点坐标（像素）：{ux}, {uy}, 深度：{dis}")

                        if dis > 0:  # 确保深度有效
                            # 将像素坐标和深度转换为相机坐标系下的3D坐标
                            camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis)
                            camera_xyz = np.array(camera_xyz)  # 转换为numpy数组

                            # 转换到世界坐标系（利用手眼矩阵和机械臂当前姿态）
                            given_conf = self.rm_controller.get_joint_values() if self.rm_controller else None

                            # 将单个点转换为点云格式（N, 3），以便使用 transform_point_cloud_handeye
                            camera_point_cloud = camera_xyz.reshape(1, 3)

                            # 使用手眼变换函数（考虑机械臂当前姿态）
                            # 注意：使用已绑定的类方法，而不是直接调用 yoloutils 的函数
                            world_point_cloud = self.transform_point_cloud_handeye(
                                handeye_mat=self.HANDEYE,
                                pcd=camera_point_cloud,
                                given_conf=given_conf
                            )
                            # mgm.gen_pointcloud(world_point_cloud)
                            # mgm.gen_sphere(pos=)
                            target_world_pos = world_point_cloud[0]  # 提取第一个点

                            print(f"目标世界坐标：{target_world_pos}")

                            # 6. 可视化检测结果
                            # 绘制检测结果（包含边界框和关键点）
                            im_array = results[0].plot()  # 绘制检测结果

                            # 在图像上标注关键点
                            cv2.circle(im_array, (ux, uy), 5, (0, 255, 0), -1)  # 绿色圆点标记关键点
                            cv2.putText(im_array, f"Depth: {dis:.3f}m", (ux + 10, uy - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # 显示检测结果
                            cv2.imshow('YOLO Detection', im_array)
                            cv2.waitKey(1)  # 非阻塞显示，等待1ms

                            return target_world_pos
                        else:
                            print(f"[Warning] 关键点 ({ux}, {uy}) 深度无效")
                            continue

                # 如果没有找到有效关键点
                if target_world_pos is None:
                    print("[Warning] 未找到有效的关键点（可见性为1）")
                    # 仍然显示检测结果（即使没有有效关键点）
                    if results and len(results) > 0:
                        im_array = results[0].plot()
                        cv2.imshow('YOLO Detection', im_array)
                        cv2.waitKey(1)
                    return None
            else:
                print("[Info] No target detected by YOLO.")
                # 显示原始图像（无检测结果）
                cv2.imshow('YOLO Detection', color_img)
                cv2.waitKey(1)
                return None

        else:
            print("Simulation mode: Using mock detection.")
            # 仿真模式下返回假数据
            return np.array([0.45, -0.1, 0.3])

    def pick_place_pepper(self,
                          approach_dis=0.05,
                          plane_normal=np.array([1.0, 0, 0]),
                          bound_lower=-40,
                          bound_upper=40,
                          grad=50,
                          viewpoint_joint=np.array([-0.08028514,
                                                    -1.05934503,
                                                    1.49318156,
                                                    0.08553859,
                                                    1.19422415,
                                                    -0.04305727]),
                          place_joint=np.array([1.57254163e+00, 6.96909976e-02, 1.92076978e+00, 1.71216801e-02, 7.77579056e-01, 2.96705989e-04]),
                          ):
        """
        执行一次完整的采摘任务，并使用时间戳记录本次采摘耗时。
        日志会以 CSV 形式追加写入当前目录下的 pepper_pick_log.txt：
        time_start, duration_sec, result
        """
        # 记录开始时间
        _task_start_time = time.time()
        _task_success = False
        robot = self.robot
        rrtc_planner = self.rrtc_planner

        # if self.real_robot and self.rm_controller is not None:
        #     self.rm_controller.move_j(joint_angles=np.degrees(viewpoint_joint))

        # 同步真机状态到仿真
        if self.real_robot and self.rm_controller is not None:
            start_conf = self.rm_controller.get_joint_values()
        else:
            start_conf = robot.get_jnt_values()

        robot.goto_given_conf(start_conf)

        # ------------------------------------------------------------------
        # 1. 移动到观测点 (Viewpoint)
        # ------------------------------------------------------------------
        print("--- Phase 1: Moving to Viewpoint ---")
        if np.linalg.norm(np.array(start_conf) - np.array(viewpoint_joint)) > 0.03:
            print(start_conf)
            mot_data_view = rrtc_planner.plan(start_conf=start_conf,
                                              goal_conf=np.degrees(viewpoint_joint),
                                              ext_dist=.1,
                                              max_time=300)
            if mot_data_view is not None:
                robot.goto_given_conf(viewpoint_joint)
                if self.real_robot:
                    self.rm_controller.move_jntspace_path(mot_data_view.jv_list,control_frequency=0.8)
                    time.sleep(3.0)
            else:
                print("[Error] Failed to plan to viewpoint.")
                return
        else:
            print("Already at viewpoint.")

        print("--- Phase 2: Detecting Target ---")
        tgt_pos = None

        if self.real_robot:
            tgt_pos = self.scan_and_detect()
            if tgt_pos is None:
                print("[Info] No pepper detected, retreating to pick pose...")

                self.retreat_to_pick_pose(pick_joint=np.degrees(viewpoint_joint))
                return
        else:
            tgt_pos = np.array([-0.45, -0.1, 0.3])
            print(f"[Sim] Using mock target: {tgt_pos}")

        # ------------------------------------------------------------------
        # 3. 搜索抓取姿态并规划路径 (IK Search Loop)
        # ------------------------------------------------------------------
        print(f"--- Phase 3: Searching Grasp Pose at {tgt_pos} ---")

        start_conf = robot.get_jnt_values()

        is_success = False
        final_approach_path = None
        final_linear_path = None

        for theta in range(bound_lower, bound_upper + 1, grad):
            hand_x = np.array([0, 0, -1])
            hand_z = np.array([-1, 0, 0])
            hand_y = np.cross(hand_z, hand_x)
            tgt_rotmat_base = np.array([hand_x, hand_y, hand_z]).T
            tgt_rotmat = rm.rotmat_from_axangle(hand_y, np.radians(theta)) @ tgt_rotmat_base
            mgm.gen_frame(tgt_pos, tgt_rotmat).attach_to(base)
            grasp_conf = robot.ik(tgt_pos=tgt_pos,
                                  tgt_rotmat=tgt_rotmat,
                                  seed_jnt_values=start_conf)
            if grasp_conf is not None:
                robot.goto_given_conf(grasp_conf)
                if not robot.is_collided():
                    # 预接近点
                    tgt_pos_approach = tgt_pos - hand_z * approach_dis
                    approach_conf = robot.ik(tgt_pos_approach, tgt_rotmat, seed_jnt_values=grasp_conf)

                    if approach_conf is not None:
                        robot.goto_given_conf(approach_conf)
                        if not robot.is_collided():
                            # RRT: Start -> Approach
                            mot_data_rrt = rrtc_planner.plan(start_conf=start_conf,
                                                             goal_conf=approach_conf,
                                                             obstacle_list=[self.shelf_model],
                                                             ext_dist=.1,
                                                             max_time=300)

                            if mot_data_rrt is not None:
                                # Linear: Approach -> Grasp
                                mot_data_linear = self.adp_planner.gen_linear_motion(
                                    start_conf=approach_conf,
                                    goal_conf=grasp_conf,
                                    granularity=0.01,
                                    obstacle_list=[self.shelf_model]
                                )

                                if mot_data_linear is not None:
                                    final_approach_path = mot_data_rrt
                                    final_linear_path = mot_data_linear
                                    self.vis_path(mot_data_rrt, rm.const.winter_map)
                                    self.vis_path(mot_data_linear, rm.const.wistia_map)
                                    is_success = True
                                    break
                                else:
                                    print(f"Theta {theta}: Linear motion failed.")
                            else:
                                print(f"Theta {theta}: RRT motion failed.")
                        else:
                            print(f"Theta {theta}: Approach conf collided.")
                    else:
                        print(f"Theta {theta}: Approach conf IK failed.")
                else:
                    print(f"Theta {theta}: Grasp conf collided.")
            else:
                pass

        # ------------------------------------------------------------------
        # 4. 规划放置路径和执行 (Planning Placement Path and Execution)
        # ------------------------------------------------------------------
        mot_data_place = None
        if is_success:
            print("--- Phase 4: Planning Placement Route ---")

            # 4.1 规划放置路径 (Planning the Placement Path - 发生在仿真和真机模式下)
            current_conf_for_place_planning = final_approach_path.jv_list[-1]
            robot.goto_given_conf(current_conf_for_place_planning)

            mot_data_place = rrtc_planner.plan(start_conf=current_conf_for_place_planning,
                                               goal_conf=place_joint,
                                               obstacle_list=[self.shelf_model],
                                               ext_dist=.1,
                                               max_time=300)

            if mot_data_place is not None:
                self.vis_path(mot_data_place, rm.const.copper_map)
                print("Placement path planned and visualized (Copper Map).")
            else:
                print("[Error] Failed to plan to place position.")
                is_success = False  # 放置路径规划失败，任务失败

        # 4.2 执行 (Execution)
        if is_success and mot_data_place is not None and self.real_robot:
            print("--- Executing Grasp and Place ---")
            pc = self.rm_controller

            # A. 打开夹爪
            print("Opening Gripper...")
            claw_controller.open_gripper()
            time.sleep(0.5)

            # 定义一个工具函数：依次执行 MoveJ 轨迹
            # def exec_path(path, speed):
            #     for conf in path:
            #         self.rm_controller.move_j(joint_angles=np.degrees(conf),espeed=sped)
            #         time.sleep(0.01)

            # B. 移动到预接近点 (RRT Path)
            print("Moving to Approach Point...")
            self.rm_controller.move_jntspace_path(final_approach_path.jv_list,control_frequency=0.8)
            time.sleep(2)
            # C. 直线插入 (Linear Path)
            print("Inserting to Grasp...")
            self.rm_controller.move_jntspace_path(final_linear_path.jv_list,control_frequency=0.8)
            time.sleep(2)
            # D. 闭合夹爪
            print("Closing Gripper...")
            claw_controller.close_gripper()
            time.sleep(1.0)

            # E. 直线撤退 (Reverse Linear Path)
            print("Retreating...")
            self.rm_controller.move_jntspace_path(final_linear_path.jv_list[::-1],control_frequency=0.8)
            time.sleep(2)
            # F. 移动到放置点 (Move to Place)
            print("Moving to Place...")
            self.rm_controller.move_jntspace_path(mot_data_place.jv_list,control_frequency=0.8)
            time.sleep(2)

            # G. 放下
            claw_controller.open_gripper()
            time.sleep(0.5)
            print("Task Completed.")

        elif is_success and mot_data_place is not None and not self.real_robot:
            print("Simulation Success.")
            # 在仿真中将机械臂移动到最终放置位置
            robot.goto_given_conf(place_joint)

        elif not is_success:
            print("Task Failed: No valid path found or placement path failed.")

        # ================= 时间统计与日志记录 =================
        _task_end_time = time.time()
        _duration = _task_end_time - _task_start_time
        # 认为只要规划和执行阶段标记为 is_success 且（真机或仿真）路径规划完整，即为成功
        if is_success:
            _task_success = True
        _ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_task_start_time))
        _result_str = "success" if _task_success else "fail"
        log_line = f"{_ts_str},{_duration:.3f},{_result_str}\n"
        try:
            with open("pepper_pick_log.txt", "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception as e:
            print(f"[Warn] Failed to write pick log: {e}")
        print(f"[Log] Pick task { _result_str }, duration: {_duration:.3f} s")

    def vis_path(self, mot_data, colormap):
        """辅助函数：可视化轨迹"""
        if mot_data:
            n = len(mot_data.mesh_list)
            for i, model in enumerate(mot_data.mesh_list):
                model.rgb = colormap(i / n)
                model.alpha = 0.3
                model.attach_to(base)

    # def execute_path_point_by_point(self, path, speed=5, is_radians=True):
    #     for jv in path:
    #         self.rm_controller.move_j(
    #             joint_angles=jv,
    #             speed=speed,
    #         )
    #     time.sleep(0.03)

    def retreat_to_pick_pose(self, pick_joint, obstacle_list=None, control_frequency=0.8):
        """
        没检测到辣椒时，把机械臂收回到“采摘点位姿”(你定义的 pick_joint)。
        - 真机：RRT规划 + move_jntspace_path 执行
        - 仿真：robot.goto_given_conf
        """
        if obstacle_list is None:
            obstacle_list = [self.shelf_model]

        robot = self.robot

        # 当前关节（真机优先）
        if self.real_robot and self.rm_controller is not None:
            start_conf = self.rm_controller.get_joint_values()
        else:
            start_conf = robot.get_jnt_values()

        # 规划回到采摘点位姿
        mot = self.rrtc_planner.plan(
            start_conf=start_conf,
            goal_conf=pick_joint,
            obstacle_list=obstacle_list,
            ext_dist=.1,
            max_time=300
        )

        if mot is None:
            print("[Error] Failed to plan retreat to pick pose.")
            # 兜底：仿真可直接goto；真机建议不要强行硬走
            if not self.real_robot:
                robot.goto_given_conf(pick_joint)
            return False

        # 执行
        robot.goto_given_conf(pick_joint)
        if self.real_robot and self.rm_controller is not None:
            self.rm_controller.move_jntspace_path(mot.jv_list, control_frequency=control_frequency)
            time.sleep(1.0)

        print("[Info] Retreated to pick pose.")
        return True


if __name__ == '__main__':
    from wrs.drivers.devices.realsense.realsense_d400s import RealSenseD405
    import wrs.Pepper.yoloutils.yolo_utils as yoloutils
    picker = RealmanPepperPicker(real_robot=True)
    robot = Realman()
    robot.gen_meshmodel(toggle_jnt_frames=True).attach_to(base)
    view_point = np.radians([-0.16688839, -1.59552764,  1.95066723,  0.09162979,  1.25496153, -0.07829547])
    place_point = np.array([1.57254163e+00, 6.96909976e-02, 1.92076978e+00, 1.71216801e-02, 7.77579056e-01, 2.96705989e-04])

    picker.pick_place_pepper(
        approach_dis=0.2,
        plane_normal=np.array([1, 0, 0]),
        bound_lower=-40,
        bound_upper=40,
        grad=3,
        viewpoint_joint=view_point,
        place_joint=place_point
    )
    base.run()
# if __name__ == '__main__':
#     import wrs.Pepper.yoloutils.yolo_utils as yoloutils
#     from wrs.drivers.devices.realsense.realsense_d400s import RealSenseD405
#
#     picker = RealmanPepperPicker(real_robot=True)  # ✅ 有 self.robot / self.rm_controller / self.HANDEYE
#
#     camera = RealSenseD405()
#     pcd_raw, pcd_color, _depth_img, color_img = camera.get_pcd_texture_depth()
#
#     given_conf = picker.rm_controller.get_joint_values() if picker.rm_controller else picker.robot.get_jnt_values()
#
#     pcd_r = picker.transform_point_cloud_handeye(
#         handeye_mat=picker.HANDEYE,
#         pcd=pcd_raw,
#         given_conf=given_conf
#     )
#     pcd_color_rgba = np.append(pcd_color, np.ones((len(pcd_color), 1)), axis=1)
#     mgm.gen_pointcloud(pcd_r,rgba=pcd_color_rgba).attach_to(base)
#     mgm.gen_sphere(pos=np.array([-0.438   ,  0.17446    , 0.40553]),radius=0.005,rgb=np.array([1,0,0])).attach_to(base)
#     base.run()
