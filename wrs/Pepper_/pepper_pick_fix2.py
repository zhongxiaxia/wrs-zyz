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
import wrs.Pepper.yoloutils.yolo_utils4 as yoloutils
from wrs.Pepper.Grapper.Grapper import ClawController


base = wd.World(cam_pos=rm.vec(2, 0, 1), lookat_pos=rm.vec(0, 0, 0.5))
mgm.gen_frame().attach_to(base)


DEFAULT_GRIPPER_COM = "COM3"  # 如现场端口不同，改这里
# claw_id = 9  # 夹爪ID
# claw_controller = ClawController(claw_id, com_port=DEFAULT_GRIPPER_COM)  # 创建夹爪控制器对象
# claw_controller.connect_gripper()  # 连接夹爪
# claw_controller.enable_gripper()  # 使能夹爪


class RealmanPepperPicker():
    def __init__(self, real_robot=True):
        # 绑定yolo_utils的手眼变换函数，让它成为类方法
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
        self.HANDEYE = np.array([[-0.013634785865167112, 0.9993155344887175, 0.03438829951851894, -0.09899999999999978], [-0.9990437250270998, -0.012186134398893706, -0.041989684593803374, 0.02599999999999998], [-0.04154188366317311, -0.03492793520631615, 0.9985260693860681, -0.027000000000000093], [0.0, 0.0, 0.0, 1.0]])

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

    def scan_and_detect(self):
        """
        获取图像，通过YOLO识别目标（使用关键点检测），只返回 2D 像素坐标 (u, v)。
        深度转换在外部单独处理，避免深度失败导致整条流程中断。
        """
        if not self.real_robot:
            print("Simulation mode: Using mock pixel.")
            return np.array([320, 240])  # 仿真兜底

        # 1. 检查相机流
        if self.rs_pipeline is None or self.rs_align is None:
            print("[Error] Camera pipeline not initialized.")
            return None

        try:
            # 2. 获取对齐帧
            frames = self.rs_pipeline.wait_for_frames()
            aligned = self.rs_align.process(frames)
            self._last_depth_frame = aligned.get_depth_frame()  # 缓存，供后续 3D 转换用
            color_frame = aligned.get_color_frame()

            if not color_frame:
                print("[Error] Failed to get color frame.")
                return None

            color_img = np.asanyarray(color_frame.get_data())
            self._last_color_frame = color_frame  # 缓存
            self._last_color_img = color_img

        except Exception as e:
            print(f"[Error] Camera grab failed: {e}")
            return None

        # 3. YOLO 推理
        print("--- Running YOLO Inference ---")
        results = self.yolo_model.predict(color_img, save=False, show_conf=False)

        # 4. 提取关键点
        keypoints_array = self.get_Keyresult(results)

        if len(keypoints_array) == 0:
            print("[Info] No keypoint detected by YOLO.")
            cv2.imshow("YOLO Detection", color_img)
            cv2.waitKey(1)
            return None

        # 只取第一个关键点
        u, v, conf = keypoints_array[0]

        if conf < 0.5:
            print(f"[Info] Low confidence keypoint: {conf:.2f}")
            cv2.imshow("YOLO Detection", results[0].plot())
            cv2.waitKey(1)
            return None

        u, v = int(u), int(v)
        print(f"[OK] YOLO 2D keypoint: ({u}, {v})")

        # 可视化
        im_array = results[0].plot()
        cv2.circle(im_array, (u, v), 5, (0, 255, 0), -1)
        cv2.imshow("YOLO Detection", im_array)
        cv2.waitKey(1)

        return np.array([u, v])  # ✅ 只返回像素坐标

    def pixel_to_world(self, u, v, neighborhood_size=1):
        """
        将 2D 像素坐标转换为世界坐标系下的 3D 点。
        直接使用点云矩阵，不依赖 get_distance()。
        返回 world_xyz (3,) 或 None（点云无效时）
        """
        depth_frame = getattr(self, '_last_depth_frame', None)
        color_frame = getattr(self, '_last_color_frame', None)

        if depth_frame is None or color_frame is None:
            print("[Error] No cached depth/color frame.")
            return None

        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        h, w = depth_intrin.height, depth_intrin.width

        # 边界检查
        if not (0 <= u < w and 0 <= v < h):
            print(f"[Warning] 像素越界 ({u}, {v})")
            return None

        # 直接生成点云矩阵 (H, W, 3)，不走 get_distance
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        pcd_cam = np.asanyarray(points.get_vertices()).view(
            np.float32
        ).reshape(h, w, 3)

        # 方案1：直接取该像素对应的点云坐标
        camera_xyz = pcd_cam[v, u]  # 注意：numpy 索引是 [row, col] = [v, u]

        # 检查是否有效（无效点通常是 [0,0,0]）
        if np.linalg.norm(camera_xyz) < 1e-6:
            # 方案2：邻域均值估计兜底
            print(f"[Info] 像素 ({u}, {v}) 点云无效，尝试邻域估计...")
            camera_xyz = yoloutils._estimate_point_from_neighborhood(
                target_pixel=(u, v),
                pcd_matrix=pcd_cam,
                neighborhood_size=neighborhood_size
            )

            if camera_xyz is None or np.any(np.isnan(camera_xyz)):
                print(f"[Warning] 邻域估计也失败 ({u}, {v})")
                return None

            camera_xyz = np.asarray(camera_xyz)

        print(f"采摘点相机坐标: {camera_xyz}")

        # 相机 → 世界坐标（手眼）
        given_conf = (
            self.rm_controller.get_joint_values()
            if self.rm_controller else None
        )

        camera_point_cloud = camera_xyz.reshape(1, 3)

        world_point_cloud = self.transform_point_cloud_handeye(
            handeye_mat=self.HANDEYE,
            pcd=camera_point_cloud,
            given_conf=given_conf
        )

        pcd_r = picker.transform_point_cloud_handeye(
            handeye_mat=picker.HANDEYE,
            pcd=pcd_cam.reshape(-1,3),
            given_conf=given_conf
        )
        color_image = np.asanyarray(color_frame.get_data())
        cw, ch = color_image.shape[:2][::-1]
        v, t = points.get_vertices(), points.get_texture_coordinates()
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
        np.clip(u, 0, ch - 1, out=u)
        np.clip(v, 0, cw - 1, out=v)
        pc_color = color_image[u, v] / 255
        pc_color[:, [0, 2]] = pc_color[:, [2, 0]]
        pcd_color_rgba = np.append(pc_color, np.ones((len(pc_color), 1)), axis=1)
        mgm.gen_pointcloud(pcd_r, rgba=pcd_color_rgba).attach_to(base)

        world_xyz = world_point_cloud[0]
        print(f"目标世界坐标: {world_xyz}")

        return world_xyz

    def select_target_manual(self, neighborhood_size=5):
        """
        手动选点模式：
        - 用 RealSense 拍一帧图像
        - 弹出大图让你用鼠标点击一个像素点
        - 将点击像素转换为世界坐标 (x, y, z) 并返回
        """
        if not self.real_robot:
            print("Simulation mode: manual select uses mock point.")
            return np.array([-0.45, -0.1, 0.3])

        # 1. 检查相机
        if self.rs_pipeline is None or self.rs_align is None:
            print("[Error] Camera pipeline not initialized.")
            return None

        try:
            # warm up: 丢掉前 N 帧，让自动曝光稳定
            for _ in range(30):
                frames = self.rs_pipeline.wait_for_frames()
            frames = self.rs_pipeline.wait_for_frames()
            aligned = self.rs_align.process(frames)
            self._last_depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not color_frame or not self._last_depth_frame:
                print("[Error] Failed to get depth/color frames.")
                return None

            color_img = np.asanyarray(color_frame.get_data())

            self._last_color_frame = color_frame
            self._last_color_img = color_img
        except Exception as e:
            print(f"[Error] Camera grab failed: {e}")
            return None

        # 2. 弹出图像，让你点击一个像素点
        try:
            pixels = yoloutils.pick_one_pixel_from_image(color_img)
        except Exception as e:
            print(f"[Error] Manual pixel selection failed: {e}")
            return None

        if not pixels:
            print("[Info] No pixel selected.")
            return None

        u, v = pixels[0]
        print(f"[Manual] Selected pixel: ({u}, {v})")

        # 3. 像素 → 世界坐标
        world_xyz = self.pixel_to_world(int(u), int(v), neighborhood_size=neighborhood_size)
        if world_xyz is None:
            print("[Info] Failed to convert manual pixel to world coordinate.")
            return None

        return world_xyz

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

        print("--- Phase 2: Selecting Target (Manual Click) ---")
        tgt_pos = None

        if self.real_robot:
            tgt_pos = self.select_target_manual()
            # tgt_pos[1] -= 0.05
            if tgt_pos is None:
                print("[Info] No valid target selected, retreating to pick pose...")
                self.retreat_to_pick_pose(pick_joint=np.degrees(viewpoint_joint))
                return
        else:
            tgt_pos = np.array([-0.45, -0.1, 0.3])
            print(f"[Sim] Using mock target: {tgt_pos}")

        # # ------------------------------------------------------------------
        # # 3. 搜索抓取姿态并规划路径 (IK Search Loop)
        # # ------------------------------------------------------------------
        # print(f"--- Phase 3: Searching Grasp Pose at {tgt_pos} ---")
        #
        # start_conf = robot.get_jnt_values()
        #
        # is_success = False
        # final_approach_path = None
        # final_linear_path = None
        #
        # for theta in range(bound_lower, bound_upper + 1, grad):
        #     hand_x = np.array([0, 0, -1])
        #     hand_z = np.array([-1, 0, 0])
        #     hand_y = np.cross(hand_z, hand_x)
        #     tgt_rotmat_base = np.array([hand_x, hand_y, hand_z]).T
        #     tgt_rotmat = rm.rotmat_from_axangle(hand_y, np.radians(theta)) @ tgt_rotmat_base
        #     mgm.gen_frame(tgt_pos, tgt_rotmat).attach_to(base)
        #     grasp_conf = robot.ik(tgt_pos=tgt_pos,
        #                           tgt_rotmat=tgt_rotmat,
        #                           seed_jnt_values=start_conf)
        #     if grasp_conf is not None:
        #         robot.goto_given_conf(grasp_conf)
        #         if not robot.is_collided():
        #             # 预接近点
        #             tgt_pos_approach = tgt_pos - hand_z * approach_dis
        #             approach_conf = robot.ik(tgt_pos_approach, tgt_rotmat, seed_jnt_values=grasp_conf)
        #
        #             if approach_conf is not None:
        #                 robot.goto_given_conf(approach_conf)
        #                 if not robot.is_collided():
        #                     # RRT: Start -> Approach
        #                     mot_data_rrt = rrtc_planner.plan(start_conf=start_conf,
        #                                                      goal_conf=approach_conf,
        #                                                      obstacle_list=[self.shelf_model],
        #                                                      ext_dist=.1,
        #                                                      max_time=300)
        #
        #                     if mot_data_rrt is not None:
        #                         # Linear: Approach -> Grasp
        #                         mot_data_linear = self.adp_planner.gen_linear_motion(
        #                             start_conf=approach_conf,
        #                             goal_conf=grasp_conf,
        #                             granularity=0.01,
        #                             obstacle_list=[self.shelf_model]
        #                         )
        #
        #                         if mot_data_linear is not None:
        #                             final_approach_path = mot_data_rrt
        #                             final_linear_path = mot_data_linear
        #                             self.vis_path(mot_data_rrt, rm.const.winter_map)
        #                             self.vis_path(mot_data_linear, rm.const.wistia_map)
        #                             is_success = True
        #                             break
        #                         else:
        #                             print(f"Theta {theta}: Linear motion failed.")
        #                     else:
        #                         print(f"Theta {theta}: RRT motion failed.")
        #                 else:
        #                     print(f"Theta {theta}: Approach conf collided.")
        #             else:
        #                 print(f"Theta {theta}: Approach conf IK failed.")
        #         else:
        #             print(f"Theta {theta}: Grasp conf collided.")
        #     else:
        #         pass
        #
        # # ------------------------------------------------------------------
        # # 4. 规划放置路径和执行 (Planning Placement Path and Execution)
        # # ------------------------------------------------------------------
        # mot_data_place = None
        # if is_success:
        #     print("--- Phase 4: Planning Placement Route ---")
        #
        #     # 4.1 规划放置路径 (Planning the Placement Path - 发生在仿真和真机模式下)
        #     current_conf_for_place_planning = final_approach_path.jv_list[-1]
        #     robot.goto_given_conf(current_conf_for_place_planning)
        #
        #     mot_data_place = rrtc_planner.plan(start_conf=current_conf_for_place_planning,
        #                                        goal_conf=place_joint,
        #                                        obstacle_list=[self.shelf_model],
        #                                        ext_dist=.1,
        #                                        max_time=300)
        #
        #     if mot_data_place is not None:
        #         self.vis_path(mot_data_place, rm.const.copper_map)
        #         print("Placement path planned and visualized (Copper Map).")
        #     else:
        #         print("[Error] Failed to plan to place position.")
        #         is_success = False  # 放置路径规划失败，任务失败
        #
        # # 4.2 执行 (Execution)
        # if is_success and mot_data_place is not None and self.real_robot:
        #     print("--- Executing Grasp and Place ---")
        #     pc = self.rm_controller
        #
        #     # A. 打开夹爪
        #     # print("Opening Gripper...")
        #     # claw_controller.open_gripper()
        #     # time.sleep(0.5)
        #
        #     # 定义一个工具函数：依次执行 MoveJ 轨迹
        #     # def exec_path(path, speed):
        #     #     for conf in path:
        #     #         self.rm_controller.move_j(joint_angles=np.degrees(conf),espeed=sped)
        #     #         time.sleep(0.01)
        #
        #     # B. 移动到预接近点 (RRT Path)
        #     print("Moving to Approach Point...")
        #     self.rm_controller.move_jntspace_path(final_approach_path.jv_list,control_frequency=0.8)
        #     time.sleep(2)
        #     # C. 直线插入 (Linear Path)
        #     print("Inserting to Grasp...")
        #     self.rm_controller.move_jntspace_path(final_linear_path.jv_list,control_frequency=0.8)
        #     time.sleep(2)
        #     # # D. 闭合夹爪
        #     # print("Closing Gripper...")
        #     # claw_controller.close_gripper()
        #     # time.sleep(1.0)
        #
        #     # E. 直线撤退 (Reverse Linear Path)
        #     print("Retreating...")
        #     self.rm_controller.move_jntspace_path(final_linear_path.jv_list[::-1],control_frequency=0.8)
        #     time.sleep(2)
        #     # F. 移动到放置点 (Move to Place)
        #     print("Moving to Place...")
        #     self.rm_controller.move_jntspace_path(mot_data_place.jv_list,control_frequency=0.8)
        #     time.sleep(2)
        #
        #     # G. 放下
        #     # claw_controller.open_gripper()
        #     # time.sleep(0.5)
        #     # print("Task Completed.")
        #
        # elif is_success and mot_data_place is not None and not self.real_robot:
        #     print("Simulation Success.")
        #     # 在仿真中将机械臂移动到最终放置位置
        #     robot.goto_given_conf(place_joint)
        #
        # elif not is_success:
        #     print("Task Failed: No valid path found or placement path failed.")

        # ================= 时间统计与日志记录 =================
        # _task_end_time = time.time()
        # _duration = _task_end_time - _task_start_time
        # # 认为只要规划和执行阶段标记为 is_success 且（真机或仿真）路径规划完整，即为成功
        # if is_success:
        #     _task_success = True
        # _ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_task_start_time))
        # _result_str = "success" if _task_success else "fail"
        # log_line = f"{_ts_str},{_duration:.3f},{_result_str}\n"
        # try:
        #     with open("pepper_pick_log.txt", "a", encoding="utf-8") as f:
        #         f.write(log_line)
        # except Exception as e:
        #     print(f"[Warn] Failed to write pick log: {e}")
        # print(f"[Log] Pick task { _result_str }, duration: {_duration:.3f} s")

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


# if __name__ == '__main__':
#     picker = RealmanPepperPicker(real_robot=True)
#     robot = Realman()
#     robot.gen_meshmodel(toggle_jnt_frames=True).attach_to(base)
#     view_point = np.radians([-0.16688839, -1.59552764,  1.95066723,  0.09162979,  1.25496153,
#        -0.07829547])
#     place_point = np.array([1.57254163e+00, 6.96909976e-02, 1.92076978e+00, 1.71216801e-02, 7.77579056e-01, 2.96705989e-04])
#
#     picker.pick_place_pepper(
#         approach_dis=0.2,
#         plane_normal=np.array([1, 0, 0]),
#         bound_lower=-40,
#         bound_upper=40,
#         grad=3,
#         viewpoint_joint=view_point,
#         place_joint=place_point
#     )
#     base.run()
if __name__ == '__main__':
    import wrs.Pepper.yoloutils.yolo_utils as yoloutils
    from wrs.drivers.devices.realsense.realsense_d400s import RealSenseD405

    picker = RealmanPepperPicker(real_robot=True)  # ✅ 有 self.robot / self.rm_controller / self.HANDEYE

    tgt_pos = picker.select_target_manual()
    #
    # camera = RealSenseD405(resoultion='mid')
    # pcd_raw, pcd_color, _depth_img, color_img = camera.get_pcd_texture_depth()
    # pcd_raw, pcd_color, _depth_img, color_img = camera.get_pcd_texture_depth()
    # pcd_raw, pcd_color, _depth_img, color_img = camera.get_pcd_texture_depth()


    given_conf = picker.rm_controller.get_joint_values() if picker.rm_controller else picker.robot.get_jnt_values()


    # mgm.gen_sphere(pos=np.array([-0.438   ,  0.17446    , 0.40553]),radius=0.005,rgb=np.array([1,0,0])).attach_to(base)
    mgm.gen_sphere(pos=tgt_pos, radius=0.005, rgb=np.array([1, 0, 0])).attach_to(base)
    base.run()
