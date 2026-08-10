'''
@ Time: 2023/6/4 15:09
@ Author: Huang Yikun
@ Tile:
'''
# 从Realsense 获取一帧图像并使用YOLO识别目标，返回目标的坐标信息
# 按下回车后才进行下一帧


import time
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import json
from wrs.robot_sim.manipulators.realman_arm.realman_planning_peppershelf import PepperPicker

# 相机状态定义
CamState = 1  # 默认启动后为采集图像状态
pepperList = []  # 存储识别到的目标信息列表
# 用来存储关键点数据的数组
allkeypoints_array = []
place_joint = np.array([1.57254163e+00, 6.96909976e-02, 1.92076978e+00, 1.71216801e-02, 7.77579056e-01, 2.96705989e-04])
basket_position = place_joint
# 加载YOLOv8模型
model = YOLO(r"F:\WRS\wrs-virtual\wrs\Pepper\best.pt")


def load_calibration_matrix_from_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    mat_list = data.get('affine_mat', None)
    if mat_list is None:
        raise ValueError("JSON 文件中没有找到 'affine_mat' 字段")
    return np.array(mat_list)


# 获取对齐的RGB-D图像
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


def get_Keyresult(results):
    keypoints_array = []
    target_class_id = 0
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

                kpts = keypoints[i].data[0].cpu().numpy()
                for j, kpt in enumerate(kpts):
                    print(f"关键点 {j}: X={kpt[0]:.2f}, Y={kpt[1]:.2f}, 可见={kpt[2]:.4f}")
                    if j == 0:
                        # 将关键点数据添加到数组中
                        keypoints_array.append([kpt[0], kpt[1], kpt[2]])
    # # 输出存储的关键点数据
    # print("\n存储的关键点数据：")
    # for kpt in keypoints_array:
    #     print(f"X={kpt[0]:.2f}, Y={kpt[1]:.2f}, Z={kpt[2]:.4f}")
    return keypoints_array


# 处理图像
def process_image(depth_intrin, img_color, aligned_depth_frame):
    rob_xyz = []
    """处理单张图像，包括目标检测与坐标转换"""
    # img_color=r"C:\Users\Administrator\Desktop\rgb_20241111_160706.jpg"
    results = model.predict(img_color, save=False, show_conf=False)
    keypoints_array = get_Keyresult(results)

    # detections 用来存储目标信息
    detections = []
    for keypoint in keypoints_array:
        if round(keypoint[2]) == 1:

            ux, uy = int(keypoint[0]), int(keypoint[1])
            # 添加边界检查
            if not (0 <= ux < depth_intrin.width and 0 <= uy < depth_intrin.height):
                print(f"警告：坐标超出范围 ({ux}, {uy})")
                continue  # 跳过无效坐标

            dis = aligned_depth_frame.get_distance(ux, uy)
            print(dis)
            print("采摘点坐标：" + str(ux) + "," + str(uy) + "," + str(dis))
            # camera_xyz 是相机坐标系下的坐标  xyz
            camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis)
            camera_xyz = np.round(np.array(camera_xyz), 3)  # 转换为毫米并保留三位小数
            # 这里要修改
            # rob_xyz = camera_xyz
            rob_xyz.append(camera_xyz)
    # 原来
    for result in results:
        boxes = result.boxes.xywh.tolist()
        # boxes = result.boxes
        # plot 是绘制目标框的函数
        im_array = result.plot()

        # 获取所有检测框的类别编号（整数形式）
        class_ids = result.boxes.cls.tolist()  # 返回浮点数列表
        class_ids = [int(x) for x in class_ids]  # 转换为整数

        # # 遍历每个目标
        # for i in range(len(boxes)):
        #     ux, uy = int(boxes[i][0]), int(boxes[i][1])
        #     dis = aligned_depth_frame.get_distance(ux, uy)
        #     # camera_xyz 是相机坐标系下的坐标
        #     camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis)
        #     camera_xyz = np.round(np.array(camera_xyz), 3) * 1000  # 转换为毫米并保留三位小数

        #     # 假设类别信息存储于result.names中
        #     label = result.names[int(result.boxes.cls[i])]

        #     if label == 'PepperTOP':
        #         detections.append({'label': label, 'coordinates': camera_xyz})
        #     elif label == 'Pepper':
        #         # 对于Pepper仅记录，不做其他操作
        #         pass
    print(rob_xyz)
    return detections, im_array, rob_xyz

    # 备份
    # for result in results:
    #     boxes = result.boxes.xywh.tolist()
    #     # plot 是绘制目标框的函数
    #     im_array = result.plot()
    #     # 遍历每个目标
    #     for i in range(len(boxes)):
    #         ux, uy = int(boxes[i][0]), int(boxes[i][1])
    #         dis = aligned_depth_frame.get_distance(ux, uy)
    #         # camera_xyz 是相机坐标系下的坐标
    #         camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis)
    #         camera_xyz = np.round(np.array(camera_xyz), 3) * 1000  # 转换为毫米并保留三位小数

    #         # 假设类别信息存储于result.names中
    #         label = result.names[int(result.boxes.cls[i])]

    #         if label == 'PepperTOP':
    #             detections.append({'label': label, 'coordinates': camera_xyz})
    #         elif label == 'Pepper':
    #             # 对于Pepper仅记录，不做其他操作
    #             pass

    # return detections, im_array


def Camear_main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 启动流
    pipe_profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    time.sleep(2)  # 等待相机初始化

    frame_buffer = []
    detection_buffer = []
    try:
        while True:
            if CamState == 1:
                depth_intrin, img_color, aligned_depth_frame = get_aligned_images(pipeline, align)
                # CamState=0
                # 处理图像，返回检测结果和处理后的图像
                # detections是检测结果列表， im_array是处理后的图像
                detections, im_array, rob_xyz = process_image(depth_intrin, img_color, aligned_depth_frame)
                if len(rob_xyz) > 0:  # 确保至少有一个点被检测到
                    tgt_pos = np.array(rob_xyz[0])
                    # tgt_pos = rob_xyz[0]  # 选取第一个检测到的点
                    print(f"选择了坐标点(相机坐标系): {rob_xyz[0].tolist()}")  # 如果是三维点，输出类似 [0.5, 1.2, 0.7]
                else:
                    continue

                # 不再使用帧缓存和计算平均值
                # 直接处理当前帧的检测结果
                for det in detections:
                    label = det['label']
                    # coordinates是检测到的目标坐标，(x,y,z)
                    coordinates = det['coordinates']
                    print(f"{label} Detected at Coordinates: {coordinates}")

                    # 如果是PepperTOP，则更新pepperList
                    if label == 'PepperTOP':
                        pepperList.append(coordinates)

                cv2.imshow('detection', im_array)
                # cv2.waitKey(1)

                #  通知机械臂
                # rob_xyz, 
                print("通知机械臂")
                print(rob_xyz)
                # 加载初始标定矩阵
                init_mat = load_calibration_matrix_from_json("manual_calibration_realman.json")  # 机械臂末端到相机坐标矩阵
                picker = PepperPicker(init_mat, use_pytracik=True)  # 启用 pytracik IK 求解器
                # picker = PepperPicker(init_mat)

                # 调试：打印当前末端位姿（世界系）和即将到达的目标点
                # try:
                #     cur_pos, cur_rot = picker.realman_controller.get_pose()
                #     tgt_pos_world = picker.align_pcd(tgt_pos)
                #     print(f"[INFO] 当前末端位置 (world): {cur_pos}")
                #     print(f"[INFO] 目标位置 (world, from cam_tgt_pos): {tgt_pos_world}")
                # except Exception as e:
                #     print(f"[WARN] 无法获取当前末端/目标位姿: {e}")

                picker.pick_place_pepper(cam_tgt_pos=tgt_pos,
                                         approach_dis=.15,
                                         bound_lower=-30,
                                         bound_upper=30,
                                         grad = 2,
                                         place_joint=basket_position,
                                         viewpoint_joint=np.array([-0.08028514,
                                                                   -1.05934503,
                                                                   1.49318156,
                                                                   0.08553859,
                                                                   1.19422415,
                                                                   -0.04305727]))

                # 等待回车键按下【等待机械臂返回信息后继续处理下一帧图像】
                key = cv2.waitKey(0)  # 等待按键
                if key == 13:  # 回车键对应的键值为 13
                    continue  # 按下回车后继续处理下一帧图像
                if key == 27:  # ESC键对应的键值为 27
                    # 退出程序
                    cv2.destroyAllWindows()
                    break
    finally:
        pipeline.stop()


if __name__ == '__main__':
    CamState = 1
    Camear_main()
