# 不同线程调用
import threading
import time
import queue
import sys
import keyboard  # 新增：用于监听键盘事件

import time
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from wrs.robot_sim.manipulators.realman_arm.realman_planning_peppershelf import PepperPicker

# 相机状态定义
CamState = 1  # 默认启动后为采集图像状态
pepperList = []  # 存储识别到的目标信息列表
# 线程停止标志
stop_thread = False




# 用来存储关键点数据的数组
allkeypoints_array = []
place_joint=np.array([1.57254163e+00, 6.96909976e-02, 1.92076978e+00, 1.71216801e-02, 7.77579056e-01, 2.96705989e-04])
basket_position = place_joint
# 加载YOLOv8模型
model = YOLO(r"K:\BaiduSyncdisk\HYKPro\Project\SLPepper\best.pt")


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
    # img_color = r"C:\Users\Administrator\Desktop\rgb_20241111_160706.jpg"
    results = model.predict(img_color, save=False, show_conf=False)
    keypoints_array = get_Keyresult(results)

    # detections 用来存储目标信息
    detections = []
    for keypoint in keypoints_array:
        if round(keypoint[2]) == 1:
            ux, uy = int(keypoint[0]), int(keypoint[1])
            dis = aligned_depth_frame.get_distance(ux, uy)
            print("采摘点坐标：" + str(ux) + "," + str(uy) + "," + str(dis))
            # camera_xyz 是相机坐标系下的坐标
            camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis)
            camera_xyz = np.round(np.array(camera_xyz), 3) * 1000  # 转换为毫米并保留三位小数
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
                tgt_pos = np.array(rob_xyz)
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
                # PepperPicker.pick_place_pepper(tgt_pos, basket_position)

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

################################
# 假设YOLO检测方法（这里只是一个模拟）
def yolo_detection(camera_data, output_queue, event):
    global stop_thread

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


    while not stop_thread:
        # 模拟相机捕获图像
        depth_intrin, img_color, aligned_depth_frame = get_aligned_images(pipeline, align)
        # CamState=0
        # 处理图像，返回检测结果和处理后的图像
        # detections是检测结果列表， im_array是处理后的图像
        detections, im_array, rob_xyz = process_image(depth_intrin, img_color, aligned_depth_frame)
        tgt_pos = np.array(rob_xyz)
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
        print("发送数据给机械臂")
        # 将坐标数据传送到队列，供机械臂使用
        output_queue.put(rob_xyz)



        # 等待回车键按下【等待机械臂返回信息后继续处理下一帧图像】
        key = cv2.waitKey(0)  # 等待按键
        if key == 13:  # 回车键对应的键值为 13
            continue  # 按下回车后继续处理下一帧图像
        if key == 27:  # ESC键对应的键值为 27
            # 退出程序
            cv2.destroyAllWindows()
            break


        # 等待机械臂完成任务（等待信号）
        event.wait()

        # 机械臂完成任务后，重置事件，准备下一轮捕获
        event.clear()
        print("相机准备捕获下一帧图像...")


# 模拟机械臂控制
def robotic_arm_control(input_queue, event):
    global stop_thread
    while not stop_thread:
        if not input_queue.empty():
            coordinates = input_queue.get()  # 获取相机传输过来的目标坐标
            print(f"机械臂接收到坐标: {coordinates}")

            # 检查是否要退出
            if stop_thread:
                break

            # 模拟机械臂控制
            print(f"机械臂正在移动到坐标 {coordinates} ...")

            # 机械臂调用的方法，需要调整
            PepperPicker.pick_place_pepper(coordinates, basket_position)


            # 完成后，通知相机捕获下一帧
            event.set()
            print("机械臂完成任务，通知相机继续捕获下一帧图像")
        else:
            print("等待接收目标坐标...")
            time.sleep(1)


# 监听按键函数
def listen_for_exit_key():
    global stop_thread
    try:
        # 按下 'q' 键时退出
        keyboard.wait('q')  # 用户按下 q 键时触发
        print("\n检测到按键 'q'，正在终止所有线程...")
        stop_thread = True

        if keyboard.is_pressed('a'):
            print("检测到按键 'space'，正在终止所有线程...")
    except:
        pass


if __name__ == '__main__':
    # 创建线程通信的队列
    coordinate_queue = queue.Queue()

    # 创建控制信号的事件
    event = threading.Event()

    # 创建相机线程，负责YOLO目标检测
    camera_thread = threading.Thread(target=yolo_detection, args=("Camera", coordinate_queue, event))

    # 创建机械臂线程，负责控制机械臂的运动
    arm_thread = threading.Thread(target=robotic_arm_control, args=(coordinate_queue, event))

    # 创建监听按键的线程
    key_listener_thread = threading.Thread(target=listen_for_exit_key)

    # 启动线程
    camera_thread.start()
    arm_thread.start()
    key_listener_thread.start()

    # 启动时，允许相机线程先捕获图像
    event.set()

    # 主线程等待子线程完成
    camera_thread.join()
    arm_thread.join()
    key_listener_thread.join()

    print("程序已终止。")