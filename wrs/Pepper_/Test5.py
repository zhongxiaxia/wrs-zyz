
# 不同线程调用
import threading
import time
import queue
import sys
import keyboard  # 新增：用于监听键盘事件

# 线程停止标志
stop_thread = False

# 假设YOLO检测方法（这里只是一个模拟）
def yolo_detection(camera_data, output_queue, event):
    global stop_thread
    while not stop_thread:
        # 模拟相机捕获图像
        print("相机正在捕获图像并进行YOLO检测...")
        time.sleep(2)  # 模拟每2秒钟捕获一帧图像
        
        # 假设通过YOLO检测到目标坐标
        detected_coordinates = (100, 150)  # 示例坐标
        print(f"检测到目标坐标: {detected_coordinates}")
        
        # 将坐标数据传送到队列，供机械臂使用
        output_queue.put(detected_coordinates)
        
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
            time.sleep(3)  # 模拟机械臂的移动时间
            
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

        if keyboard.is_pressed('a')  :
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