# import cv2
# import numpy as np
# import time
# import pyrealsense2 as rs
#
# class RLtime:
#     def __init__(self,picker, target_distance_thresh = 0.8,yaw_offset_deg = 20.0):
#         self.picker = picker
#         self.target_distance_thresh = target_distance_thresh
#         self.yaw_offset_deg = yaw_offset_deg
#         self.last_best_view_idx = 0
#
#     def capture_view_confidences(self):
#
#         def _jnts_within_limits()
#             jnts_rad = np
#
#
#     conf_select = np.zeros(3)
#     joint_slect = np.zeros((6,3))
#     tcp_select = np.zeros((6,3))
#
#     rm_controller = self.picker.rm_controller
#
#     if self.picker.rs_pipeline is not None and self.picker.rs_align is not None:
#         try:
#             time.sleep(0.3)
#             frames = self.picker.rs_pipeline.wait_for_frames()
#
#             aligned = self.picker.rs_pipeline.align(frames)
#             depth_frame = aligned.get_depth_frame()
#             color_frame = aligned.get_color_frame()
#
#             if color_frame and depth_frame:
#                 color_img = np.asanyarray(color_frame.get_data())
#                 depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
#
#                 pc = rs.pointcloud()
#                 pc.map_to(color_frame)
#                 points = pc.calculate(depth_frame)
#                 pcd_cam = np.asanyarray(points.get_vertices()).view(np.float32).reshape(depth_intrin.height, depth_intrin.width, depth_intrin.height,depth_intrin.width,3)
#
#                 results = self.picker.yolo_model.predict(color_img,save=False,show_conf=False)
#                 try:
#                     if results:
#                         vis_img = results[0].plot()
#                         cv2.imshow('vis_img',vis_img)
#                     else:
#                         cv2.imshow('vis_img',color_img)
#                     cv2.waitKey(1)
#                 except Exception as e:
#                     print(f"{e}")
#
#                 if results and results[0].boxes is not None and len(results[0].boxes) > 0:
#                     res = results[0]
#                     boxes = res.boxes
#                     kpts_all = res.keypoints
#
#                     best_conf = 0.0
#                     min_distance = float('inf')
#
#                     for i in range(len(boxes)):
#                         conf = float(boxes.conf[i].item())
#                         if conf < 0.3:
#                             continue
#
#                         if kpts_all
#
