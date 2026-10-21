from scene_assets import load_plant_scene, PROJECT_ROOT
import colorsys
from panda3d.core import (
    Geom, GeomNode, GeomPoints, GeomVertexFormat, GeomVertexData,
    GeomVertexWriter, LineSegs, OmniBoundingVolume, GeomEnums
)

import bbox
import TSDF
import raycast
from voxel_viz import VoxelGridVisualizer
import nbv_tsdf as nbv_mod

# ==============================================================================
# 点云可视化器
# ==============================================================================
class PointCloudVisualizer:
    def __init__(self, base, max_points=100000, point_size=4):
        self.max_points = max_points
        self.node = GeomNode('point_cloud')
        self.np = base.render.attachNewNode(self.node)
        self.np.setRenderModeThickness(point_size)
        self.np.setLightOff()
        self.node.setBounds(OmniBoundingVolume())
        self.node.setFinal(True)
        self.vdata = GeomVertexData('pc_data', GeomVertexFormat.getV3c4(), GeomEnums.UHDynamic)
        self.vdata.setNumRows(self.max_points)
        self.geom = Geom(self.vdata)
        self.points = GeomPoints(GeomEnums.UHStatic)
        self.geom.addPrimitive(self.points)
        self.node.addGeom(self.geom)
        self.writer_v = GeomVertexWriter(self.vdata, 'vertex')
        self.writer_c = GeomVertexWriter(self.vdata, 'color')

    def update(self, points_world, colors=None):
        num_new_points = len(points_world)
        if num_new_points > self.max_points: num_new_points = self.max_points; points_world = points_world[:self.max_points]
        if num_new_points == 0: self.points.clearVertices(); return
        self.writer_v.setRow(0); self.writer_c.setRow(0)
        for i, p in enumerate(points_world[:num_new_points]):
            self.writer_v.addData3f(p[0], p[1], p[2])
            if colors is not None: r, g, b, a = colors[i]
            else: r, g, b, a = 1.0, 1.0, 1.0, 1.0
            self.writer_c.addData4f(r, g, b, a)
        self.points.clearVertices()
        self.points.addNextVertices(num_new_points)

if __name__ == "__main__":
    import cv2
    import numpy as np
    import os
    import torch
    from ultralytics import YOLO
    from wrs import wd, rm, mgm
    from wrs.visualization.panda.panda3d_utils import VirtualDepthCamera, ExtraWindow
    

    # --- 设置场景 ---
    base = wd.World(cam_pos=[1, 0, 1], lookat_pos=[0, 0, .05], w=640, h=480)
    base.enableMouse()
    base.setBackgroundColor(1, 1, 1)

    # 加载模型
    # Render v1 semantic plant meshes; keep single-target ROI references hidden.
    obj1, obj2, plant_models = load_plant_scene(base)

    obj1_min, obj1_max = bbox.world_aabb_from_obj(obj1)
    obj2_min, obj2_max = bbox.world_aabb_from_obj(obj2)
    
    # --- 初始化 TSDF ---
    global_min = np.minimum(obj1_min, obj2_min) - 0.05
    global_max = np.maximum(obj1_max, obj2_max) + 0.05
    # 注意：TSDF voxel_size 建议小一点，比如 5mm
    voxel_size = 0.005 
    dims = np.ceil((global_max - global_min) / voxel_size).astype(int)
    dims = np.maximum(dims, 1)  # 防止 0
    
    tsdf_vol = TSDF.TSDFVolume(
        origin=global_min,
        voxel_size=voxel_size,
        dims=dims,
        trunc_margin=voxel_size * 5  # 截断距离
    )
    print(f"TSDF Initialized: {dims}")

    # --- 相机 ---
    resolution = (640, 480)
    vcam = VirtualDepthCamera(cam_pos=[0.4, 0, 0.65], lookat_pos=[0, 0, 0.4],
                              resolution=resolution, screen_size=np.array((0.05, 0.05)),
                              fov=45, depth_far=2.0)
    
    # # 让主窗口相机跟随 vcam，便于看到 NBV 移动
    # base.disableMouse()
    # def sync_main_cam():
    #     base.cam.setMat(vcam._pdnp.getMat())
    # sync_main_cam()
    
    # --- NBV Solver ---
    K = vcam.intrinsics
    img_size = (resolution[0], resolution[1])
    
    nbv_solver = nbv_mod.SimpleNBV(
        tsdf_vol=tsdf_vol,
        obj_min=obj1_min, obj_max=obj1_max,
        K=K, img_size=img_size,
        use_gpu=True, obj1_weight=2.0
    )

    # --- YOLO ---
    yolo_model_path = str(PROJECT_ROOT / "yolo11/runs/detect/train3/weights/best.pt")
    yolo_model = YOLO(yolo_model_path)
    yolo_class_names = ['pepper', 'stem']
    
    # --- 可视化 ---
    ew = ExtraWindow(base, cam_pos=[.5, 0, .5], lookat_pos=[0, 0, .03], w=640, h=480)
    ew.set_origin((np.array([0, 40])))
    pc_viz = PointCloudVisualizer(ew)
    
    vox_viz = VoxelGridVisualizer(base, window_origin=(640, 40), size=(640, 480),
                                  cam_pos=(0.5, -0.5, 0.3), lookat=(0, 0, 0.05),
                                  voxel_size=tsdf_vol.voxel_size, max_voxels=100000)

    # 轨迹可视化
    cam_trace_pts = [vcam.cam_pos.copy()]
    cam_path_node = [None]
    
    def _refresh_cam_trace():
        # 清理旧节点
        if cam_path_node[0] is not None:
            cam_path_node[0].removeNode()
            cam_path_node[0] = None
        
        if len(cam_trace_pts) == 0:
            return
        
        # 画路径
        ls = LineSegs("cam_path")
        ls.setThickness(2.0)
        ls.setColor(1.0, 0.5, 0.0, 0.8)  # 橙色半透明
        first = True
        for p in cam_trace_pts:
            if first:
                ls.moveTo(p[0], p[1], p[2])
                first = False
            else:
                ls.drawTo(p[0], p[1], p[2])
        cam_path_node[0] = base.render.attachNewNode(ls.create())
    
    _refresh_cam_trace()

    # --- 辅助功能 ---
    def detect_with_yolo(img_rgb):
        if img_rgb is None: return img_rgb, []
        img_vis = img_rgb.copy()
        results = yolo_model(img_rgb[:,:,:3], conf=0.25, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0])
                cls_name = yolo_class_names[cls] if cls < len(yolo_class_names) else str(cls)
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                detections.append({'class': cls_name, 'bbox': [x1, y1, x2, y2]})
        return img_vis, detections

    # --- 辅助函数：生成 LookAt 旋转矩阵 ---
    def _look_at(cam_pos, target):
        cam_pos = np.asarray(cam_pos, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        forward = target - cam_pos
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)
        return np.column_stack([right, forward, up]).astype(np.float32)

    # --- TSDF 版 NBV 策略（与 policy.NBVPolicy 接口相似） ---
    class NBVPolicyTSDF:
        def __init__(self, tsdf_vol, obj_min, obj_max, K, img_size, vcam):
            self.tsdf_vol = tsdf_vol
            self.obj_min = np.asarray(obj_min, dtype=np.float32)
            self.obj_max = np.asarray(obj_max, dtype=np.float32)
            self.vcam = vcam
            self.nbv_solver = nbv_mod.SimpleNBV(
                tsdf_vol=self.tsdf_vol,
                obj_min=self.obj_min,
                obj_max=self.obj_max,
                K=K.astype(np.float32),
                img_size=img_size,
                use_gpu=True,
                obj1_weight=2.0,
            )

        def _generate_candidate_views(self, center, radius=0.35, n_az=16):
            center = np.asarray(center, dtype=np.float32)
            cam_poses = []
            for k in range(n_az):
                phi = 2.0 * np.pi * k / n_az
                cam_pos = np.array([
                    center[0] + radius * np.cos(phi),
                    center[1] + radius * np.sin(phi),
                    center[2] + 0.0,
                ], dtype=np.float32)
                R_c2w = _look_at(cam_pos, center)
                cam_poses.append((cam_pos, R_c2w))
            return cam_poses

        def plan_and_move(self, radius=0.35):
            target_center = 0.5 * (self.obj_min + self.obj_max)
            cam_poses = self._generate_candidate_views(target_center, radius=radius, n_az=16)
            best_idx, best_gain = self.nbv_solver.best_view(cam_poses)
            if best_idx is None:
                print("[NBV] 没有可用候选视角")
                return None, 0.0
            best_pos, best_rot = cam_poses[best_idx]
            print(f"[NBV] best gain = {best_gain:.4f}, pos = {best_pos}")
            self.vcam.cam_pos = best_pos
            self.vcam.cam_rotmat = best_rot
            return best_pos, best_gain

    # 创建 TSDF 版策略实例，接口与 policy.NBVPolicy 类似
    nbv_policy = NBVPolicyTSDF(tsdf_vol, obj1_min, obj1_max, K, img_size, vcam)

    # --- NBV 执行函数 ---
    def on_find_nbv():
        print("m pressed")
        best_pos, gain = nbv_policy.plan_and_move(radius=0.35)
        if best_pos is not None:
            cam_trace_pts.append(best_pos)
            _refresh_cam_trace()

    base.accept("m", on_find_nbv)

    # Create the visualization of the camera (Frustum)
    # We will update its pose in the loop
    vcam_viz_node = [vcam.gen_meshmodel(alpha=0.5)]
    vcam_viz_node[0].attach_to(base)

    # --- 主循环 ---
    def update(task):
        # Update Visualization
        vcam_viz_node[0].detach()
        vcam_viz_node[0] = vcam.gen_meshmodel(alpha=0.5)
        vcam_viz_node[0].attach_to(base)
        
        # 1. 获取图像
        img_rgb = vcam.get_rgb_image()
        img_depth = vcam.get_depth_image() # (H, W) 米
        
        # 2. YOLO 检测并生成 2D 语义 Mask
        img_vis, detections = detect_with_yolo(img_rgb)
        
        # 构建语义概率图 (0~1)
        sem_mask = np.zeros(img_depth.shape, dtype=np.float32)
        for det in detections:
            if det['class'] == 'pepper':
                x1, y1, x2, y2 = det['bbox']
                # 简单处理：框内概率为 0.9
                # 边界检查
                x1, x2 = max(0, x1), min(img_depth.shape[1], x2)
                y1, y2 = max(0, y1), min(img_depth.shape[0], y2)
                sem_mask[y1:y2, x1:x2] = 0.9 

        # 3. TSDF 融合 [核心修改]
        # 构造 4x4 相机姿态矩阵
        cam_pose = np.eye(4)
        # integrate() projects OpenCV [right, down, forward] coordinates;
        # VirtualDepthCamera supplies Panda [right, forward, up] columns.
        panda_rot = vcam.cam_rotmat
        cam_pose[:3, :3] = np.column_stack(
            [panda_rot[:, 0], -panda_rot[:, 2], panda_rot[:, 1]])
        cam_pose[:3, 3] = vcam.cam_pos
        
        # 融合一帧
        tsdf_vol.integrate(img_depth, vcam.intrinsics, cam_pose, sem_img=sem_mask)

        # 4. 可视化 TSDF
        # 提取表面点云 (weight > 0 且 near surface)
        pts, probs = tsdf_vol.get_point_cloud(filter_val=tsdf_vol.voxel_size * 2) 
        
        # 在 ExtraWindow 显示点云
        if len(pts) > 0:
            # 根据语义上色：prob > 0.5 (pepper) -> Red, else -> Green
            colors = np.zeros((len(pts), 4))
            is_pepper = probs > 0.5
            colors[is_pepper] = [1, 0, 0, 1]
            colors[~is_pepper] = [0, 1, 0, 1]
            pc_viz.update(pts, colors)
        
        # 在 Voxel Window 显示体素 (为了 debugging)
        # 这里我们只采样一部分或者显示 Weight = 0 的空洞
        # TSDF 也可以像 Occupancy Grid 一样画，但为了性能，我们这里画表面
        if len(pts) > 0:
             # 为了 VoxelViz，我们需要 rgba 颜色
             vox_colors = colors # 复用上面的颜色
             # 下采样一点以免太卡
             skip = 1
             vox_viz.update(pts[::skip], vox_colors[::skip])
        else:
             vox_viz.clear()

        # 5. 显示 RGB 和 Depth
        cv2.imshow("RGB + YOLO", img_vis)
        d_vis = (img_depth / 2.0 * 255).astype(np.uint8)
        cv2.imshow("Depth", cv2.applyColorMap(d_vis, cv2.COLORMAP_JET))
        
        key = cv2.waitKey(1)
        return task.cont

    base.taskMgr.add(update, "update_loop")
    
    # 键盘控制 (手动微调)
    move_speed = 0.05
    rot_speed = np.radians(5)

    def move_cam(dx=0, dy=0, dz=0):
        vcam.cam_pos = vcam.cam_pos + np.array([dx, dy, dz])
        print(f"[Camera] pos = {vcam.cam_pos}, rotmat = \n{vcam.cam_rotmat}")
        sync_main_cam()

    def yaw_cam(delta):
        vcam.cam_rotmat = rm.rotmat_from_euler(0, 0, delta) @ vcam.cam_rotmat
        print(f"[Camera] pos = {vcam.cam_pos}, rotmat = \n{vcam.cam_rotmat}")
        sync_main_cam()

    base.accept("w", lambda: move_cam(0,  move_speed, 0))
    base.accept("s", lambda: move_cam(0, -move_speed, 0))
    base.accept("a", lambda: move_cam(-move_speed, 0, 0))
    base.accept("d", lambda: move_cam( move_speed, 0, 0))
    base.accept("q", lambda: move_cam(0, 0,  move_speed))   # 上升
    base.accept("e", lambda: move_cam(0, 0, -move_speed))   # 下降
    base.accept("arrow_left",  lambda: yaw_cam(rot_speed))
    base.accept("arrow_right", lambda: yaw_cam(-rot_speed))

    print("Running... Press 'm' to trigger NBV calculation.")
    base.run()
