from scene_assets import load_plant_scene, PROJECT_ROOT
import colorsys
from panda3d.core import (
    Geom, GeomNode, GeomPoints, GeomVertexFormat, GeomVertexData,
    GeomVertexWriter, LineSegs
)

import bbox
import voxel_grid  
import policy
from voxel_viz import VoxelGridVisualizer, uncertainty_to_rgba

def heat_map_color(z, min_z=0.0, max_z=0.3):
    """Generates a rainbow color based on height."""
    # Normalize z
    t = (z - min_z) / (max_z - min_z)
    t = np.clip(t, 0, 1)
    # Hue: 0.0 (Red) to 0.66 (Blue).
    # Let's go Blue(0.66) -> Red(0.0) for height (Standard heatmap)
    hue = (1.0 - t) * 0.66
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return r, g, b, 1.0


from panda3d.core import OmniBoundingVolume, GeomEnums


class PointCloudVisualizer:
    def __init__(self, base, max_points=100000, point_size=4):
        self.max_points = max_points

        # 1. Create Node and disable culling
        self.node = GeomNode('point_cloud')
        self.np = base.render.attachNewNode(self.node)
        self.np.setRenderModeThickness(point_size)
        self.np.setLightOff()

        # Force "Infinite" bounds so it never disappears
        self.node.setBounds(OmniBoundingVolume())
        self.node.setFinal(True)

        # 2. Setup Data Container (Fixed Size Pool)
        # usage=GeomEnums.UHDynamic ensures the GPU expects frequent updates
        self.vdata = GeomVertexData('pc_data', GeomVertexFormat.getV3c4(), GeomEnums.UHDynamic)

        # CRITICAL: Pre-allocate the maximum size ONCE.
        # We will never resize this buffer again. Resizing is slow.
        self.vdata.setNumRows(self.max_points)

        # 3. Setup Primitive (The "Drawer")
        self.geom = Geom(self.vdata)
        self.points = GeomPoints(GeomEnums.UHStatic)  # The indices don't change often, the data does
        self.geom.addPrimitive(self.points)
        self.node.addGeom(self.geom)

        # 4. Create Writers (Keep these persistent)
        self.writer_v = GeomVertexWriter(self.vdata, 'vertex')
        self.writer_c = GeomVertexWriter(self.vdata, 'color')

    def update(self, points_world, colors=None):
        """
        Updates the point cloud geometry.
        points_world: (N, 3) numpy array
        colors: (N, 4) numpy array, 每个点的 RGBA，可选
        """
        num_new_points = len(points_world)
        # Safety Check
        if num_new_points > self.max_points:
            print(f"Warning: Point cloud truncated! {num_new_points} > {self.max_points}")
            num_new_points = self.max_points
            points_world = points_world[:self.max_points]
        if num_new_points == 0:
            # If no points, just clear the drawing instruction
            self.points.clearVertices()
            return
        # 1. Reset Writers to the beginning of the buffer
        self.writer_v.setRow(0)
        self.writer_c.setRow(0)
        # 2. Write Data (Only the active points)
        # This overwrites the first N rows of the buffer.
        # Rows N to Max are left as stale garbage (we just won't draw them).
        for i, p in enumerate(points_world[:num_new_points]):
            self.writer_v.addData3f(p[0], p[1], p[2])

            if colors is not None:
                r, g, b, a = colors[i]
            else:
                # 原来的按高度着色
                z_norm = (p[2] - 0.0) / 0.2
                z_norm = max(0.0, min(1.0, z_norm))
                r, g, b, a = z_norm, 0.0, 1.0 - z_norm, 1.0
                
            self.writer_c.addData4f(r, g, b, a)
        # 3. Update the Drawing Primitive
        # CRITICAL: We clear the "Draw List" and tell it to only draw indices 0 to N
        self.points.clearVertices()
        self.points.addNextVertices(num_new_points)


if __name__ == "__main__":
    import cv2
    import numpy as np
    from wrs import wd, rm, mgm
    from wrs.visualization.panda.panda3d_utils import VirtualDepthCamera, ExtraWindow
    import os
    from ultralytics import YOLO  # 添加YOLO导入
    
    def save_ply_with_colors(filename, points, colors):
        """保存点云为 PLY 格式（带颜色）"""
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for i in range(len(points)):
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                       f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")

    base = wd.World(cam_pos=[1, 0, 1], lookat_pos=[0, 0, .05],
                    w=640, h=480)
    
    # 启用鼠标控制（轨道控制）
    base.enableMouse()
    base.setBackgroundColor(1, 1, 1)  # 白色背景
    # Render v1 semantic plant meshes; keep single-target ROI references hidden.
    obj1, obj2, plant_models = load_plant_scene(base)

    # ===== 计算两个物体在世界坐标系下的 AABB =====
    obj1_min, obj1_max = bbox.world_aabb_from_obj(obj1)
    print(f"[DEBUG] obj1_min = {obj1_min}, obj1_max = {obj1_max}")
    print(f"[DEBUG] obj1 center = {0.5 * (obj1_min + obj1_max)}")
    obj2_min, obj2_max = bbox.world_aabb_from_obj(obj2)
    print(f"[DEBUG] obj2_min = {obj2_min}, obj2_max = {obj2_max}")
    print(f"[DEBUG] obj2 center = {0.5 * (obj2_min + obj2_max)}")
    # ==============================================
    # 计算全局 bbox 作为体素网格范围
    global_min = np.minimum(obj1_min, obj2_min) - 0.02  # 略微扩大一点
    global_max = np.maximum(obj1_max, obj2_max) + 0.02
    voxel_size = 0.01  # 1cm 一格，自行调整

    dims = np.ceil((global_max - global_min) / voxel_size).astype(int)
    dims = np.maximum(dims, 1)  # 防止 0

    vg = voxel_grid.VoxelGrid3D(
        origin=global_min,
        voxel_size=voxel_size,
        dims=dims,
        p0=0.5,
        p_hit=0.7,
        p_miss=0.4,
    )
    # ==============================================
    mgm.gen_frame(ax_length=0.2, ax_radius=0.005).attach_to(base)
    resolution = (640, 480)
    vcam = VirtualDepthCamera(cam_pos=[0.4, 0, 0.65],
                              lookat_pos=[0, 0, 0.4],
                              resolution=resolution,
                              screen_size=np.array((0.05, 0.05)),
                              fov=45,
                              depth_far=5.0)  # Clip depth at 5 meters
                              
    #####################################
    K = vcam.intrinsics
    img_size = (resolution[0], resolution[1])

    nbv_policy = policy.NBVPolicy(
        vg=vg,
        obj_min=obj1_min,
        obj_max=obj1_max,
        K=K,
        img_size=img_size,
        vcam=vcam,
    )

    # 加载训练好的YOLO模型
    yolo_model_path = str(PROJECT_ROOT / "yolo11/runs/detect/train3/weights/best.pt")
    yolo_model = YOLO(yolo_model_path)
    print(f"[YOLO] Model loaded from {yolo_model_path}")
    
    # 类别名称（根据data.yaml）
    yolo_class_names = ['pepper', 'stem']
    yolo_class_colors = {
        'pepper': (0, 255, 0),    # 绿色 (BGR格式)
        'stem': (0, 165, 255),    # 橙色 (BGR格式)
    }

    # 点云和图像保存计数器
    pcd_save_counter = [0]
    pcd_save_dir = "saved_pointclouds"
    os.makedirs(pcd_save_dir, exist_ok=True)
    
    # RGB图像保存计数器
    rgb_save_counter = [0]
    rgb_save_dir = "saved_rgb_images"
    os.makedirs(rgb_save_dir, exist_ok=True)

    def detect_with_yolo(img_rgb):
        """使用YOLO模型检测图像中的辣椒和茎"""
        if img_rgb is None or len(img_rgb) == 0:
            return img_rgb, []
        
        # 确保图像是3通道
        img_rgb_3ch = img_rgb[:, :, :3] if img_rgb.shape[2] > 3 else img_rgb
        
        # YOLO检测
        results = yolo_model(img_rgb_3ch, conf=0.25, verbose=False)  # conf是置信度阈值
        
        detections = []
        img_with_detections = img_rgb_3ch.copy()
        
        # 绘制检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # 获取类别名称
                if class_id < len(yolo_class_names):
                    class_name = yolo_class_names[class_id]
                    color = yolo_class_colors.get(class_name, (255, 255, 255))
                    
                    # 绘制边界框
                    cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), color, 2)
                    
                    # 绘制标签
                    label = f"{class_name} {confidence:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img_with_detections, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(img_with_detections, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        return img_with_detections, detections

    def save_rgb_image():
        """保存当前视角的RGB图像"""
        img_rgb = vcam.get_rgb_image()  # 获取RGB图像
        if img_rgb is not None and len(img_rgb) > 0:
            # 使用YOLO检测
            img_rgb_with_detections, detections = detect_with_yolo(img_rgb)
            
            # 添加信息文本
            cam_pos = vcam.cam_pos
            info_txt = f"Pos: {cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}"
            if len(detections) > 0:
                info_txt += f" | Detections: {len(detections)}"
            img_rgb_copy = img_rgb_with_detections.copy()
            cv2.putText(img_rgb_copy, info_txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
            img_rgb_copy = img_rgb_copy[:, :, :3]  # 确保是3通道
            
            # 保存RGB图像为JPG格式
            rgb_img_filename = os.path.join(rgb_save_dir, f"rgb_{rgb_save_counter[0]:04d}.jpg")
            cv2.imwrite(rgb_img_filename, img_rgb_copy)
            print(f"[Save] RGB image saved to {rgb_img_filename} (Detections: {len(detections)})")
            rgb_save_counter[0] += 1

    # 按键触发一次 NBV 规划（例如按 'm' 键）
    def on_find_nbv():
        print("m pressed")
        nbv_policy.plan_and_move(radius=0.1)
        cam_trace_pts.append(nbv_policy.vcam.cam_pos.copy())
        _refresh_cam_trace()

        # 保存RGB图像（NBV移动相机后）
        save_rgb_image()
        
        # 获取当前视角的点云并创建可视化图像
        pcd = vcam.get_point_cloud()
        if len(pcd) == 0:
            print("[Save] No point cloud to save")
            return
        
        # 创建点云可视化图像
        img_size = (640, 480)  # 与相机分辨率一致
        pcd_img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255  # 白色背景
        
        # 将点云投影到图像平面
        K = vcam.intrinsics
        cam_pos = vcam.cam_pos
        cam_rotmat = vcam.cam_rotmat
        
        # 转换到相机坐标系（policy坐标系：x=右, y=前, z=上）
        pcd_cam_policy = (pcd - cam_pos) @ cam_rotmat
        
        # 转换到OpenCV坐标系（x=右, y=下, z=前）
        pcd_cam_cv = np.zeros_like(pcd_cam_policy)
        pcd_cam_cv[:, 0] = pcd_cam_policy[:, 0]   # x: 右 -> 右
        pcd_cam_cv[:, 1] = -pcd_cam_policy[:, 2]  # y: -z (上 -> 下)
        pcd_cam_cv[:, 2] = pcd_cam_policy[:, 1]   # z: y (前 -> 前)
        
        # 过滤掉相机后面的点
        valid_mask = pcd_cam_cv[:, 2] > 0.01  # z坐标（前向）> 0.01m
        pcd_cam_valid = pcd_cam_cv[valid_mask]
        pcd_valid = pcd[valid_mask]
        
        num_points_drawn = 0
        if len(pcd_cam_valid) > 0:
            # 投影到图像平面（OpenCV标准投影公式）
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            x_img = (fx * pcd_cam_valid[:, 0] / pcd_cam_valid[:, 2] + cx).astype(int)
            y_img = (fy * pcd_cam_valid[:, 1] / pcd_cam_valid[:, 2] + cy).astype(int)
            
            # 过滤在图像范围内的点
            in_image = (x_img >= 0) & (x_img < img_size[0]) & (y_img >= 0) & (y_img < img_size[1])
            x_img = x_img[in_image]
            y_img = y_img[in_image]
            pcd_valid_img = pcd_valid[in_image]
            
            # 根据obj1/obj2分类着色
            mask1_pcd = (
                (pcd_valid_img[:, 0] >= obj1_min[0]) & (pcd_valid_img[:, 0] <= obj1_max[0]) &
                (pcd_valid_img[:, 1] >= obj1_min[1]) & (pcd_valid_img[:, 1] <= obj1_max[1]) &
                (pcd_valid_img[:, 2] >= obj1_min[2]) & (pcd_valid_img[:, 2] <= obj1_max[2])
            )
            
            # 绘制点：obj1=红色，obj2=绿色
            for i, (x, y) in enumerate(zip(x_img, y_img)):
                if mask1_pcd[i]:
                    cv2.circle(pcd_img, (x, y), 2, (0, 0, 255), -1)  # 红色 (BGR格式)
                else:
                    cv2.circle(pcd_img, (x, y), 2, (0, 255, 0), -1)  # 绿色 (BGR格式)
            
            num_points_drawn = len(pcd_valid_img)
        
        # 添加信息文本
        cam_pos = vcam.cam_pos
        info_txt = f"Pos: {cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f} | Points: {num_points_drawn}"
        cv2.putText(pcd_img, info_txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
        
        # 保存点云图像为JPG
        pcd_img_filename = os.path.join(pcd_save_dir, f"pcd_view_{pcd_save_counter[0]:04d}.jpg")
        cv2.imwrite(pcd_img_filename, pcd_img)
        print(f"[Save] Point cloud image saved to {pcd_img_filename} ({num_points_drawn} points)")
        
        # 保存当前视角的点云
        pcd = vcam.get_point_cloud()
        if len(pcd) > 0:
            # 添加颜色：obj1=红色，obj2=绿色
            mask1_pcd = (
                (pcd[:, 0] >= obj1_min[0]) & (pcd[:, 0] <= obj1_max[0]) &
                (pcd[:, 1] >= obj1_min[1]) & (pcd[:, 1] <= obj1_max[1]) &
                (pcd[:, 2] >= obj1_min[2]) & (pcd[:, 2] <= obj1_max[2])
            )
            colors_pcd = np.zeros((len(pcd), 3), dtype=np.uint8)
            colors_pcd[mask1_pcd] = [255, 0, 0]  # obj1：红色
            colors_pcd[~mask1_pcd] = [0, 255, 0]  # obj2：绿色
            
            # 保存为 PLY 格式
            filename = os.path.join(pcd_save_dir, f"pcd_view_{pcd_save_counter[0]:04d}.ply")
            save_ply_with_colors(filename, pcd, colors_pcd)
            print(f"[Save] Point cloud saved to {filename} ({len(pcd)} points)")
            pcd_save_counter[0] += 1

    base.accept("m", on_find_nbv)
    ######################################
    move_speed = 0.05
    rot_speed = np.radians(5)

    def move_cam(dx=0, dy=0, dz=0):
        vcam.cam_pos = vcam.cam_pos + np.array([dx, dy, dz])
        print(f"[Camera] pos = {vcam.cam_pos}, rotmat = \n{vcam.cam_rotmat}")
        save_rgb_image()  # 移动相机后保存图像

    def yaw_cam(delta):
        vcam.cam_rotmat = rm.rotmat_from_euler(0, 0, delta) @ vcam.cam_rotmat
        print(f"[Camera] pos = {vcam.cam_pos}, rotmat = \n{vcam.cam_rotmat}")
        save_rgb_image()  # 旋转相机后保存图像

    base.accept("w", lambda: move_cam(0,  move_speed, 0))
    base.accept("s", lambda: move_cam(0, -move_speed, 0))
    base.accept("a", lambda: move_cam(-move_speed, 0, 0))
    base.accept("d", lambda: move_cam( move_speed, 0, 0))
    base.accept("q", lambda: move_cam(0, 0,  move_speed))   # 上升
    base.accept("e", lambda: move_cam(0, 0, -move_speed))   # 下降
    base.accept("arrow_left",  lambda: yaw_cam(rot_speed))
    base.accept("arrow_right", lambda: yaw_cam(-rot_speed))
    ######################################


    # Create the visualization of the camera (Frustum)
    # We will update its pose in the loop
    vcam_viz_node = [vcam.gen_meshmodel(alpha=0.5)]
    vcam_viz_node[0].attach_to(base)
    vcam_pcd_node = [mgm.gen_pointcloud(vcam.get_point_cloud(), point_size=.001)]
    vcam_pcd_node[0].attach_to(base)
    ray_nodes = []  # 存放射线可视化节点
    cam_trace_pts = [vcam.cam_pos.copy()]  # 记录相机轨迹
    cam_marker_nodes = []
    cam_path_node = [None]
    # target_point = obj.trm_mesh.center_mass
    min_bounds, max_bounds = obj2.trm_mesh.bounds
    center_local = 0.5 * (min_bounds + max_bounds)
    target_point = center_local @ obj2.rotmat.T + obj2.pos

    vcam_pcd_node[0].detach()

    ew = ExtraWindow(base,
                     cam_pos=[.5, 0, .5], lookat_pos=[0, 0, .03],
                     w=640, h=480,)
    ew.set_origin((np.array([0, 40])))
    pc_viz = PointCloudVisualizer(ew)
    pc_viz.np.setPos(0, 0, 0)

    vox_viz = VoxelGridVisualizer(
        base,
        window_origin=(640, 40),
        size=(640, 480),
        cam_pos=(0.5, -0.5, 0.3),
        lookat=(0, 0, 0.05),
        voxel_size=vg.voxel_size,
        max_voxels=50000
    )    

    def _refresh_cam_trace():
        # 清理旧节点
        for n in cam_marker_nodes:
            n.removeNode()
        cam_marker_nodes.clear()
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

        # 画每个位置的小三棱锥（线框）
        size = 0.02
        for p in cam_trace_pts:
            x, y, z = p
            tip = np.array([x, y, z + size])
            b1 = np.array([x + size, y, z - size])
            b2 = np.array([x - size, y + size, z - size])
            b3 = np.array([x - size, y - size, z - size])
            ls_m = LineSegs("cam_marker")
            ls_m.setThickness(1.5)
            ls_m.setColor(0.2, 0.8, 1.0, 0.9)  # 青色
            for a, b in [(tip, b1), (tip, b2), (tip, b3), (b1, b2), (b2, b3), (b3, b1)]:
                ls_m.moveTo(a[0], a[1], a[2])
                ls_m.drawTo(b[0], b[1], b[2])
            cam_marker_nodes.append(base.render.attachNewNode(ls_m.create()))

    _refresh_cam_trace()


    def update(task):
        t = task.time
        vcam_viz_node[0].detach()
        # --- A. Orbit Logic ---
        radius = 0.4
        speed = 1
        angle = t * speed
        # Calculate new position (Circular orbit at height 0.2)
        new_x = target_point[0] + radius * np.cos(angle)
        new_y = target_point[1] + radius * np.sin(angle)
        new_z = 0.545 + 0.1 * np.sin(angle * 0.5)  # Bobbing up and down slightly

        # Update Camera Pose

        # vcam.cam_pos = np.array([new_x, new_y, new_z])
        # vcam.look_at(target_point)  # 注释掉，让 NBV 和手动控制完全控制相机
        
        # Update Visualization
        vcam_viz_node[0] = vcam.gen_meshmodel(alpha=0.5)
        vcam_viz_node[0].attach_to(base)
        # Get Images
        img_rgb = vcam.get_rgb_image()  # (H, W, 3) uint8
        img_depth = vcam.get_depth_image()  # (H, W) float32 in meters
        
        # 使用YOLO进行检测
        img_rgb_with_detections, yolo_detections = detect_with_yolo(img_rgb)
        if len(yolo_detections) > 0:
            print(f"[YOLO] Detected {len(yolo_detections)} objects: {[d['class'] for d in yolo_detections]}")
        #########################
        # pcd = vcam.get_point_cloud()
        # step = 10
        # pcd_sparse = pcd[::step]
        # pc_viz.update(pcd_sparse)
        pcd = vcam.get_point_cloud()
        step = 10
        pcd_sparse = pcd[::step]

        pts = pcd_sparse

        # 属于 obj1 的点：在 obj1 AABB 里面
        mask1 = (
            (pts[:, 0] >= obj1_min[0]) & (pts[:, 0] <= obj1_max[0]) &
            (pts[:, 1] >= obj1_min[1]) & (pts[:, 1] <= obj1_max[1]) &
            (pts[:, 2] >= obj1_min[2]) & (pts[:, 2] <= obj1_max[2])
        )
        # 其余归为 obj2（简单做法）
        mask2 = ~mask1

        colors = np.zeros((len(pts), 4), dtype=np.float32)
        colors[mask1] = np.array([1.0, 0.0, 0.0, 1.0])  # obj1：红色
        colors[mask2] = np.array([0.0, 1.0, 0.0, 1.0])  # obj2：绿色

        # 语义标签与概率（供 NBV 使用）
        sem_probs = np.ones((len(pts),), dtype=np.float32)
        sem_classes = np.full((len(pts),), -1, dtype=np.int8)
        sem_classes[mask1] = 0  # obj1
        sem_classes[mask2] = 1  # obj2

        pc_viz.update(pcd_sparse, colors=colors)

        # --- 可视化相机射线（取子样本，避免过多线段） ---
        # for n in ray_nodes:
        #     n.removeNode()
        # ray_nodes.clear()
        # if len(pts) > 0:
        #     max_rays = 200
        #     stride = max(1, len(pts) // max_rays)
        #     sample_pts = pts[::stride]
        #     ls = LineSegs("rays")
        #     ls.setThickness(1.0)
        #     ls.setColor(0.5, 0.5, 0.5, 0.2)  # 低透明度的灰色
        #     cam_pos = vcam.cam_pos
        #     for p in sample_pts:
        #         ls.moveTo(cam_pos[0], cam_pos[1], cam_pos[2])
        #         ls.drawTo(p[0], p[1], p[2])
        #     node = base.render.attachNewNode(ls.create())
        #     ray_nodes.append(node)
        # ===================================================
        # 用这一帧的点云更新体素概率
        # vg.update_hits(pcd_sparse)
        vg.update_hits_with_free_space(pcd_sparse, vcam.cam_pos)
        # 更新语义（不修改占据概率）
        vg.update_semantics(pcd_sparse, sem_probs, sem_classes)
        # ===================================================
        occ = vg.get_occupancy()
        # 只绘制确定的体素：高占据 / 高空闲（不显示不确定的）
        mask_occupied = occ > 0.8
        mask_free = occ < 0.2
        mask_interest = mask_occupied | mask_free

        idxs_interest = np.column_stack(np.nonzero(mask_interest))

        if len(idxs_interest) > 0:
            centers = vg.origin + vg.voxel_size * (idxs_interest + 0.5)
            probs = occ[idxs_interest[:, 0], idxs_interest[:, 1], idxs_interest[:, 2]]

            vis_margin = 0.03
            in_obj1 = (
                (centers[:, 0] >= obj1_min[0] - vis_margin) & (centers[:, 0] <= obj1_max[0] + vis_margin) &
                (centers[:, 1] >= obj1_min[1] - vis_margin) & (centers[:, 1] <= obj1_max[1] + vis_margin) &
                (centers[:, 2] >= obj1_min[2] - vis_margin) & (centers[:, 2] <= obj1_max[2] + vis_margin)
            )

            centers = centers[in_obj1]
            probs = probs[in_obj1]

            if len(centers) > 0:
                # 判断占据体素是否在obj1的AABB内（目标区域）
                in_obj1_aabb = (
                    (centers[:, 0] >= obj1_min[0]) & (centers[:, 0] <= obj1_max[0]) &
                    (centers[:, 1] >= obj1_min[1]) & (centers[:, 1] <= obj1_max[1]) &
                    (centers[:, 2] >= obj1_min[2]) & (centers[:, 2] <= obj1_max[2])
                )
                
                # 只显示确定的体素：占据=蓝色/红色，空闲=绿色
                colors = np.zeros((len(centers), 4), dtype=np.float32)
                mask_occ = probs > 0.8
                mask_free_viz = probs < 0.2
                
                # 占据体素：目标区域（obj1内）=红色，非目标区域=蓝色
                mask_occ_obj1 = mask_occ & in_obj1_aabb  # 占据且在obj1内
                mask_occ_other = mask_occ & ~in_obj1_aabb  # 占据但不在obj1内
                colors[mask_occ_obj1] = np.array([1.0, 0.0, 0.0, 0.5])  # 目标区域：红色
                colors[mask_occ_other] = np.array([0.0, 0.0, 1.0, 0.5])  # 非目标区域：蓝色
                colors[mask_free_viz] = np.array([0.0, 1.0, 0.0, 0.3])  # 空闲：绿色
                vox_viz.update(centers, colors)
            else:
                vox_viz.clear()
        else:
            vox_viz.clear()
        #########################
        depth_vis = img_depth.copy()
        img_depth[img_depth > 4] = 0
        max_vis_dist = np.max(img_depth)
        depth_vis = np.clip(depth_vis, 0, max_vis_dist)
        depth_norm = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)
        # Apply false color: Blue = Close, Red = Far
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        K = vcam.intrinsics
        p_world = target_point
        # Transform world point to camera frame
        # P_cam = R^T * (P_world - t_cam)
        p_cam = vcam.cam_rotmat.T @ (p_world - vcam.cam_pos)
        verts = obj2.trm_mesh.vertices
        faces = obj2.trm_mesh.faces
        # Manual projection math (using K)
        verts_world = verts @ obj2.rotmat.T + obj2.pos
        # Vectorized World -> Camera (Panda convention: X-Right, Z-Up, Y-Fwd)
        # We need Camera Space for projection (X-Right, Y-Down, Z-Fwd)
        # But we can just use the manual logic from previous code or K directly
        # Let's use the manual logic which matches K derivation
        verts_cam_panda = (verts_world - vcam.cam_pos) @ vcam.cam_rotmat
        verts_cv_x = verts_cam_panda[:, 0]
        verts_cv_y = -verts_cam_panda[:, 2]  # Panda Z is CV -Y
        verts_cv_z = verts_cam_panda[:, 1]  # Panda Y is CV Z
        z_safe = verts_cv_z.copy()
        z_safe[z_safe < 0.001] = 0.001

        u_coords = (K[0, 0] * verts_cv_x) / z_safe + K[0, 2]
        v_coords = (K[1, 1] * verts_cv_y) / z_safe + K[1, 2]

        uv_coords = np.stack((u_coords, v_coords), axis=1).astype(np.int32)
        face_polygons = uv_coords[faces]

        # Draw contours on RGB
        # 使用带检测结果的图像
        img_rgb_display = img_rgb_with_detections if len(yolo_detections) > 0 else img_rgb
        mask = np.zeros(img_rgb_display.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, face_polygons, 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img_rgb_display, contours, -1, (0, 255, 0), 2)

        # --- G. Final Display ---
        # Add Overlay Text
        info_txt = f"Pos: {new_x:.2f}, {new_y:.2f}, {new_z:.2f}"
        if len(yolo_detections) > 0:
            info_txt += f" | YOLO: {len(yolo_detections)}"
        cv2.putText(img_rgb_display, info_txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
        cv2.putText(depth_colored, "Depth (Jet Colormap)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        img_rgb_display = img_rgb_display[:, :, :3]
        
        combined_img = np.hstack((img_rgb_display, depth_colored))
        cv2.imshow("Virtual Depth Camera (RGB | Depth)", combined_img)
        key = cv2.waitKey(1) & 0xFF
        return task.cont


    # Add task to Panda3D manager
    base.taskMgr.add(update, "vcam_update_loop")
    print("Running... Press Ctrl+C in console to stop, or close the main window.")
    base.run()
