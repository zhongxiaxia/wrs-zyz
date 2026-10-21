import numpy as np

import voxel_grid
import bbox


class SimpleNBV:
    """
    基于 voxel_grid.VoxelGrid3D 的简易 NBV 计算器：
    - 网格里每个体素有一个占据概率 p
    - 我们假设“信息增益”≈ 当前视角能看到的、处于“未确认”状态的目标体素个数
      （例如 p 在 [p_min, p_max] 区间的体素）。
    """

    def __init__(self, vg: voxel_grid.VoxelGrid3D,
                 obj_min, obj_max,
                 K, img_size,
                 p_min=0.3, p_max=0.7,
                 downsample=8,
                 t_step_factor=np.sqrt(3)):
        """
        vg       : VoxelGrid3D 实例（在 pepper_cloud.py 里创建的 vg）
        obj_min  : 目标 obj1 的 AABB 最小值 (3,)
        obj_max  : 目标 obj1 的 AABB 最大值 (3,)
        K        : 相机 3x3 内参矩阵 vcam.intrinsics
        img_size : (W, H)
        p_min,p_max : 认为是“还不确定”的概率区间
        downsample  : 射线在像素平面上的步长，用于加速，如 8 表示每 8 像素取一个 ray
        t_step_factor: ray marching 的步长系数，一般 √3 * voxel_size
        """
        self.vg = vg
        self.obj_min = np.asarray(obj_min, dtype=np.float32)
        self.obj_max = np.asarray(obj_max, dtype=np.float32)
        self.K = K.astype(np.float32)
        self.W, self.H = int(img_size[0]), int(img_size[1])
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.downsample = int(downsample)
        self.t_step = float(t_step_factor * self.vg.voxel_size)

        # 预先缓存一些量
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

    def _raycast_gain_for_view(self, cam_pos, cam_rotmat):
        """
        对单个视角做简化射线投射：
        - 在像素平面上网格采样 (u, v)
        - 在相机坐标系构造方向向量 d_cam
        - 转到世界系 d_world, 然后在 voxel_grid 里沿 ray 做步进
        - 在步进过程中，如果某个体素处于 [p_min,p_max] 且在 obj1 bbox 内，则记一次“潜在增益”
        返回：gain (float)
        """
        # cam_pos = np.asarray(cam_pos, dtype=np.float32)
        # cam_R = np.asarray(cam_rotmat, dtype=np.float32)  # 3x3, Panda3D 世界->相机的转置要注意，这里假定传入的是 R_world_cam^T 或者你自己在外面处理

        # occ = self.vg.get_occupancy()
        # gain = 0

        # # 下采样后的像素范围
        # us = np.arange(0, self.W, self.downsample, dtype=np.int32)
        # vs = np.arange(0, self.H, self.downsample, dtype=np.int32)

        # rays_in_bbox = 0
        # uncertain_voxels = 0

        cam_pos = np.asarray(cam_pos, dtype=np.float32)
        cam_R = np.asarray(cam_rotmat, dtype=np.float32)

        # cam_R_c2w = cam_R.T
        
        # ===== 调试：打印关键信息 =====
        print(f"\n[DEBUG] cam_pos = {cam_pos}")
        print(f"[DEBUG] obj_min = {self.obj_min}, obj_max = {self.obj_max}")
        print(f"[DEBUG] cam_R shape = {cam_R.shape}")
        # 检查目标中心是否在 AABB 内
        center = 0.5 * (self.obj_min + self.obj_max)
        print(f"[DEBUG] target center = {center}")
        print(f"[DEBUG] center in bbox? {np.all(center >= self.obj_min) and np.all(center <= self.obj_max)}")
        # ================================
        
        occ = self.vg.get_occupancy()
        gain = 0
        rays_in_bbox = 0
        uncertain_voxels = 0
        
        us = np.arange(0, self.W, self.downsample, dtype=np.int32)
        vs = np.arange(0, self.H, self.downsample, dtype=np.int32)
        
        # 测试：只检查图像中心的一条射线
        u_test = self.W // 2
        v_test = self.H // 2
        dir_cam_cv = np.array([(u_test - self.cx) / self.fx,
                               (v_test - self.cy) / self.fy,
                               1.0], dtype=np.float32)
        dir_cam_cv /= np.linalg.norm(dir_cam_cv)
        # OpenCV -> policy 坐标系
        dir_cam_policy = np.array([
            dir_cam_cv[0],   # x: 右
            dir_cam_cv[2],   # z: 前 -> y
            -dir_cam_cv[1]   # -y: -下 -> z (上)
        ], dtype=np.float32)
        dir_world = cam_R @ dir_cam_policy
        print(f"[DEBUG] center ray: dir_cam_cv = {dir_cam_cv}, dir_cam_policy = {dir_cam_policy}, dir_world = {dir_world}")
        
        # 验证：从相机到目标中心的方向
        center = 0.5 * (self.obj_min + self.obj_max)
        expected_dir = center - cam_pos
        expected_dir /= np.linalg.norm(expected_dir)
        print(f"[DEBUG] expected direction to center = {expected_dir}")
        print(f"[DEBUG] dot product = {np.dot(dir_world, expected_dir):.3f} (should be close to 1.0)")
        
        # 沿着这条测试射线走几步，看是否进入 bbox
        for step in range(10):
            t = step * 0.1
            p = cam_pos + t * dir_world
            in_bbox = np.all(p >= self.obj_min) and np.all(p <= self.obj_max)
            print(f"  t={t:.2f}, p={p}, in_bbox={in_bbox}")
            if in_bbox:
                break

        for u in us:
            for v in vs:
                # 像素 -> 相机坐标系射线方向 (OpenCV: x 右, y 下, z 前)
                dir_cam_cv = np.array([(u - self.cx) / self.fx,
                                      (v - self.cy) / self.fy,
                                      1.0], dtype=np.float32)
                dir_cam_cv /= np.linalg.norm(dir_cam_cv)

                # OpenCV 相机坐标 -> policy.py 定义的相机坐标
                # policy.py 中 R_c2w 的列是 [right, forward, up]
                # OpenCV: [x=右, y=下, z=前] -> policy: [x=右, y=前, z=上]
                # 所以映射是: [x_cv, y_cv, z_cv] -> [x_cv, z_cv, -y_cv]
                dir_cam_policy = np.array([
                    dir_cam_cv[0],   # x: 右 -> 右
                    dir_cam_cv[2],   # z: 前 -> 前 (y)
                    -dir_cam_cv[1]   # -y: -下 -> 上 (z)
                ], dtype=np.float32)

                # 相机坐标 -> 世界坐标
                # cam_R (R_c2w) 的列是 [right, forward, up]，对应 [x, y, z]
                dir_world = cam_R @ dir_cam_policy

                # Ray marching: p(t) = cam_pos + t * dir_world
                t = 0.0
                t_max = 5.0  # 最大射线长度，可以根据场景大小调节
                while t < t_max:
                    p = cam_pos + t * dir_world
                    t += self.t_step

                    # 只考虑目标 AABB 内的点
                    if np.any(p < self.obj_min) or np.any(p > self.obj_max):
                        continue

                    idx, valid_mask = self.vg.world_to_index(p[None, :])
                    if not valid_mask[0]:
                        continue
                    i, j, k = idx[0]
                    p_voxel = occ[i, j, k]

                    # 如果这个体素概率处于“不确定”区间，则认为观测一次有信息增益
                    if self.p_min <= p_voxel <= self.p_max:
                        gain += 1
                        uncertain_voxels += 1
                        break
                        # 可以 break 提前结束这条 ray（只算第一个不确定体素）
                    rays_in_bbox += 1  # 统计进入 bbox 的射线数
        print(f"  rays_in_bbox={rays_in_bbox}, uncertain_voxels={uncertain_voxels}, gain={gain}")
        return float(gain)

    def evaluate_views(self, cam_poses):
        """
        cam_poses: list/array of (cam_pos, cam_rotmat) 对
        返回: gains (len(cam_poses),) 的数组
        """
        gains = []
        for cam_pos, cam_rotmat in cam_poses:
            g = self._raycast_gain_for_view(cam_pos, cam_rotmat)
            gains.append(g)
        return np.array(gains, dtype=np.float32)

    def best_view(self, cam_poses):
        """
        在给定候选视角中选出信息增益最大的 NBV。
        返回: (best_index, best_gain)
        """
        gains = self.evaluate_views(cam_poses)
        if len(gains) == 0:
            return None, 0.0
        idx = int(np.argmax(gains))
        return idx, float(gains[idx])