import numpy as np

# import nbv as nbv_mod
import nbv_gpu as nbv_mod


class NBVPolicy:
    """
    把 SimpleNBV 封装成一个“策略”，负责：
    1）给定目标 obj1 的 AABB、体素网格 vg、相机内参 K；
    2）生成一批候选视角；
    3）调用 SimpleNBV 计算每个视角的增益；
    4）把虚拟相机 vcam 移动到增益最大的视角。
    """

    def __init__(self, vg, obj_min, obj_max, K, img_size, vcam):
        """
        vg       : voxel_grid.VoxelGrid3D
        obj_min  : obj1 AABB min (3,)
        obj_max  : obj1 AABB max (3,)
        K        : 3x3 内参矩阵
        img_size : (W, H)
        vcam     : VirtualDepthCamera / VirtualCamera 实例
        """
        self.vg = vg
        self.obj_min = np.asarray(obj_min, dtype=np.float32)
        self.obj_max = np.asarray(obj_max, dtype=np.float32)
        self.K = K.astype(np.float32)
        self.img_size = img_size
        self.vcam = vcam

        self.nbv_solver = nbv_mod.SimpleNBV(
            vg=self.vg,
            obj_min=self.obj_min,
            obj_max=self.obj_max,
            K=self.K,
            img_size=self.img_size,
            p_min=0.3,
            p_max=0.7,
            downsample=8,
        )

    def _look_at(self, cam_pos, target):
        """
        给定相机位置和目标点，返回看向 target 的旋转矩阵（列向量: right, forward, up）。
        """
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
        R_c2w = np.column_stack([right, forward, up]).astype(np.float32)
        return R_c2w

    def _generate_candidate_views(self, center, radius=0.4, n_az=12, n_el=1):
        """
        生成一批候选相机位姿 (cam_pos, cam_R_c2w)：
        - center: 目标中心（obj1 AABB 中点）
        - radius: 水平半径
        - n_az  : 水平方向离散个数
        - n_el  : 垂直方向层数（先设 1 即可）
        """
        center = np.asarray(center, dtype=np.float32)
        cam_poses = []

        # 简单：只在水平面上围绕一圈
        for k in range(n_az):
            phi = 2.0 * np.pi * k / n_az
            cam_pos = np.array([
                center[0] + radius * np.cos(phi),
                center[1] + radius * np.sin(phi),
                center[2] + 0.0,   # 相机高度 = 目标中心高度，可按需加偏移
            ], dtype=np.float32)

            R_c2w = self._look_at(cam_pos, center)

            cam_poses.append((cam_pos, R_c2w))

        return cam_poses

    def _generate_views_around_current_cam(self, radius=0.2, n_az=8):
        """
        围绕当前相机位置生成候选视角：
        - radius: 在当前相机附近的扰动半径（米）
        - n_az  : 在水平面上采样多少个点
        """
        cam_poses = []

        # 当前相机的位姿
        base_pos = self.vcam.cam_pos.astype(np.float32)
        base_R   = self.vcam.cam_rotmat.astype(np.float32)

        # 仍然让所有候选视角“看向 obj1 的中心”
        obj_center = 0.5 * (self.obj_min + self.obj_max)

        for k in range(n_az):
            phi = 2.0 * np.pi * k / n_az

            # 在当前相机的水平面上绕一圈（相对当前相机做平面位移）
            offset = np.array([
                radius * np.cos(phi),
                radius * np.sin(phi),
                0.0,
            ], dtype=np.float32)

            cam_pos = base_pos + offset

            # 构造一个“看向 obj_center” 的旋转矩阵（和原来一样的逻辑）
            R_c2w = self._look_at(cam_pos, obj_center)

            cam_poses.append((cam_pos, R_c2w))

        return cam_poses

    def _generate_random_views_around_current_cam(self, radius=0.4, n=16, z_offset=0.0):
        """
        围绕当前相机位置的圆盘随机采样候选视角（均匀分布），相机朝向 obj1 中心。
        """
        base_pos = self.vcam.cam_pos.astype(np.float32)
        obj_center = 0.5 * (self.obj_min + self.obj_max)
        cam_poses = []
        us = np.random.rand(n)
        vs = np.random.rand(n)
        rs = radius * np.sqrt(us)  # 均匀圆盘采样
        thetas = 2.0 * np.pi * vs
        for r, theta in zip(rs, thetas):
            dx = r * np.cos(theta)
            dy = r * np.sin(theta)
            cam_pos = base_pos + np.array([dx, dy, z_offset], dtype=np.float32)
            R_c2w = self._look_at(cam_pos, obj_center)
            cam_poses.append((cam_pos, R_c2w))
        return cam_poses

    def _integrate_current_observation(self, step=8, update_free_space=True):
        """
        用当前虚拟相机的一帧点云更新体素网格：
        - step: 下采样步长，防止一次性点太多
        - update_free_space: True 则调用 update_hits_with_free_space
        """
        pcd = self.vcam.get_point_cloud()          # (N,3) 世界坐标
        if len(pcd) == 0:
            return
        if step > 1:
            pcd = pcd[::step]

        # 只保留 obj1 AABB 内的点，避免把其它物体也写进 vg（按需保留/删除）
        mask_obj1 = (
            (pcd[:, 0] >= self.obj_min[0]) & (pcd[:, 0] <= self.obj_max[0]) &
            (pcd[:, 1] >= self.obj_min[1]) & (pcd[:, 1] <= self.obj_max[1]) &
            (pcd[:, 2] >= self.obj_min[2]) & (pcd[:, 2] <= self.obj_max[2])
        )
        pts_obj1 = pcd[mask_obj1]
        if len(pts_obj1) == 0:
            return

        if update_free_space:
            self.vg.update_hits_with_free_space(pts_obj1, self.vcam.cam_pos)
        else:
            self.vg.update_hits(pts_obj1)


    def plan_and_move(self, radius=0.4):
        """
        计算一次 NBV，并把 vcam 移到该视角。
        """
        # # 目标中心：用 obj1 的 AABB 中点
        # center = 0.5 * (self.obj_min + self.obj_max)


        # # 1) 生成候选视角
        # cam_poses = self._generate_candidate_views(center, radius=radius, n_az=16)

        # 0) 先把上一帧观测融合进体素网格
        self._integrate_current_observation(step=8, update_free_space=True)

        # 1) 围绕当前相机，在圆盘内随机生成候选视角
        cam_poses = self._generate_random_views_around_current_cam(radius=radius, n=16, z_offset=0.0)

        # 2) 计算每个视角的增益
        best_idx, best_gain = self.nbv_solver.best_view(cam_poses)
        if best_idx is None:
            print("[NBV] 没有可用候选视角")
            return

        best_pos, best_R_c2w = cam_poses[best_idx]
        print(f"[NBV] best gain = {best_gain:.1f}, pos = {best_pos}")

        # 3) 移动虚拟相机到最佳视角
        self.vcam.cam_pos = best_pos
        self.vcam.cam_rotmat = best_R_c2w