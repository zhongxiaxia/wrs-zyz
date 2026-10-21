import numpy as np
import torch

import voxel_grid
import bbox

import raycast

class SimpleNBVGPU:
    """
    基于 voxel_grid.VoxelGrid3D 的 GPU 加速 NBV 计算器（使用 PyTorch）。
    与 SimpleNBV 接口兼容，但使用 GPU 加速 raycast 计算。
    """

    def __init__(self, vg: voxel_grid.VoxelGrid3D,
                 obj_min, obj_max,
                 K, img_size,
                 p_min=0.3, p_max=0.7,
                 downsample=8,
                 t_step_factor=np.sqrt(3),
                 use_gpu=True,
                 obj1_class_id=0,
                 obj1_weight=0.2):
        """
        vg       : VoxelGrid3D 实例
        obj_min  : 目标 obj1 的 AABB 最小值 (3,)
        obj_max  : 目标 obj1 的 AABB 最大值 (3,)
        K        : 相机 3x3 内参矩阵 vcam.intrinsics
        img_size : (W, H)
        p_min,p_max : 认为是"还不确定"的概率区间
        downsample  : 射线在像素平面上的步长
        t_step_factor: ray marching 的步长系数
        use_gpu  : 是否使用 GPU（如果可用）
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
        # 语义：obj1 类别 ID 与权重（鼓励能看到更多 obj1）
        self.obj1_class_id = int(obj1_class_id)
        self.obj1_weight = float(obj1_weight)

        # 预先缓存一些量
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

        # 设置设备
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"[NBV_GPU] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            if use_gpu:
                print("[NBV_GPU] GPU requested but not available, using CPU")
            else:
                print("[NBV_GPU] Using CPU")

        # 预加载体素网格到 GPU（如果网格不是太大）
        self._preload_voxel_grid()

    def _preload_voxel_grid(self):
        """预加载体素网格数据到 GPU（初始化时调用）"""
        self._update_voxel_grid()

    def _update_voxel_grid(self):
        """更新 GPU 上的体素网格数据（每次计算前调用，确保数据是最新的）"""
        # 重新获取最新的占据概率
        self.occ_t = torch.tensor(
            self.vg.get_occupancy(),
            dtype=torch.float32,
            device=self.device
        )
        # 这些值通常不会变，但为了安全也更新一下
        self.origin_t = torch.tensor(
            self.vg.origin,
            dtype=torch.float32,
            device=self.device
        )
        self.voxel_size_t = torch.tensor(
            self.vg.voxel_size,
            dtype=torch.float32,
            device=self.device
        )
        self.dims_t = torch.tensor(
            self.vg.dims,
            dtype=torch.long,
            device=self.device
        )
        # 语义类别（-1: 背景，0: obj1，1: obj2）
        self.sem_class_t = torch.tensor(
            getattr(self.vg, "sem_class", np.full(self.vg.dims, -1, dtype=np.int8)),
            dtype=torch.long,
            device=self.device
        )
        self.obj_min_t = torch.tensor(
            self.obj_min,
            dtype=torch.float32,
            device=self.device
        )
        self.obj_max_t = torch.tensor(
            self.obj_max,
            dtype=torch.float32,
            device=self.device
        )

    def _raycast_gain_for_view(self, cam_pos, cam_rotmat):
        """
        GPU 加速的 raycast 计算。
        返回：gain (float)
        """
        # 关键：每次计算前重新加载最新的体素网格数据
        self._update_voxel_grid()
        
        cam_pos = np.asarray(cam_pos, dtype=np.float32)
        cam_R = np.asarray(cam_rotmat, dtype=np.float32)

        # 转换为 PyTorch tensor（仅用于方向计算）
        cam_R_t = torch.tensor(cam_R, dtype=torch.float32, device=self.device)

        # 1) 生成所有像素坐标（向量化）
        us = torch.arange(0, self.W, self.downsample, dtype=torch.float32, device=self.device)
        vs = torch.arange(0, self.H, self.downsample, dtype=torch.float32, device=self.device)
        u_grid, v_grid = torch.meshgrid(us, vs, indexing='ij')
        u_flat = u_grid.flatten()  # (N,)
        v_flat = v_grid.flatten()  # (N,)
        N_rays = len(u_flat)

        # 2) 批量计算所有射线的方向（OpenCV 相机坐标）
        dir_cam_cv = torch.stack([
            (u_flat - self.cx) / self.fx,
            (v_flat - self.cy) / self.fy,
            torch.ones(N_rays, dtype=torch.float32, device=self.device)
        ], dim=1)  # (N, 3)
        dir_cam_cv = dir_cam_cv / torch.norm(dir_cam_cv, dim=1, keepdim=True)

        # 3) OpenCV -> policy 坐标系
        dir_cam_policy = torch.stack([
            dir_cam_cv[:, 0],   # x: 右 -> 右
            dir_cam_cv[:, 2],   # z: 前 -> 前 (y)
            -dir_cam_cv[:, 1]   # -y: -下 -> 上 (z)
        ], dim=1)  # (N, 3)

        # 4) 批量转换到世界坐标
        dir_world = (cam_R_t @ dir_cam_policy.T).T  # (N, 3)

        # 将方向转为 numpy 以传给通用 raycast 函数
        dir_world_np = dir_world.detach().cpu().numpy()

        # 5) 使用通用 GPU raycast 计算每条射线第一个经过的体素
        voxel_indices = raycast.raycast_voxels_gpu(
            origin=self.vg.origin,
            voxel_size=self.vg.voxel_size,
            dims=self.vg.dims,
            cam_pos=cam_pos,
            directions=dir_world_np,
            t_min=self.t_step,
            t_max=5.0,
            t_step=self.t_step,
            obj_min=self.obj_min,
            obj_max=self.obj_max,
            return_first_only=True,
            device=self.device
        )  # (N, 3)，无效射线为 (-1, -1, -1)

        # 过滤无效体素
        valid_mask = ~(voxel_indices == -1).all(axis=1)
        if not valid_mask.any():
            return 0.0

        idx_valid_np = voxel_indices[valid_mask]  # (M', 3)

        # 10) 批量查询体素概率（使用 GPU 上的 occ_t）
        idx_valid_t = torch.tensor(idx_valid_np, dtype=torch.long, device=self.device)
        p_voxels = self.occ_t[idx_valid_t[:, 0], idx_valid_t[:, 1], idx_valid_t[:, 2]]  # (M',)
        c_voxels = self.sem_class_t[idx_valid_t[:, 0], idx_valid_t[:, 1], idx_valid_t[:, 2]]  # (M',)

        # 11) 计算香农熵作为信息增益
        # 香农熵: H(p) = -p*log(p) - (1-p)*log(1-p)
        # 只对不确定区间的体素计算熵（参考原代码的逻辑）
        uncertain_mask = (p_voxels >= self.p_min) & (p_voxels <= self.p_max)
        p_uncertain = p_voxels[uncertain_mask]
        entropy_sum = 0.0
        if len(p_uncertain) > 0:
            # 避免 log(0)，使用小的 epsilon
            eps = 1e-8
            p_clamped = torch.clamp(p_uncertain, eps, 1.0 - eps)
            # 计算香农熵
            entropy = -(p_clamped * torch.log(p_clamped) + (1.0 - p_clamped) * torch.log(1.0 - p_clamped))
            entropy_sum = float(entropy.sum().item())

        # 12) 统计可见的 obj1 体素数量，鼓励看到更多 obj1
        obj1_count = (c_voxels == self.obj1_class_id).sum().item()

        # 避免 log(0)，使用小的 epsilon
        gain = entropy_sum + self.obj1_weight * float(obj1_count)

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


# 为了兼容性，提供一个别名
SimpleNBV = SimpleNBVGPU

