import numpy as np
import torch

import TSDF
import raycast


class SimpleNBVGPU_TSDF:
    """
    基于 TSDF 的 GPU 加速 NBV 计算器。
    原理：寻找 TSDF 权重较低（未观测或观测不足）的表面区域，优先观测目标物体。
    """

    def __init__(self, tsdf_vol: TSDF.TSDFVolume,
                 obj_min, obj_max,
                 K, img_size,
                 downsample=8,
                 t_step_factor=1.0,
                 use_gpu=True,
                 obj1_weight=2.0,
                 k_decay=0.2):
        """
        tsdf_vol : TSDFVolume 实例
        obj_min  : 目标 BBox 最小值 (3,) (用于射线裁剪优化)
        obj_max  : 目标 BBox 最大值 (3,)
        K        : 相机内参
        img_size : (W, H)
        downsample: 像素采样间隔
        obj1_weight: 对辣椒 (Class 0) 的关注度加成
        k_decay    : 增益衰减系数，G = exp(-k * weight)
        """
        self.tsdf_vol = tsdf_vol
        self.obj_min = np.asarray(obj_min, dtype=np.float32)
        self.obj_max = np.asarray(obj_max, dtype=np.float32)
        self.K = K.astype(np.float32)
        self.W, self.H = int(img_size[0]), int(img_size[1])
        self.downsample = int(downsample)
        self.t_step = float(t_step_factor * self.tsdf_vol.voxel_size)
        self.obj1_weight = float(obj1_weight)
        self.k_decay = float(k_decay)

        # 预先缓存相机参数
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

        # 设置设备
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"[NBV_TSDF] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            if use_gpu:
                print("[NBV_TSDF] GPU requested but not available, using CPU")
            else:
                print("[NBV_TSDF] Using CPU")

        # 预加载体素网格到 GPU（如果网格不是太大）
        self._preload_voxel_grid()

    def _preload_voxel_grid(self):
        """预加载体素网格数据到 GPU（初始化时调用）"""
        self._update_voxel_grid()

    def _update_voxel_grid(self):
        """更新 GPU 上的体素网格数据（每次计算前调用，确保数据是最新的）"""
        # 1. 获取权重 (Weights) - 使用getter方法
        weights_np = self.tsdf_vol.get_weights()
        self.weights_t = torch.tensor(
            weights_np, dtype=torch.float32, device=self.device
        )

        # 2. 获取TSDF值 - 需要直接访问（TSDF类没有提供getter）
        tsdf_np = self.tsdf_vol._tsdf_vol
        self.tsdf_t = torch.tensor(
            tsdf_np, dtype=torch.float32, device=self.device
        )

        # 3. 获取语义 (Semantics) - 需要直接访问（TSDF类没有提供getter）
        sem_np = self.tsdf_vol._sem_vol
        self.sem_t = torch.tensor(
            sem_np, dtype=torch.float32, device=self.device
        )

        # 4. 基础几何信息（这些值通常不会变，但为了安全也更新一下）
        self.origin_t = torch.tensor(
            self.tsdf_vol.origin, dtype=torch.float32, device=self.device
        )
        self.voxel_size_t = torch.tensor(
            self.tsdf_vol.voxel_size, dtype=torch.float32, device=self.device
        )
        self.dims_t = torch.tensor(
            self.tsdf_vol.dims, dtype=torch.long, device=self.device
        )
        self.obj_min_t = torch.tensor(
            self.obj_min, dtype=torch.float32, device=self.device
        )
        self.obj_max_t = torch.tensor(
            self.obj_max, dtype=torch.float32, device=self.device
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
            origin=self.tsdf_vol.origin,
            voxel_size=self.tsdf_vol.voxel_size,
            dims=self.tsdf_vol.dims,
            cam_pos=cam_pos,
            directions=dir_world_np,
            t_min=self.tsdf_vol.voxel_size,
            t_max=3.0,
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

        # 6) 批量查询体素数据（使用 GPU 上的 tensor）
        idx_valid_t = torch.tensor(idx_valid_np, dtype=torch.long, device=self.device)
        tsdf_voxels = self.tsdf_t[idx_valid_t[:, 0], idx_valid_t[:, 1], idx_valid_t[:, 2]]  # (M',)
        w_voxels = self.weights_t[idx_valid_t[:, 0], idx_valid_t[:, 1], idx_valid_t[:, 2]]  # (M',)
        sem_prob_voxels = self.sem_t[idx_valid_t[:, 0], idx_valid_t[:, 1], idx_valid_t[:, 2]]  # (M',)
        
        # 调试信息（仅在第一次调用时打印）
        if not hasattr(self, '_debug_printed'):
            print(f"[NBV_TSDF Debug] Found {len(idx_valid_np)} valid voxels")
            print(f"[NBV_TSDF Debug] Weight range: [{w_voxels.min().item():.3f}, {w_voxels.max().item():.3f}]")
            print(f"[NBV_TSDF Debug] TSDF range: [{tsdf_voxels.min().item():.3f}, {tsdf_voxels.max().item():.3f}]")
            print(f"[NBV_TSDF Debug] Sem prob range: [{sem_prob_voxels.min().item():.3f}, {sem_prob_voxels.max().item():.3f}]")
            self._debug_printed = True

        # 7) 计算增益 (TSDF 核心逻辑)
        # [公式 1] 基础增益：基于权重的指数衰减
        # W=0 (未知) -> Gain=1.0
        # W大 (已知) -> Gain -> 0
        base_gain = torch.exp(-self.k_decay * w_voxels)

        # [公式 2] 表面重要性：TSDF值接近0的点更重要（表面区域）
        # |TSDF| 越小，越接近表面，重要性越高
        # 对于未观测区域（TSDF=1.0），给予中等重要性（0.5），鼓励观测
        # 对于已观测区域，根据TSDF值计算重要性
        tsdf_abs = torch.abs(tsdf_voxels)
        # 未观测区域（权重=0）或TSDF接近1的区域，给予基础重要性
        is_unobserved = (w_voxels < 0.1) | (tsdf_abs > 0.9)
        surface_importance = torch.where(
            is_unobserved,
            torch.ones_like(tsdf_abs) * 0.5,  # 未观测区域：中等重要性
            1.0 - torch.clamp(tsdf_abs, 0.0, 1.0)  # 已观测区域：根据TSDF值
        )

        # [公式 3] 语义加成：如果是辣椒 (prob > 0.5)，给予额外关注
        is_obj1 = (sem_prob_voxels > 0.5).float()

        # 最终得分 = 基础分 * 表面重要性 * (1 + 辣椒加成)
        # 权重低且接近表面的区域，如果是目标物体，获得最高增益
        final_scores = base_gain * surface_importance * (1.0 + self.obj1_weight * is_obj1)

        # 求和得到总增益
        total_gain = final_scores.sum().item()

        return float(total_gain)

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
SimpleNBV = SimpleNBVGPU_TSDF