import numpy as np

import raycast

def _logit(p):
    return np.log(p / (1.0 - p))

class VoxelGrid3D:
    """
    简单 3D 概率体素网格（占据栅格），用 log-odds 做递推更新。
    这里只根据点云“命中”更新占据概率，没有做“空闲空间沿射线”的更新，先保证简单可用。
    """
    def __init__(
        self,
        origin,          # 体素网格世界坐标原点 (3,)
        voxel_size=0.01, # 单个体素边长 [m]
        dims=(80, 80, 80),  # 网格尺寸 (Nx, Ny, Nz)
        p0=0.5,          # 初始占据概率
        p_hit=0.7,       # 观察到点时的占据概率
        p_miss=0.4       # 没观测到（当前没用到，预留）
    ):
        self.origin = np.asarray(origin, dtype=np.float32)
        self.voxel_size = float(voxel_size)
        self.dims = np.array(dims, dtype=np.int32)

        # log odds 参数
        self.l0 = _logit(p0)
        self.l_hit = _logit(p_hit) - self.l0
        self.l_miss = _logit(p_miss) - self.l0

        # 网格：log-odds
        self.log_odds = np.full(self.dims, self.l0, dtype=np.float32)
        # 语义：概率与类别（-1: 背景/未知，0: obj1，1: obj2）
        self.sem_prob = np.zeros(self.dims, dtype=np.float32)
        self.sem_class = np.full(self.dims, -1, dtype=np.int8)

    def world_to_index(self, pts):
        """
        pts: (N,3) 世界坐标
        返回: (N,3) int 索引，以及合法 mask
        """
        pts = np.asarray(pts, dtype=np.float32)
        idx = ((pts - self.origin) / self.voxel_size).astype(np.int32)
        mask = (
            (idx[:, 0] >= 0) & (idx[:, 0] < self.dims[0]) &
            (idx[:, 1] >= 0) & (idx[:, 1] < self.dims[1]) &
            (idx[:, 2] >= 0) & (idx[:, 2] < self.dims[2])
        )
        return idx, mask

    def update_hits(self, pts_world):
        """
        用一帧点云更新占据概率（把有点的体素 log-odds 往“占据”方向推）。
        """
        if len(pts_world) == 0:
            return
        idx, mask = self.world_to_index(pts_world)
        idx = idx[mask]
        if len(idx) == 0:
            return

        # 为了效率，使用 np.unique 做一次聚合
        uniq, counts = np.unique(idx, axis=0, return_counts=True)
        for (i, j, k), cnt in zip(uniq, counts):
            self.log_odds[i, j, k] += self.l_hit * cnt

    def get_occupancy(self):
        """
        返回占据概率 (Nx,Ny,Nz)，值域 [0,1]
        """
        # 数值稳定：避免 exp 溢出
        log_odds_clipped = np.clip(self.log_odds, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-log_odds_clipped))

    def get_occupied_points(self, p_thresh=0.6):
        """
        返回占据概率大于阈值的体素中心点坐标 (M,3)，方便可视化。
        """
        occ = self.get_occupancy()
        mask = occ > p_thresh
        idxs = np.column_stack(np.nonzero(mask))
        if len(idxs) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        centers = self.origin + self.voxel_size * (idxs + 0.5)
        return centers.astype(np.float32)

    def get_semantic_points(self, p_thresh=0.6):
        """
        返回占据概率大于阈值的体素中心点坐标，以及语义概率与类别。
        """
        occ = self.get_occupancy()
        mask = occ > p_thresh
        idxs = np.column_stack(np.nonzero(mask))
        if len(idxs) == 0:
            return (np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int8))
        centers = self.origin + self.voxel_size * (idxs + 0.5)
        sem_probs = self.sem_prob[mask]
        sem_classes = self.sem_class[mask]
        return (centers.astype(np.float32),
                sem_probs.astype(np.float32),
                sem_classes.astype(np.int8))

    def update_semantics(self, pts_world, sem_probs, sem_classes):
        """
        仅更新语义概率/类别，不修改占据 log-odds。
        sem_probs: (N,) [0,1]
        sem_classes: (N,) int8，-1/0/1
        """
        if len(pts_world) == 0:
            return
        pts_world = np.asarray(pts_world, dtype=np.float32)
        sem_probs = np.asarray(sem_probs, dtype=np.float32)
        sem_classes = np.asarray(sem_classes, dtype=np.int8)
        idx, mask = self.world_to_index(pts_world)
        idx = idx[mask]
        sem_probs = sem_probs[mask]
        sem_classes = sem_classes[mask]
        if len(idx) == 0:
            return
        for (i, j, k), p_s, c_s in zip(idx, sem_probs, sem_classes):
            self.sem_prob[i, j, k] = float(p_s)
            self.sem_class[i, j, k] = int(c_s)

    def update_hits_with_semantics(self, pts_world, sem_probs=None, sem_classes=None):
        """
        同时更新占据与语义。
        sem_probs: (N,) 属于该类的概率 [0,1]
        sem_classes: (N,) 语义类别，-1/0/1
        """
        if len(pts_world) == 0:
            return
        pts_world = np.asarray(pts_world, dtype=np.float32)
        idx, mask = self.world_to_index(pts_world)
        idx = idx[mask]
        if len(idx) == 0:
            return

        uniq, counts = np.unique(idx, axis=0, return_counts=True)
        for (i, j, k), cnt in zip(uniq, counts):
            self.log_odds[i, j, k] += self.l_hit * cnt

        if sem_probs is not None and sem_classes is not None:
            sem_probs = np.asarray(sem_probs, dtype=np.float32)[mask]
            sem_classes = np.asarray(sem_classes, dtype=np.int8)[mask]
            for (i, j, k), p_s, c_s in zip(idx, sem_probs, sem_classes):
                self.sem_prob[i, j, k] = float(p_s)
                self.sem_class[i, j, k] = int(c_s)


    def update_hits_with_free_space(self, pts_world, cam_pos,
                                    max_points=200,
                                    t_step_factor=0.5):
        """
        用点云更新占据概率，同时更新射线路径上的空闲空间（使用 raycast.py 中的 GPU 加速函数）。

        pts_world: (N, 3) 世界坐标点云
        cam_pos  : (3,) 相机位置（世界坐标）
        max_points: 为了性能，最多使用多少个点进行空闲空间更新
        t_step_factor: 射线步长 = voxel_size * t_step_factor
        """
        if len(pts_world) == 0:
            return

        cam_pos = np.asarray(cam_pos, dtype=np.float32)
        pts_world = np.asarray(pts_world, dtype=np.float32)

        # 使用 raycast_batch_points_gpu 计算所有射线路径上的体素
        free_space_indices, hit_indices = raycast.raycast_batch_points_gpu(
            origin=self.origin,
            voxel_size=self.voxel_size,
            dims=self.dims,
            cam_pos=cam_pos,
            target_points=pts_world,
            max_points=max_points,
            t_step=self.voxel_size * t_step_factor,
            device=None
        )

        # 1) 更新路径上的体素为“空闲”（概率往 0 推）
        for i, j, k in free_space_indices:
            self.log_odds[i, j, k] += self.l_miss  # l_miss 是负数，会降低概率

        # 2) 更新点本身为“占据”（概率往 1 推）
        for i, j, k in hit_indices:
            self.log_odds[i, j, k] += self.l_hit

        # 3) 限制 log_odds 范围，避免数值爆炸
        self.log_odds = np.clip(self.log_odds, -50.0, 50.0)