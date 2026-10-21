import numpy as np
import torch


def raycast_voxels_gpu(origin, voxel_size, dims,
                       cam_pos, directions,
                       t_min=0.0, t_max=5.0, t_step=None,
                       obj_min=None, obj_max=None,
                       return_first_only=False,
                       device=None):
    """
    GPU 加速的批量射线投射，计算多条射线经过的体素索引。
    
    参数:
        origin: (3,) 体素网格世界坐标原点
        voxel_size: float 体素边长
        dims: (3,) 体素网格尺寸 (Nx, Ny, Nz)
        cam_pos: (3,) 或 (N, 3) 相机位置（世界坐标）
        directions: (N, 3) 射线方向向量（世界坐标，已归一化）
        t_min: float 射线起始距离
        t_max: float 射线最大距离
        t_step: float 步长，如果 None 则自动计算为 voxel_size * sqrt(3)
        obj_min: (3,) 可选，AABB 最小值，只返回在 bbox 内的体素
        obj_max: (3,) 可选，AABB 最大值
        return_first_only: bool 如果 True，每条射线只返回第一个有效体素
        device: torch.device 计算设备，如果 None 则自动选择
    
    返回:
        如果 return_first_only=False:
            list of arrays，每个元素是 (M_i, 3) 的体素索引数组，对应第 i 条射线
        如果 return_first_only=True:
            (N, 3) 数组，每条射线第一个有效体素的索引，无效的为 (-1, -1, -1)
    """
    # 设置设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 转换为 tensor
    origin_t = torch.tensor(origin, dtype=torch.float32, device=device)
    voxel_size_t = torch.tensor(voxel_size, dtype=torch.float32, device=device)
    dims_t = torch.tensor(dims, dtype=torch.long, device=device)
    
    cam_pos = np.asarray(cam_pos, dtype=np.float32)
    directions = np.asarray(directions, dtype=np.float32)
    
    # 处理 cam_pos 的形状
    if cam_pos.ndim == 1:
        cam_pos = cam_pos[None, :]  # (1, 3)
        N_rays = len(directions)
        cam_pos_t = torch.tensor(cam_pos, dtype=torch.float32, device=device).expand(N_rays, -1)
    else:
        N_rays = len(cam_pos)
        cam_pos_t = torch.tensor(cam_pos, dtype=torch.float32, device=device)
    
    directions_t = torch.tensor(directions, dtype=torch.float32, device=device)
    
    # 确保方向向量归一化
    directions_t = directions_t / torch.norm(directions_t, dim=1, keepdim=True)
    
    # 计算步长
    if t_step is None:
        t_step = voxel_size * np.sqrt(3)
    t_step_t = torch.tensor(t_step, dtype=torch.float32, device=device)
    
    # 生成步进距离
    n_steps = max(1, int((t_max - t_min) / t_step))
    t_values = torch.linspace(
        t_min + t_step,
        t_max,
        n_steps,
        dtype=torch.float32,
        device=device
    )  # (n_steps,)
    
    # 扩展维度用于批量计算
    # cam_pos_t: (N, 3) -> (N, 1, 3)
    # directions_t: (N, 3) -> (N, 1, 3)
    # t_values: (n_steps,) -> (1, n_steps, 1)
    cam_pos_expanded = cam_pos_t.unsqueeze(1)  # (N, 1, 3)
    dir_expanded = directions_t.unsqueeze(1)  # (N, 1, 3)
    t_expanded = t_values.unsqueeze(0).unsqueeze(-1)  # (1, n_steps, 1)
    
    # 所有射线上的所有采样点: (N, n_steps, 3)
    points = cam_pos_expanded + t_expanded * dir_expanded
    
    # 转换到体素索引
    idx_float = (points - origin_t) / voxel_size_t  # (N, n_steps, 3)
    idx = idx_float.long()  # (N, n_steps, 3)
    
    # 检查索引有效性
    valid_idx = (idx[:, :, 0] >= 0) & (idx[:, :, 0] < dims_t[0]) & \
                (idx[:, :, 1] >= 0) & (idx[:, :, 1] < dims_t[1]) & \
                (idx[:, :, 2] >= 0) & (idx[:, :, 2] < dims_t[2])  # (N, n_steps)
    
    # 如果指定了 AABB，进一步过滤
    if obj_min is not None and obj_max is not None:
        obj_min_t = torch.tensor(obj_min, dtype=torch.float32, device=device)
        obj_max_t = torch.tensor(obj_max, dtype=torch.float32, device=device)
        in_bbox = (points >= obj_min_t) & (points <= obj_max_t)  # (N, n_steps, 3)
        in_bbox_all = in_bbox.all(dim=2)  # (N, n_steps)
        valid_idx = valid_idx & in_bbox_all
    
    if return_first_only:
        # 每条射线只返回第一个有效体素
        first_valid = torch.argmax(valid_idx.int(), dim=1)  # (N,) - 第一条有效射线的索引
        has_valid = valid_idx.any(dim=1)  # (N,) - 哪些射线有有效体素
        
        result = torch.full((N_rays, 3), -1, dtype=torch.long, device=device)
        if has_valid.any():
            valid_ray_indices = torch.nonzero(has_valid, as_tuple=False).squeeze(-1)
            result[valid_ray_indices] = idx[valid_ray_indices, first_valid[valid_ray_indices]]
        
        return result.cpu().numpy()
    else:
        # 返回每条射线的所有有效体素
        results = []
        for i in range(N_rays):
            valid_mask = valid_idx[i]  # (n_steps,)
            if valid_mask.any():
                valid_indices = idx[i][valid_mask]  # (M, 3)
                # 去重（相邻步可能落在同一个体素）
                valid_indices_np = valid_indices.cpu().numpy()
                unique_indices = np.unique(valid_indices_np, axis=0)
                results.append(unique_indices)
            else:
                results.append(np.zeros((0, 3), dtype=np.int32))
        
        return results


def raycast_single_point_gpu(origin, voxel_size, dims,
                             cam_pos, target_point,
                             t_step=None,
                             device=None):
    """
    从相机位置到单个目标点的射线投射（GPU 加速）。
    用于体素网格更新。
    
    参数:
        origin: (3,) 体素网格世界坐标原点
        voxel_size: float 体素边长
        dims: (3,) 体素网格尺寸
        cam_pos: (3,) 相机位置
        target_point: (3,) 目标点位置
        t_step: float 步长
        device: torch.device
    
    返回:
        voxel_indices: (M, 3) 射线路径上的体素索引（不包括目标点本身）
    """
    direction = target_point - cam_pos
    dist = np.linalg.norm(direction)
    
    if dist < 1e-6:
        return np.zeros((0, 3), dtype=np.int32)
    
    direction = direction / dist
    
    # 计算最大距离（在到达目标点之前停止）
    if t_step is None:
        t_step = voxel_size * 0.5
    t_max = dist - voxel_size
    
    if t_max <= 0:
        return np.zeros((0, 3), dtype=np.int32)
    
    # 使用批量 raycast（只有一条射线）
    directions = direction[None, :]  # (1, 3)
    cam_pos_array = cam_pos[None, :]  # (1, 3)
    
    results = raycast_voxels_gpu(
        origin=origin,
        voxel_size=voxel_size,
        dims=dims,
        cam_pos=cam_pos_array,
        directions=directions,
        t_min=0.0,
        t_max=t_max,
        t_step=t_step,
        return_first_only=False,
        device=device
    )
    
    if len(results) > 0 and len(results[0]) > 0:
        return results[0]
    else:
        return np.zeros((0, 3), dtype=np.int32)


def raycast_batch_points_gpu(origin, voxel_size, dims,
                              cam_pos, target_points,
                              max_points=None,
                              t_step=None,
                              device=None):
    """
    批量处理从相机到多个目标点的射线投射（GPU 加速）。
    用于体素网格的空闲空间更新。
    
    参数:
        origin: (3,) 体素网格世界坐标原点
        voxel_size: float 体素边长
        dims: (3,) 体素网格尺寸
        cam_pos: (3,) 相机位置
        target_points: (N, 3) 目标点位置数组
        max_points: int 最多处理多少个点（None 表示全部）
        t_step: float 步长
        device: torch.device
    
    返回:
        free_space_indices: (M, 3) 所有射线路径上的体素索引（去重后）
        hit_indices: (N', 3) 目标点所在的体素索引（N' <= N，只包括有效的点）
    """
    target_points = np.asarray(target_points, dtype=np.float32)
    
    if len(target_points) == 0:
        return np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3), dtype=np.int32)
    
    # 限制处理的点数
    if max_points is not None and len(target_points) > max_points:
        indices = np.linspace(0, len(target_points)-1, max_points, dtype=int)
        target_points = target_points[indices]
    
    # 计算所有射线的方向
    directions = target_points - cam_pos  # (N, 3)
    dists = np.linalg.norm(directions, axis=1, keepdims=True)
    valid_mask = dists[:, 0] > 1e-6
    
    if not valid_mask.any():
        return np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3), dtype=np.int32)
    
    directions = directions[valid_mask]
    dists = dists[valid_mask]
    target_points = target_points[valid_mask]
    
    # 归一化方向
    directions = directions / dists
    
    # 计算每条射线的最大距离
    if t_step is None:
        t_step = voxel_size * 0.5
    t_maxs = dists[:, 0] - voxel_size
    
    # 使用统一的 t_max（取最小值，确保所有射线都能处理）
    t_max = float(np.min(t_maxs[t_maxs > 0]))
    
    if t_max <= 0:
        # 如果所有点都太近，只返回目标点的体素索引
        all_indices = []
        for pt in target_points:
            idx_float = (pt - origin) / voxel_size
            idx = idx_float.astype(np.int32)
            if (idx >= 0).all() and (idx < dims).all():
                all_indices.append(idx)
        
        if len(all_indices) > 0:
            hit_indices = np.array(all_indices)
            return np.zeros((0, 3), dtype=np.int32), hit_indices
        else:
            return np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3), dtype=np.int32)
    
    # 批量 raycast
    results = raycast_voxels_gpu(
        origin=origin,
        voxel_size=voxel_size,
        dims=dims,
        cam_pos=cam_pos,
        directions=directions,
        t_min=0.0,
        t_max=t_max,
        t_step=t_step,
        return_first_only=False,
        device=device
    )
    
    # 收集所有空闲空间体素
    all_free_indices = []
    for ray_indices in results:
        if len(ray_indices) > 0:
            all_free_indices.append(ray_indices)
    
    if len(all_free_indices) > 0:
        free_space_indices = np.unique(np.vstack(all_free_indices), axis=0)
    else:
        free_space_indices = np.zeros((0, 3), dtype=np.int32)
    
    # 计算目标点的体素索引
    hit_indices = []
    for pt in target_points:
        idx_float = (pt - origin) / voxel_size
        idx = idx_float.astype(np.int32)
        if (idx >= 0).all() and (idx < dims).all():
            hit_indices.append(idx)
    
    if len(hit_indices) > 0:
        hit_indices = np.array(hit_indices)
    else:
        hit_indices = np.zeros((0, 3), dtype=np.int32)
    
    return free_space_indices, hit_indices

