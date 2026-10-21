import numpy as np
from skimage.measure import marching_cubes  # 用于提取 Mesh


class TSDFVolume:
    """
    基于稠密网格的 TSDF (Truncated Signed Distance Function) 地图。
    适用于高精度表面重建（如辣椒、细茎）。
    """

    def __init__(
            self,
            origin,  # 世界坐标原点 [x, y, z]
            voxel_size=0.005,  # 体素大小 [m]，建议设小一点 (e.g. 5mm) 以捕捉细节
            dims=(100, 100, 100),  # 网格尺寸 (Nx, Ny, Nz)
            trunc_margin=0.05  # 截断距离 [m]，通常设为 5-10 倍 voxel_size
    ):
        self.origin = np.asarray(origin, dtype=np.float32)
        self.voxel_size = float(voxel_size)
        self.dims = np.array(dims, dtype=np.int32)
        self.trunc_margin = float(trunc_margin)

        # === 核心数据存储 ===
        # 1. TSDF 值: 范围 [-1, 1]，表示归一化的距离。
        #    1.0 表示在截断距离外（空闲），0.0 表示表面，-1.0 表示在物体内部深处
        self._tsdf_vol = np.ones(self.dims, dtype=np.float32)

        # 2. 权重: 记录该体素被观测到的置信度/次数
        self._weight_vol = np.zeros(self.dims, dtype=np.float32)

        # 3. 颜色/语义: 这里演示存储语义概率 (Class 0: Pepper)
        #    范围 [0, 1]，也是通过加权平均更新
        self._sem_vol = np.zeros(self.dims, dtype=np.float32)

        # 预计算网格坐标缓存（加速积分过程）
        self._coords_world = self._precompute_grid_coords()

    def _precompute_grid_coords(self):
        """生成所有体素中心的世界坐标 (N, 3)，用于后续向量化投影"""
        # 生成网格索引 (Nx, Ny, Nz)
        x_range = np.arange(self.dims[0])
        y_range = np.arange(self.dims[1])
        z_range = np.arange(self.dims[2])
        grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range, indexing='ij')

        # 转换为世界坐标
        # coords shape: (3, Nx, Ny, Nz) -> (3, N_total)
        coords = np.stack([grid_x, grid_y, grid_z], axis=0)
        coords = coords.astype(np.float32) * self.voxel_size
        coords[0] += self.origin[0]
        coords[1] += self.origin[1]
        coords[2] += self.origin[2]

        # 展平以便矩阵乘法: (3, N)
        return coords.reshape(3, -1)

    def integrate(self, depth_img, intrinsic, cam_pose,
                  sem_img=None, obs_weight=1.0):
        """
        融合一帧深度图 (和可选的语义图) 到 TSDF。

        depth_img: (H, W) float32, 单位米. 0 代表无效值
        intrinsic: (3, 3) 相机内参矩阵 K
        cam_pose : (4, 4) 相机到世界的变换矩阵 T_world_cam
        sem_img  : (H, W) float32, 语义概率图 (可选)
        obs_weight: 当前帧的权重，通常设为 1.0
        """
        H, W = depth_img.shape

        # 1. 计算 世界 -> 相机 的变换矩阵 (T_cam_world)
        T_cam_world = np.linalg.inv(cam_pose)

        # 2. 将所有体素中心投影到相机坐标系
        # coords_cam = R * coords_world + t
        # (3, N)
        coords_world_homo = np.vstack([self._coords_world, np.ones((1, self._coords_world.shape[1]))])
        coords_cam = T_cam_world @ coords_world_homo
        coords_cam = coords_cam[:3, :]  # 去掉齐次坐标

        # 3. 投影到图像平面 (u, v)
        # z (depth)
        z_vals = coords_cam[2, :]

        # 避免除以 0 或 负深度（在相机后面的点）
        valid_z_mask = z_vals > 0.0001

        # u = fx * x / z + cx, v = fy * y / z + cy
        # 使用矩阵乘法: pixel_coords = K @ coords_cam
        pixel_coords = intrinsic @ coords_cam
        u_vals = (pixel_coords[0, :] / (pixel_coords[2, :] + 1e-8)).astype(np.int32)
        v_vals = (pixel_coords[1, :] / (pixel_coords[2, :] + 1e-8)).astype(np.int32)

        # 4. 筛选投影在图像范围内的点
        valid_uv_mask = (u_vals >= 0) & (u_vals < W) & (v_vals >= 0) & (v_vals < H)
        valid_mask = valid_z_mask & valid_uv_mask

        if not np.any(valid_mask):
            return  # 没有体素在视野内

        # 获取有效索引
        valid_indices = np.where(valid_mask)[0]
        u_valid = u_vals[valid_indices]
        v_valid = v_vals[valid_indices]
        z_valid = z_vals[valid_indices]  # 体素到相机的距离

        # 5. 读取深度图对应的值
        d_measured = depth_img[v_valid, u_valid]

        # 过滤无效深度 (深度图中值为0或极大值的点)
        valid_depth_mask = (d_measured > 0.1) & (d_measured < 5.0)  # 假设有效范围

        # 再次筛选
        final_indices = valid_indices[valid_depth_mask]
        d_measured = d_measured[valid_depth_mask]
        z_valid = z_valid[valid_depth_mask]

        if len(final_indices) == 0:
            # 调试：仅首次提示，帮助定位深度/投影问题
            if not hasattr(self, "_debug_integrate_printed"):
                print("[TSDF Debug] No valid voxels in this frame. "
                      f"depth range=({depth_img.min():.3f},{depth_img.max():.3f}), "
                      f"valid_depth_count={(valid_depth_mask).sum()}")
                self._debug_integrate_printed = True
            return

        # 6. === TSDF 核心计算 ===
        # SDF = 测量深度 - 体素深度
        # 如果 sdf > 0: 体素在表面前面（空闲）
        # 如果 sdf < 0: 体素在表面后面（物体内部）
        sdf = d_measured - z_valid

        # 如果 sdf < -trunc_margin，说明体素在表面后面很远的地方，我们不更新它
        # 因为我们只关心表面附近的窄带，且这种点可能是遮挡关系造成的
        valid_sdf_mask = sdf >= -self.trunc_margin

        update_indices = final_indices[valid_sdf_mask]
        sdf_values = sdf[valid_sdf_mask]

        # 截断并归一化到 [-1, 1]
        tsdf_values = np.clip(sdf_values / self.trunc_margin, -1, 1)

        # 7. 加权平均融合
        # D_new = (W_old * D_old + W_obs * D_obs) / (W_old + W_obs)
        # 这里使用平展的一维数组进行索引
        curr_tsdf = self._tsdf_vol.reshape(-1)[update_indices]
        curr_weight = self._weight_vol.reshape(-1)[update_indices]

        new_weight = curr_weight + obs_weight
        new_tsdf = (curr_tsdf * curr_weight + tsdf_values * obs_weight) / new_weight

        # 8. 写回数据
        np.put(self._tsdf_vol, update_indices, new_tsdf)
        np.put(self._weight_vol, update_indices, np.clip(new_weight, 0, 100))  # 权重截断防止溢出

        # 9. 语义融合 (如果有)
        if sem_img is not None:
            # 读取对应 uv 的语义概率
            # 假设 sem_img 也是 (H, W)，值在 0-1 之间
            sem_val = sem_img[v_vals[update_indices], u_vals[update_indices]]  # 注意索引顺序

            curr_sem = self._sem_vol.reshape(-1)[update_indices]
            # 同样使用加权平均
            new_sem = (curr_sem * curr_weight + sem_val * obs_weight) / new_weight
            np.put(self._sem_vol, update_indices, new_sem)

    def get_mesh(self):
        """
        使用 Marching Cubes 提取等值面 Mesh。
        返回: vertices (N, 3), faces (M, 3), normals (N, 3)
        """
        # 提取 Isosurface (SDF=0 的面)
        # 注意: skimage 的 marching_cubes 在旧版本叫 marching_cubes_lewiner
        try:
            verts, faces, normals, values = marching_cubes(self._tsdf_vol, level=0.0)
        except ValueError:
            # 如果没有表面（全是空的或全是满的）
            return None, None, None
        except ImportError:
            print("Please install scikit-image: pip install scikit-image")
            return None, None, None

        # verts 返回的是 grid index (比如 10.5, 20.1)，需要转回世界坐标
        verts_world = verts * self.voxel_size + self.origin

        return verts_world, faces, normals

    def get_point_cloud(self, filter_val=0.2):
        """
        简单的点云提取，返回所有靠近表面的体素中心。
        """
        # 找到 |SDF| 比较小 且 权重 > 0 的点
        mask = (np.abs(self._tsdf_vol) < filter_val) & (self._weight_vol > 0.5)
        idx = np.column_stack(np.nonzero(mask))

        points = idx * self.voxel_size + self.origin

        # 获取对应的语义
        semantics = self._sem_vol[mask]

        return points, semantics

    def get_weights(self):
        """NBV 需要用到权重分布"""
        return self._weight_vol

    def get_occupancy(self):
        """
        为了兼容旧的 NBV 代码，将 TSDF 转为近似占据概率。
        TSDF > 0 (空) -> P < 0.5
        TSDF = 0 (表面) -> P = 0.5 (或者高概率，看定义)
        TSDF < 0 (内部) -> P > 0.5
        """
        # 这是一个简单的映射，实际中 SDF 更精细
        # 映射: -1 -> 1.0 (Occupied), 1 -> 0.0 (Free)
        return 0.5 * (1.0 - self._tsdf_vol)

    def get_occupancy_prob(self):
        """
        为了兼容旧的 NBV 代码，将 TSDF 转为近似占据概率。
        TSDF > 0 (空) -> P < 0.5
        TSDF = 0 (表面) -> P = 0.5 (或者高概率，看定义)
        TSDF < 0 (内部) -> P > 0.5
        """
        # 这是一个简单的映射，实际中 SDF 更精细
        # 映射: -1 -> 1.0 (Occupied), 1 -> 0.0 (Free)
        return 0.5 * (1.0 - self._tsdf_vol)