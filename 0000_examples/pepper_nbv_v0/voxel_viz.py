import numpy as np
from panda3d.core import (
    Geom, GeomNode, GeomVertexData, GeomVertexFormat,
    GeomVertexWriter, GeomTriangles, OmniBoundingVolume
)
from wrs.visualization.panda.panda3d_utils import ExtraWindow


def uncertainty_to_rgba(probs, alpha=0.8):
    """
    把占据概率映射成颜色：
    - 越接近 0.5 越红（不确定）
    - 越接近 0 或 1 越蓝（确定）
    """
    probs = np.asarray(probs, dtype=np.float32)
    uncertainty = 1.0 - np.abs(probs - 0.5) / 0.5
    colors = np.zeros((len(probs), 4), dtype=np.float32)
    colors[:, 0] = uncertainty
    colors[:, 2] = 1.0 - uncertainty
    colors[:, 3] = alpha
    return colors


class VoxelGridVisualizer:
    """
    独立窗口中的体素可视化器，把每个体素画成一个小方块。
    为了效率，默认限制可视体素数量（max_voxels）。
    """

    def __init__(self, base,
                 window_origin=(640, 0),
                 size=(640, 480),
                 cam_pos=(0.5, -0.5, 0.5),
                 lookat=(0, 0, 0.05),
                 voxel_size=0.01,
                 max_voxels=200000):
        self.voxel_size = float(voxel_size)
        self.max_voxels = int(max_voxels)

        # 创建单独的可视化窗口
        self.window = ExtraWindow(base,
                                  cam_pos=list(cam_pos),
                                  lookat_pos=list(lookat),
                                  w=size[0], h=size[1])
        self.window.set_origin(np.array(window_origin))

        # # 确保窗口可拖动
        # from panda3d.core import WindowProperties
        # win_props = WindowProperties()
        # win_props.setOrigin(window_origin[0], window_origin[1])
        # win_props.setSize(size[0], size[1])
        # win_props.setUndecorated(False)  # 确保窗口有标题栏（可拖动）
        # win_props.setFixedSize(False)    # 确保窗口大小不固定
        # self.window.win.requestProperties(win_props)

        # Panda3D 节点准备
        self.node = GeomNode('voxel_grid')
        self.node.setBounds(OmniBoundingVolume())
        self.node.setFinal(True)
        self.nodepath = self.window.render.attachNewNode(self.node)

        self.vertex_format = GeomVertexFormat.getV3c4()
        self.clear()

    def clear(self):
        """清空当前可视化"""
        self.node.removeAllGeoms()

    def update(self, centers, colors=None):
        """
        用一组体素中心和颜色刷新可视化。
        centers: (N,3)
        colors : (N,4)，如果 None 则用固定颜色
        """
        if len(centers) == 0:
            self.clear()
            return

        centers = np.asarray(centers, dtype=np.float32)
        if colors is None:
            colors = np.tile(np.array([[0.9, 0.1, 0.1, 0.8]], dtype=np.float32), (len(centers), 1))
        else:
            colors = np.asarray(colors, dtype=np.float32)

        if len(centers) > self.max_voxels:
            idx = np.random.choice(len(centers), self.max_voxels, replace=False)
            centers = centers[idx]
            colors = colors[idx]

        geom = self._build_voxel_geom(centers, colors)
        self.node.removeAllGeoms()
        self.node.addGeom(geom)

    def _build_voxel_geom(self, centers, colors):
        """
        把所有体素拼成一个 Panda3D Geom；每个体素用 12 个三角面。
        """
        half = self.voxel_size * 0.5
        vdata = GeomVertexData('voxels', self.vertex_format, Geom.UHStatic)
        vdata.setNumRows(len(centers) * 8)

        writer_v = GeomVertexWriter(vdata, 'vertex')
        writer_c = GeomVertexWriter(vdata, 'color')

        triangles = GeomTriangles(Geom.UHStatic)

        # 立方体 8 个顶点的偏移
        cube_offsets = np.array([
            [-half, -half, -half],
            [ half, -half, -half],
            [ half,  half, -half],
            [-half,  half, -half],
            [-half, -half,  half],
            [ half, -half,  half],
            [ half,  half,  half],
            [-half,  half,  half],
        ], dtype=np.float32)

        # 立方体三角面（以 0..7 顶点索引）
        faces = [
            (0, 1, 2), (0, 2, 3),  # bottom
            (4, 5, 6), (4, 6, 7),  # top
            (0, 1, 5), (0, 5, 4),  # front
            (1, 2, 6), (1, 6, 5),  # right
            (2, 3, 7), (2, 7, 6),  # back
            (3, 0, 4), (3, 4, 7),  # left
        ]

        for idx, center in enumerate(centers):
            base_vert_id = idx * 8
            cube_vertices = center + cube_offsets
            for v in cube_vertices:
                writer_v.addData3f(v[0], v[1], v[2])
            for _ in cube_vertices:
                c = colors[idx]
                writer_c.addData4f(float(c[0]), float(c[1]), float(c[2]), float(c[3]))

            for tri in faces:
                triangles.addVertices(base_vert_id + tri[0],
                                      base_vert_id + tri[1],
                                      base_vert_id + tri[2])

        geom = Geom(vdata)
        triangles.closePrimitive()
        geom.addPrimitive(triangles)
        return geom