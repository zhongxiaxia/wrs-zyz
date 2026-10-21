import numpy as np

def world_aabb_from_obj(obj):
    """
    计算一个 GeometricModel 在世界坐标系下的 AABB 包围盒。

    参数:
        obj: wrs.modeling.geometric_model.GeometricModel 实例
             需要有 obj.trm_mesh, obj.pos, obj.rotmat

    返回:
        bbox_min, bbox_max: 两个 1x3 np.array，分别是 (xmin, ymin, zmin) 和 (xmax, ymax, zmax)
    """
    if obj.trm_mesh is None:
        raise ValueError("obj.trm_mesh is None, 仅适用于有 trimesh 的模型")

    # 局部顶点 -> 世界坐标
    verts_local = obj.trm_mesh.vertices              # (N, 3)
    verts_world = verts_local @ obj.rotmat.T + obj.pos

    bbox_min = verts_world.min(axis=0)
    bbox_max = verts_world.max(axis=0)
    return bbox_min, bbox_max


def aabb_from_mesh(mesh):
    """
    直接从 trimesh.Trimesh 计算局部坐标下的 AABB。
    """
    return mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)