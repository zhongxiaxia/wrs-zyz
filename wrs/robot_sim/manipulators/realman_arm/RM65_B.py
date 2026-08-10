import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim.manipulators.manipulator_interface as mi
import wrs.modeling.collision_model as mcm

# 可选：TracIK（如果装了就用它做 IK）
try:
    from trac_ik import TracIK
    _HAS_TRACIK = True
    print("Trac IK module loaded successfully")
except Exception as e:
    print(f"Trac IK module not loaded: {e}")
    _HAS_TRACIK = False


class RM65B(mi.ManipulatorInterface):
    """
    RM65-B 仿真接口（按 confirm 的 URDF 参数搭建）
    - joint origin xyz/rpy: 完全来自你贴的 RM65-B URDF
    - limit lower/upper: 完全来自 URDF（弧度）
    - link mesh: 默认读取 current_dir/meshes/*.STL（按 URDF 文件名）
    """

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 ik_solver='d',
                 name='RM65B',
                 enable_cc=False,
                 use_tracik=True):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=np.zeros(6), name=name, enable_cc=enable_cc)

        current_file_dir = os.path.dirname(__file__)
        mesh_dir = os.path.join(current_file_dir, "meshes")

        # ====== mesh 文件名映射（如你的 STL 命名不同，在这里改）======
        mesh_map = {
            "base_link": "base_link.STL",
            "link1": "link_1.STL",
            "link2": "link_2.STL",
            "link3": "link_3.STL",
            "link4": "link_4.STL",
            "link5": "link_5.STL",
            "link6": "link_6.STL",
        }

        rgba = np.array([0.79216, 0.81961, 0.93333, 1.0])

        # ================= anchor (base_link) =================
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(os.path.join(mesh_dir, mesh_map["base_link"]))
        self.jlc.anchor.lnk_list[0].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = rgba

        # ================= joint_1 / link_1 =================
        # URDF: origin xyz="0 0 0.2405" rpy="0 0 0"
        self.jlc.jnts[0].loc_pos = np.array([0.0, 0.0, 0.2405])
        self.jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[0].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        # URDF limit: lower=-3.106 upper=3.106
        self.jlc.jnts[0].motion_range = np.array([-3.106, 3.106])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(mesh_dir, mesh_map["link1"]))
        self.jlc.jnts[0].lnk.loc_pos = np.zeros(3)
        self.jlc.jnts[0].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[0].lnk.cmodel.rgba = rgba

        # ================= joint_2 / link_2 =================
        # URDF: origin xyz="0 0 0" rpy="1.5708 -1.5708 0"
        self.jlc.jnts[1].loc_pos = np.array([0.0, 0.0, 0.0])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(1.5708, -1.5708, 0.0)
        self.jlc.jnts[1].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        # URDF limit: lower=-2.2689 upper=2.2689
        self.jlc.jnts[1].motion_range = np.array([-2.2689, 2.2689])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(mesh_dir, mesh_map["link2"]))
        self.jlc.jnts[1].lnk.loc_pos = np.zeros(3)
        self.jlc.jnts[1].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[1].lnk.cmodel.rgba = rgba

        # ================= joint_3 / link_3 =================
        # URDF: origin xyz="0.256 0 0" rpy="0 0 1.5708"
        self.jlc.jnts[2].loc_pos = np.array([0.256, 0.0, 0.0])
        self.jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 1.5708)
        self.jlc.jnts[2].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        # URDF limit: lower=-2.356 upper=2.356
        self.jlc.jnts[2].motion_range = np.array([-2.356, 2.356])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(os.path.join(mesh_dir, mesh_map["link3"]))
        self.jlc.jnts[2].lnk.loc_pos = np.zeros(3)
        self.jlc.jnts[2].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[2].lnk.cmodel.rgba = rgba

        # ================= joint_4 / link_4 =================
        # URDF: origin xyz="0 -0.21 0" rpy="1.5708 0 0"
        self.jlc.jnts[3].loc_pos = np.array([0.0, -0.21, 0.0])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(1.5708, 0.0, 0.0)
        self.jlc.jnts[3].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        # URDF limit: lower=-3.106 upper=3.106
        self.jlc.jnts[3].motion_range = np.array([-3.106, 3.106])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(os.path.join(mesh_dir, mesh_map["link4"]))
        self.jlc.jnts[3].lnk.loc_pos = np.zeros(3)
        self.jlc.jnts[3].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[3].lnk.cmodel.rgba = rgba

        # ================= joint_5 / link_5 =================
        # URDF: origin xyz="0 0 0" rpy="-1.5708 0 0"
        self.jlc.jnts[4].loc_pos = np.array([0.0, 0.0, 0.0])
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(-1.5708, 0.0, 0.0)
        self.jlc.jnts[4].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        # URDF limit: lower=-2.234 upper=2.234
        self.jlc.jnts[4].motion_range = np.array([-2.234, 2.234])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(os.path.join(mesh_dir, mesh_map["link5"]))
        self.jlc.jnts[4].lnk.loc_pos = np.zeros(3)
        self.jlc.jnts[4].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[4].lnk.cmodel.rgba = rgba

        # ================= joint_6 / link_6 =================
        # URDF: origin xyz="0 -0.144 0" rpy="1.5708 0 0"
        self.jlc.jnts[5].loc_pos = np.array([0.0, -0.144, 0.0])
        # self.jlc.jnts[5].loc_pos = np.array([.0, -0.1612, .0])
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(1.5708, 0.0, 0.0)
        self.jlc.jnts[5].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        # URDF limit: lower=-6.28 upper=6.28
        self.jlc.jnts[5].motion_range = np.array([-6.28, 6.28])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(os.path.join(mesh_dir, mesh_map["link6"]))
        self.jlc.jnts[5].lnk.loc_pos = np.zeros(3)
        self.jlc.jnts[5].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[5].lnk.cmodel.rgba = rgba

        # finalize JLC (build FK/IK model)
        self.jlc.finalize(ik_solver=ik_solver, identifier_str=name)

        # TCP (默认法兰)
        self.loc_tcp_pos = np.zeros(3)
        self.loc_tcp_rotmat = np.eye(3)

        # optional TracIK (需要 realman.urdf 放在同目录或指定路径)
        self._ik_solver = None
        if use_tracik and _HAS_TRACIK:
            directory = os.path.abspath(os.path.dirname(__file__))
            urdf_path = os.path.join(directory, "RM65-B.urdf")  # 你也可以改成 "realman.urdf"
            # tip link 用 link_6 或 link6 取决于 urdf 里 link 名
            self._ik_solver = TracIK("base_link", "link_6", urdf_path,
                                     timeout=0.04, solver_type="Distance")

        # collision checking
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        lb = self.cc.add_cce(self.jlc.anchor.lnk_list[0])
        l0 = self.cc.add_cce(self.jlc.jnts[0].lnk)
        l1 = self.cc.add_cce(self.jlc.jnts[1].lnk)
        l2 = self.cc.add_cce(self.jlc.jnts[2].lnk)
        l3 = self.cc.add_cce(self.jlc.jnts[3].lnk)
        l4 = self.cc.add_cce(self.jlc.jnts[4].lnk)
        l5 = self.cc.add_cce(self.jlc.jnts[5].lnk)

        from_list = [l3, l4, l5]
        into_list = [lb, l0]
        self.cc.set_cdpair_by_ids(from_list, into_list)

    def ik(self, tgt_pos, tgt_rotmat, seed_jnt_values=None, option="empty", toggle_dbg=False):
        # 先把 TCP 偏置“剥掉”
        tgt_rotmat = tgt_rotmat @ self.loc_tcp_rotmat.T
        tgt_pos = tgt_pos - tgt_rotmat @ self.loc_tcp_pos

        if _HAS_TRACIK and self._ik_solver is not None:
            anchor_inv = np.linalg.inv(rm.homomat_from_posrot(self.jlc.anchor.pos, self.jlc.anchor.rotmat))
            tgt_hm = anchor_inv.dot(rm.homomat_from_posrot(tgt_pos, tgt_rotmat))
            p, R = tgt_hm[:3, 3], tgt_hm[:3, :3]
            seed = self.home_conf if seed_jnt_values is None else seed_jnt_values.copy()
            return self._ik_solver.ik(p, R, seed_jnt_values=seed)
        else:
            return self.jlc.ik(tgt_pos=tgt_pos,
                               tgt_rotmat=tgt_rotmat,
                               seed_jnt_values=seed_jnt_values,
                               toggle_dbg=toggle_dbg)


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd
    import wrs.modeling.geometric_model as mgm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)

    arm = RM65B(enable_cc=True)

    # 随便给一组“弧度”关节角测试
    q = np.array([0.2, 0.1, 1.2, 0.0, 0.5, 0.0])
    arm.goto_given_conf(q)

    arm.gen_meshmodel(alpha=0.6, toggle_tcp_frame=True).attach_to(base)
    base.run()
