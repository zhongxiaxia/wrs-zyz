import os
import numpy as np
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.end_effectors.grippers.gripper_interface as gpi
import wrs.modeling.model_collection as mmc


class Hand(gpi.GripperInterface):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), ik_solver='a', name='hand',
                 cdmesh_type=mcm.const.CDMeshType.DEFAULT):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        current_file_dir = os.path.dirname(__file__)
        self.coupling.loc_flange_pose_list[0] = [np.zeros(3), np.eye(3)]
        # jaw range
        self.jaw_range = np.array([.0, .03])
        self.jlc = rkjlc.JLChain(pos=self.coupling.gl_flange_pose_list[0][0],
                                 rotmat=self.coupling.gl_flange_pose_list[0][1], n_dof=0, name=name)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "linzhi.stl"), cdmesh_type=self.cdmesh_type)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = np.array([.75, .75, .75, 1])
        self.jlc.finalize()
        self.loc_acting_center_pos = np.array([0, 0, .20]) #

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.coupling.pos = self._pos
        self.coupling.rotmat = self._rotmat
        self.jlc.fix_to(self.coupling.gl_flange_pose_list[0][0], self.coupling.gl_flange_pose_list[0][1])
        self.update_oiee()

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False, name='yumi_gripper_stickmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(toggle_root_frame=False, toggle_flange_frame=False).attach_to(m_col)
        self.jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames, toggle_flange_frame=False).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name='hand_meshmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.coupling.gen_meshmodel(rgb=rgb,
                                    alpha=alpha,
                                    toggle_root_frame=False,
                                    toggle_flange_frame=False,
                                    toggle_cdmesh=toggle_cdmesh,
                                    toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.jlc.gen_meshmodel(rgb=rgb,
                               alpha=alpha,
                               toggle_flange_frame=False,
                               toggle_jnt_frames=toggle_jnt_frames,
                               toggle_cdmesh=toggle_cdmesh,
                               toggle_cdprim=toggle_cdprim).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        # oiee
        self._gen_oiee_meshmodel(m_col, rgb=rgb, alpha=alpha, toggle_cdprim=toggle_cdprim,
                                 toggle_cdmesh=toggle_cdmesh, toggle_frame=toggle_jnt_frames)
        return m_col

    def get_jaw_width(self):
        return .1

    def change_ee_values(self, ee_values):
        """
        interface
        :param ee_values:
        :return:
        """
        return ...


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    # Initialize simulation world
    base = wd.World(cam_pos=[1, 0.5, 0.3], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)

    # Create Omnipicker object
    grpr = Hand()

    # Visualize gripper model
    grpr.gen_meshmodel(alpha=0.5,toggle_tcp_frame=True).attach_to(base)

    # Run simulation
    base.run()
