from Robotic_Arm.rm_robot_interface import *
import time
import numpy as np
import wrs.basis.robot_math as rm

try:
    import wrs.motion.trajectory.piecewisepoly_toppra as pwp

    TOPPRA_EXIST = True
except:
    TOPPRA_EXIST = False


class RealmanArmController:
    def __init__(
            self,
            ip: str = "192.168.1.18",
            port: int = 8080,
            has_gripper=False,
            multiple_thread=True,
    ):
        """
        :param ip: 机器人的 IP 地址
        :param port: 机器人的端口号

        :param has_gripper: 是否安装夹爪
        """
        # 初始化 Realman 机器人控制类
        if multiple_thread:
            mode = rm_thread_mode_e.RM_TRIPLE_MODE_E
        else:
            mode = None
        self.arm = RoboticArm(mode)
        self.handle = self.arm.rm_create_robot_arm(ip, port)  # 创建机器人连接
        print(f"连接ID: {self.handle.id}")  # 打印连接 ID
        time.sleep(.5)

        self._has_gripper = has_gripper
        if has_gripper:
            # 如果安装了夹爪，初始化夹爪控制逻辑
            pass

    def move_j(self, joint_angles: list, speed: int = 5, is_block=True):
        """
        控制机器人运动到指定的关节角度位置
        :param joint_angles: 目标关节角度，六个角度值，单位为度
        :param speed: 运动速度，单位为度/秒
        :param accel: 加速度
        :param decel: 减速度
        """
        joint_angles = np.degrees(joint_angles)
        result = self.arm.rm_movej(joint_angles, speed, 50, 0, int(is_block))
        if result == 0:
            print(f"成功移动到关节位置: {joint_angles}")
        else:
            print("关节运动失败")

    def move_p(self, pose: list, speed: int = 5, accel: int = 0, decel: int = 0, connect: int = 1):
        """
        控制机器人移动到指定的位姿
        :param pose: 目标位姿，包含位置和姿态，六个元素：[x, y, z, roll, pitch, yaw]
        :param speed: 运动速度
        :param accel: 加速度
        :param decel: 减速度
        :param connect: 是否进行连接，1 表示连接，0 表示不连接
        """
        result = self.arm.rm_movej_p(pose, speed, accel, decel, connect)
        if result == 0:
            print(f"成功移动到位姿: {pose}")
        else:
            print("位姿运动失败")

    def get_joint_values(self, retries: int = 5, retry_delay_s: float = 0.05):
        """Read joint values, retrying transient SDK receive errors.

        State queries are safe to retry because they do not issue a motion
        command. A bounded retry prevents one short network/SDK receive error
        from aborting an otherwise successful trajectory.
        """
        retries = int(retries)
        retry_delay_s = float(retry_delay_s)
        if retries < 1:
            raise ValueError("retries must be at least 1")
        if not np.isfinite(retry_delay_s) or retry_delay_s < 0:
            raise ValueError("retry_delay_s must be a finite non-negative value")

        last_error = "unknown error"
        for attempt in range(1, retries + 1):
            try:
                result = self.arm.rm_get_joint_degree()
                if not isinstance(result, (tuple, list)) or len(result) < 2:
                    last_error = f"invalid SDK result: {result!r}"
                else:
                    err_code, joint_values = result[0], result[1]
                    if err_code != 0:
                        last_error = f"SDK return_code={err_code}"
                    else:
                        joint_values = np.asarray(joint_values, dtype=float).reshape(-1)
                        if joint_values.size != 6:
                            last_error = (
                                "invalid joint count: "
                                f"expected 6, got {joint_values.size}"
                            )
                        elif not np.all(np.isfinite(joint_values)):
                            last_error = "SDK returned non-finite joint values"
                        else:
                            return np.radians(joint_values)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt < retries:
                print(
                    "[RealMan][关节角读取重试] "
                    f"第 {attempt}/{retries} 次失败：{last_error}；"
                    f"{retry_delay_s:.3f} 秒后重试。"
                )
                if retry_delay_s > 0:
                    time.sleep(retry_delay_s)

        raise RuntimeError(
            f"获取关节角失败：连续 {retries} 次读取均失败；最后错误：{last_error}"
        )

    def _send_movej_canfd_checked(self,
                                  joint_values_deg,
                                  *,
                                  context: str,
                                  sample_index: int,
                                  sample_count: int):
        """Send one CAN-FD waypoint and fail immediately on an SDK error.

        Motion commands are deliberately not retried: if an acknowledgement is
        lost, the controller may already have accepted the command, so resending
        it automatically would make execution state ambiguous.
        """
        joint_values_deg = np.asarray(joint_values_deg, dtype=float).reshape(-1)
        if joint_values_deg.size != 6 or not np.all(np.isfinite(joint_values_deg)):
            raise ValueError(
                "CAN-FD trajectory sample must contain 6 finite joint values; "
                f"context={context}, sample={sample_index}/{sample_count}"
            )

        result = self.arm.rm_movej_canfd(
            joint_values_deg,
            False,
            0,
            trajectory_mode=1,
            radio=50,
        )
        if result != 0:
            raise RuntimeError(
                "CAN-FD轨迹点下发失败："
                f"轨迹段={context}，采样点={sample_index}/{sample_count}，"
                f"SDK return_code={result}。运动命令未自动重发。"
            )
        return result

    def get_pose(self) -> (np.ndarray, np.ndarray):
        """
        获取当前机器人的位姿
        :return: 返回位姿信息，只包含位置和姿态
        """
        pose_data = self.arm.rm_get_current_arm_state()  # 获取原始数据
        if pose_data[0] < 0:
            raise Exception("Getting pose error")
        pose = pose_data[1].get('pose', [0, 0, 0, 0, 0, 0])  # 从元组中提取字典，然后获取 'pose'
        pose_np = np.asarray(pose)
        return pose_np[:3], rm.rotmat_from_euler(*pose[3:])

    def get_ee_force(self):
        force_data = self.arm.rm_get_force_data()
        return np.asarray(force_data[1].get('force_data', [0, 0, 0, 0, 0, 0]))

    def wait_until_idle(self, timeout: float = 30.0, poll_dt: float = 0.05, stable_count: int = 3):
        """Best-effort wait until the robot stops changing pose.

        The Realman SDK wrapper used in this project does not expose a clearly documented
        blocking-motion-complete API here, so we infer idleness by polling the current arm
        state/pose and waiting until it stays stable for several consecutive samples.
        """
        t0 = time.time()
        last_pose = None
        stable = 0
        while True:
            state = self.arm.rm_get_current_arm_state()
            if state[0] < 0:
                raise RuntimeError("Failed to query current arm state while waiting for motion to finish")

            pose_dict = state[1] if len(state) > 1 and isinstance(state[1], dict) else {}
            pose = np.asarray(pose_dict.get('pose', []), dtype=float)

            if pose.size >= 6:
                cur_pose = pose[:6]
                if last_pose is not None and np.allclose(cur_pose, last_pose, atol=1e-4, rtol=0.0):
                    stable += 1
                else:
                    stable = 0
                last_pose = cur_pose
                if stable >= stable_count:
                    return True
            else:
                # If pose is not available, fall back to waiting a short fixed interval.
                # This still blocks the caller and is safer than returning immediately.
                time.sleep(poll_dt)
                if time.time() - t0 > timeout:
                    raise TimeoutError("wait_until_idle timeout while pose data was unavailable")
                continue

            if time.time() - t0 > timeout:
                raise TimeoutError("wait_until_idle timeout")
            time.sleep(poll_dt)

    def wait_until_joint_target(self,
                                target_jnt_values,
                                timeout: float = 10.0,
                                poll_dt: float = 0.05,
                                tolerance_deg: float = 0.5,
                                stable_count: int = 3,
                                motion_tolerance_deg: float = None):
        """Wait until measured joints reach the target and stop moving.

        Joint-target confirmation is more reliable than inferring idleness from
        Cartesian pose.  When ``motion_tolerance_deg`` is provided, each joint
        must also change by no more than that amount between consecutive samples.
        Several consecutive valid samples are required so one noisy reading
        cannot report completion prematurely.
        """
        target = np.asarray(target_jnt_values, dtype=float).reshape(-1)
        if target.size != 6 or not np.all(np.isfinite(target)):
            raise ValueError("target_jnt_values must contain 6 finite values")
        if timeout <= 0 or poll_dt <= 0 or tolerance_deg <= 0 or stable_count <= 0:
            raise ValueError("joint-target wait parameters must be positive")
        if motion_tolerance_deg is not None and motion_tolerance_deg <= 0:
            raise ValueError("motion_tolerance_deg must be positive when provided")

        tolerance_rad = np.radians(float(tolerance_deg))
        motion_tolerance_rad = (
            None
            if motion_tolerance_deg is None
            else np.radians(float(motion_tolerance_deg))
        )
        deadline = time.monotonic() + float(timeout)
        stable = 0
        previous = None
        last_error_deg = float("inf")
        last_motion_deg = float("inf")
        while time.monotonic() < deadline:
            current = np.asarray(self.get_joint_values(), dtype=float).reshape(-1)
            if current.size != 6 or not np.all(np.isfinite(current)):
                stable = 0
                previous = None
            else:
                max_error_rad = float(np.max(np.abs(current - target)))
                last_error_deg = float(np.degrees(max_error_rad))
                motion_confirmed = motion_tolerance_rad is None
                if previous is not None:
                    max_motion_rad = float(np.max(np.abs(current - previous)))
                    last_motion_deg = float(np.degrees(max_motion_rad))
                    motion_confirmed = (
                        motion_tolerance_rad is None
                        or max_motion_rad <= motion_tolerance_rad
                    )
                if max_error_rad <= tolerance_rad and motion_confirmed:
                    stable += 1
                    if stable >= int(stable_count):
                        return True
                else:
                    stable = 0
                previous = current.copy()
            time.sleep(float(poll_dt))

        message = (
            "wait_until_joint_target timeout: "
            f"max_joint_error={last_error_deg:.3f} deg, "
            f"required<={float(tolerance_deg):.3f} deg"
        )
        if motion_tolerance_deg is not None:
            message += (
                f", max_sample_motion={last_motion_deg:.3f} deg, "
                f"required<={float(motion_tolerance_deg):.3f} deg"
            )
        raise TimeoutError(message)

    def move_jntspace_path(self,
                           path,
                           max_jntvel: list = None,
                           max_jntacc: list = None,
                           start_frame_id=1,
                           toggle_debug=False,
                           control_frequency=0.3,
                           use_toppra: bool = True,
                           canfd_dt: float = 0.02):
        """
        :param path: [jnt_values0, jnt_values1, ...], results of motion planning
        :param max_jntvel: 1x6 list to describe the maximum joint speed for the arm
        :param max_jntacc: 1x6 list to describe the maximum joint acceleration for the arm
        :param start_frame_id:
        :return:
        author: weiwei
        """
        if TOPPRA_EXIST and use_toppra:
            if path is None:
                raise ValueError("The given is incorrect!")
            path = np.array(path)
            # # Refer to https://www.ufactory.cc/_files/ugd/896670_9ce29284b6474a97b0fc20c221615017.pdf
            # # the robotic arm can accept joint position commands sent at a fixed high frequency like 100Hz
            control_frequency = control_frequency
            tpply = pwp.PiecewisePolyTOPPRA()
            interpolated_path = tpply.interpolate_by_max_spdacc(path=path,
                                                                ctrl_freq=control_frequency,
                                                                max_vels=max_jntvel,
                                                                max_accs=max_jntacc,
                                                                toggle_debug=False)
            goal_jnt_values = np.asarray(path[-1], dtype=float).copy()
            interpolated_path = interpolated_path[start_frame_id:]
            interpolated_path = np.rad2deg(interpolated_path)
            sample_count = len(interpolated_path)
            for sample_index, jnt_values in enumerate(interpolated_path, start=1):
                self._send_movej_canfd_checked(
                    jnt_values,
                    context="move_jntspace_path/toppra",
                    sample_index=sample_index,
                    sample_count=sample_count,
                )
                time.sleep(.1)
            self.wait_until_joint_target(
                goal_jnt_values,
                timeout=max(5.0, len(interpolated_path) * 0.1 + 2.0),
            )
            return

        # Stream waypoints directly (keeps the original joint-space polyline shape).
        # This is important for paths generated from Cartesian-linear planners: TOPPRA uses splines and may
        # distort the joint-space path, which can bend the TCP trajectory.
        if path is None or len(path) == 0:
            raise ValueError("move_jntspace_path: received empty path")
        path = np.asarray(path, dtype=float)
        path = path[start_frame_id:]
        if path.ndim != 2 or path.shape[1] != 6:
            raise ValueError("move_jntspace_path: expected path shape (N,6)")
        path_deg = np.degrees(path)
        dt = float(canfd_dt) if canfd_dt is not None else 0.02
        dt = max(0.0, dt)
        sample_count = len(path_deg)
        for sample_index, jnt_values_deg in enumerate(path_deg, start=1):
            self._send_movej_canfd_checked(
                jnt_values_deg,
                context="move_jntspace_path/direct",
                sample_index=sample_index,
                sample_count=sample_count,
            )
            if dt > 0:
                time.sleep(dt)
        return

    def move_jntspace_path_sequence(self,
                                    segments,
                                    control_frequency=0.1,
                                    wait_at_end=True):
        """Retime several path sections and stream them as one CAN-FD trajectory.

        Each item in ``segments`` is a mapping containing ``path``,
        ``max_jntvel`` and ``max_jntacc``.  Sections are retimed independently so
        a transfer section and a near-target section can keep different motion
        limits.  All samples are prepared before the robot starts moving, then
        sent without an idle wait at section junctions.  Only the final section
        waits for the arm to settle.
        """
        if not TOPPRA_EXIST:
            raise RuntimeError(
                "Continuous multi-profile execution requires TOPPRA; "
                "install TOPPRA before executing this trajectory"
            )
        if not segments:
            raise ValueError("move_jntspace_path_sequence: segments must not be empty")

        dt = float(control_frequency)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("control_frequency must be a positive finite sample period")

        prepared = []
        prepared_sample_info = []
        previous_end = None
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                raise TypeError(f"trajectory segment {index} must be a dict")

            path = np.asarray(segment.get("path"), dtype=float)
            if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 6:
                raise ValueError(
                    f"trajectory segment {index} must have shape (N, 6), N >= 2"
                )
            if not np.all(np.isfinite(path)):
                raise ValueError(f"trajectory segment {index} contains non-finite values")
            if previous_end is not None and not np.allclose(
                    previous_end, path[0], atol=1e-5, rtol=0.0):
                gap_deg = float(np.degrees(np.max(np.abs(previous_end - path[0]))))
                raise ValueError(
                    f"trajectory junction {index - 1}->{index} is discontinuous "
                    f"(max gap {gap_deg:.6f} deg)"
                )

            max_vel = np.asarray(segment.get("max_jntvel"), dtype=float).reshape(-1)
            max_acc = np.asarray(segment.get("max_jntacc"), dtype=float).reshape(-1)
            if max_vel.size != 6 or not np.all(np.isfinite(max_vel)) or np.any(max_vel <= 0):
                raise ValueError(f"trajectory segment {index} has invalid joint velocity limits")
            if max_acc.size != 6 or not np.all(np.isfinite(max_acc)) or np.any(max_acc <= 0):
                raise ValueError(f"trajectory segment {index} has invalid joint acceleration limits")

            tpply = pwp.PiecewisePolyTOPPRA()
            sampled = np.asarray(
                tpply.interpolate_by_max_spdacc(
                    path=path,
                    ctrl_freq=dt,
                    max_vels=max_vel,
                    max_accs=max_acc,
                    toggle_debug=False,
                ),
                dtype=float,
            )
            # Very short sections can produce only the initial TOPPRA sample.
            # Sending the endpoint one sample period later remains within the
            # requested limit because TOPPRA's duration was shorter than dt.
            if sampled.ndim != 2 or sampled.shape[1] != 6:
                raise RuntimeError(f"TOPPRA returned an invalid segment {index}")
            if len(sampled) < 2:
                sampled = np.vstack((path[0], path[-1]))
            else:
                sampled[0] = path[0]
                sampled[-1] = path[-1]

            # Skip only the first trajectory's current-position sample.  Keep
            # later sections' first sample so the shared endpoint is commanded
            # for one control period.  This realizes the zero boundary velocity
            # assumed by the two independently retimed sections without an
            # additional wait_until_idle pause.
            prepared_segment = sampled[1:] if index == 0 else sampled
            prepared.append(prepared_segment)
            segment_name = str(segment.get("name") or f"segment_{index + 1}")
            segment_sample_count = len(prepared_segment)
            prepared_sample_info.extend(
                (segment_name, sample_index, segment_sample_count)
                for sample_index in range(1, segment_sample_count + 1)
            )
            previous_end = path[-1]

        command_path = np.vstack(prepared)
        command_path_deg = np.degrees(command_path)
        global_sample_count = len(command_path_deg)
        for global_sample_index, (jnt_values_deg, sample_info) in enumerate(
                zip(command_path_deg, prepared_sample_info), start=1):
            segment_name, segment_sample_index, segment_sample_count = sample_info
            self._send_movej_canfd_checked(
                jnt_values_deg,
                context=(
                    f"move_jntspace_path_sequence/{segment_name} "
                    f"(全局采样点 {global_sample_index}/{global_sample_count})"
                ),
                sample_index=segment_sample_index,
                sample_count=segment_sample_count,
            )
            time.sleep(dt)

        if wait_at_end:
            self.wait_until_joint_target(
                previous_end,
                timeout=max(5.0, len(command_path_deg) * dt + 2.0),
            )
        return


    def move_jntspace_path_hyk(self,
                           path,
                           max_jntvel: list = None,
                           max_jntacc: list = None,
                           start_frame_id=1,
                           toggle_debug=False,
                           control_frequency=0.3):
        """
        执行一段关节空间路径。

        两种模式:
        1) 如果安装了 TOPPRA (高频插补 / canfd 轨迹下发) -> 走原来的分支
        2) 如果没装 TOPPRA -> 使用官方 rm_movej() 直接去该段轨迹的最终关节角位姿

        参数:
            path: 规划器返回的关节轨迹, 例如 mot_data.jv_list
                  结构通常是 [q0, q1, q2, ...]，每个qi是长度6的关节角，单位: 弧度
            max_jntvel, max_jntacc: 最高速度加速度 (给toppra用的)，可忽略
            start_frame_id: toppra插补起始帧
            control_frequency: toppra分支里用来插补的频率 (Hz倒数)
        """
        if path is None or len(path) == 0:
            raise ValueError("move_jntspace_path: 收到的path为空，无法执行")

        if TOPPRA_EXIST:
            # ★ 原逻辑：使用toppra插补 + canfd逐点下发
            path = np.array(path)
            tpply = pwp.PiecewisePolyTOPPRA()
            interpolated_path = tpply.interpolate_by_max_spdacc(
                path=path,
                ctrl_freq=control_frequency,
                max_vels=max_jntvel,
                max_accs=max_jntacc,
                toggle_debug=toggle_debug
            )
            interpolated_path = interpolated_path[start_frame_id:]
            # toppra返回的是弧度，机械臂SDK需要度
            interpolated_path_deg = np.degrees(interpolated_path)

            sample_count = len(interpolated_path_deg)
            for sample_index, jnt_values_deg in enumerate(
                    interpolated_path_deg, start=1):
                # canfd高频下发接口
                self._send_movej_canfd_checked(
                    jnt_values_deg,
                    context="move_jntspace_path_hyk/toppra",
                    sample_index=sample_index,
                    sample_count=sample_count,
                )
                time.sleep(.05)
            return

        # -------- Fallback (无TOPPRA) --------
        # 没有toppra，我们就让机械臂直接去path的最后一个点。
        # 由控制柜自己做插补运动，这样依然是平滑的工业轨迹。
        last_q_rad = np.asarray(path[-1], dtype=float)   # 弧度
        last_q_deg = np.degrees(last_q_rad).tolist()     # 转成度给SDK

        # 速度/加速度这些参数你可以按现场改，
        # 这里我用 speed=20, accel=50, decel=0, block=1 只是示例:
        speed = 20   # 角速度 (deg/s)，可以先设小一点以确保安全
        accel = 50   # 加速度，跟SDK的单位对应
        decel = 0
        block_flag = 1  # 1=阻塞，等到位才返回

        print("[RealmanArmController] Fallback move_jntspace_path -> rm_movej(last waypoint):", last_q_deg)
        ret = self.arm.rm_movej(last_q_deg, speed, accel, decel, block_flag)
        if ret != 0:
            print("⚠ rm_movej 执行返回非0, 请检查控制柜/限位/安全区")
        self.wait_until_idle(timeout=10.0)
        return ret


    def close_connection(self):
        """
        删除机器人连接，释放资源
        """
        self.arm.rm_delete_robot_arm()
        print("机器人连接已删除")


if __name__ == "__main__":
    armx = rgt_arm_x = RealmanArmController(ip="192.168.1.18", port=8080, multiple_thread=True)
    # lft_arm_x = RealmanArmController(ip="192.168.0.18", port=8080, multiple_thread=False)
    print(armx.get_pose())
    print(repr(armx.get_joint_values()))
    print(rgt_arm_x.get_pose())
    # print(lft_arm_x.get_pose())
    # print(lft_arm_x.get_joint_values())
