import sys
import numpy as np
import os
sys.path.insert(0, os.getcwd())
from simulation.controller.ik_solver import FrankaSolver, MobileFrankaSolver
import h5py
from tqdm import tqdm
from simulation.controller.utils import rotate_quaternion_around_world_z_axis
import torch
from termcolor import cprint
try:
    import rospy
    from std_msgs.msg import Float64MultiArray, Bool
    ROSPY_ENABLED = True
except:
    ROSPY_ENABLED = False
    print("rospy not loaded. Teleop mode will be disabled.")

class franka_controller:
    def __init__(self, scene, scene_dict, scene_config, robot_config, close_thres, teleop=None, evaluation=False):
        """
        Initialize the Franka controller with the specified IK type and simulation settings.

        Args:
            ik_type (str): Type of IK solver to use. Options: "ik_solver" or "motion_gen"
            ik_sim (bool): Whether to use simulation mode.
            simulator (str): The simulator to use, e.g., "genesis".
        """
        self.scene_config = scene_config
        self.robot_config = robot_config
        self.scene = scene
        self.franka = scene_dict["robot"]
        self.ee_link = robot_config["ee_link"]
        self.default_ee_quat = self.franka.get_link(self.ee_link).get_quat().cpu().numpy()
        self.default_ee_pos = self.franka.get_link(self.ee_link).get_pos().cpu().numpy()
        if robot_config["name"] == "mobile_franka":
            self.franka_solver = MobileFrankaSolver(ik_type="motion_gen", no_solver=False, scene_config=self.scene_config, robot_config=self.robot_config)
            cprint("mobile franka solver loaded", "green")
        else:
            self.franka_solver = FrankaSolver(ik_type="motion_gen", ik_sim=True, simulator="genesis", no_solver=False)
            cprint("franka solver loaded", "green")
        # self.real_franka_solver = FrankaSolver(ik_type="motion_gen", ik_sim=False, simulator=None, no_solver=True)
        self.record_started = False
        self.default_joint_positions = robot_config["default_joint_positions"]
        self.close_state = [close_thres / 100, close_thres / 100]
        self.open_state = [0.04, 0.04]
        self.current_control = np.array(self.default_joint_positions[:-2])
        self.default_gripper_state = False
        self.current_gripper_control = False
        self.teleop = teleop
        self.evaluation = evaluation
        self.all_dof_ids = [self.franka.get_joint(name).dof_idx for name in self.robot_config["joint_names"]]
        if self.default_gripper_state:
            self.franka.set_dofs_position(
                self.close_state,
                self.all_dof_ids[-2:], 
            )
            self.default_joint_positions[-2:] = self.close_state
        else:
            self.franka.set_dofs_position(
                self.open_state,
                self.all_dof_ids[-2:], 
            )
            self.default_joint_positions[-2:] = self.open_state
        self.franka.set_dofs_position(
                self.default_joint_positions[:-2],
                self.all_dof_ids[:-2], 
            )
        self.franka.control_dofs_position(
            self.default_joint_positions,
            self.all_dof_ids,
        )
        if teleop or evaluation:
            if ROSPY_ENABLED:
                self.init_teleop()
            else:
                raise RuntimeError("rospy is not enabled! Install ROS first if in teleop mode.")
        
    def _callback_joint_control(self, msg):
        self.current_control = np.array(msg.data)
        self.franka.control_dofs_position(
            np.array(msg.data),
            self.all_dof_ids[:-2],
        )
    def _callback_ee_control(self, msg):
        ee_control = np.array(msg.data)
        joint_pos = self.franka.get_dofs_position().cpu().numpy()
        current_joint_angles = joint_pos[:-2]
        result = self.franka_solver.solve_ik_by_motion_gen(
            curr_joint_state=current_joint_angles, 
            target_trans=ee_control[:3],
            target_quat=ee_control[3:],
        )
        self.current_control = result[-1]
        self.franka.control_dofs_position(
            result[-1],
            self.all_dof_ids[:-2],
        )
    def _callback_gripper_control(self, msg):
        if not self.default_gripper_state:
            if msg.data:
                self.current_gripper_control = True
                self.franka.control_dofs_position(
                    self.close_state,
                    self.all_dof_ids[-2:], 
                )
            else:
                self.current_gripper_control = False
                self.franka.control_dofs_position(
                    self.open_state,
                    self.all_dof_ids[-2:], 
                )
            
    def init_teleop(self):
        rospy.init_node('genesis_sim', anonymous=True)
        # use ros publish to send the pred_mano_params to the client
        self.pub_joint = rospy.Publisher('/genesis/joint_states', Float64MultiArray, queue_size=1)
        self.pub_ee    = rospy.Publisher('/genesis/ee_states', Float64MultiArray, queue_size=1)
        self.sub_joint_control = rospy.Subscriber(
            "/genesis/joint_control",
            Float64MultiArray,
            self._callback_joint_control,
            queue_size=1,
        )
        self.sub_ee_control = rospy.Subscriber(
            "/genesis/ee_control",
            Float64MultiArray,
            self._callback_ee_control,
            queue_size=1,
        )
        self.sub_gripper_control = rospy.Subscriber(
            "/genesis/gripper_control",
            Bool,
            self._callback_gripper_control,
            queue_size=1,
        )

    def publish_states(self):
        joint_pos = self.franka.get_dofs_position()
        joint_pos = joint_pos.cpu().numpy()
        joint_pos_list = joint_pos.flatten().tolist()
        joint_pos_msg = Float64MultiArray(data=joint_pos_list)
        self.pub_joint.publish(joint_pos_msg)
        
        # joint_pos = self.franka.get_dofs_position().cpu().numpy()
        trans, rot_quat = self.franka_solver.compute_fk(joint_pos)
        current_ee_pose = np.concatenate([trans, rot_quat])
        current_ee_pose_list = current_ee_pose.flatten().tolist()
        current_ee_msg = Float64MultiArray(data=current_ee_pose_list)
        self.pub_ee.publish(current_ee_msg)

    def move_to_goal(self, pos, quat, gripper_open=True, quick=False):
        """
        Move the Franka robot to the specified goal position and orientation.

        Args:
            pos (list or np.ndarray): The target position in Cartesian coordinates.
            quat (list or np.ndarray): The target orientation as a quaternion.
        """
        joint_pos = self.franka.get_dofs_position().cpu().numpy()
        current_joint_angles = joint_pos[:-2]  # 当前关节角度
        result = self.franka_solver.solve_ik_by_motion_gen(
            curr_joint_state=current_joint_angles, 
            target_trans=pos,
            target_quat=quat,
        )
        if gripper_open:
            self.current_gripper_control = False # gripper open --> gripper = False
        else:
            self.current_gripper_control = True # gripper close --> gripper = True
            
        if result and len(result):
            if quick:
                try:
                    waypoint = np.array(result[-1]).flatten()
                except Exception:
                    waypoint = np.array(result[-1])
                full_target = (waypoint + np.array(self.open_state) if gripper_open else waypoint + np.array(self.close_state))
                self.franka.control_dofs_position(full_target)
                # wait for the desired end-effector pose (use provided pos/quaternion)
                ee_target = np.concatenate([np.array(pos).flatten(), np.array(quat).flatten()])
                self.wait_until_control_reached(ee_target)
            else:
                print("Moving to goal with {} waypoints.".format(len(result)))
                for waypoint in result:
                    self.current_control = np.array(waypoint)
                    if gripper_open:
                        self.current_gripper_control = False # gripper open --> gripper = False
                    else:
                        self.current_gripper_control = True # gripper close --> gripper = True
                    self.franka.control_dofs_position((waypoint+self.open_state if gripper_open else waypoint+self.close_state))
                    self.step()
                    target_ee_pose = self.franka_solver.compute_fk(waypoint)[0]
                    # import ipdb; ipdb.set_trace()
                    current_ee_pose = self.franka_solver.compute_fk(self.franka.get_dofs_position().cpu().numpy()[:-2])[0]
                    while(target_ee_pose is not None and not np.allclose(np.array(current_ee_pose).flatten(), np.array(target_ee_pose).flatten(), atol=1e-2)):
                        self.step()
                        current_ee_pose = self.franka_solver.compute_fk(self.franka.get_dofs_position().cpu().numpy()[:-2])[0]
        else:
            raise RuntimeError("IK Failed")
    
    def close_gripper(self, wait_steps=100):
        """
        Close the gripper of the Franka robot.
        """
        joint_pos = self.franka.get_dofs_position().cpu().numpy()
        current_joint_angles = joint_pos
        current_joint_angles[-2:] = self.close_state
        self.current_gripper_control = True
        self.franka.control_dofs_position(current_joint_angles)
        for i in range(wait_steps):
            self.current_gripper_control = True
            self.step()

    def open_gripper(self, wait_steps=100):
        """
        Open the gripper of the Franka robot.
        """
        joint_pos = self.franka.get_dofs_position().cpu().numpy()
        current_joint_angles = joint_pos
        current_joint_angles[-2:] = self.open_state
        self.current_gripper_control = False
        self.franka.control_dofs_position(current_joint_angles)
        for i in range(wait_steps):
            self.current_gripper_control = False
            self.step()

    def reset_franka(self):
        """
        Reset the Franka robot to its initial state.
        """
        self.current_control = np.array(self.default_joint_positions[:-2])
        self.current_gripper_control = self.default_gripper_state
        self.franka.set_dofs_position(self.default_joint_positions)
        self.scene.step()
        
    def step(self):
        """
        Step the simulation forward.
        """
        if self.default_gripper_state:
            self.franka.control_dofs_position(
                self.close_state,
                self.all_dof_ids[-2:], 
            )
        if self.teleop or self.evaluation:
            self.publish_states()
        self.scene.step()
        if self.record_started:
            self.record_step()

    def wait_until_control_reached(self, target, tol=5e-3, max_steps=500, ori_tol=1e-2):
        """
        Wait until the robot reaches the target end-effector pose (position + quaternion)
        within given tolerances. By default the function uses EE-pose check. It will also
        accept a DOF target and fall back to DOF comparison if provided.

        Args:
            target (array-like): If length==7 treated as EE pose [x,y,z,qx,qy,qz,qw] or
                                [pos(3), quat(4)] (order follows compute_fk output). Otherwise
                                treated as DOF joint target and compared to get_dofs_position().
            tol (float): Position tolerance in meters for EE comparison or absolute per-DOF
                         tolerance for DOF comparison.
            max_steps (int): Maximum simulation steps to wait before giving up.
            ori_tol (float): Orientation tolerance (radians). Quaternion angle difference is
                             used to compare orientation.

        Returns:
            bool: True if reached within max_steps, False on timeout or invalid input.
        """
        # normalize target to numpy array
        try:
            target_arr = np.array(target).flatten()
        except Exception:
            return False

        use_ee = target_arr.size == 7

        for i in range(max_steps):
            if use_ee:
                # get current joint positions and compute FK
                cur_joints = self.franka.get_dofs_position()
                try:
                    cur_joints_arr = cur_joints.cpu().numpy()
                except Exception:
                    cur_joints_arr = np.array(cur_joints)
                trans, rot_quat = self.franka_solver.compute_fk(cur_joints_arr)
                cur_ee = np.concatenate([np.array(trans).flatten(), np.array(rot_quat).flatten()])

                # compare position
                pos_diff = np.linalg.norm(cur_ee[:3] - target_arr[:3])
                # compare orientation using quaternion dot -> angle
                q1 = cur_ee[3:7]
                q2 = target_arr[3:7]
                # ensure unit quaternions
                try:
                    q1 = q1 / np.linalg.norm(q1)
                    q2 = q2 / np.linalg.norm(q2)
                except Exception:
                    pass
                dot = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)
                ang = 2.0 * np.arccos(dot)

                if pos_diff <= tol and ang <= ori_tol:
                    return True
            else:
                # DOF comparison fallback
                cur = self.franka.get_dofs_position()
                try:
                    cur_arr = cur.cpu().numpy().flatten()
                except Exception:
                    cur_arr = np.array(cur).flatten()

                if cur_arr.shape == target_arr.shape:
                    if np.allclose(cur_arr, target_arr, atol=tol):
                        return True
                else:
                    m = min(cur_arr.size, target_arr.size)
                    if np.allclose(cur_arr[:m], target_arr[:m], atol=tol):
                        return True

            # not reached yet -> step once (this will also publish states/record if enabled)
            print("Waiting for control to reach target... Step {}".format(i + 1))
            self.step()

        cprint(f"Warning: wait_until_control_reached timed out after {max_steps} steps", "yellow")
        return False

    def record_step(self):
        raise NotImplementedError("Recording functionality is not implemented yet.")
            
class pick_and_place_controller(franka_controller):
    """Controller for pick and place operations using the Franka robot.
    Inherits from the `franka_controller` class.
    """
    def __init__(self, scene, scene_dict, scene_config, robot_config, default_poses, close_thres=1, teleop=False, evaluation=False):
        """
        Initialize the pick and place controller.

        Args:
            scene: The simulation scene.
            robot: The Franka robot instance.
            ik_type (str): Type of IK solver to use. Options: "ik_solver" or "motion_gen".
            simulator (str): The simulator to use, e.g., "genesis".
        """
        super().__init__(scene, scene_dict, scene_config, robot_config, close_thres, teleop=teleop, evaluation=evaluation)
        self.object_active = scene_dict["object_active"]
        self.object_passive = scene_dict["object_passive"]
        self.default_poses = default_poses
        
        
        self.scene.step()
        self.record_started = False
        self.timestamp = 0
        self.record = []
        self.traj_cnt = 0
        
    def reset_scene(self):
        def set_friction_ratio(rigid_entity, friction_ratio, link_indices, envs_idx=None):
            """
            Set the friction ratio of the geoms of the specified links.
            Parameters
            ----------
            friction_ratio : torch.Tensor, shape (n_envs, n_links)
                The friction ratio
            link_indices : array_like
                The indices of the links to set friction ratio.
            envs_idx : None | array_like, optional
                The indices of the environments. If None, all environments will be considered. Defaults to None.
            """
            geom_indices = []
            for i in link_indices:
                for j in range(rigid_entity._links[i].n_geoms):
                    geom_indices.append(rigid_entity._links[i]._geom_start + j)
            rigid_entity._solver.set_geoms_friction_ratio(
                torch.cat(
                    [
                        ratio.unsqueeze(-1).repeat(1, rigid_entity._links[j].n_geoms)
                        for j, ratio in zip(link_indices, friction_ratio.unbind(-1))
                    ],
                    dim=-1,
                ).squeeze(0),
                geom_indices,
                envs_idx,
            )
            return rigid_entity

        self.franka.set_dofs_position(self.default_joint_positions)
        self.franka.control_dofs_position(self.default_joint_positions)
        self.current_control = self.default_joint_positions
        self.current_gripper_control = self.default_gripper_state
        active_x_min = self.scene_config["object_active"]["pos_range"]["x"][0] / 100
        active_x_max = self.scene_config["object_active"]["pos_range"]["x"][1] / 100
        active_y_min = self.scene_config["object_active"]["pos_range"]["y"][0] / 100
        active_y_max = self.scene_config["object_active"]["pos_range"]["y"][1] / 100
        rand_x = np.random.rand() * (active_x_max - active_x_min) + active_x_min
        rand_y = np.random.rand() * (active_y_max - active_y_min) + active_y_min
        fixed_z = self.default_poses["active_pos"][2]
        self.object_active.set_pos(np.array([rand_x, rand_y, fixed_z]))

        passive_x_min = self.scene_config["object_passive"]["pos_range"]["x"][0] / 100
        passive_x_max = self.scene_config["object_passive"]["pos_range"]["x"][1] / 100
        passive_y_min = self.scene_config["object_passive"]["pos_range"]["y"][0] / 100
        passive_y_max = self.scene_config["object_passive"]["pos_range"]["y"][1] / 100
        rand_x = np.random.rand() * (passive_x_max - passive_x_min) + passive_x_min
        rand_y = np.random.rand() * (passive_y_max - passive_y_min) + passive_y_min
        fixed_z = self.default_poses["passive_pos"][2]
        self.object_passive.set_pos(np.array([rand_x, rand_y, fixed_z]))


        rand_angle = np.random.rand() * 360
        # wxyz --> xyzw
        quat_scipy = np.array([self.default_poses["active_quat"][1], self.default_poses["active_quat"][2],self.default_poses["active_quat"][3],self.default_poses["active_quat"][0]])
        quat_scipy = rotate_quaternion_around_world_z_axis(quat_scipy, rand_angle)
        # xyzw --> wxyz
        rand_quat = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
        self.object_active.set_quat(rand_quat)

        rand_angle = np.random.rand() * 360
        # wxyz --> xyzw
        quat_scipy = np.array([self.default_poses["passive_quat"][1], self.default_poses["passive_quat"][2],self.default_poses["passive_quat"][3],self.default_poses["passive_quat"][0]])
        quat_scipy = rotate_quaternion_around_world_z_axis(quat_scipy, rand_angle)
        # xyzw --> wxyz
        rand_quat = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
        self.object_passive.set_quat(rand_quat)
        
        # set physics
        # set_friction_ratio(self.object_active, torch.tensor([self.physics_params["friction_ratio"]]).to("cuda"), link_indices = [0],)
        # self.object_active.set_mass_shift(torch.tensor([self.physics_params["mass_shift"]]).to("cuda"), link_indices = [0])
        # self.object_active.set_COM_shift(
        #     com_shift=torch.tensor(self.physics_params["com_shift"]).to("cuda").unsqueeze(0),
        #     link_indices = [0],
        # )
        if not self.teleop and not self.evaluation:
            self.scene.step()
    
    def get_states(self):
        object_active_pos_state = self.object_active.get_pos()
        object_active_quat_state = self.object_active.get_quat()
        object_passive_pos_state = self.object_passive.get_pos()
        object_passive_quat_state = self.object_passive.get_quat()
        
        joint_pos = self.franka.get_dofs_position().cpu().numpy()
        trans, rot_quat = self.franka_solver.compute_fk(joint_pos)
        current_ee_pose = np.concatenate([trans, rot_quat])
        current_ee_control = np.concatenate(self.franka_solver.compute_fk(self.current_control))
        states = {
            "timestamp": np.array([self.timestamp]),
            "joint_states": joint_pos, 
            "ee_states": current_ee_pose,
            "joint_control": self.current_control,
            "ee_control": current_ee_control,
            "gripper_control": np.array([self.current_gripper_control]),
            "object_states": {
                "active": np.concatenate([object_active_pos_state.cpu().numpy(), object_active_quat_state.cpu().numpy()]),
                "passive": np.concatenate([object_passive_pos_state.cpu().numpy(), object_passive_quat_state.cpu().numpy()]),
            },
        }
        return states
    
    def record_step(self):
        self.record.append(self.get_states())
        self.timestamp += 1

    def save_traj(self, output_dir):
        def save_dict_to_hdf5(dic, filename):
            with h5py.File(filename, 'w') as h5file:
                _save_dict_to_hdf5(dic, h5file)

        def _save_dict_to_hdf5(dic, h5grp):
            for key, item in dic.items():
                if isinstance(item, dict):
                    # 创建组并递归保存子字典
                    subgroup = h5grp.create_group(key)
                    _save_dict_to_hdf5(item, subgroup)
                elif isinstance(item, np.ndarray) or isinstance(item, list):
                    # 保存numpy数组（启用压缩）
                    h5grp.create_dataset(key, data=np.array(item), compression="gzip")
                else:
                    print(f"Unknown item {key}, {type(item)}")
        self.record_started = False
        current_traj_dir = os.path.join(output_dir, "{:05d}".format(self.traj_cnt))
        os.makedirs(current_traj_dir, exist_ok=True)
        for idx, state in enumerate(self.record):
            save_dict_to_hdf5(state, os.path.join(current_traj_dir, f"{idx}.h5"))
        self.clean_traj()
        self.traj_cnt += 1

    def clean_traj(self):
        self.record = []
    
    def start_record(self):
        self.record_started = True
        self.record = []
        self.timestamp = 0

    def end_record(self):
        self.record_started = False
