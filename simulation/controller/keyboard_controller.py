import numpy as np
from termcolor import cprint
try:
    import rospy
    from std_msgs.msg import Float64MultiArray, Bool
    KEYBOARD_ROS_ENABLED = True
except:
    KEYBOARD_ROS_ENABLED = False
class keyboard_teleop_controller:
    def __init__(self, robot, robot_config, franka_genesis_controller):
        self.robot = robot
        self.robot_config = robot_config
        self.franka_genesis_controller = franka_genesis_controller
        self.pos_step = 0.01
        self.dpos = np.zeros(3, dtype=np.float32)
        # sub_joint = rospy.Subscriber("/genesis/joint_states", Float64MultiArray, queue_size=1)
        # sub_ee = rospy.Subscriber("/genesis/ee_states", Float64MultiArray, queue_size=1)
        self.pub_joint_control = rospy.Publisher("/genesis/joint_control", Float64MultiArray, queue_size=1)
        self.pub_gripper_control = rospy.Publisher("/genesis/gripper_control", Bool, queue_size=1)
        self.default_joint_pos = self.franka_genesis_controller.default_joint_positions[:-2]
        self.default_pos, self.default_quat = self.franka_genesis_controller.franka_solver.compute_fk(self.default_joint_pos)
        cprint("W/S: +/- X, A/D: +/- Y, R/F: +/- Z, Z: close gripper, X: open gripper", "cyan")

    def keyboard_listener(self, c):
        gripper_open = self.franka_genesis_controller.current_gripper_control
        if c == "w":
            self.dpos[0] += self.pos_step
        elif c == "s":
            self.dpos[0] -= self.pos_step
        elif c == "a":
            self.dpos[1] += self.pos_step
        elif c == "d":
            self.dpos[1] -= self.pos_step        
        elif c == "r":
            self.dpos[2] += self.pos_step
        elif c == "f":
            self.dpos[2] -= self.pos_step
        elif c == "z":
            gripper_open = False
        elif c == "x":
            gripper_open = True
        else:
            return
        target_pos = self.default_pos + self.dpos
        current_joint_positions = self.franka_genesis_controller.franka.get_dofs_position().cpu().numpy()[:-2]
        result = self.franka_genesis_controller.franka_solver.solve_ik_by_motion_gen(
            curr_joint_state=current_joint_positions,
            target_trans=target_pos,
            target_quat=self.default_quat,
        )
        if not result:
            return
        self.pub_joint_control.publish(data=result[-1])
        if gripper_open:
            self.pub_gripper_control.publish(data=True)
        else:
            self.pub_gripper_control.publish(data=False)

    def reset(self):
        self.dpos = np.zeros(3, dtype=np.float32)
