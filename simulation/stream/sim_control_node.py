#!/usr/bin/env python3
import os
import sys
import threading
import yaml
from pathlib import Path

import rospy
from std_msgs.msg import Float64MultiArray, Int32
from termcolor import cprint
sys.path.insert(0, os.getcwd())

import genesis as gs
from simulation.utils.scene_utils import design_pnp_scene, init_scene_physic_params
from simulation.stream.franka_genesis_controller import pick_and_place_controller

from sensor_msgs.msg import Image
try:
    from cv_bridge import CvBridge
    import numpy as np
    _bridge = CvBridge()
except Exception:
    _bridge = None
    np = None


class CmdStateBuffer:
    """缓存来自 keyboard_node 的控制命令 (ee + gripper)"""
    def __init__(self):
        self.lock = threading.Lock()
        self.last_cmd = None  # Float64MultiArray.data length = 8

    def set_cmd(self, arr):
        with self.lock:
            self.last_cmd = arr

    def pop_cmd(self):
        with self.lock:
            return self.last_cmd


class RecordCmdBuffer:
    """缓存 1/2/3/4 等录制控制命令"""
    def __init__(self):
        self.lock = threading.Lock()
        self.last_cmd = None  # Int32: 1,2,3,4

    def set_cmd(self, v: int):
        with self.lock:
            self.last_cmd = v

    def pop_cmd(self):
        with self.lock:
            v = self.last_cmd
            self.last_cmd = None
            return v


def design_scene(scene_config, robot_config):
    scene, scene_dict, cams, default_poses = design_pnp_scene(scene_config, robot_config, show_viewer=True)
    left_cam = scene.add_camera(
        res    = (1280, 720),
        pos    = (0.62, -0.5, 0.5),
        lookat = (0.62, 0, 0),
        fov    = 60,
        GUI    = True,
    )
    front_cam = scene.add_camera(
        res    = (1280, 720),
        pos    = (1.5, -0.05, 0.3),
        lookat = (0.5, -0.05, 0.3),
        fov    = 30,
        GUI    = True,
    )
    cams["anno_cam_1"] = left_cam
    cams["anno_cam_2"] = front_cam
    return scene, scene_dict, cams, default_poses


def get_robot_state_vector(controller):
    """
    使用 pick_and_place_controller 现有成员获取 robot state：
      [q7, ee_pos3, ee_quat4, gripper_open1]
    """
    import numpy as _np

    # 关节角
    joint_pos = controller.franka.get_dofs_position().cpu().numpy().reshape(-1)
    if joint_pos.shape[0] < 7:
        raise RuntimeError(f"Expected at least 7 joints, got {joint_pos.shape[0]}")
    q7 = joint_pos[:7]

    # EE 位姿：用 franka_solver.compute_fk
    trans, rot_quat = controller.franka_solver.compute_fk(joint_pos)
    ee_pos = _np.array(trans).reshape(3)
    ee_quat = _np.array(rot_quat).reshape(4)

    # gripper_open：current_gripper_control=True 表示 close
    gripper_closed = bool(controller.current_gripper_control)
    gripper_open = float(not gripper_closed)

    return _np.concatenate([q7, ee_pos, ee_quat, [gripper_open]]).astype(float).tolist()


def apply_cmd_state_to_controller(controller, cmd_state):
    """
    cmd_state: [ee_pos(3), ee_quat(4), gripper_open(0/1)]
    不改 franka，只用现有 move_to_goal。
    """
    import numpy as _np
    arr = _np.array(cmd_state, dtype=float)
    if arr.size < 8:
        return
    ee_pos = arr[0:3]
    ee_quat = arr[3:7]
    g_open = bool(arr[7])
    try:
        controller.move_to_goal(pos=ee_pos, quat=ee_quat, gripper_open=g_open, quick=False)
    except Exception as e:
        print("[WARN] move_to_goal failed:", e)


def camera_to_ros_image(cam, rgb=True):
    """
    尝试多种 Genesis Camera 接口：
      - RGB: get_color_rgba / get_color_bg
      - Depth: get_depth
    然后用 cv_bridge 转成 ROS Image。
    """
    if _bridge is None or np is None:
        return None

    try:
        if rgb:
            frame = None
            encoding = "bgr8"
            if hasattr(cam, "get_color_rgba"):
                rgba = cam.get_color_rgba()  # HxWx4, float [0,1] 或 uint8
                rgba = np.asarray(rgba)
                if rgba.dtype != np.uint8:
                    rgba = (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)
                # RGBA -> BGR
                frame = rgba[..., :3][..., ::-1]
            elif hasattr(cam, "get_color_bg"):
                frame = np.asarray(cam.get_color_bg())
            else:
                return None
        else:
            if not hasattr(cam, "get_depth"):
                return None
            depth = cam.get_depth()
            depth = np.asarray(depth).astype(np.float32)
            frame = depth
            encoding = "32FC1"

        msg = _bridge.cv2_to_imgmsg(frame, encoding=encoding)
        return msg
    except Exception as e:
        print("[WARN] camera_to_ros_image failed:", e)
        return None


def reset_layout(controller, scene_config):
    obj_passive = controller.scene_dict["object_passive"]
    obj_active  = controller.scene_dict["object_active"]
    import numpy as _np
    while True:
        controller.reset_scene()
        passive_pos = obj_passive.get_pos().cpu().numpy()
        active_pos  = obj_active.get_pos().cpu().numpy()
        passive_aabb = obj_passive.get_link("object").get_AABB()
        active_aabb  = obj_active.get_link("object").get_AABB()

        if not (
            active_aabb[0, 0] > scene_config["object_active"]["pos_range"]["x"][0] / 100
            and active_aabb[0, 1] > scene_config["object_active"]["pos_range"]["y"][0] / 100
            and active_aabb[1, 0] < scene_config["object_active"]["pos_range"]["x"][1] / 100
            and active_aabb[1, 1] < scene_config["object_active"]["pos_range"]["y"][1] / 100
        ):
            continue
        if not (
            passive_aabb[0, 0] > scene_config["object_passive"]["pos_range"]["x"][0] / 100
            and passive_aabb[0, 1] > scene_config["object_passive"]["pos_range"]["y"][0] / 100
            and passive_aabb[1, 0] < scene_config["object_passive"]["pos_range"]["x"][1] / 100
            and passive_aabb[1, 1] < scene_config["object_passive"]["pos_range"]["y"][1] / 100
        ):
            continue

        active_xyaabb = active_aabb.cpu().numpy()[:2, :2]
        passive_xyaabb = passive_aabb.cpu().numpy()[:2, :2]
        x_overlap = active_xyaabb[0, 0] < passive_xyaabb[1, 0] and active_xyaabb[1, 0] > passive_xyaabb[0, 0]
        y_overlap = active_xyaabb[0, 1] < passive_xyaabb[1, 1] and active_xyaabb[1, 1] > passive_xyaabb[0, 1]
        if x_overlap and y_overlap:
            continue

        if _np.linalg.norm(passive_pos - active_pos) > scene_config["far_threshold"]:
            break


def run_simulation(cmd_buffer: CmdStateBuffer,
                   record_buffer: RecordCmdBuffer,
                   controller,
                   cams,
                   process_output_dir,
                   scene_config,
                   pub_rgb,
                   pub_depth,
                   pub_state):
    cprint("Simulation loop started.", "green")
    rate = rospy.Rate(500)

    while not rospy.is_shutdown():
        # 渲染相机
        for cam_key in list(cams.keys()):
            if str(cam_key).startswith("anno"):
                cams[cam_key].render()

        # 录制命令 1/2/3/4
        cmd_rec = record_buffer.pop_cmd()
        if cmd_rec is not None:
            if cmd_rec == 1:
                print('recollect !!!')
                controller.clean_traj()
                reset_layout(controller, scene_config)
            elif cmd_rec == 2:
                print('start recording !!!')
                controller.start_record()
            elif cmd_rec == 3:
                print('end recording !!!')
                controller.end_record()
            elif cmd_rec == 4:
                print('save record ...')
                controller.save_traj(process_output_dir)
                reset_layout(controller, scene_config)

        # ee + gripper 命令
        cmd = cmd_buffer.pop_cmd()
        if cmd is not None:
            apply_cmd_state_to_controller(controller, cmd)

        # 控制器步进
        controller.step()

        # 发布 robot_state
        state_vec = get_robot_state_vector(controller)
        pub_state.publish(Float64MultiArray(data=state_vec))

        # 发布 RGB/Depth
        cam_rgb = cams.get("anno_cam_1", None)
        cam_depth = cams.get("anno_cam_2", None)
        if cam_rgb is not None:
            img_msg = camera_to_ros_image(cam_rgb, rgb=True)
            if img_msg is not None:
                pub_rgb.publish(img_msg)
        if cam_depth is not None:
            depth_msg = camera_to_ros_image(cam_depth, rgb=False)
            if depth_msg is not None:
                pub_depth.publish(depth_msg)

        rate.sleep()


def main():
    rospy.init_node('teleop_sim_node', anonymous=True)

    output_dir = rospy.get_param('~output_dir', 'datasets/records')
    scene_cfg_path = rospy.get_param('~scene_cfg_path', 'simulation/configs/scene/banana_plate.yaml')
    robot_cfg_path = rospy.get_param('~robot_cfg_path', 'simulation/configs/robot/mobile_franka.yaml')

    scene_config = yaml.safe_load(Path(scene_cfg_path).open('r'))
    robot_config = yaml.safe_load(Path(robot_cfg_path).open('r'))

    from datetime import datetime
    now = datetime.now()
    milliseconds = now.microsecond // 1000
    timestamp = now.strftime("%Y%m%d_%H%M%S") + f"_{milliseconds:03d}"
    process_output_dir = os.path.join(output_dir, scene_config["task_name"], timestamp)
    os.makedirs(process_output_dir, exist_ok=True)

    cprint("Initializing Genesis", "green")
    gs.init(backend=gs.gpu, logging_level='warning')
    scene, scene_dict, cams, default_poses = design_scene(scene_config, robot_config)
    scene.build()
    init_scene_physic_params(scene, scene_dict, scene_config, robot_config)

    controller = pick_and_place_controller(
        scene=scene,
        scene_dict=scene_dict,
        scene_config=scene_config,
        robot_config=robot_config,
        default_poses=default_poses,
        close_thres=robot_config["close_thres"],
        teleop="keyboard",
    )

    # 命令缓存
    cmd_buffer = CmdStateBuffer()
    record_buffer = RecordCmdBuffer()

    def on_cmd_state(msg: Float64MultiArray):
        cmd_buffer.set_cmd(msg.data)

    def on_record_cmd(msg: Int32):
        record_buffer.set_cmd(msg.data)

    rospy.Subscriber('/teleop/cmd_state', Float64MultiArray, on_cmd_state, queue_size=10)
    rospy.Subscriber('/teleop/record_cmd', Int32, on_record_cmd, queue_size=10)

    pub_rgb = rospy.Publisher('/sim/rgb', Image, queue_size=1)
    pub_depth = rospy.Publisher('/sim/depth', Image, queue_size=1)
    pub_state = rospy.Publisher('/sim/robot_state', Float64MultiArray, queue_size=10)

    sim_thread = threading.Thread(
        target=run_simulation,
        args=(cmd_buffer, record_buffer, controller, cams, process_output_dir, scene_config, pub_rgb, pub_depth, pub_state),
        daemon=True,
    )
    sim_thread.start()

    cprint('Simulation control node started.', 'cyan')
    rospy.spin()
    cprint('Simulation control node exiting...', 'yellow')


if __name__ == '__main__':
    main()
