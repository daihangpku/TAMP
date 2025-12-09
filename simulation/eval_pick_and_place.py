import os
import argparse
import torch
import numpy as np
import open3d as o3d
import sys

sys.path.insert(0, os.getcwd())

import genesis as gs
import yaml
from pathlib import Path
from simulation.controller.franka_genesis_controller import pick_and_place_controller
from simulation.utils.scene_utils import design_pnp_scene, init_scene_physic_params
from simulation.render_pick_and_place import render_scene
from termcolor import cprint
from tqdm import tqdm
from pynput import keyboard

import queue
import threading
import time

try:
    import rospy
    from sensor_msgs.msg import Image
    ROSPY_ENABLED = True
except:
    ROSPY_ENABLED = False
    raise NotImplementedError
        

def design_scene(scene_config, robot_config, show_viewer=False):
    scene, scene_dict, cams, default_poses = design_pnp_scene(scene_config, robot_config, show_viewer=show_viewer)
    return scene, scene_dict, cams, default_poses

image_queue = queue.Queue(maxsize=1) 

def consumer_thread():
    import cv2
    """
    消费者线程：从队列中获取图像并显示。
    """
    try:
        cv2.namedWindow("Live Stream", cv2.WINDOW_AUTOSIZE)

        while True:
            # 尝试从队列中获取图像，设置1秒超时
            try:
                # 消费者线程会阻塞在这里，直到队列中有数据
                img = image_queue.get(timeout=1)
                if img is not None:
                    cv2.imshow("Live Stream", img)
            except queue.Empty:
                # 队列为空时，不做任何事，继续等待
                pass
            
            # 窗口事件处理
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Consumer error: {e}")
    finally:
        cv2.destroyAllWindows()
        
def main(args):
    scene_config = yaml.safe_load(Path(args.scene_cfg_path).open('r'))
    robot_config = yaml.safe_load(Path(args.robot_cfg_path).open('r'))
    
    from datetime import datetime
    now = datetime.now()
    milliseconds = now.microsecond // 1000
    timestamp = now.strftime("%Y%m%d_%H%M%S") + f"_{milliseconds:03d}"
    task_name = scene_config["task_name"]
    process_output_dir = os.path.join(args.output_dir, task_name, timestamp)
    os.makedirs(process_output_dir, exist_ok=True)
                             
    cprint("*" * 40, "green")
    cprint("  Initializing Genesis", "green")
    cprint("*" * 40, "green")
    
    # Init Genesis
    gs.init(backend=gs.gpu, logging_level = 'warning')
    scene, scene_dict, cams, default_poses = design_scene(scene_config, robot_config, show_viewer=False)
    scene.build()
    

    # Assets
    robot = scene_dict["robot"]
    object_active = scene_dict["object_active"]
    object_passive = scene_dict["object_passive"]
    init_scene_physic_params(scene, scene_dict, scene_config, robot_config)

    cprint("*" * 40, "green")
    cprint("  Initializing Controller", "green")
    cprint("*" * 40, "green")
    
    # Init Controller
    controller = pick_and_place_controller(
        scene=scene,
        scene_dict=scene_dict,
        scene_config=scene_config,
        robot_config=robot_config,
        default_poses=default_poses,
        close_thres=robot_config["close_thres"],
        teleop=False,
        evaluation=True
    )
    def reset_layout():
        # Reset Layout
        while True:
            controller.reset_scene()
            passive_pos = object_passive.get_pos().cpu().numpy()
            active_pos = object_active.get_pos().cpu().numpy()
            passive_aabb = object_passive.get_link("object").get_AABB()
            active_aabb = object_active.get_link("object").get_AABB()
            
            if (
                active_aabb[0, 0] > scene_config["object_active"]["pos_range"]["x"][0] / 100
                and active_aabb[0, 1] > scene_config["object_active"]["pos_range"]["y"][0] / 100
                and active_aabb[1, 0] < scene_config["object_active"]["pos_range"]["x"][1] / 100
                and active_aabb[1, 1] < scene_config["object_active"]["pos_range"]["y"][1] / 100
            ):
                pass
            else:
                cprint("active out of range", "yellow")
                continue
            if (
                passive_aabb[0, 0] > scene_config["object_passive"]["pos_range"]["x"][0] / 100
                and passive_aabb[0, 1] > scene_config["object_passive"]["pos_range"]["y"][0] / 100
                and passive_aabb[1, 0] < scene_config["object_passive"]["pos_range"]["x"][1] / 100
                and passive_aabb[1, 1] < scene_config["object_passive"]["pos_range"]["y"][1] / 100
            ):
                pass
            else:
                cprint("passive out of range", "yellow")
                continue
            active_xyaabb = active_aabb.cpu().numpy()[:2, :2]
            passive_xyaabb = passive_aabb.cpu().numpy()[:2, :2]
            x_overlap = active_xyaabb[0, 0] < passive_xyaabb[1, 0] and active_xyaabb[1, 0] > passive_xyaabb[0, 0]
            y_overlap = active_xyaabb[0, 1] < passive_xyaabb[1, 1] and active_xyaabb[1, 1] > passive_xyaabb[0, 1]
            if x_overlap and y_overlap:
                cprint("active-passive overlap box", "yellow")
                continue
            if np.linalg.norm(passive_pos - active_pos) > scene_config.far_threshold:
                cprint("active-passive too near", "yellow")
                break
            
    def on_press(key):
        pass
        # print(f"Key pressed: {key}")

    def on_release(key):
        nonlocal process_output_dir
        nonlocal controller
        nonlocal reset_layout
        if key.char == "1":
            print('recollect !!!')
            reset_layout()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    pbar = tqdm(total=None, bar_format='Teleop: {rate_fmt}', unit='frames')

    # 创建并启动消费者线程
    consumer = threading.Thread(target=consumer_thread, daemon=True)
    consumer.start()
    
    pub_image = rospy.Publisher('/genesis/color_image', Image, queue_size=1)
    
    while True:
        if not listener.is_alive():
            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()
        rgb_image, depth_image = render_scene(scene, scene_dict, None, cams["desk_cam"])
        
        if not image_queue.empty():
            try:
                image_queue.get_nowait()
            except queue.Empty:
                pass
        image_queue.put(rgb_image[:, :, ::-1])
        
        ros_image = Image()
        ros_image.header.stamp = rospy.Time.now()
        ros_image.header.frame_id = "camera_frame"
        ros_image.height = rgb_image.shape[0]
        ros_image.width = rgb_image.shape[1]
        ros_image.encoding = "rgb8"
        ros_image.step = ros_image.width * 3
        ros_image.data = rgb_image.tobytes()
        pub_image.publish(ros_image)
        
        controller.step()
        pbar.update()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="datasets/records")
    parser.add_argument("--scene_cfg_path", type=str, default="simulation/configs/scene/banana_plate.yaml")
    parser.add_argument("--robot_cfg_path", type=str, default="simulation/configs/robot/panda_xml.yaml")
    args = parser.parse_args()
    main(args)