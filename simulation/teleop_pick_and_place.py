import os
import argparse
import time
import torch
import numpy as np
import open3d as o3d
from PIL import Image
import sys
import queue

sys.path.insert(0, os.getcwd())
import genesis as gs
import yaml
from pathlib import Path
from easydict import EasyDict

from simulation.controller.franka_genesis_controller import pick_and_place_controller
from simulation.controller.keyboard_controller import keyboard_teleop_controller
from simulation.utils.scene_utils import design_pnp_scene, init_scene_physic_params
from termcolor import cprint
from tqdm import tqdm
from pynput import keyboard
try:
    import rospy
    from std_msgs.msg import Float64MultiArray, Bool
    KEYBOARD_ROS_ENABLED = True
except:
    KEYBOARD_ROS_ENABLED = False
    print("rospy not loaded. Keyboard teleop mode will be disabled.")


def design_scene(scene_config, robot_config, show_viewer=True):
    scene, scene_dict, cams, default_poses = design_pnp_scene(scene_config, robot_config, show_viewer=show_viewer)
    left_cam = scene.add_camera(
        res    = (1280, 720),
        pos    = (0, -0.5, 0.5 + robot_config["robot_to_table_height"] / 100),
        lookat = (0, 0, robot_config["robot_to_table_height"] / 100),
        fov    = 60,
        GUI    = True,
    )
    front_cam = scene.add_camera(
        res    = (1280, 720),
        pos    = (1.5, -0.05, 0.3 + robot_config["robot_to_table_height"] / 100),
        lookat = (0.5, -0.05, 0.3 + robot_config["robot_to_table_height"] / 100),
        fov    = 30,
        GUI    = True,
    )
    cams["anno_cam_1"] = left_cam
    cams["anno_cam_2"] = front_cam
    return scene, scene_dict, cams, default_poses

def main(args):
    scene_config = yaml.safe_load(Path(args.scene_cfg_path).open('r'))
    robot_config = yaml.safe_load(Path(args.robot_cfg_path).open('r'))

    from datetime import datetime
    now = datetime.now()
    milliseconds = now.microsecond // 1000
    timestamp = now.strftime("%Y%m%d_%H%M%S") + f"_{milliseconds:03d}"
    process_output_dir = os.path.join(args.output_dir, scene_config["task_name"], timestamp)
    os.makedirs(process_output_dir, exist_ok=True)

    if args.mode == "keyboard" and not KEYBOARD_ROS_ENABLED:
        raise RuntimeError("rospy is not enabled! Install ROS first if using keyboard teleop mode.")

    cprint("*" * 40, "green")
    cprint("  Initializing Genesis", "green")
    cprint("*" * 40, "green")

    gs.init(backend=gs.gpu, logging_level='warning')
    scene, scene_dict, cams, default_poses = design_scene(scene_config, robot_config, show_viewer=True)
    scene.build()

    robot = scene_dict["robot"]
    object_active = scene_dict["object_active"]
    object_passive = scene_dict["object_passive"]
    init_scene_physic_params(scene, scene_dict, scene_config, robot_config)
    cprint("*" * 40, "green")
    cprint("  Initializing Controller", "green")
    cprint("*" * 40, "green")
    
    controller = pick_and_place_controller(
        scene=scene,
        scene_dict=scene_dict,
        scene_config=scene_config,
        robot_config=robot_config,
        default_poses=default_poses,
        close_thres=robot_config["close_thres"],
        teleop=args.mode,
    )
    if args.mode == "keyboard":
        keyboard_controller = keyboard_teleop_controller(robot, robot_config, controller)
    def reset_layout():
        nonlocal keyboard_controller
        keyboard_controller.reset()
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
            # if (
            #     passive_aabb[0, 0] > scene_config["object_passive"]["pos_range"]["x"][0] / 100
            #     and passive_aabb[0, 1] > scene_config["object_passive"]["pos_range"]["y"][0] / 100
            #     and passive_aabb[1, 0] < scene_config["object_passive"]["pos_range"]["x"][1] / 100
            #     and passive_aabb[1, 1] < scene_config["object_passive"]["pos_range"]["y"][1] / 100
            # ):
            #     pass
            # else:
            #     cprint("passive out of range", "yellow")
            #     continue
            active_xyaabb = active_aabb.cpu().numpy()[:2, :2]
            passive_xyaabb = passive_aabb.cpu().numpy()[:2, :2]
            x_overlap = active_xyaabb[0, 0] < passive_xyaabb[1, 0] and active_xyaabb[1, 0] > passive_xyaabb[0, 0]
            y_overlap = active_xyaabb[0, 1] < passive_xyaabb[1, 1] and active_xyaabb[1, 1] > passive_xyaabb[0, 1]
            if x_overlap and y_overlap:
                cprint("active-passive overlap box", "yellow")
                continue
            if np.linalg.norm(passive_pos - active_pos) > scene_config["far_threshold"]:
                cprint("active-passive too near", "yellow")
            break

    
    input_queue = queue.Queue()

    def on_press(key):
        try:
            c = key.char
            input_queue.put(('press', c))
        except AttributeError:
            pass

    def on_release(key):
        try:
            c = key.char
            input_queue.put(('release', c))
        except AttributeError:
            pass

    cprint("1: reset & recollect, 2: start record, 3: end record, 4: save & reset", "cyan")
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    pbar = tqdm(total=None, bar_format='Teleop: {rate_fmt}', unit='frames')

    while True:
        if not listener.is_alive():
            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()
        
        while not input_queue.empty():
            event_type, c = input_queue.get()
            if event_type == 'press':
                if args.mode == "keyboard":
                    keyboard_controller.keyboard_listener(c)
            elif event_type == 'release':
                if c == "1":
                    print("recollect !!!")
                    controller.clean_traj()
                    reset_layout()
                elif c == "2":
                    print("start recording !!!")
                    controller.start_record()
                elif c == "3":
                    print("end recording !!!")
                    controller.end_record()
                elif c == "4":
                    print("save record ...")
                    controller.save_traj(process_output_dir)
                    reset_layout()

        controller.step()
        for cam_key in cams.keys():
            if cam_key.startswith("anno"):
                cams[cam_key].render()
        pbar.update()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="datasets/records")
    parser.add_argument("--scene_cfg_path", type=str, default="simulation/configs/scene/banana_plate_room.yaml")
    parser.add_argument("--robot_cfg_path", type=str, default="simulation/configs/robot/mobile_franka.yaml")
    parser.add_argument("--mode", type=str, default="keyboard", choices=["pico", "keyboard"])
    args = parser.parse_args()
    main(args)