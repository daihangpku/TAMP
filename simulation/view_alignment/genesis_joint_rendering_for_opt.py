# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
import argparse
import os
import json
from tqdm import tqdm
# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--logs", type=str, default="logs", help="")
parser.add_argument("--background_urdf", type=str, default='outputs/background-20250409_100100_125/splitted_point_cloud/object.urdf')
parser.add_argument("--intr", type=str, default='datasets/records/franka-track/cam_K.txt')

args_cli = parser.parse_args()
os.makedirs(args_cli.logs, exist_ok=True)
    
import genesis as gs
import torch
import sys
import numpy as np
import time
sys.path.insert(0, os.getcwd())
from simulation.gs_viewer import render_and_save_specific_view, GenesisGaussianViewer
from simulation.utils.constants import FR3_DEFAULT_CFG
from scipy.spatial.transform import Rotation

def load_extrinsics(file_path):
    T = np.loadtxt(file_path)
    assert T.shape == (4, 4), "input must be 4*4 matrix"

    R_inv = T[:3, :3].T  # R^T
    t_inv = -R_inv @ T[:3, 3]  # -R^T * t

    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv

    position = T_inv[:3, 3].tolist()

    rotation_matrix = T_inv[:3, :3]
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat().tolist()  # (x, y, z, w)

    quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]


    return position, quaternion,T[:3, :3],T[:3, 3]

def design_scene():
    genesis_scene_path_dict = {
        "robot": "assets/fr3/fr3.urdf",
        "table": args_cli.background_urdf,
    }
    scene = gs.Scene(
        viewer_options = gs.options.ViewerOptions(
            camera_pos    = (0, -3.5, 2.5),
            camera_lookat = (0.0, 0.0, 0.5),
            camera_fov    = 30,
            max_FPS       = 60,
        ),
        sim_options = gs.options.SimOptions(
            dt = 0.01,
        ),
        show_viewer = False,
    )
    franka = scene.add_entity(
        gs.morphs.URDF(
            file  = genesis_scene_path_dict["robot"],
            pos   = (0, 0, 0),
            quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
            scale = 1.0,
            merge_fixed_links=False,
            fixed=True,
            
        ),
        material=gs.materials.Rigid(gravity_compensation=1.0),
    )
    table = scene.add_entity(
        gs.morphs.URDF(
            file  = genesis_scene_path_dict["table"],
            pos   = (0, 0, 0),
            quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
            scale = 1.0,
            fixed=True,
            convexify=False,
        ),
    )
    genesis_scene_dict = {
        "robot": franka,
        "table": table,
    }
    gs_viewer = GenesisGaussianViewer(genesis_scene_path_dict, 
                                      genesis_scene_dict,
                                      args_cli=args_cli,)
    camera_intr = np.loadtxt(args_cli.intr) # json.load(open("assets/realsense/cam_intr.json"))
    camera_intr_dict = dict()
    camera_intr_dict["fx"] = camera_intr[0, 0]
    camera_intr_dict["fy"] = camera_intr[1, 1]
    camera_intr_dict["cx"] = camera_intr[0, 2]
    camera_intr_dict["cy"] = camera_intr[1, 2]
    camera_intr_dict["image_width"] = 1280
    camera_intr_dict["image_height"] = 720
    return scene, gs_viewer, genesis_scene_dict, camera_intr_dict

def run_simulator(scene, gs_viewer, genesis_scene_dict, camera_intr):
    gs_viewer.update()
    robot = genesis_scene_dict["robot"]

    count = 0
    jnt_names = [
        'fr3_joint1',
        'fr3_joint2',
        'fr3_joint3',
        'fr3_joint4',
        'fr3_joint5',
        'fr3_joint6',
        'fr3_joint7',
        'fr3_finger_joint1',
        'fr3_finger_joint2',
    ]
    pbar = tqdm(total=None, desc='frames')
    default_joint_pos = [FR3_DEFAULT_CFG[name] for name in jnt_names]
    cnt = 0
    gap = 50
    # Simulation loop
    robot.set_dofs_position(torch.tensor(default_joint_pos).to("cuda"))
    while True:
        cnt += 1
        if cnt>gap :
            while True:
                if os.path.exists(f"{args_cli.logs}/main.txt") and os.path.exists(f"{args_cli.logs}/sub.txt"):
                    T = np.loadtxt(f"{args_cli.logs}/main.txt")
                    if T[0] == 1:
                        iter = int(T[1])
                        num_particles = int(T[2])
                        break
                    else:
                        time.sleep(1)
            if args_cli.logs is not None:
                    render_dir = os.path.join(args_cli.logs,f"{iter}/images_ori")
                    matrix_dir = os.path.join(args_cli.logs,f"{iter}/matrix")
                    os.makedirs(render_dir, exist_ok=True)
                    try:
                        for i in tqdm(range(1, num_particles+1), desc="rendering views", unit="view"):
                            _, _, R, T = load_extrinsics(os.path.join(matrix_dir,f"1{int(i):04d}.txt"))
                            render_and_save_specific_view(
                                gs_viewer.viewer_renderer,
                                torch.device("cuda:0"),
                                os.path.join(render_dir,f"1{int(i):04d}.png"),
                                camera_intr,
                                R, T,
                                )
                        np.savetxt(f"{args_cli.logs}/main.txt", np.array([0, iter]), fmt="%d")
                        np.savetxt(f"{args_cli.logs}/sub.txt", np.array([1, iter]), fmt="%d")
                        time.sleep(0.1)
                    except:
                        raise ValueError("render error")

        count += 1
        # Update buffers
        scene.step()
        gs_viewer.update()
        pbar.update(1)
        
if __name__ == "__main__":
    gs.init(backend=gs.gpu, logging_level = 'warning')
    scene, gs_viewer, genesis_scene_dict, camera_intr = design_scene()
    scene.build()
    run_simulator(scene, gs_viewer, genesis_scene_dict, camera_intr)
