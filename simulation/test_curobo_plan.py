import os
import sys
sys.path.insert(0, os.getcwd())
import yaml
from pathlib import Path
from termcolor import cprint

import genesis as gs
from simulation.utils.scene_utils import design_pnp_scene, init_scene_physic_params
from simulation.controller.franka_genesis_controller import franka_controller

# 目标末端位姿（w,x,y,z 四元数）
GOAL_POS = [1.0, 0.0, 0.4]
GOAL_QUAT = [1.0, 0.0, 0.0, 0.0]  # 若保持当前朝向，可忽略此值

def main(scene_cfg_path="simulation/configs/scene/banana_plate_room.yaml",
         robot_cfg_path="simulation/configs/robot/mobile_franka.yaml"):
    # 加载配置
    scene_config = yaml.safe_load(Path(scene_cfg_path).open('r'))
    robot_config = yaml.safe_load(Path(robot_cfg_path).open('r'))

    # 初始化 Genesis
    gs.init(backend=gs.gpu, logging_level='warning')
    cprint("Initializing scene...", "green")

    # 使用 teleop 的构建函数，场景内已有桌面/房间/机器人
    scene, scene_dict, cams, default_poses = design_pnp_scene(scene_config, robot_config, show_viewer=True)

    # 相机（可选）
    scene.add_camera(
        res=(1280, 720),
        pos=(1.5, 0.5, 0.8 + robot_config["robot_to_table_height"] / 100),
        lookat=(0.0, 0.0, robot_config["robot_to_table_height"] / 100),
        fov=45,
        GUI=True,
    )

    scene.build()

    # 初始化物理参数（KP/KV/力范围等）
    init_scene_physic_params(scene, scene_dict, scene_config, robot_config)

    # 控制器（内部使用 MobileFrankaSolver 的 MotionGen）
    controller = franka_controller(
        scene=scene,
        scene_dict=scene_dict,
        scene_config=scene_config,
        robot_config=robot_config,
        close_thres=robot_config.get("close_thres", 1.0),
        teleop=None,
        evaluation=False,
    )

    # 重置到默认位姿
    controller.reset_franka()
    for _ in range(50):
        controller.step()

    joint_pos = controller.franka.get_dofs_position().cpu().numpy()
    curr_pos, curr_quat = controller.franka_solver.compute_fk(joint_pos)

    # goal_pos = [curr_pos[0] + DELTA_POS[0],
    #             curr_pos[1] + DELTA_POS[1],
    #             curr_pos[2] + DELTA_POS[2]]
    goal_pos = GOAL_POS
    goal_quat = GOAL_QUAT

    cprint(f"Current EE pos: {curr_pos} quat: {curr_quat}, goal pos: {goal_pos} quat: {goal_quat}", "cyan")

    # 规划并移动到目标（小幅偏移）
    cprint("Planning and moving to nearby goal...", "cyan")
    try:
        controller.move_to_goal(pos=goal_pos, quat=goal_quat, gripper_open=True, quick=False)
        cprint("Reached nearby goal.", "green")
    except Exception as e:
        cprint(f"Planning failed: {e}", "red")

    # 停留观察
    for _ in range(300):
        controller.step()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_cfg_path", type=str, default="simulation/configs/scene/banana_plate_room.yaml")
    parser.add_argument("--robot_cfg_path", type=str, default="simulation/configs/robot/mobile_franka.yaml")
    args = parser.parse_args()
    main(args.scene_cfg_path, args.robot_cfg_path)