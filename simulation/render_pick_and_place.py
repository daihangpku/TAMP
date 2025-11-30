import argparse
import os
import json
import genesis as gs
import torch
import sys
import numpy as np
from tqdm import tqdm
import imageio
from scipy.spatial.transform import Rotation
import h5py
import cv2
import yaml
from pathlib import Path
from easydict import EasyDict
import open3d as o3d
os.environ["PYOPENGL_PLATFORM"] = "egl"
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulation.auto_collect_pick_and_place import design_scene as design_pnp_scene

def render_scene(scene_dict, data, grasp_cam, args_cli=None, znear=0.1, render_types=["rgb", "depth"]):
    if data is not None:
        robot = scene_dict["robot"]
        object_active = scene_dict["object_active"]
        object_passive = scene_dict["object_passive"]
        robot.set_dofs_position(data["joint_states"])
        passive_pos = data["object_states"]["passive"][:3]
        passive_quat = data["object_states"]["passive"][3:7]
        active_pos = data["object_states"]["active"][:3]
        active_quat = data["object_states"]["active"][3:7]
        object_passive.set_pos(passive_pos)
        object_passive.set_quat(passive_quat)
        object_active.set_pos(active_pos)
        object_active.set_quat(active_quat)

    rgb_image = None
    depth_image = None
    need_rgb = "rgb" in render_types
    need_depth = "depth" in render_types
    if not (need_rgb or need_depth):
        return rgb_image, depth_image

    # 使用仿真相机渲染
    import ipdb; ipdb.set_trace()
    rgb_raw, depth_raw = grasp_cam.render(rgb=need_rgb, depth=need_depth)
    sim_rgb = rgb_raw if need_rgb else None
    sim_depth = depth_raw if need_depth else None

    if sim_rgb is not None:
        rgb_np = sim_rgb
        if isinstance(rgb_np, torch.Tensor):
            rgb_np = rgb_np.detach().cpu().numpy()
        if rgb_np.dtype != np.uint8:
            max_val = float(rgb_np.max()) if rgb_np.size > 0 else 1.0
            if max_val <= 1.0 + 1e-3:
                rgb_np = (rgb_np * 255.0).clip(0, 255).astype(np.uint8)
            else:
                rgb_np = rgb_np.clip(0, 255).astype(np.uint8)
        rgb_image = np.ascontiguousarray(rgb_np[..., :3])

    if sim_depth is not None:
        depth_np = sim_depth
        if isinstance(depth_np, torch.Tensor):
            depth_np = depth_np.detach().cpu().numpy()
        depth_image = np.clip(depth_np * 1000.0, 0, 65535).astype('uint16')

    return rgb_image, depth_image
            
def read_h5_file(file_path):
    """
    读取层级结构的HDF5文件，返回嵌套字典，其中值为NumPy数组。
    
    参数:
        file_path (str): HDF5文件的路径
        
    返回:
        dict: 嵌套字典，包含文件中的所有组和数据集
    """
    def traverse_group(group):
        data = {}
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                # 读取数据集并存储为NumPy数组
                data[key] = item[()]
            elif isinstance(item, h5py.Group):
                # 递归遍历子组
                data[key] = traverse_group(item)
        return data
    
    with h5py.File(file_path, 'r') as f:
        return traverse_group(f)

def load_extrinsics(file_path):
    T = np.loadtxt(file_path)
    assert T.shape == (4, 4), "输入必须是 4×4 矩阵"

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

def visualize_depth(depth_map, max_depth=None, view=True):
    """
    可视化深度图
    :param depth_map: 输入的深度图（单通道）
    :param max_depth: 可选参数，指定最大深度值进行归一化
    :return: 可视化后的彩色深度图
    """

    # 如果未指定max_depth，则使用深度图的最大值
    if max_depth is None:
        max_depth = np.max(depth_map)
    
    # 将深度值归一化到0-255范围并转换为8位无符号整型
    depth_vis = np.clip(depth_map, 0, max_depth)  # 限制最大深度
    depth_vis = (depth_vis / max_depth * 255).astype(np.uint8)
    
    # 应用颜色映射（这里使用JET颜色映射）
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    if view:
        cv2.imshow("Depth Visualization", depth_colormap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return depth_colormap

def visualize_rgb(rgb):
    cv2.imshow("RGB Visualization", rgb[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def visualize_rgbd(rgb, depth, camera_intr, depth_scale=1000.0):
    """
    输入:
        rgb: HxWx3的RGB图像 (np.uint8格式, 0-255)
        depth: HxW的深度图 (np.float32, 单位通常为毫米)
        depth_scale: 深度值缩放比例（默认1000表示深度单位为毫米）
    """
    # 1. 创建Open3D的RGB和深度图像对象
    rgb_o3d = o3d.geometry.Image(rgb)
    depth_o3d = o3d.geometry.Image(depth)

    # 2. 构建RGBD图像
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=rgb_o3d,
        depth=depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=3.0,  # 深度截断值（根据实际调整）
        convert_rgb_to_intensity=False
    )

    # 3. 生成点云并可视化
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=camera_intr["image_width"],
        height=camera_intr["image_height"],
        fx=camera_intr["fx"],  # 默认焦距（需根据相机参数调整）
        fy=camera_intr["fy"],
        cx=camera_intr["cx"],
        cy=camera_intr["cy"]
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic
    )
    
    # 4. 可视化
    o3d.io.write_point_cloud("debug.ply", pcd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_dir", type=str, default=None)
    parser.add_argument('--cfg_path', type=str, default=None)
    parser.add_argument("--mod", type=int, default=5)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--demo_min_idx", type=int, default=0)
    parser.add_argument('--demo_num', type=int, default=200)
    parser.add_argument("--nots", action="store_true", default=False)
    parser.add_argument('--render_types', default=["rgb","depth"], type=list, help='render types, comma separated, options: rgb, depth')
    args_cli = parser.parse_args()
    args_cli.render_types = args_cli.render_types.split(',')
    scene_config = EasyDict(yaml.safe_load(Path(args_cli.cfg_path).open('r')))
    
    gs.init(backend=gs.gpu, logging_level = 'error')
    scene, scene_config, scene_dict, scene_asset_path_dict, grasp_cam, default_poses = design_pnp_scene(scene_config, show_viewer=False)
    scene.build()

    intr_mat = np.array(grasp_cam.intrinsics)
    width, height = grasp_cam.res
    camera_intr = {
        "image_width": width,
        "image_height": height,
        "fx": intr_mat[0, 0],
        "fy": intr_mat[1, 1],
        "cx": intr_mat[0, 2],
        "cy": intr_mat[1, 2],
    }
    
    def render_demos(demo_dirs):
        for demo_dir in tqdm(demo_dirs, desc="demo"):
            h5_file_idxs = [int(file.replace(".h5", "")) for file in os.listdir(demo_dir) if file.endswith(".h5") and int(file.replace(".h5", "")) % args_cli.mod == 0]
            h5_file_idxs = sorted(h5_file_idxs)
            for h5_file_idx in tqdm(h5_file_idxs, desc="frame"):
                h5_path = os.path.join(demo_dir, f"{h5_file_idx}.h5")
                data = read_h5_file(h5_path)
                rgb_image, depth_image = render_scene(scene_dict, data, grasp_cam, args_cli, render_types=args_cli.render_types)
                if args_cli.debug:
                    if "rgb" in args_cli.render_types and "depth" in args_cli.render_types:
                        visualize_rgbd(rgb_image, depth_image, camera_intr)
                        print("saved rgbd to debug.ply in current directory")
                    if "rgb" in args_cli.render_types:
                        visualize_rgb(rgb_image)
                    if "depth" in args_cli.render_types:                        
                        visualize_depth(depth_image)
                else:
                    if "rgb" in args_cli.render_types:                        
                        imageio.imwrite(h5_path.replace(".h5", ".jpg"), rgb_image, quality=90)
                    if "depth" in args_cli.render_types:
                        cv2.imwrite(h5_path.replace(".h5", "_depth.png"), depth_image)
                        depth_vis = visualize_depth(depth_image, view=False)
                        cv2.imwrite(h5_path.replace(".h5", "_depth_render.jpg"), depth_vis)
    def collect_demo_dirs():
        if not args_cli.nots:
            demo_dirs = []
            timestamps = os.listdir(args_cli.record_dir)
            for timestamp in timestamps:
                timestamp_dir = os.path.join(args_cli.record_dir, timestamp)
                sub_demo_dirs = [os.path.join(timestamp_dir, x) for x in sorted(os.listdir(timestamp_dir))]
                demo_dirs.extend(sub_demo_dirs)
        else:
            demo_dirs = [os.path.join(args_cli.record_dir, x) for x in sorted(os.listdir(args_cli.record_dir))]
        return demo_dirs
    demo_dirs = collect_demo_dirs()
    demo_dirs = demo_dirs[args_cli.demo_min_idx : args_cli.demo_min_idx + args_cli.demo_num]
    print(f"Render {len(demo_dirs)} demos.")
    render_demos(demo_dirs)