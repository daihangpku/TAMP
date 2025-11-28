import os
import torch
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Literal, List
import cv2
import json
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fast_gaussian_model_manager import FastGaussianModelManager, construct_from_ply, matrix_to_quaternion
from gs_viewer import render_and_save_specific_view
from utils.gs_viewer_utils import ClientThread, ViewerRenderer, GSPlatRenderer
from utils.gs_viewer_utils import *
import imageio

def readTransformsJson(transforms_json_path, matrix_format="c2w", coord_system="opengl"):
    """
    从transforms.json文件中读取相机参数
    
    Args:
        transforms_json_path: transforms.json文件路径
        matrix_format: "c2w" (camera-to-world, NeRF格式) 或 "w2c" (world-to-camera, COLMAP格式)
        coord_system: "opengl" (OpenGL/NeRF坐标系) 或 "opencv" (OpenCV坐标系)
    """
    cam_infos = []
    
    with open(transforms_json_path, 'r') as f:
        transforms_data = json.load(f)
    
    # 获取内参
    if 'fl_x' in transforms_data["frames"][0] and 'fl_y' in transforms_data["frames"][0]:
        fx = transforms_data["frames"][0]['fl_x']
        fy = transforms_data["frames"][0]['fl_y']
        cx = transforms_data["frames"][0].get('cx', transforms_data["frames"][0]['w'] / 2.0)
        cy = transforms_data["frames"][0].get('cy', transforms_data["frames"][0]['h'] / 2.0)
        width = int(transforms_data["frames"][0]['w'])
        height = int(transforms_data["frames"][0]['h'])
    else:
        # 如果没有明确的焦距信息，尝试从camera_angle_x计算
        camera_angle_x = transforms_data.get('camera_angle_x', None)
        try:
            width = int(transforms_data["frames"][0]['w'])
            height = int(transforms_data["frames"][0]['h'])
        except:
            width = int(transforms_data['w'])
            height = int(transforms_data['h'])
        if camera_angle_x is not None:
            fx = width / (2.0 * np.tan(camera_angle_x / 2.0))
            fy = fx  # 假设方形像素
            print(f"Calculated fx/fy from camera_angle_x: {fx}")
        else:
            # 使用默认值
            fx = width / 2.0
            fy = height / 2.0
            print(f"Using default fx/fy: fx={fx}, fy={fy}")
        cx = width / 2.0
        cy = height / 2.0
    
    intrinsic = {
        "image_height": height,
        "image_width": width,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy
    }
    
    print(f"Camera intrinsics: {intrinsic}")
    print(f"Camera angle x: {transforms_data.get('camera_angle_x', 'N/A')}")
    
    frames = transforms_data['frames']
    for idx, frame in enumerate(frames):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(frames)))
        sys.stdout.flush()
        
        # 获取变换矩阵
        transform_matrix = np.array(frame['transform_matrix'])
        
        if matrix_format == "c2w":
            # NeRF格式：transform_matrix是c2w，需要转换为w2c
            c2w = transform_matrix
            w2c = np.linalg.inv(c2w)
        else:  # matrix_format == "w2c"
            # COLMAP格式：transform_matrix已经是w2c
            w2c = transform_matrix
            c2w = np.linalg.inv(w2c)
        
        # 坐标系转换
        if coord_system == "opencv":
            # 从OpenGL坐标系转换到OpenCV坐标系
            # OpenGL: +Y向上, +Z向屏幕内
            # OpenCV: +Y向下, +Z向屏幕外
            opengl_to_opencv = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
            
            if matrix_format == "c2w":
                # 对c2w矩阵应用转换
                c2w = c2w @ opengl_to_opencv
                w2c = np.linalg.inv(c2w)
            else:
                # 对w2c矩阵应用转换
                w2c = opengl_to_opencv @ w2c
        
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        
        # 使用文件名作为key
        key = frame.get('file_path', f"frame_{idx:04d}")
        if key.startswith('./'):
            key = key[2:]
        
        cam_info = (key, intrinsic, R, T)
        cam_infos.append(cam_info)
    
    return cam_infos

        
def _load_rigid_from_file(load_from: str, device: torch.device=torch.device("cuda:0")):
    model = construct_from_ply(
        ply_path=Path(load_from), 
        device=device,
    )
    return model


class GenesisGaussianViewer:
    def __init__(
        self, 
        ply_path: str = None,
        device: torch.device=torch.device("cuda:0"),
        background_color: Tuple = (0, 0, 0),
        image_format: Literal["jpeg", "png"] = "jpeg",
        active_sh_degree: int = 3,
        num_envs: int = 1,
    ):
        super().__init__()
        self.device = device
        self.asset_names = []
        self.gs_model_list = []
        self.part_names = []
        self.num_envs = num_envs
        self.urdfs = dict()
        
        if ply_path and os.path.exists(ply_path):
            gs_model = _load_rigid_from_file(load_from=ply_path, device=device)
            self.gs_model_list.append(gs_model)
        
        self.env_gs_model_manager = FastGaussianModelManager(
            gaussian_models=self.gs_model_list, 
            num_envs=self.num_envs,
            device=device,
            active_sh_degree=active_sh_degree,
        )
        self.renderer = GSPlatRenderer()
        self.background_color = background_color
        self.image_format = image_format
        self.viewer_renderer = ViewerRenderer(
            self.env_gs_model_manager,
            self.renderer,
            torch.tensor(self.background_color, dtype=torch.float, device=self.device),
        )


if __name__ == "__main__":
    # Example usage
    
    # 使用同一目录下的transforms文件
    object_name = "pink-20250627_184711_116"
    path = f"/home/daihang/workspace/twinaligner/gaussian-splatting/datasets/{object_name}"
    output_dir = f"/home/daihang/workspace/twinaligner/gaussian-splatting/output/{object_name}"
    output_path = os.path.join(output_dir, "rendered_depth")
    transforms_json_file = os.path.join(path, "transforms.json")
    ply_path = os.path.join(output_dir, "point_cloud/iteration_30000/point_cloud.ply")
    
    # 尝试不同的矩阵格式和坐标系
    matrix_format = "c2w"  # 尝试 "c2w" 或 "w2c"
    coord_system = "opencv"  # 尝试 "opengl" 或 "opencv"
    print(f"Using matrix format: {matrix_format}")
    print(f"Using coordinate system: {coord_system}")
    
    viewer = GenesisGaussianViewer(ply_path=ply_path)
    # 检查transforms文件是否存在
    if not os.path.exists(transforms_json_file):
        print(f"Transforms file not found at {transforms_json_file}")
        print("Please provide a transforms_train.json file in the same directory")
        sys.exit(1)
    
    cameras_info = readTransformsJson(transforms_json_file, matrix_format=matrix_format, coord_system=coord_system)
    os.makedirs(output_path, exist_ok=True)
    from tqdm import tqdm
    for cam_info in tqdm(cameras_info):
        key, intr, R, T = cam_info
        # 创建完整的文件路径并确保目录存在
        full_output_path = os.path.join(output_path, f"{key}.png")
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        
        rendered_nontable = render_and_save_specific_view(
            viewer.viewer_renderer, 
            torch.device("cuda"),
            full_output_path,
            intr,
            R = R,
            T = T,
            verbose=False,
            render_alpha=True,
            render_depth=True,
            return_outputs=True,
            save=True,
        )
        # rgb_image = rendered_nontable["rgb_image"]
        # depth_image = rendered_nontable["depth_image"]
    # alpha_image = rendered_nontable["alpha_image"]
    