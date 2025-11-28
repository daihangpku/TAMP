import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import os
import subprocess
import torch
import torch.nn.functional as F
import cv2
import shutil
import sys
import time
import re
import atexit
sys.path.insert(0, os.getcwd())
from simulation.view_alignment.grounded_sam_for_opt import do_grounded_sam, load_model_once
processes = []
def cleanup():
    for proc in processes:
        proc.terminate()
        print(f"Terminated process with PID: {proc.pid}")
atexit.register(cleanup)
sam2_model_cfg, predictor, model, sam2_checkpoint=load_model_once()
def save_transformation_matrix(matrix, output_txt_file):
    try:
        np.savetxt(output_txt_file, matrix, fmt='%.10f')  
    except Exception as e:
        print(f"❌ saving failed: {e}")
        exit()
def mask_loss(pred_mask, true_mask):
    loss = F.binary_cross_entropy(pred_mask, true_mask)
    return loss
def load_mask_from_file(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  

    _, binary_mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask / 255.0 

    mask = torch.tensor(binary_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return mask

def loss_function_batch(rotation_matrices, args = None, iter = 0):
    losses = []
    iter_dir = f"{args.logs}/{iter}"
    os.makedirs(iter_dir, exist_ok=True)
    import shutil
    shutil.rmtree(iter_dir, ignore_errors=True)
    os.makedirs(iter_dir, exist_ok=True)
    matrix_dir = f"{args.logs}/{iter}/matrix"
    

    os.makedirs(matrix_dir, exist_ok=True)
    for i in range(rotation_matrices.shape[0]):
        save_transformation_matrix(rotation_matrices[i], os.path.join(matrix_dir, f"1{(i+1):04d}.txt"))

    np.savetxt(f"{args.logs}/main.txt", np.array([1, iter, args.num_particles]), fmt="%d")
    np.savetxt(f"{args.logs}/sub.txt", np.array([0, iter]), fmt="%d")
    
    while True:
        if os.path.exists(f"{args.logs}/main.txt") and os.path.exists(f"{args.logs}/sub.txt"):
            T = np.loadtxt(f"{args.logs}/sub.txt")
            if T[0] == 1:
                break
            else:
                time.sleep(1)
    args.output_dir = f'{args.logs}/{iter}'
    args.input_dir = f'{args.logs}/{iter}/images_ori'
    src_image = args.real_picture
    dest_image = args.input_dir
    shutil.copy(src_image, os.path.join(dest_image, "00000.png"))
    shutil.copy(src_image, os.path.join(dest_image, "00001.png"))
    do_grounded_sam(args, sam2_model_cfg, predictor, model, sam2_checkpoint)
    masks_dir = f'{args.logs}/{iter}/masks'
    target_mask = []
    mask_files = sorted([f for f in os.listdir(masks_dir) if re.match(r"00001_\d+\.png", f)])
    for mask_file in mask_files:
        pred_mask = load_mask_from_file(os.path.join(masks_dir, mask_file))
        target_mask.append(pred_mask)

    for i in range(1, args.num_particles+1):
        loss = 0
        mask_files = sorted([f for f in os.listdir(masks_dir) if re.match(fr"1{(i):04d}_\d+\.png", f)])
        for id, mask_file in enumerate(mask_files):
            pred_mask = load_mask_from_file(os.path.join(masks_dir, mask_file))
            loss += mask_loss(pred_mask, target_mask[id])
        losses.append(loss.item())
    print(f"loss: {losses}")
    return losses

def load_transformation_matrix(file_path):
    try:
        if file_path.endswith('.npy'):
            initial_transform = np.load(file_path)
            if initial_transform.ndim == 3:
                initial_transform = initial_transform[0]
        elif file_path.endswith('.txt'):
            initial_transform = np.loadtxt(file_path)
        else:
            raise ValueError("❌ unsupported format! plz using .txt or .npy file")
        
        if initial_transform.shape != (4, 4):
            raise ValueError("❌ matrix should be 4x4!")
        return initial_transform
    except Exception as e:
        print(f"❌ reading failed: {e}")
        exit()


def pso_optimization(loss_function, w=0.8, c1=0.8, c2=0.8, max_distance=0.1, initial_matrix=None, args = None):
    USE_QUAT = args.use_quat
    if USE_QUAT:
        print("Using quaternion representation for rotation.")
    else:
        print("Using Euler angles representation for rotation.")
    # initialize particles
    init_rotation = R.from_matrix(initial_matrix[:3, :3])
    init_translation = initial_matrix[:3, 3]
    if USE_QUAT:
        init_quat = init_rotation.as_quat()
        particles_quat = np.random.randn(args.num_particles, 4) * max_distance + init_quat
        particles_quat[0] = init_quat
        for i in range(args.num_particles):
            particles_quat[i] = particles_quat[i] / np.linalg.norm(particles_quat[i])
        particles_dquat = np.zeros_like(particles_quat)  # init velocity=0
    else:
        init_angle = init_rotation.as_euler('xyz', degrees=False)
        particles_angle = np.random.randn(args.num_particles, 3) * max_distance + init_angle
        particles_angle[0] = init_angle
        particles_dangle = np.zeros_like(particles_angle)
    #init_source_to_target[:3, :3] = R.from_euler('xyz', [init_x_angle, init_y_angle, init_z_angle], degrees=True).as_matrix()
    
    

    particles_translation = np.random.randn(args.num_particles, 3) * max_distance + init_translation
    
    particles_dtranslation = np.zeros_like(particles_translation)  # init velocity=0
    particles_position = np.zeros((args.num_particles, 4, 4)) 

    for i in range(args.num_particles):
        if USE_QUAT:
            particles_position[i, :3, :3] = R.from_quat(particles_quat[i]).as_matrix()
        else:
            particles_position[i, :3, :3] = R.from_euler('xyz', particles_angle[i], degrees=False).as_matrix()
        particles_position[i, :3, 3 ] = particles_translation[i]
        particles_position[i, 3, 3 ] = 1
        particles_position[i, 3, :3 ] = 0

    personal_best_position = particles_position.copy()
    personal_best_loss = loss_function(particles_position, args = args, iter = 0)  
    
    global_best_position = personal_best_position[np.argmin(personal_best_loss)]
    global_best_loss = np.min(personal_best_loss)
    
    for iteration in range(1,args.num_iterations+1):

        current_loss = loss_function(particles_position, args=args, iter = iteration)
        
        # update personal best
        for i in range(args.num_particles):
            if current_loss[i] < personal_best_loss[i]:
                personal_best_position[i] = particles_position[i]
                personal_best_loss[i] = current_loss[i]
        
        # update global best
        min_loss_index = np.argmin(current_loss)
        iter_best_position = particles_position[min_loss_index]
        if current_loss[min_loss_index] < global_best_loss:
            global_best_position = particles_position[min_loss_index]
            global_best_loss = current_loss[min_loss_index]
        
        # update particles
        for i in range(args.num_particles):
            if USE_QUAT:
                inertia_quat = w * particles_dquat[i] 
                cognitive_quat = c1 * np.random.random(4)*(R.from_matrix(personal_best_position[i, :3, :3]).as_quat() - R.from_matrix(particles_position[i, :3, :3]).as_quat())
                social_quat = c2 * np.random.random(4) *(R.from_matrix(global_best_position[:3, :3]).as_quat() - R.from_matrix(particles_position[i, :3, :3]).as_quat())
                particles_dquat[i] = inertia_quat + cognitive_quat + social_quat
                particles_quat[i] += particles_dquat[i]
                # Normalize quaternion
                particles_quat[i] /= np.linalg.norm(particles_quat[i])
                particles_position[i, :3, :3] = R.from_quat(particles_quat[i]).as_matrix()
            else:
                inertia_angle = w * particles_dangle[i]
                cognitive_angle = c1 * np.random.random(3) * (R.from_matrix(personal_best_position[i, :3, :3]).as_euler('xyz', degrees=False) - R.from_matrix(particles_position[i, :3, :3]).as_euler('xyz', degrees=False))
                social_angle = c2 * np.random.random(3) * (R.from_matrix(global_best_position[:3, :3]).as_euler('xyz', degrees=False) - R.from_matrix(particles_position[i, :3, :3]).as_euler('xyz', degrees=False))
                particles_dangle[i] = inertia_angle + cognitive_angle + social_angle
                particles_angle[i] += particles_dangle[i]
                particles_position[i, :3, :3] = R.from_euler('xyz', particles_angle[i], degrees=False).as_matrix()

            inertia_translation = w * particles_dtranslation[i]
            
            cognitive_translation = c1 * np.random.random(3) * (personal_best_position[i, :3, 3] - particles_position[i, :3, 3])
            social_translation = c2 * np.random.random(3) * (global_best_position[:3, 3] - particles_position[i, :3, 3])
            particles_dtranslation[i] = inertia_translation + cognitive_translation + social_translation
            particles_translation[i] += particles_dtranslation[i]

            
            particles_position[i, :3, 3] = particles_translation[i]

            particles_position[i, 3, :3] = 0
            particles_position[i, 3, 3] = 1

        print(f"Iteration {iteration}, Global Best Loss: {global_best_loss}, iter best index: {min_loss_index+1}")
        iter_best_dir = f"{args.logs}/{iteration}/iter_best"
        os.makedirs(iter_best_dir, exist_ok=True)
        save_transformation_matrix(iter_best_position, os.path.join(iter_best_dir, f"best.txt"))
        save_transformation_matrix([min_loss_index+1, current_loss[min_loss_index]], os.path.join(iter_best_dir, f"loss.txt"))
        shutil.copy(args.real_picture, os.path.join(iter_best_dir, "00000.png"))
        shutil.copy(f"{args.logs}/{iteration}/images_ori/1{(min_loss_index+1):04d}.png", os.path.join(iter_best_dir, f"1{(min_loss_index+1):04d}.png"))
        np.savetxt(f"{args.logs}/best.txt", global_best_position)
    return global_best_position

parser = argparse.ArgumentParser(description="")
parser.add_argument("-it", "--init_trans", type=str, default="/home/daihang/twinmanip/logs/results/best.txt", help="")
parser.add_argument("-l", "--logs", type=str, default=None, help="")
parser.add_argument("-rp", "--real_picture", type=str, required=True, help="real picture directory")
parser.add_argument('--output_dir', type=str, default="", help='')# no need to provide
parser.add_argument('--input_dir', type=str, default="", help='')# no need to provide
parser.add_argument('--segment_prompt', type=str, required=True, help='segment_prompt')
parser.add_argument('--bg_color', type=str, choices=['white', 'black'], default='black', help='background color')
parser.add_argument('--track_sam', action='store_true', help='track sam')
parser.add_argument("--interactive_mask", action="store_true")
parser.add_argument("--mask_only", action="store_true")
parser.add_argument("--num_particles", type=int, default=100)
parser.add_argument("--num_iterations", type=int, default=20)
parser.add_argument("--max_distance", type=float, default=0.01)
parser.add_argument("--c1", type=float, default=0.2)
parser.add_argument("--c2", type=float, default=0.5)
parser.add_argument("--use_quat", action="store_true")
parser.add_argument('--background_urdf', type=str, default="outputs/background-20250409_100100_125/splitted_point_cloud/object.urdf")
parser.add_argument('--intr', type=str, default="datasets/records/franka-track/cam_K.txt")
args = parser.parse_args()
import subprocess, threading
def render_process(args):
    proc=subprocess.Popen(['python', 'simulation/view_alignment/genesis_joint_rendering_for_opt.py', '--logs', f'{args.logs}', '--background_urdf', args.background_urdf,
                           '--intr', args.intr])
    processes.append(proc)
thread = threading.Thread(target=render_process, args=(args,))
thread.daemon = True  # Daemonize thread
thread.start()
transformation_txt = args.init_trans
os.makedirs(args.logs, exist_ok=True)
os.makedirs(os.path.join(args.logs, "matrix"), exist_ok=True)
initial_matrix = load_transformation_matrix(transformation_txt)

optimized_matrix = pso_optimization(loss_function_batch,  max_distance=args.max_distance, c1 = args.c1, c2= args.c2, initial_matrix = initial_matrix ,args = args)

print("Optimized Matrix:")
print(optimized_matrix)
