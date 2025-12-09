# Installation
Install ros/noetic first
```bash
echo "deb [trusted=yes arch=amd64] http://deb.repo.autolabor.com.cn jammy main" | sudo tee /etc/apt/sources.list.d/autolabor.list
sudo apt update
sudo apt install ros-noetic-autolabor
roscore
```
```bash
# sudo apt install git-lfs 
git clone https://github.com/daihangpku/TAMP.git
cd TAMP 
git submodule update --init --recursive
# git lfs install

conda create -n genesis python=3.10 -y
conda activate genesis
pip install -r requirements.txt
cd simulation/curobo
pip install -e . --no-build-isolation
cd ../..

## Run teleop with ROS split processes
In two terminals:
```bash
# Terminal 1: keyboard publisher
rosrun TAMP keyboard_node.py
# Terminal 2: simulation subscriber (replace with your sim integration inside sim_control_node)
rosrun TAMP sim_control_node.py
```
# Usage
## teleoperate using keyboard
```bash
export SETUPTOOLS_USE_DISTUTILS=stdlib
python simulation/teleop_pick_and_place.py --mode "keyboard"
```
## render
```bash
python simulation/render_pick_and_place.py --record_dir datasets/records/banana_plate/ --cfg_path simulation/configs/banana_plate.yaml
```
## dp 
### training
```bash
cd policy/diffusion_policy_tamp
conda activate robodiff
bash scripts/generate_data.sh -i ../../datasets/records/banana_plate --gripper True --sim # convert2zarr
bash scripts/train.sh --input /home/daihang/school/core/TAMP/datasets/records/banana_plate_zarr_dp_sim_demonum2/train --task pnp # train
```
### eval
in one terminal at home dir with genesis env:
```bash
python simulation/eval_pick_and_place.py 
```
in another at policy/diffusion_policy_tamp with robodiff env:
```bash
bash scripts/eval.sh --input data/outputs/2025.12.09/14.30.09_train_diffusion_unet_image_real_image/checkpoints/latest.ckpt -g -s
```