# Installation
The whole project is tested on cuda-11.8.
## Install ros/noetic first
```bash
echo "deb [trusted=yes arch=amd64] http://deb.repo.autolabor.com.cn jammy main" | sudo tee /etc/apt/sources.list.d/autolabor.list
sudo apt update
sudo apt install ros-noetic-autolabor
roscore
```
## Install TAMP
```bash
# sudo apt install git-lfs 
git clone https://github.com/daihangpku/TAMP.git
cd TAMP 
git submodule update --init --recursive
# git lfs install

conda create -n tamp python=3.10 -y
conda activate tamp
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
cd simulation/curobo
pip install -e . --no-build-isolation
cd ../..
```
## Install Anygrasp
In tamp env:
```bash
# Install libssl
sudo add-apt-repository ppa:nrbrtx/libssl1
sudo apt update
sudo apt install libssl1.1

# Install graspnetAPI
pip install git+https://github.com/hwfan/graspnetAPI.git

# Install pointnet2
pushd simulation/anygrasp_sdk/pointnet2 && python setup.py install --verbose && popd

# Install MinkowskiEngine
conda install openblas-devel -c anaconda -y
# pip install git+https://github.com/NVIDIA/MinkowskiEngine.git --global-option="--blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas" --verbose
BLAS_INCLUDE_DIRS="${CONDA_PREFIX}/include" BLAS=openblas pip install -v --no-build-isolation git+https://github.com/NVIDIA/MinkowskiEngine.git
# Copy license
your_license_path=~/workspace/anygrasp_sdk/license_registration/
cp -r $your_license_path/license simulation/anygrasp_sdk/grasp_detection
cp -r $your_license_path/license simulation/anygrasp_sdk/license_registration

# Copy checkpoint
your_checkpoint_path=~/workspace/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar
mkdir -p simulation/anygrasp_sdk/grasp_detection/log/
cp -r $your_checkpoint_path simulation/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar

# Copy model
cp simulation/anygrasp_sdk/grasp_detection/gsnet_versions/gsnet.cpython-310-x86_64-linux-gnu.so simulation/anygrasp_sdk/grasp_detection/gsnet.so
cp simulation/anygrasp_sdk/license_registration/lib_cxx_versions/lib_cxx.cpython-310-x86_64-linux-gnu.so simulation/anygrasp_sdk/grasp_detection/lib_cxx.so

# To check license
simulation/anygrasp_sdk/license_registration/license_checker -c simulation/anygrasp_sdk/license_registration/license/licenseCfg.json
```
In two terminals:
```bash
# Terminal 1: keyboard publisher
rosrun TAMP keyboard_node.py
# Terminal 2: simulation subscriber (replace with your sim integration inside sim_control_node)
rosrun TAMP sim_control_node.py
```

## Install DP3
```bash
cd policy/3D-Diffusion-Policy
conda remove -n dp3 --all
conda create -n dp3 python=3.8
conda activate dp3
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
cd 3D-Diffusion-Policy && pip install -e . && cd ..
cd third_party/pytorch3d_simplified && pip install -e . && cd ../..
pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor h5py open3d opencv-python huggingface_hub==0.11.1
pip install kaleido plotly
cd visualizer && pip install -e . && cd ..
pip install pynput rospkg
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```
## Install DP
```bash
cd policy/diffusion_policy_tamp
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
## dp3
```bash
cd policy/3D-Diffusion-Policy
python scripts/generate_zarr.py --data_dir ../../datasets/records/banana_plate --save_dir ../../datasets/records --env_name banana_plate
bash scripts/train_policy.sh -a dp3 -t pick -i 0112 -s 0 -g 0 --zarr_path ../../../datasets/records/banana_plate_zarr_dp3_sim
```