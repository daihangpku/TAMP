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