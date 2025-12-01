# Installation
Install ros/noetic first
```bash
echo "deb [trusted=yes arch=amd64] http://deb.repo.autolabor.com.cn jammy main" | sudo tee /etc/apt/sources.list.d/autolabor.list
sudo apt update
sudo apt install ros-noetic-autolabor
roscore
```
```bash
conda create -n genesis python=3.10 -y
conda activate genesis
pip install -r requirements.txt
cd simulation/curobo
pip install -e . --no-build-isolation
cd ../..
```
# Usage
## teleoperate using keyboard
```bash
python simulation/teleop_pick_and_place.py --mode "keyboard"
```
## render
```bash
python simulation/render_pick_and_place.py --record_dir datasets/records/banana_plate/ --cfg_path simulation/configs/banana_plate.yaml
```