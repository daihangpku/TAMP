# Installation
Install ros/noetic first
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
python simulation/teleop_pick_and_place_keyboard.py
```
## render
```bash
python simulation/render_pick_and_place.py 
```