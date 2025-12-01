FR3_DEFAULT_CFG = {
    "fr3_joint1": 0.03995192,
    "fr3_joint2": -0.00592799,#
    "fr3_joint3": -0.18448183,
    "fr3_joint4": -2.21950033,
    "fr3_joint5": -0.00492253,
    "fr3_joint6": 2.22304146,
    "fr3_joint7": 0.69084185,
    "fr3_finger_joint1": 0.04,
    "fr3_finger_joint2": 0.04,
}
XML_DEFAULT_CFG = {
    "joint1": 0.03995192,
    "joint2": -0.00592799,#
    "joint3": -0.18448183,
    "joint4": -2.21950033,
    "joint5": -0.00492253,
    "joint6": 2.22304146,
    "joint7": 0.69084185,
    "finger_joint1": 0.04,
    "finger_joint2": 0.04,
}
MOBILE_DEFAULT_CFG = {
    "base_x": 0.0,
    "base_y": 0.0,
    "base_z": 0.0,
    "panda_joint1": 0.03995192,
    "panda_joint2": -0.00592799,#
    "panda_joint3": -0.18448183,
    "panda_joint4": -2.21950033,
    "panda_joint5": -0.00492253,
    "panda_joint6": 2.22304146,
    "panda_joint7": 0.69084185,
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}
BEST_PARAMS = {
    "kp": [
        2424.657254246728,
        16976.159595959594,
        12101.932123237386,
        10031.517126148701,
        1238.0951824004453,
        69.25030758714965,
        396.801903749272
    ],
    "kv": [
        2003.6817042606517,
        3271.0,
        1537.591478696742,
        1456.187969924812,
        460.44862155388466,
        13.506265664160395,
        80.49874686716791
    ],
}

CAMERA_INTRINSICS_DIR = "assets/realsense/cam_intr.json"
CAMERA_EXTRINSICS_DIR = "assets/realsense/cam_extr_init.txt"
OBJECT_URDF = {
    "gold": "outputs/gold-202504041202/object.urdf",
    "wood": "outputs/wood-20250409_223302_621/object.urdf",
    "box": "outputs/box-202504041230/object.urdf",
    "oreo": "outputs/box-202504192344_oreo/object.urdf",
    "ovaltine": "outputs/box-202504192344_ovaltine/object.urdf",
    "milk": "outputs/milk-box-20250827_180907_055/object.urdf",
    "old_milk": "outputs/milk-202504192349_milk/object.urdf",
    "old_oreo": "outputs/box-202504192344_oreo/object.urdf",
    "old_ovaltine": "outputs/box-202504192344_ovaltine/object.urdf",
    "old_gold": "outputs/gold-202504041202/object.urdf",
    "old_oreo_pinwm": "outputs/old_oreo_pinwm/object.urdf",
    "old_milk_pinwm": "outputs/old_milk_pinwm/object.urdf",
    "old_ovaltine_pinwm": "outputs/old_ovaltine_pinwm/object.urdf",
    "old_gold_pinwm": "outputs/old_gold_pinwm/object.urdf",
}

JOINT_NAMES = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'joint7',
    'finger_joint1',
    'finger_joint2',
]
DEFAULT_JOINT_ANGLES = [XML_DEFAULT_CFG[joint_name] for joint_name in JOINT_NAMES]
