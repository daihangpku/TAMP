import numpy as np

def rotate_axis_quaternion(ori_axis):
    if ori_axis == 'x':
        return (np.sqrt(2) / 2, 0, -np.sqrt(2) / 2, 0)
    elif ori_axis == 'y':
        return (np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0)
    elif ori_axis == 'z':
        return (1, 0, 0, 0)
    elif ori_axis == "-x":
        return (np.sqrt(2) / 2, 0, np.sqrt(2) / 2, 0)
    elif ori_axis == "-y":
        return (np.sqrt(2) / 2, -np.sqrt(2) / 2, 0, 0)
    elif ori_axis == "-z":
        return (0, 1, 0, 0)
    else:
        return (1, 0, 0, 0)