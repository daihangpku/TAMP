#!/usr/bin/env python3
import sys
import termios
import tty
import signal
import os
import select

import rospy
from std_msgs.msg import Float64MultiArray, Int32
from sensor_msgs.msg import Image

sys.path.insert(0, os.getcwd())

# 缓存来自 sim 的状态
latest_rgb = None
latest_depth = None
latest_robot_state = None  # Float64MultiArray, len=15: [q7, pos3, quat4, grip1]


def rgb_cb(msg: Image):
    global latest_rgb
    latest_rgb = msg


def depth_cb(msg: Image):
    global latest_depth
    latest_depth = msg


def robot_state_cb(msg: Float64MultiArray):
    global latest_robot_state
    latest_robot_state = msg


def get_key(timeout=0.01):
    """
    非阻塞读取一个按键:
      - timeout 秒内无按键返回 None
      - Ctrl-C 对应 '\x03'
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([fd], [], [], timeout)
        if rlist:
            ch = sys.stdin.read(1)
        else:
            ch = None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def main():
    rospy.init_node('teleop_keyboard_node', anonymous=True)

    pub_cmd = rospy.Publisher('/teleop/cmd_state', Float64MultiArray, queue_size=10)
    pub_rec = rospy.Publisher('/teleop/record_cmd', Int32, queue_size=10)

    # 订阅 sim 的输出
    rospy.Subscriber('/sim/rgb', Image, rgb_cb, queue_size=1)
    rospy.Subscriber('/sim/depth', Image, depth_cb, queue_size=1)
    rospy.Subscriber('/sim/robot_state', Float64MultiArray, robot_state_cb, queue_size=10)

    print('Keyboard node started. Press Ctrl-C to exit.')
    print('Subscribed to /sim/rgb, /sim/depth, /sim/robot_state')
    print('Publishing ee+gripper cmd to /teleop/cmd_state and record cmds to /teleop/record_cmd')
    print('Key mapping (与 teleop 一致):')
    print('  W/S: +/- X, A/D: +/- Y, R/F: +/- Z')
    print('  Z: close gripper, X: open gripper')
    print('  1: reset & recollect, 2: start record, 3: end record, 4: save & reset')

    running = True

    def handle_sigint(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    rate = rospy.Rate(100)
    import numpy as np

    # 累积 Δpos（类似 keyboard_controller.dpos）
    dpos = np.zeros(3, dtype=np.float32)
    pos_step = 0.01
    gripper_open = True  # True=open, False=close

    while not rospy.is_shutdown() and running:
        try:
            k = get_key(timeout=0.01)

            if k is None:
                rate.sleep()
                continue

            # Ctrl-C 在 raw 模式下是 '\x03'
            if k == '\x03':
                running = False
                break

            c = k.lower()

            # 录制控制：1/2/3/4
            if c in ['1', '2', '3', '4']:
                pub_rec.publish(Int32(data=int(c)))
                rate.sleep()
                continue

            moved = False
            if c == 'w':
                dpos[0] += pos_step; moved = True
            elif c == 's':
                dpos[0] -= pos_step; moved = True
            elif c == 'a':
                dpos[1] += pos_step; moved = True
            elif c == 'd':
                dpos[1] -= pos_step; moved = True
            elif c == 'r':
                dpos[2] += pos_step; moved = True
            elif c == 'f':
                dpos[2] -= pos_step; moved = True
            elif c == 'z':
                gripper_open = False; moved = True
            elif c == 'x':
                gripper_open = True; moved = True
            else:
                rate.sleep()
                continue

            if not moved or latest_robot_state is None:
                rate.sleep()
                continue

            state_arr = np.array(latest_robot_state.data, dtype=float)
            if state_arr.size < 15:
                rate.sleep()
                continue

            # 当前 EE 位姿（sim 里用 franka_solver.compute_fk 生成）
            curr_ee_pos  = state_arr[7:10].copy()
            curr_ee_quat = state_arr[10:14].copy()

            target_pos  = curr_ee_pos + dpos
            target_quat = curr_ee_quat
            target_grip = 1.0 if gripper_open else 0.0

            cmd_arr = np.concatenate([target_pos, target_quat, [target_grip]]).astype(float).tolist()
            pub_cmd.publish(Float64MultiArray(data=cmd_arr))

        except Exception:
            # 避免小错误打断循环
            pass

        rate.sleep()

    print("Keyboard node exiting...")
    rospy.signal_shutdown("keyboard node stopped")


if __name__ == '__main__':
    main()
