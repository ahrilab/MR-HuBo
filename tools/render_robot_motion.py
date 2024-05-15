"""
Render the motion of the robot with pybullet simulator and save it as a gif or mp4 file.

Usage:
    python tools/render_robot_motion.py -r ROBOT_TYPE -mi MOTION_IDX [-ef] -e EXTENTION --fps FPS [-s]  # for pred_motion
    python tools/render_robot_motion.py -r ROBOT_TYPE -gt -mi MOTION_IDX -e EXTENTION --fps FPS [-s]    # for gt_motion

Example:
    # render for prediction motion
    python tools/render_robot_motion.py -r COMAN -mi 13_08 -ef -e mp4 --fps 120 -s
    python tools/render_robot_motion.py -r COMAN -mi 13_18 -e mp4 --fps 120 -s

    # render for GT motion
    python tools/render_robot_motion.py -r=COMAN -gt -mi="13_08" -e mp4 --fps 120 -s
    python tools/render_robot_motion.py -r=COMAN -gt -mi="13_18" -e mp4 --fps 120
"""

import argparse
import sys
import pickle
import imageio
import os
import os.path as osp
from typing import Dict, List

sys.path.append("./src")
from visualize.pybullet_render import pybullet_render
from utils.types import RobotType, PybulletRenderArgs
from utils.RobotConfig import RobotConfig
from utils.consts import *


def main(args: PybulletRenderArgs):
    # robot의 정보를 가져옴
    robot_config = RobotConfig(args.robot_type)
    robot_name = robot_config.robot_type.name

    # load motion data (joint) and set output path
    # if ground truth, load ground truth motion data
    if args.ground_truth:
        motions: List[Dict[str, float]] = pickle.load(open(GT_PATH, "rb"))
        motion_idx = args.motion_idx
        robot_name_for_gt = args.robot_type.name[0] + args.robot_type.name[1:].lower()
        motions = motions[robot_name_for_gt][motion_idx]["q"]

    # if not ground truth, load predicted motion data
    else:
        # fmt: off
        motions_dir = PRED_MOTIONS_DIR(robot_name, args.extreme_filter)
        motion_name = PRED_MOTION_NAME(robot_name, args.extreme_filter, args.motion_idx)
        motion_path = osp.join(motions_dir, motion_name)
        motions: List[Dict[str, float]] = pickle.load(open(motion_path, "rb"))
        # fmt: on

    frames = pybullet_render(motions, robot_config, args.smooth)

    # Save the frames as a gif or mp4 file
    # fmt: off
    if args.ground_truth:
        output_dir = PYBULLET_GT_VID_DIR(robot_name)
        output_name = PYBULLET_GT_VID_NAME(robot_name, motion_idx, args.extention)
        output_path = osp.join(output_dir, output_name)
    else:
        output_dir = PYBULLET_PRED_VID_DIR(robot_name, args.extreme_filter)
        output_name = PYBULLET_PRED_VID_NAME(robot_name, args.extreme_filter, args.motion_idx, args.extention)
        output_path = osp.join(output_dir, output_name)
    # fmt: on

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    extension = output_path.split(".")[-1]
    if extension == "gif":
        imageio.mimwrite(output_path, frames, duration=1000 / args.fps)
    elif extension == "mp4":
        imageio.mimsave(output_path, frames, fps=args.fps)

    # save frames as images
    frame_dir = f"{output_dir}/{args.motion_idx}"
    os.makedirs(frame_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        imageio.imwrite(osp.join(frame_dir, f"frame_{i:04}.png"), frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for render with pybullet")

    parser.add_argument("--robot-type", "-r", type=RobotType, default=RobotType.COMAN)
    parser.add_argument("--ground-truth", "-gt", action="store_true")
    parser.add_argument("--motion-idx", "-mi", type=str, default="02_05")
    parser.add_argument("--extreme-filter", "-ef", action="store_true")
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--smooth", "-s", action="store_true")
    parser.add_argument("--extention", "-e", type=str, default="gif")

    args: PybulletRenderArgs = parser.parse_args()
    main(args)
