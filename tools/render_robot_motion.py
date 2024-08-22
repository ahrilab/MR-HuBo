"""
Render the motion of the robot with pybullet simulator and save it as a gif or mp4 file.

Usage:
    python tools/render_robot_motion.py -r ROBOT_TYPE -e EXTENTION --fps FPS [-mi MOTION_IDX] [-ef-off] [-s-off]    # for pred_motion
    python tools/render_robot_motion.py -r ROBOT_TYPE -gt -e EXTENTION --fps FPS [-mi MOTION_IDX] [-s-off]          # for gt_motion

Example:
    # render for prediction motion
    python tools/render_robot_motion.py -r COMAN -mi 13_08 -e mp4 --fps 120
    python tools/render_robot_motion.py -r COMAN -mi 13_18 -e mp4 --fps 120 -ef-off -s-off
    python tools/render_robot_motion.py -r COMAN -e mp4 --fps 120

    # render for GT motion
    python tools/render_robot_motion.py -r=COMAN -gt -mi="13_08" -e mp4 --fps 120 -s-off
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
    # if the motion index is not given, render all the motions
    if args.motion_idx is None:
        motion_idxs = GT_MOTION_IDXS
    else:
        # check the motion index is valid
        if args.motion_idx not in GT_MOTION_IDXS:
            raise ValueError(f"Invalid motion index: {args.motion_idx}")

        # if the motion index is given, render only the given motion
        motion_idxs = [args.motion_idx]

    for motion_idx in motion_idxs:
        render_motion(args, motion_idx)


def render_motion(args: PybulletRenderArgs, motion_idx: str):
    # load the robot configuration
    robot_config = RobotConfig(args.robot_type)
    robot_name = robot_config.robot_type.name

    # load motion data (joint) and set output path
    # if ground truth, load ground truth motion data
    if args.ground_truth:
        motions: List[Dict[str, float]] = pickle.load(open(GT_PATH, "rb"))
        robot_name_for_gt = args.robot_type.name[0] + args.robot_type.name[1:].lower()
        motions = motions[robot_name_for_gt][motion_idx]["q"]

    # if not ground truth, load predicted motion data
    else:
        # fmt: off
        motions_dir = PRED_MOTIONS_DIR(robot_name, args.one_stage, args.extreme_filter_off)
        motion_name = PRED_MOTION_NAME(robot_name, args.extreme_filter_off, motion_idx)
        motion_path = osp.join(motions_dir, motion_name)
        motions: List[Dict[str, float]] = pickle.load(open(motion_path, "rb"))
        # fmt: on

    # render the motion with pybullet
    # The first frame has an issue with the camera view, so we skip the first frame
    frames = pybullet_render(motions, robot_config, args.smooth_off)[1:]

    # Save the frames as a gif or mp4 file
    # fmt: off
    if args.ground_truth:
        output_dir = PYBULLET_GT_VID_DIR(robot_name)
        output_name = PYBULLET_GT_VID_NAME(robot_name, motion_idx, args.extention)
        output_path = osp.join(output_dir, output_name)
    else:
        output_dir = PYBULLET_PRED_VID_DIR(robot_name, args.one_stage, args.extreme_filter_off)
        output_name = PYBULLET_PRED_VID_NAME(robot_name, args.extreme_filter_off, motion_idx, args.extention)
        output_path = osp.join(output_dir, output_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    extension = output_path.split(".")[-1]
    if extension == "gif":
        imageio.mimwrite(output_path, frames, duration=1000 / args.fps)
    elif extension == "mp4":
        imageio.mimsave(output_path, frames, fps=args.fps)
    # fmt: on

    # save frames as images
    # frame_dir = f"{output_dir}/{args.motion_idx}"
    # os.makedirs(frame_dir, exist_ok=True)
    # for i, frame in enumerate(frames):
    #     imageio.imwrite(osp.join(frame_dir, f"frame_{i:04}.png"), frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for render with pybullet")

    parser.add_argument("--robot-type", "-r", type=RobotType, default=RobotType.COMAN)
    parser.add_argument("--ground-truth", "-gt", action="store_true")
    parser.add_argument("--motion-idx", "-mi", type=str)
    parser.add_argument("--extreme-filter-off", "-ef-off", action="store_true")
    parser.add_argument("--one-stage", "-os", action="store_true")
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--smooth-off", "-s-off", action="store_true")
    parser.add_argument("--extention", "-e", type=str, default="gif")

    args: PybulletRenderArgs = parser.parse_args()
    main(args)
