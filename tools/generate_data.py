"""
Generate <Robot-Human> paired pose data

1. Generate robot pose data using random sampling of joint angles and forward kinematics.
2. Generate human pose data using VPoser IK solver.

Usage:
    python tools/generate_data.py -r [robot_type] -s [num_seeds] -p [poses_per_seed] -d [device] -i [restart_idx]

Example:
    python tools/generate_data.py -r REACHY -s 1000 -p 2000 -d cuda -i 0
    python tools/generate_data.py -r NAO -s 1000 -p 2000 -d cuda:1 -i 500

"""

import argparse
import sys
import pickle
import os
import os.path as osp
import numpy as np
from tqdm import tqdm

sys.path.append("./src")
from process_data.sample_robot_data import sample_robot_data
from process_data.fit2smpl import fit2smpl
from utils.types import RobotType, GenerateDataArgs
from utils.RobotConfig import RobotConfig
from utils.consts import *


def generate_data(args: GenerateDataArgs):
    # load the robot configurations
    robot_config = RobotConfig(args.robot_type)

    # make directories for saving data
    os.makedirs(robot_config.XYZS_REPS_PATH, exist_ok=True)
    os.makedirs(robot_config.ANGLES_PATH, exist_ok=True)
    os.makedirs(robot_config.SMPL_PARAMS_PATH, exist_ok=True)

    # sample robot data iteratively for number of seeds
    for seed in tqdm(range(args.num_seeds)):

        # skip if seed < restart_idx
        if seed < args.restart_idx:
            continue

        # sample robot data
        angles_list, xyzs_array, reps_array, xyzs4smpl_array = sample_robot_data(
            args.robot_type,
            args.poses_per_seed,
        )

        # fits the robot joints to SMPL parameters
        smpl_data = fit2smpl(robot_config, xyzs4smpl_array, args.device)

        # save robot's xyz + rep data file
        # file name: DATA_PATH/xyzs+reps_0000.npz
        np.savez(
            osp.join(robot_config.XYZS_REPS_PATH, robot_xyzs_reps_path(seed)),
            xyzs=xyzs_array,
            reps=reps_array,
            xyzs4smpl=xyzs4smpl_array,
        )

        # save robot's angle data file
        # file name: DATA_PATH/angles_0000.pkl
        pickle.dump(
            angles_list,
            open(osp.join(robot_config.ANGLES_PATH, robot_angles_path(seed)), "wb"),
        )

        # save SMPL parameters
        # file name: DATA_PATH/params_0000.npz
        np.savez(
            osp.join(robot_config.SMPL_PARAMS_PATH, smpl_params_path(seed)),
            **smpl_data,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="args for generating <Robot-Human> paired motion data"
    )

    parser.add_argument(
        "--robot-type",
        "-r",
        type=RobotType,
        required=True,
        help=f"Select the robot type: {RobotType._member_names_}",
    )
    parser.add_argument(
        "--num-seeds",
        "-s",
        type=int,
        default=NUM_SEEDS,
        help="number of seeds for sampling",
    )
    parser.add_argument(
        "--poses-per-seed",
        "-p",
        type=int,
        default=POSE_PER_SEED,
        help="number of poses samples for each seed",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=DEVICE,
    )
    parser.add_argument(
        "--restart-idx",
        "-i",
        type=int,
        default=0,
    )

    args: GenerateDataArgs = parser.parse_args()
    generate_data(args)
