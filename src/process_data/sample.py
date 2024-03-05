"""
Sample random Reachy poses for training and testing.
Data number: 1,000 (number of seeds) * 2,000 (poses per seed) = 2,000,000

1. Sample random values within the range of each joint (Reachy) -> 17 elements
2. Get xyz pos and rotation (Quaternion Representation) of 31 joints (Reachy) using forward kenematics
3. Quaternion -> 6D rotation
4. Create xyzs4smpl (SMPL) with 21 elements using Reachy joint xyz

=> Store xyzs, reps, xyzs4smpl into xyzs+reps.npy file,
   Store angle into angle.pkl file (Total number of files: 1,000 + 1,000).

Usage:
    python sample.py -r [robot_type] -s [num_seeds] -m [motions_per_seed] -mc -nc [num_cores]

Example:
    python sample.py -r NAO
    python sample.py -r NAO -mc -nc 20
"""

import numpy as np
import kinpy as kp
from tqdm import tqdm
import os
import os.path as osp
import pickle
import argparse
import sys
from multiprocessing import Pool

sys.path.append("./src")
from utils.consts import *
from utils.types import RobotType, SampleArgs
from utils.RobotConfig import RobotConfig
from utils.forward_kinematics import forward_kinematics


def main(args: SampleArgs, core_idx: int):
    # load the robot configurations which is matched with the robot type
    robot_config = RobotConfig(args.robot_type)

    # create a directory for saving the robot data
    os.makedirs(robot_config.ANGLES_PATH, exist_ok=True)
    os.makedirs(robot_config.XYZS_REPS_PATH, exist_ok=True)

    # fmt: off
    num_seeds = args.num_seeds                  # how many random seed to be used for sampling
    motions_per_seed = args.motions_per_seed    # how many poses to be sampled for each random seed.
    # fmt: on

    # build a kinematic chain from robot's urdf
    chain = kp.build_chain_from_urdf(open(robot_config.URDF_PATH).read())

    # only sample the data for the current core_idx.
    task_list = range(num_seeds)
    task_list = [seed for seed in task_list if seed % args.num_cores == core_idx]

    for seed in tqdm(task_list):
        # Ensuring that random values are uniform for the same seed,
        # but different for different seeds.
        np.random.seed(seed)

        xyzs_list = []
        reps_list = []
        angles_list = []
        xyzs4smpl_list = []

        for i in range(motions_per_seed):
            # fmt: off
            # angles: list of joints angle dicts (num_iter, joint_num) of {k: joint, v: angle}
            #         (roll, pitch, yaw of joints)
            # k: joint key, v[0]: MIN, v[1]: MAX,
            # v[1] - v[0]: Maximum range of each joint key.
            # np.random.rand() * (MAX - MIN) + MIN: Random value in range (MIN, MAX).
            angles = {
                k: ((np.random.rand() * (v[1] - v[0])) + v[0])
                for k, v in robot_config.joi_range.items()
            }
            # fmt: on

            xyzs, reps, xyzs4smpl = forward_kinematics(robot_config, chain, angles)

            # Append the iteration items into the total list
            angles_list.append(angles)
            xyzs_list.append(xyzs)
            reps_list.append(reps)
            xyzs4smpl_list.append(xyzs4smpl)

        # fmt: off
        xyzs_list = np.asarray(xyzs_list)            # shape: (num_iter, # of robot links, 3)
        reps_list = np.asarray(reps_list)            # shape: (num_iter, # of robot links, 6)
        xyzs4smpl_list = np.asarray(xyzs4smpl_list)  # shape: (num_iter, 21, 3)
        # fmt: on

        # save robot's xyz + rep data file
        # file name: DATA_PATH/xyzs+reps_0000.npz
        # data keys in a file: xyzs, reps, xyzs4smpl
        np.savez(
            osp.join(robot_config.XYZS_REPS_PATH, robot_xyzs_reps_path(seed)),
            xyzs=xyzs_list,
            reps=reps_list,
            xyzs4smpl=xyzs4smpl_list,
        )

        # save robot's angle data file
        # file name: DATA_PATH/angles_0000.pkl
        pickle.dump(
            angles_list,
            open(osp.join(robot_config.ANGLES_PATH, robot_angles_path(seed)), "wb"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for sampling reachy data")

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
        "--motions-per-seed",
        "-m",
        type=int,
        default=MOTION_PER_SEED,
        help="number of motion samples for each seed",
    )
    parser.add_argument(
        "--multi-cpu",
        "-mc",
        action="store_true",
        help="use multiple cpus for sampling",
    )
    parser.add_argument(
        "--num-cores",
        "-nc",
        type=int,
        default=1,
        help="number of cores for multiprocessing",
    )

    args: SampleArgs = parser.parse_args()

    if args.multi_cpu:
        pool = Pool(args.num_cores)
        args_with_idx_list = []
        for i in range(args.num_cores):
            args_with_idx_list.append((args, i))

        pool.starmap(main, args_with_idx_list)

    else:
        main(args, 0)
