"""
1. Load robot joint angles data from .pkl file
2. Generate link positions and orientations using forward kinematics
3. Save the generated link positions and orientations into .npz file

Usage:
    python src/process_data/fk_with_angles.py -r <robot_type> -mc -nc <num_cores>

Example:
    python src/process_data/fk_with_angles.py -r REACHY
    python src/process_data/fk_with_angles.py -r NAO -mc -nc 20
"""

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
from utils.types import RobotType, FKWithAnglesArgs
from utils.RobotConfig import RobotConfig
from utils.forward_kinematics import forward_kinematics


def main(args: FKWithAnglesArgs, core_idx: int):
    # load the robot configurations which is matched with the robot type
    robot_config = RobotConfig(args.robot_type)
    chain = kp.build_chain_from_urdf(open(robot_config.URDF_PATH).read())

    cf_xyzs_reps_path = robot_config.CF_XYZS_REPS_PATH
    cf_angles_files = sorted(
        [f for f in os.listdir(robot_config.CF_ANGLES_PATH) if f.endswith(".pkl")]
    )
    cf_angles_files = cf_angles_files[args.start_idx :]

    # for multiprocessing
    task_list = range(len(cf_angles_files))
    task_list = [idx for idx in task_list if idx % args.num_cores == core_idx]
    cf_angles_files = [cf_angles_files[idx] for idx in task_list]

    for cf_angles_file in tqdm(cf_angles_files):

        # Extract the data index from the filename
        # e.g. "cf_angles_0001.pkl" -> "0001"
        data_idx = cf_angles_file.split(".")[0].split("_")[-1]

        # Load the .pkl file
        cf_angles_path = osp.join(robot_config.CF_ANGLES_PATH, cf_angles_file)
        angles_list: List[dict] = pickle.load(open(cf_angles_path, "rb"))

        xyzs_list = []
        reps_list = []
        xyzs4smpl_list = []

        for angles in angles_list:
            # Get the robot's link positions and orientations using forward kinematics
            xyzs, reps, xyzs4smpl = forward_kinematics(robot_config, chain, angles)

            xyzs_list.append(xyzs)
            reps_list.append(reps)
            xyzs4smpl_list.append(xyzs4smpl)

        xyzs_list = np.array(xyzs_list)
        reps_list = np.array(reps_list)
        xyzs4smpl_list = np.array(xyzs4smpl_list)

        # Save the generated link positions and orientations into .npz file
        npz_file_path = osp.join(cf_xyzs_reps_path, f"cf_xyzs+reps_{data_idx}.npz")
        np.savez(
            npz_file_path, xyzs=xyzs_list, reps=reps_list, xyzs4smpl=xyzs4smpl_list
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for sampling reachy data")

    parser.add_argument(
        "--robot-type",
        "-r",
        type=RobotType,
        default=RobotType.REACHY,
        help=f"Select the robot type: {RobotType._member_names_}",
    )
    parser.add_argument(
        "--start-idx",
        "-s",
        type=int,
        default=0,
        help="start index for sampling",
    )
    parser.add_argument(
        "--multi-cpu",
        "-mc",
        action="store_true",
        help="use multiple cpus for sampling",
        default=False,
    )
    parser.add_argument(
        "--num-cores",
        "-nc",
        type=int,
        default=1,
        help="number of cores for multiprocessing",
    )

    args: FKWithAnglesArgs = parser.parse_args()

    if args.multi_cpu:
        pool = Pool(args.num_cores)
        args_with_idx_list = []
        for i in range(args.num_cores):
            args_with_idx_list.append((args, i))

        pool.starmap(main, args_with_idx_list)

    else:
        main(args, 0)
