"""
Convert robot joint angles data saved as .mat files into .pkl files.

Usage:
    python src/process_data/convert_mat2pkl.py -r <robot_type>

Example:
    python src/process_data/convert_mat2pkl.py -r REACHY
    python src/process_data/convert_mat2pkl.py -r COMAN
    python src/process_data/convert_mat2pkl.py -r NAO
"""

import argparse
import os
import scipy.io
import pickle
import sys
from tqdm import tqdm

sys.path.append("./src")
from utils.types import RobotType, ConvertMat2PklArgs
from utils.consts import *
from utils.RobotConfig import RobotConfig


def convert_mat2pkl(args: ConvertMat2PklArgs):
    robot_config = RobotConfig(args.robot_type)
    mats_folder = robot_config.CF_MATS_PATH
    robot_angles_folder = robot_config.CF_ANGLES_PATH

    mat_files = sorted([f for f in os.listdir(mats_folder) if f.endswith(".mat")])

    for mat_file in tqdm(mat_files):
        # Extract the data index from the filename
        # e.g. "reachy_angles_0001.mat" -> "0001"
        data_idx = mat_file.split(".")[0].split("_")[-1]

        # Load the .mat file
        mat_file_path = os.path.join(mats_folder, mat_file)
        mat_data = scipy.io.loadmat(mat_file_path)

        # Extract the joint names and the joint angles from the .mat file
        q_names = [arr[0] for arr in mat_data["jnames_ctrl"].flatten().tolist()]
        cf_data = mat_data["q_traj_cf"]

        # Convert the data into a list of dictionaries
        converted_data = []
        for i in range(len(cf_data)):
            converted_item = {}

            # The first three joint angles are ignored for COMAN
            if args.robot_type == RobotType.COMAN:
                for j in range(len(q_names))[3:]:
                    converted_item.update({q_names[j]: cf_data[i, j]})
            # Otherwise, all joint angles are included
            else:
                for j in range(len(q_names)):
                    converted_item.update({q_names[j]: cf_data[i, j]})

            converted_data.append(converted_item)

        # Save the converted data as a .pkl file
        pickle_file_path = os.path.join(
            robot_angles_folder, f"cf_angles_{data_idx}.pkl"
        )
        with open(pickle_file_path, "wb") as f:
            pickle.dump(converted_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert robot joint angles data saved as .mat files into .pkl files."
    )
    parser.add_argument(
        "-r",
        "--robot_type",
        type=RobotType,
        choices=list(RobotType),
        default=RobotType.REACHY,
        help="The type of robot whose data is to be converted.",
    )
    args: ConvertMat2PklArgs = parser.parse_args()

    convert_mat2pkl(args)
