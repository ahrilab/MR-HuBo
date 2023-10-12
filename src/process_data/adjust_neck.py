"""
Adjust reachy raw neck angle to human possible neck angle.

1. Get neck axis angle from smpl data and convert it to euler angle.
2. Set the SMPL nect euler angle as reachy neck angle. (reachy's neck angle is adjusted to human possible angle.)
3. Obtain forward kinematics result of reachy using fixed angle data.
4. Save the fixed angle & forward kinematics result of reachy.
"""

import numpy as np
import kinpy as kp
import torch
import os
import os.path as osp
import pickle
import argparse
import sys
import glob
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_euler_angles

sys.path.append("./src")
from utils.transform import quat2rep
from utils.consts import *
from utils.types import RobotType, AdjustNeckArgs
from utils.RobotConfig import RobotConfig


def main(args: AdjustNeckArgs):
    robot_config = RobotConfig(args.robot_type)

    # create directory for results
    os.makedirs(robot_config.FIX_DATA_PATH, exist_ok=True)

    # kinematic chain from robot's urdf
    chain = kp.build_chain_from_urdf(open(robot_config.URDF_PATH).read())

    # raw_angle_files: files of reachy raw angles
    raw_angle_files = sorted(glob.glob(osp.join(robot_config.RAW_DATA_PATH, "*.pkl")))

    for f in raw_angle_files:  # Reachy Files
        total_xyzs = []
        total_reps = []
        total_angles = []
        total_xyzs4smpl = []

        # raw_angles: list of joints angle dicts (num_iter, num_joints) of {k: joint, v: angle}
        # smpl_data: SMPL data generated from fit2smpl.py
        raw_angles = pickle.load(open(f, "rb"))
        data_idx = f.split("/")[-1].split("_")[-1][:4]
        smpl_data = np.load(osp.join(HUMAN_PARAM_PATH, human_params_path(data_idx)))

        for i in range(len(raw_angles)):  # len(raw_angles) = 2000
            # 1. Get neck axis angle from smpl data and convert it to euler angle.
            # 2. Set the SMPL nect euler angle as reachy neck angle. (reachy's neck angle is adjusted to human possible angle.)
            # 3. Obtain forward kinematics result of reachy using fixed angle data.
            # 4. Save the fixed angle & forward kinematics result of reachy.

            # theta: angles of reachy raw data.
            # smpl_data['poses']: smpl xyz data (<- reachy xyz data). shape: (2000, 165)
            # In the same data index of file, reachy data and smpl data are paired.

            theta = raw_angles[i]

            # neck axis angle. shape: (3,)
            neck_aa = smpl_data["poses"][i].reshape(-1, 3)[SMPL_NECK_IDX]

            # axis angle -> matrix -> euler angle (roll, pitch, yaw)
            neck_euler = matrix_to_euler_angles(
                axis_angle_to_matrix(torch.Tensor(neck_aa)), "ZXY"
            )

            theta["neck_roll"] = neck_euler[0]
            theta["neck_pitch"] = neck_euler[1]
            theta["neck_yaw"] = neck_euler[2]

            fixed_fk_result = chain.forward_kinematics(theta)

            xyzs = list()
            reps = list()

            for k, v in fixed_fk_result.items():
                curr_xyz = v.pos  # should change to 1 2 0
                curr_quat = v.rot
                curr_rep = quat2rep(curr_quat)

                xyzs.append(curr_xyz)
                reps.append(curr_rep)

            xyzs = np.vstack(xyzs)
            reps = np.asarray(reps)

            xyzs4smpl = np.asarray(robot_config.convert_xyzs(xyzs))
            total_xyzs.append(xyzs)
            total_reps.append(reps)
            total_xyzs4smpl.append(xyzs4smpl)
            total_angles.append(theta)
        total_xyzs = np.asarray(total_xyzs)
        total_reps = np.asarray(total_reps)
        total_xyzs4smpl = np.asarray(total_xyzs4smpl)

        np.savez(
            osp.join(robot_config.FIX_DATA_PATH, robot_xyzs_reps_path(data_idx)),
            xyzs=total_xyzs,
            reps=total_reps,
            xyzs4smpl=total_xyzs4smpl,
        )
        pickle.dump(
            total_angles,
            open(
                osp.join(robot_config.FIX_DATA_PATH, robot_angles_path(data_idx)), "wb"
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for adjust raw neck from smpl")
    parser.add_argument(
        "--robot-type",
        "-r",
        type=RobotType,
        help=f"Select the robot type: {RobotType._member_names_}",
    )
    args: AdjustNeckArgs = parser.parse_args()

    main(args)
