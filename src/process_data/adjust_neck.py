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


def main(args):
    os.makedirs(args.reachy_fix_path, exist_ok=True)

    chain = kp.build_chain_from_urdf(open(REACHY_URDF_PATH).read())

    # raw_angle_files: files of reachy raw angles
    raw_angle_files = sorted(glob.glob(osp.join(args.reachy_raw_path, "*.pkl")))

    for f in raw_angle_files:  # Reachy Files
        all_xyzs = []
        all_reps = []
        all_angles = []
        all_xyzs4smpl = []

        # raw_angles: list of joints angle dicts (num_iter, 17) of {k: joint, v: angle}
        # smpl_data: SMPL data generated from fit2smpl.py
        raw_angles = pickle.load(open(f, "rb"))
        data_idx = f.split("/")[-1].split("_")[-1][:3]
        smpl_data = np.load(osp.join(args.human_path, human_params_path(data_idx)))

        for i in range(len(raw_angles)):  # len(raw_angles) = 2000
            # 1. Get neck axis angle from smpl data and convert it to euler angle.
            # 2. Set the SMPL nect euler angle as reachy neck angle. (reachy's neck angle is adjusted to human possible angle.)
            # 3. Obtain forward kinematics result of reachy using fixed angle data.
            # 4. Save the fixed angle & forward kinematics result of reachy.

            # theta: angles of reachy raw data.
            # smpl_data['poses']: smpl xyz data (<- reachy xyz data). shape: (2000, 165)
            # In the same data index of file, reachy data and smpl data are paired.

            theta = raw_angles[i]

            neck_aa = smpl_data["poses"][i].reshape(-1, 3)[SMPL_NECK_IDX]  # neck axis angle. shape: (3,)
            neck_euler = matrix_to_euler_angles(
                axis_angle_to_matrix(torch.Tensor(neck_aa)), "ZXY"
            )  # axis angle -> matrix -> euler angle (roll, pitch, yaw)

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

            xyzs4smpl = np.asarray(get_xyzs4smpl(xyzs))
            all_xyzs.append(xyzs)
            all_reps.append(reps)
            all_xyzs4smpl.append(xyzs4smpl)
            all_angles.append(theta)
        all_xyzs = np.asarray(all_xyzs)
        all_reps = np.asarray(all_reps)
        all_xyzs4smpl = np.asarray(all_xyzs4smpl)

        np.savez(
            osp.join(args.reachy_fix_path, reachy_xyzs_reps_path(data_idx)),
            xyzs=all_xyzs,
            reps=all_reps,
            xyzs4smpl=all_xyzs4smpl,
        )
        pickle.dump(all_angles, open(osp.join(args.reachy_fix_path, reachy_angles_path(data_idx)), "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for adjust raw neck from smpl")
    parser.add_argument("--reachy-raw-path", type=str, default=REACHY_RAW_PATH)
    parser.add_argument("--human-path", type=str, default=HUMAN_PARAM_PATH)
    parser.add_argument("--reachy-fix-path", type=str, default=REACHY_FIX_PATH)
    args = parser.parse_args()

    main(args)
