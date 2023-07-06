"""
Sample random Reachy poses for training and testing.
Data number: 500 (number of seeds) * 2000 (poses per seed) = 1,000,000

1. Sample random values within the range of each joint (Reachy) -> 17 elements
2. Get xyz pos and rotation (Quaternion Representation) of 31 joints (Reachy) using forward kenematics
3. Quaternion -> 6D rotation
4. Create xyzs4smpl (SMPL) with 21 elements using Reachy joint xyz

=> Store xyzs, reps, xyzs4smpl into xyzs+reps.npy file,
   Store angle into angle.pkl file (Total number of files: 500 + 500).
"""

import numpy as np
import kinpy as kp
from tqdm import tqdm
import os
import os.path as osp
import pickle
import argparse
import sys

sys.path.append("./src")
from utils.misc import joint_range
from utils.transform import quat2rep
from utils.consts import *


def main(args):
    os.makedirs(args.save_path, exist_ok=True)

    max_seed = args.max_seed  # how many random seed to be used for sampling
    num_per_seed = args.num_per_seed  # how many poses to be sampled for each random seed.

    chain = kp.build_chain_from_urdf(open(REACHY_URDF_PATH).read())
    # Reachy urdf: Definition of 31 joints, 31 links for reachy robot.

    for seed in range(max_seed):
        # Ensuring that random values are uniform for the same seed,
        # but different for different seeds.
        np.random.seed(seed)

        all_xyzs = []
        all_reps = []
        all_angles = []
        all_xyzs4smpl = []

        for i in tqdm(range(num_per_seed)):
            # fmt: off
            # theta: {roll, pitch, yaw of joints} -> len: 17
            # k: joint key, v[0]: MIN, v[1]: MAX,
            # v[1] - v[0]: Maximum range of each joint key.
            # np.random.rand() * (MAX - MIN) + MIN: Random value in range (MIN, MAX).
            theta = {
                k: ((np.random.rand() * (v[1] - v[0])) + v[0])
                for k, v in joint_range.items()
            }
            # fmt: on

            # fk_result: forward kinematics result of chain (Reachy) -> len: 31
            #            keys of each element: pos (xyz position), rot (rotation vector: quaternion representation)
            fk_result = chain.forward_kinematics(theta)

            xyzs = list()
            reps = list()

            for k, v in fk_result.items():
                curr_xyz = v.pos  # should change to 1 2 0  => Why don't we save this way? => It needs to be checked
                curr_quat = v.rot  # Quaternion Representation. shape: (4,)
                curr_rep = quat2rep(curr_quat)  # Transform Quaternion into 6D Rotation Representation

                xyzs.append(curr_xyz)
                reps.append(curr_rep)
                # xyzs & reps are lists of 31 elements, each element is a 3D or 6D numpy array.

            # Q: In this case, `vstack` and `asarray` works the same?
            xyzs = np.vstack(xyzs)  # xyzs.shape: (31, 3)
            reps = np.asarray(reps)  # reps.shape: (31, 6)
            #########################################################

            xyzs4smpl = np.asarray(get_xyzs4smpl(xyzs))  # shape: (21, 3)
            all_xyzs.append(xyzs)
            all_reps.append(reps)
            all_xyzs4smpl.append(xyzs4smpl)
            all_angles.append(theta)  # list of joints angle dicts (num_iter, 17) of {k: joint, v: angle}
        all_xyzs = np.asarray(all_xyzs)  # shape: (num_iter, 31, 3)
        all_reps = np.asarray(all_reps)  # shape: (num_iter, 31, 6)
        all_xyzs4smpl = np.asarray(all_xyzs4smpl)  # shape: (num_iter, 21, 3)
        np.savez(
            osp.join(args.save_path, reachy_xyzs_reps_path(seed)),
            xyzs=all_xyzs,
            reps=all_reps,
            xyzs4smpl=all_xyzs4smpl,
        )
        pickle.dump(
            all_angles,
            open(osp.join(args.save_path, reachy_angles_path(seed)), "wb"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for sampling reachy data")
    parser.add_argument("--save-path", type=str, default=REACHY_RAW_PATH)
    parser.add_argument("--max-seed", type=int, default=500, help="maximum seeds for sampling")
    parser.add_argument("--num-per-seed", type=int, default=2000, help="number of samples for each seed")
    args = parser.parse_args()

    main(args)
