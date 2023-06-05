import numpy as np
import kinpy as kp
from tqdm import tqdm
import os
import os.path as osp
import pickle
import argparse
import sys

sys.path.append('.')
from src.misc import joint_range
from src.transform import quat2rep

def main(args):
    os.makedirs(args.save_path, exist_ok=True)

    max_seed = args.max_seed
    num_per_seed = args.num_per_seed

    chain = kp.build_chain_from_urdf(open('./reachy.urdf').read())

    for seed in range(max_seed):
        np.random.seed(seed)
        all_xyzs = []
        all_reps = []
        all_angles = []
        all_xyzs4smpl = []

        for i in tqdm(range(num_per_seed)):
            th = {k: np.random.rand()*(v[1]-v[0]) + v[0] for k, v in joint_range.items()}
            ret = chain.forward_kinematics(th)

            xyzs = list()
            reps = list()

            for k, v in ret.items():
                print(k)
                curr_xyz = v.pos # should change to 1 2 0 
                curr_quat = v.rot
                curr_rep = quat2rep(curr_quat)

                xyzs.append(curr_xyz)
                reps.append(curr_rep)
            print()

            xyzs = np.vstack(xyzs)
            reps = np.asarray(reps)
            reachy_to_smpl_idx=[0, 2, 1, 5, 4, 9, 12, 17, 19, 21, 53, 40, 42, 16, 18, 20, 38, 25, 27, 24, 23]

            # I tried to map XYZ points of robot body to SMPL joints. The joint order needs to be aligned properly.
            # TODO: plot the robot XYZ positions to be saved, and find out whether the robot joints are well aligned with SMPL joints.
            xyzs4smpl = [np.array([0.0, 0.0, 0.65]), # pelvis
                        np.array([0.0, -0.1, 0.65]), # right hip
                        np.array([0.0, 0.1, 0.65]), # left hip
                        np.array([0.0, -0.1, 0.36]), #right knee
                        np.array([0.0, 0.1, 0.36]), # left knee
                        np.array([0.0, 0.0, 0.9]), # spine 3
                        np.array([0.0, 0.0, 1.05]), # neck
                        xyzs[2], xyzs[5], xyzs[7], xyzs[8], 
                        xyzs[9], xyzs[10], xyzs[11], xyzs[14],
                        xyzs[16], xyzs[17], xyzs[18], xyzs[19],
                        np.array([xyzs[27][0]-0.01, xyzs[27][1], xyzs[27][2]]), # righ camera
                        np.array([xyzs[26][0]-0.01, xyzs[26][1], xyzs[26][2]]), # left camera                                                    
            ]
            xyzs4smpl = np.asarray(xyzs4smpl)
            all_xyzs.append(xyzs)
            all_reps.append(reps)
            all_xyzs4smpl.append(xyzs4smpl)
            all_angles.append(th)
        all_xyzs = np.asarray(all_xyzs)
        all_reps = np.asarray(all_reps)
        all_xyzs4smpl = np.asarray(all_xyzs4smpl)
        np.savez(osp.join(args.save_path, 'xyzs+reps_{:03}.npz'.format(seed)),
                  xyzs=all_xyzs, reps=all_reps, xyzs4smpl=all_xyzs4smpl)
        pickle.dump(all_angles, open(osp.join(args.save_path, 'angles_{:03}.pkl'.format(seed)), 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for sampling reachy data')
    parser.add_argument('--save-path', type=str, default='./data/reachy/raw')
    parser.add_argument('--max-seed', type=int, default=500, help='maximum seeds for sampling')
    parser.add_argument('--num-per-seed', type=int, default=2000, help='number of samples for each seed')
    args = parser.parse_args()

    main(args)