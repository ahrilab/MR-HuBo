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
        all_rots = []
        all_angles = []
        all_xyzs4smpl = []

        for i in tqdm(range(num_per_seed)):
            th = {k: np.random.rand()*(v[1]-v[0]) + v[0] for k, v in joint_range.items()}
            ret = chain.forward_kinematics(th)

            xyzs = list()
            rots = list()

            for k, v in ret.items():
                curr_xyz = v.pos # should change to 1 2 0 
                curr_quat = v.rot
                curr_rep = quat2rep(curr_quat)

                xyzs.append(curr_xyz)
                rots.append(curr_rep)

            xyzs = np.vstack(xyzs)
            rots = np.asarray(rots)

            xyzs4smpl = [np.array([0.0, 0.0, 0.65]),
                        np.array([0.0, -0.1, 0.65]),
                        np.array([0.0, 0.1, 0.65]),
                        np.array([0.0, -0.1, 0.36]),
                        np.array([0.0, 0.1, 0.36]),
                        np.array([0.0, 0.0, 0.9]),
                        np.array([0.0, 0.0, 1.05]),
                        xyzs[2], xyzs[5], xyzs[7], xyzs[8], 
                        xyzs[9], xyzs[10], xyzs[11], xyzs[14],
                        xyzs[16], xyzs[17], xyzs[18], xyzs[19],
                        np.array([xyzs[27][0]-0.01, xyzs[27][1], xyzs[27][2]]),
                        np.array([xyzs[26][0]-0.01, xyzs[26][1], xyzs[26][2]]),                                                    
            ]
            xyzs4smpl = np.asarray(xyzs4smpl)
            all_xyzs.append(xyzs)
            all_rots.append(rots)
            all_xyzs4smpl.append(xyzs4smpl)
            all_angles.append(th)
        all_xyzs = np.asarray(all_xyzs)
        all_rots = np.asarray(all_rots)
        all_xyzs4smpl = np.asarray(all_xyzs4smpl)
        np.savez(osp.join(args.save_path, 'xyzs+rots_{:03}.npz'.format(seed)),
                  xyzs=all_xyzs, rots=all_rots, xyzs4smpl=all_xyzs4smpl)
        pickle.dump(all_angles, open(osp.join(args.save_path, 'angles_{:03}.pkl'.format(seed)), 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for sampling reachy data')
    parser.add_argument('--save-path', type=str, default='./data/reachy/raw')
    parser.add_argument('--max-seed', type=int, default=100, help='maximum seeds for sampling')
    parser.add_argument('--num-per-seed', type=int, default=1000, help='number of samples for each seed')
    args = parser.parse_args()

    main(args)