'''
seed 개수 (500) * seed당 pose 개수 (2000) 만큼
Random한 pose를 생성 ?

1. 17개의 Joint (Reachy) 에서 가능한 Range 안에서 임의의 값을 뽑음 
2. forward kenematics를 이용해서 31개의 joint (Reachy) 의 xyz pos와 rotation (Quaternion Representation) 얻음
3. Quaternion -> 6D rotation
4. Reachy joint의 xyz를 이용해 21개의 element가 있는 xyzs4smpl (SMPL)를 생성

=> Seed 개수만큼 num_per_seed 개의 {xyzs, reps, xyzs4smpl} -> (.npy) 와 angle -> (.pkl) 를 생성
'''

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

    max_seed = args.max_seed            # how many random seed to be used for sampling
    num_per_seed = args.num_per_seed    # how many poses to be sampled for each random seed.

    chain = kp.build_chain_from_urdf(open(REACHY_URDF_PATH).read())
    # Reachy urdf: Definition of 31 joints, 31 links for reachy robot.

    for seed in range(max_seed):
        np.random.seed(seed)            # Ensuring that random values are uniform for the same seed, but different for different seeds.
        all_xyzs = []
        all_reps = []
        all_angles = []
        all_xyzs4smpl = []

        for i in tqdm(range(num_per_seed)):
            # k: joint key, v[0]: MIN, v[1]: MAX,
            # v[1] - v[0]: Maximum range of each joint key.
            # np.random.rand() * (MAX - MIN) + MIN: Random value in range (MIN, MAX).
            # 의문 [해결]: 동일한 random seed에서 np.random.rand()는 항상 같은 값을 가져야 하지 않나요?
            # 그렇다면 num_per_seed 동안 모든 iteration에서 동일한 결과가 나와야 할 것 같은데 왜 그렇지 않나요?
            # -> np.random.rand()는 같은 런타임 상에서는 실행 시마다 다른 값을 리턴함.
            # 하지만 다른 런타임일 때에는 항상 이전 런타임과 동일한 순서로 같은 값을 리턴함.
            th = {k: ((np.random.rand() * (v[1] - v[0])) + v[0]) for k, v in joint_range.items()}
            ret = chain.forward_kinematics(th)
            # What is 'th' and 'ret' stand for?
            # th: {roll, pitch, yaw of joints} -> len: 17
            # ret: len: 31 (reachy.urdf 에 joint와 link가 각각 31개 씩 있음)
            #      (추측) Reachy의 Joint에 대해 각각 pos, rot, rotation_vec를 가지고 있음.

            xyzs = list()
            reps = list()

            for k, v in ret.items():
                curr_xyz = v.pos                # should change to 1 2 0  => Why don't we save this way?
                                                # (Current: change like this way when we load data.)
                curr_quat = v.rot               # Quaternion Representation. shape: (4,)
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
            all_angles.append(th)                   # list of joints angle dicts (num_iter, 17) of {k: joint, v: angle}
        all_xyzs = np.asarray(all_xyzs)             # shape: (num_iter, 31, 3)
        all_reps = np.asarray(all_reps)             # shape: (num_iter, 31, 6)
        all_xyzs4smpl = np.asarray(all_xyzs4smpl)   # shape: (num_iter, 21, 3)
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