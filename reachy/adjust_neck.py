import numpy as np
import kinpy as kp
import torch
import os
import os.path as osp
import pickle
import argparse
import sys
import glob 

sys.path.append('.')
from src.misc import joint_range
from src.transform import quat2rep
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_euler_angles

def main(args):
    os.makedirs(args.save_path, exist_ok=True)

    chain = kp.build_chain_from_urdf(open('./reachy.urdf').read())

    files = sorted(glob.glob(osp.join(args.reachy_path, '*.pkl')))  # /data/reachy/raw -> all angles

    for f in files:         # Reachy Files
        all_xyzs = []
        all_reps = []
        all_angles = []
        all_xyzs4smpl = []

        raw_angles = pickle.load(open(f, 'rb'))     # list of joints angle dicts (num_iter, 17) of {k: joint, v: angle}
        data_idx = f.split('/')[-1].split('_')[-1][:3]
        smpl_data = np.load(osp.join(args.smpl_path, 'params_{}.npz'.format(data_idx)))     # fit2smpl에서 생성한 SMPL 데이터

        # 왜 360 이상인 데이터만 진행한 것인가요?
        if 360 <= int(data_idx):                    # data index는 0~499
            print(data_idx)
            for i in range(len(raw_angles)):        # 각각의 data 파일 안에 2000개의 pose 데이터가 있음.
                
                # th는 reachy의 angle 데이터
                # smpl_data['poses']는 reachy xyz -> smpl xyz 로 만든 데이터
                # reachy 데이터와 smpl 데이터는 각각 2000개 이며, 서로 pair임.

                # smpl_data['poses'] 에서 neck에 해당하는 (index: 12) axis angle 정보를 euler angle로 변환하고,
                # 이 값을 reachy의 neck angle로 설정.
                # 즉 reachy의 neck angle은 smpl이 가능한 (사람이 가능한) neck angle로 설정됨.
                
                th = raw_angles[i]

                # 원래 smpl_data['poses']의 shape: (2000, 165)
                neck_aa = smpl_data['poses'][i].reshape(-1, 3)[12]                                      # neck axis angle
                neck_euler = matrix_to_euler_angles(axis_angle_to_matrix(torch.Tensor(neck_aa)), 'ZXY') # axis angle -> matrix -> euler (오일러)
    
                th['neck_roll'] = neck_euler[0]
                th['neck_pitch'] = neck_euler[1]
                th['neck_yaw'] = neck_euler[2]

                ret = chain.forward_kinematics(th)

                xyzs = list()
                reps = list()

                for k, v in ret.items():
                    curr_xyz = v.pos # should change to 1 2 0 
                    curr_quat = v.rot
                    curr_rep = quat2rep(curr_quat)

                    xyzs.append(curr_xyz)
                    reps.append(curr_rep)

                xyzs = np.vstack(xyzs)
                reps = np.asarray(reps)

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
                all_reps.append(reps)
                all_xyzs4smpl.append(xyzs4smpl)
                all_angles.append(th)
            all_xyzs = np.asarray(all_xyzs)
            all_reps = np.asarray(all_reps)
            all_xyzs4smpl = np.asarray(all_xyzs4smpl)

            np.savez(osp.join(args.save_path, 'xyzs+reps_{}.npz'.format(data_idx)),
                    xyzs=all_xyzs, reps=all_reps, xyzs4smpl=all_xyzs4smpl)
            pickle.dump(all_angles, open(osp.join(args.save_path, 
                                                'angles_{}.pkl'.format(data_idx)), 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for adjust raw neck from smpl')
    parser.add_argument('--reachy-path', type=str, default='./data/reachy/raw')
    parser.add_argument('--smpl-path', type=str, default='./data/human')
    parser.add_argument('--save-path', type=str, default='./data/reachy/fix')
    args = parser.parse_args()

    main(args)