"""
    run_inverse_kenematics_engine
xyz4smpl            =>           smpl_data
"""

import argparse
import numpy as np
import os.path as osp
import os

import sys
sys.path.append('.')

from utils.hbp import run_ik_engine, make_vids
import glob

def main(args):
    num_betas = 16
    batch_size = 100
    device = 'cuda'

    if osp.isdir(args.reachy_path):
        os.makedirs(args.res_path, exist_ok=True)                       # ?
        os.makedirs(args.vid_path, exist_ok=True)                       # visualize 시, 저장할 비디오 폴더

        files = sorted(glob.glob(osp.join(args.reachy_path, '*.npz')))  # reachy의 xyz + reps 데이터
    else:
        files = [args.reachy_path]

    for f in files:
        data_idx = f.split('/')[-1].split('_')[-1][:3]              # DATA_PATH/xyzs+reps_000.npz => 000

        if int(data_idx) > -1:
            reachy_data = np.load(f)

            motion = reachy_data['xyzs4smpl']
            motion = np.zeros_like(reachy_data['xyzs4smpl'])        # shape: (Num_iters, 21, 3)
            # (x, y, z) => (y, z, x)
            motion[:, :, 0] = reachy_data['xyzs4smpl'][:, :, 1]
            motion[:, :, 1] = reachy_data['xyzs4smpl'][:, :, 2]
            motion[:, :, 2] = reachy_data['xyzs4smpl'][:, :, 0]

            # running IK from the code makes huge memory usage. Doesn't it empty cache?
            # TODO: memory leak seems to happen in codes from VPoser. Any possible solution?
            smpl_data = run_ik_engine(osp.join(args.res_path, 'params_{}.npz'.format(data_idx)), 
                                    motion, batch_size, args.smpl_path, args.vposer_path, num_betas, device, args.verbosity)
            
            print('trans: {}'.format(smpl_data['trans'].shape))
            print('betas: {}'.format(smpl_data['betas'].shape))
            print('root_orient: {}'.format(smpl_data['root_orient'].shape))
            print('poZ_body: {}'.format(smpl_data['poZ_body'].shape))
            print('pose_body: {}'.format(smpl_data['pose_body'].shape))
            print('poses: {}'.format(smpl_data['poses'].shape))

            # print(smpl_data)
            # {
            #   trans: (2000, 3),
            #   betas: (16,),
            #   root_orient: (2000, 3),
            #   poZ_body: (2000, 32),
            #   pose_body: (2000, 63),
            #   poses: (2000, 165),
            #   surface_model_type: 'smplx',
            #   gender: 'neutral',
            #   mocap_frame_rate: 30,
            #   num_betas: 16
            # }
            
        if args.visualize:
            print('start visualizing...')
            make_vids(osp.join(args.vid_path, '{}.mp4'.format(data_idx)),
                      smpl_data, len(motion), args.smpl_path, num_betas, args.fps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for fitting reachy to smpl')
    parser.add_argument('--vposer-path', type=str, default='./data/vposer_v2_05')
    parser.add_argument('--smpl-path', type=str, default='./data/bodymodel/smplx/neutral.npz')
    parser.add_argument('--reachy-path', type=str, default='./data/reachy/raw')                 # sample.py에서 저장한 path
    parser.add_argument('--res-path', type=str, default='./data/human/')                        # result path: 생성한 SMPL을 저장할 path
    parser.add_argument('--vid-path', type=str, default='./vids/human/')
    parser.add_argument('--visualize', type=int, default=0)
    parser.add_argument('--verbosity', type=int, default=0)
    parser.add_argument('--fps', type=int, default=1)

    args = parser.parse_args()

    main(args)
