import argparse
import numpy as np
import os.path as osp
import os

import sys
sys.path.append('.')

from src.hbp import run_ik_engine, make_vids
import glob

def main(args):
    num_betas = 16
    batch_size = 64
    device = 'cuda'

    if osp.isdir(args.reachy_path):
        os.makedirs(args.res_path, exist_ok=True)       
        os.makedirs(args.vid_path, exist_ok=True)

        files = sorted(glob.glob(osp.join(args.reachy_path, '*.npz')))
    else:
        files = [args.reachy_path]    

    for f in files:
        data_idx = f.split('/')[-1].split('_')[-1][:3]
        reachy_data = np.load(f)

        motion = reachy_data['xyzs4smpl']
        motion = np.zeros_like(reachy_data['xyzs4smpl'])
        motion[:, :, 0] = reachy_data['xyzs4smpl'][:, :, 1]
        motion[:, :, 1] = reachy_data['xyzs4smpl'][:, :, 2]
        motion[:, :, 2] = reachy_data['xyzs4smpl'][:, :, 0]

        smpl_data = run_ik_engine(osp.join(args.res_path, 'params_{}.npz'.format(data_idx)), 
                                  motion, batch_size, args.smpl_path, args.vposer_path, num_betas, device, args.verbosity)
            
        if args.visualize:
            print('start visualizing...')
            make_vids(osp.join(args.vid_path, '{}.mp4'.format(data_idx)),
                      smpl_data, len(motion), args.smpl_path, num_betas, args.fps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for fitting reachy to smpl')
    parser.add_argument('--vposer-path', type=str, default='./data/vposer_v2_05')
    parser.add_argument('--smpl-path', type=str, default='./data/bodymodel/smplx/neutral.npz')
    parser.add_argument('--reachy-path', type=str, default='./data/reachy/raw')
    parser.add_argument('--res-path', type=str, default='./data/human/raw')
    parser.add_argument('--vid-path', type=str, default='./vids/human/')
    parser.add_argument('--visualize', type=int, default=1)
    parser.add_argument('--verbosity', type=int, default=1)
    parser.add_argument('--fps', type=int, default=1)

    args = parser.parse_args()

    main(args)