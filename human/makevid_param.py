import argparse
import joblib
import numpy as np
import os.path as osp
import glob

import sys
sys.path.append('.')
from src.hbp import make_vids

def main(args):
    num_betas = 16

    files = sorted(glob.glob(osp.join(args.res_path, '*.npz')))

    for f in files:
        data_idx = f.split('/')[-1].split('_')[-1][:3]
        smpl_data = np.load(f)

        print('Visualizing {}....'.format(f))
        make_vids(osp.join(args.vid_path, '{}.mp4'.format(data_idx)), 
                  smpl_data, len(smpl_data['trans']), args.smpl_path, num_betas, args.fps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for fitting reachy to smpl')
    parser.add_argument('--smpl-path', type=str, default='./data/bodymodel/smplx/neutral.npz')
    parser.add_argument('--vid-path', type=str, default='./vids/human')
    parser.add_argument('--res-path', type=str, default='./data/human')
    parser.add_argument('--fps', type=int, default=1)

    args = parser.parse_args()

    main(args)