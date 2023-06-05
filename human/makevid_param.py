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

    if args.res_path[-3:] == 'npz':
        files = [args.res_path]
    else:
        files = sorted(glob.glob(osp.join(args.res_path, '*.npz')))

    for f in files:
        if len(files) > 1:
            data_idx = f.split('/')[-1].split('_')[-1][:3]
        else:
            data_idx = '000'
        smpl_data = np.load(f)
        smpl_data = {'pose_body': smpl_data['pose_body']}
        smpl_data['betas'] = np.zeros(num_betas)

        print('Visualizing {}....'.format(f))
        make_vids(osp.join(args.vid_path, '{}.mp4'.format(data_idx)), 
                  smpl_data, len(smpl_data['pose_body']), args.smpl_path, num_betas, args.fps, False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for fitting reachy to smpl')
    parser.add_argument('--smpl-path', type=str, default='./data/bodymodel/smplx/neutral.npz')
    parser.add_argument('--vid-path', type=str, default='./vids/human')
    parser.add_argument('--res-path', type=str, default='./pymaf_smpl_recon.npz')
    parser.add_argument('--fps', type=int, default=30)

    args = parser.parse_args()

    main(args)