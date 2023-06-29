import argparse
import joblib
import numpy as np

import sys
sys.path.append('.')
from utils.hbp import transform_smpl_coordinate, make_vids

def main(args):
    num_betas = 16
    result = joblib.load(open(args.res_path, 'rb'))

    smpl_data = {}
    smpl_data['betas'] = np.zeros((num_betas, ))
    smpl_data['pose_body'] = result['pose'][:, 3:66]
    smpl_data['root_orient'] = np.zeros((len(result['pose']), 3))
    smpl_data['trans'] = np.zeros((len(result['pose']), 3))
    smpl_data['trans'][:, 1] += 0.5

    transformed_d = transform_smpl_coordinate(bm_fname=args.smpl_path, 
                                            trans=smpl_data['trans'], 
                                            root_orient=smpl_data['root_orient'],
                                            betas=smpl_data['betas'], 
                                            rotxyz=[90, 0, 0])
    smpl_data.update(transformed_d)

    smpl_data['surface_model_type'] = 'smplx'
    smpl_data['gender'] = 'neutral'
    smpl_data['mocap_frame_rate'] = 30
    smpl_data['num_betas'] = num_betas

    make_vids(args.vid_path, smpl_data, len(result['pose']), args.smpl_path, num_betas, args.fps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for fitting reachy to smpl')
    parser.add_argument('--smpl-path', type=str, default='./data/bodymodel/smplx/neutral.npz')
    parser.add_argument('--vid-path', type=str, default='./pymaf.mp4')
    parser.add_argument('--res-path', type=str, default='./pymaf.pkl')
    parser.add_argument('--fps', type=int, default=30)

    args = parser.parse_args()

    main(args)