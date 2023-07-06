"""
Make video of reachy motions from pickle files of angles.

- pickle with angles -> render -> save -> makevid
"""

import argparse
import pickle
import kinpy as kp
import os
import os.path as osp
import glob
from moviepy.editor import *
import sys

sys.path.append("./src")
from utils.viz import draw_imgs
from utils.consts import *


def main(args):
    os.makedirs(args.tmp_path, exist_ok=True)

    chain = kp.build_chain_from_urdf(open(REACHY_URDF_PATH).read())

    if osp.isdir(args.file_path):
        os.makedirs(args.save_path, exist_ok=True)
        files = sorted(glob.glob(osp.join(args.file_path, '*.pkl')))
    else:
        files = [args.file_path]

    for f in files:
        if len(files) > 1:
            data_idx = f.split('/')[-1].split('_')[-1][:3]
            save_path = osp.join(args.save_path, '{}.mp4'.format(data_idx))
        else:
            save_path = args.save_path

        # TODO: Check how below "angles" look like.
        angles = pickle.load(open(f, 'rb'))
        # ./pymaf_robot_v2.pkl 의 경우 길이 785짜리의 list였고,
        # 각각의 element는 17개의 key-data로 이루어진 dictionary이며,
        # 각각은 reachy의 yaw, pitch, roll에 대한 정보가 있는 angle 데이터이다.

        draw_imgs(angles, chain, args.tmp_path, args.resolution, args.smooth)
        
        clip = ImageSequenceClip(args.tmp_path, fps=args.fps)
        clip.write_videofile(save_path, fps=args.fps)

        if args.delete:
            d_files = glob.glob(osp.join(args.tmp_path, '*.png'))
            for df in d_files:
                os.remove(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for making video of sampled angles')
    parser.add_argument('--fps', type=int, default=60, help='fps for rendering')
    parser.add_argument('--file-path', type=str, default='./pymaf_robot_v2.pkl')    # 이 친구는 갑자기 어디서 나왔을까요? -> test 결과물
    parser.add_argument('--save-path', type=str, default='./pymaf_robot_v2.mp4')
    parser.add_argument('--tmp-path', type=str, default='./raw_results')
    parser.add_argument('--resolution', type=int, default=1280, help='resolution for rendering')
    parser.add_argument('--delete', type=int, default=0)
    parser.add_argument('--smooth', type=int, default=1)

    args = parser.parse_args()

    main(args)