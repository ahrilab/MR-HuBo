# pickle with angles -> render -> save -> makevid
import argparse
import pickle
import kinpy as kp
import os
import os.path as osp
import glob
from moviepy.editor import *

from src.viz import draw_imgs

def main(args):
    os.makedirs(args.tmp_path, exist_ok=True)
    os.makedirs('./vids', exist_ok=True)
    save_path = osp.join('./vids', args.save_path)

    angles = pickle.load(open(args.file_path, 'rb'))
    
    chain = kp.build_chain_from_urdf(open('./reachy.urdf').read())

    draw_imgs(angles, chain, args.tmp_path, args.resolution)
    
    clip = ImageSequenceClip(args.tmp_path, fps=args.fps)
    clip.write_videofile(save_path, fps=args.fps)

    if args.delete:
        files = glob.glob(osp.join(args.tmp_path, '*.png'))
        for f in files:
            os.remove(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for making video of sampled angles')
    parser.add_argument('--fps', type=int, default=1, help='fps for rendering')
    parser.add_argument('--file-path', type=str, default='./data/reachy/raw/angles_000.pkl', help='tmp dir to save imgs')
    parser.add_argument('--save-path', type=str, default='res.mp4', help='tmp dir to save imgs')
    parser.add_argument('--tmp-path', type=str, default='./tmp_imgs', help='tmp dir to save imgs')
    parser.add_argument('--resolution', type=int, default=1280, help='resolution for rendering')
    parser.add_argument('--delete', type=int, default=1)

    args = parser.parse_args()

    main(args)