# pickle with angles -> render -> save -> makevid
import argparse
import pickle
import kinpy as kp
import os
import os.path as osp
import glob
from moviepy.editor import *

sys.path.append('.')
from src.viz import draw_imgs

def main(args):
    os.makedirs(args.tmp_path, exist_ok=True)

    chain = kp.build_chain_from_urdf(open('./reachy.urdf').read())

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

        angles = pickle.load(open(f, 'rb'))        

        draw_imgs(angles, chain, args.tmp_path, args.resolution)
        
        clip = ImageSequenceClip(args.tmp_path, fps=args.fps)
        clip.write_videofile(save_path, fps=args.fps)

        if args.delete:
            d_files = glob.glob(osp.join(args.tmp_path, '*.png'))
            for df in d_files:
                os.remove(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for making video of sampled angles')
    parser.add_argument('--fps', type=int, default=30, help='fps for rendering')
    parser.add_argument('--file-path', type=str, default='./pymaf_robot.pkl')
    parser.add_argument('--save-path', type=str, default='./pymaf_robot.mp4')
    parser.add_argument('--tmp-path', type=str, default='./tmp_imgs')
    parser.add_argument('--resolution', type=int, default=1280, help='resolution for rendering')
    parser.add_argument('--delete', type=int, default=0)

    args = parser.parse_args()

    main(args)