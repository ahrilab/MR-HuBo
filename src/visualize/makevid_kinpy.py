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
from typing import List

sys.path.append("./src")
from utils.viz import draw_imgs
from utils.consts import *
from utils.RobotConfig import RobotConfig
from utils.types import RobotType, MakeRobotVideoArgs


def main(args: MakeRobotVideoArgs):
    robot_config = RobotConfig(args.robot_type)

    os.makedirs(args.tmp_path, exist_ok=True)

    chain = kp.build_chain_from_urdf(open(robot_config.URDF_PATH).read())

    if osp.isdir(args.motion_path):
        os.makedirs(args.result_path, exist_ok=True)
        files = sorted(glob.glob(osp.join(args.motion_path, "*.pkl")))
    else:
        files = [args.motion_path]

    for f in files:
        if len(files) > 1:
            data_idx = f.split("/")[-1].split("_")[-1][:4]
            save_path = osp.join(args.result_path, "{}.mp4".format(data_idx))
        else:
            save_path = args.result_path

        # TODO: Check how below "angles" look like.
        angles: List[dict[str, float]] = pickle.load(open(f, "rb"))

        draw_imgs(
            angles,
            chain,
            args.tmp_path,
            args.resolution,
            args.smooth,
            robot_config.joi_keys,
        )

        clip = ImageSequenceClip(args.tmp_path, fps=args.fps)
        clip.write_videofile(save_path, fps=args.fps)

        if args.delete:
            d_files = glob.glob(osp.join(args.tmp_path, "*.png"))
            for df in d_files:
                os.remove(df)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="args for making video of sampled angles")
    parser.add_argument("--fps", type=int, default=20, help="fps for rendering")
    parser.add_argument("--motion-path", "-mp", type=str, default="./out/pymaf_robot_v2.pkl")
    parser.add_argument("--result-path", "-rp", type=str, default="./out/pymaf_robot_v2.mp4")
    parser.add_argument("--tmp-path", "-tp", type=str, default=TMP_FRAME_PATH)
    parser.add_argument("--resolution", "-res", type=int, default=1280, help="resolution for rendering")
    parser.add_argument("--delete", "-d", action="store_true", help="delete tmp files")
    parser.add_argument("--smooth", "-s", action="store_true", help="smooth motion")
    parser.add_argument(
        "--robot-type", "-r",
        type=RobotType,
        help=f"Select the robot type: {RobotType._member_names_}",
    )
    # fmt: on

    args: MakeRobotVideoArgs = parser.parse_args()

    main(args)
