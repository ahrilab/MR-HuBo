"""
This script is for fitting reachy's pose data to SMPL parameters
by running Inverse Kinematics engine of VPoser.

smpl_params = run_ik_engine(xyzs4smpl of reachy)
"""

import argparse
import numpy as np
import os.path as osp
import os
import glob
import sys

sys.path.append("./src")
from utils.hbp import run_ik_engine, make_vids
from utils.consts import *
from utils.types import RobotType, Fit2SMPLArgs
from utils.RobotConfig import RobotConfig


def main(args: Fit2SMPLArgs):
    robot_config = RobotConfig(args.robot_type)
    batch_size = 100
    device = "cuda"
    video_path = args.video_result_path
    motion_len = 0

    # create directory for results
    os.makedirs(robot_config.ROBOT_TO_SMPL_PATH, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)

    # data files of robot's xyzs + reps
    robot_xyzs_reps_files = sorted(
        glob.glob(osp.join(robot_config.RAW_DATA_PATH, "*.npz"))
    )

    for f in robot_xyzs_reps_files:
        # DATA_PATH/xyzs+reps_0000.npz => 0000
        # data keys in a file: 'xyzs', 'reps', 'xyzs4smpl'
        data_idx = f.split("/")[-1].split("_")[-1][:4]

        if int(data_idx) > -1:
            robot_xyzs_reps_data = np.load(f)

            # xyzs4smpl shape: (Num_iters, 21, 3)
            xyzs4smpl = np.zeros_like(robot_xyzs_reps_data["xyzs4smpl"])
            motion_len = len(xyzs4smpl)

            # Convert (x, y, z) => (y, z, x)
            xyzs4smpl[:, :, 0] = robot_xyzs_reps_data["xyzs4smpl"][:, :, 1]
            xyzs4smpl[:, :, 1] = robot_xyzs_reps_data["xyzs4smpl"][:, :, 2]
            xyzs4smpl[:, :, 2] = robot_xyzs_reps_data["xyzs4smpl"][:, :, 0]

            # running IK from the code makes huge memory usage. Doesn't it empty cache?
            # TODO: memory leak seems to happen in codes from VPoser. Any possible solution?
            smpl_data = run_ik_engine(
                res_path=osp.join(
                    robot_config.ROBOT_TO_SMPL_PATH, smpl_params_path(data_idx)
                ),
                motion=xyzs4smpl,
                batch_size=batch_size,
                smpl_path=SMPL_PATH,
                vposer_path=VPOSER_PATH,
                num_betas=NUM_BETAS,
                device=device,
                verbosity=args.verbosity,
                smpl_joint_idx=robot_config.smpl_joint_idx,
            )
            # smpl_data: {
            #   trans: (2000, 3),
            #   betas: (16,),
            #   root_orient: (2000, 3),
            #   poZ_body: (2000, 32),
            #   pose_body: (2000, 63),
            #   poses: (2000, 165),
            #   surface_model_type: 'smplx',
            #   gender: 'neutral',
            #   mocap_frame_rate: 30,
            #   num_betas: 16}

        if args.visualize:
            print("start visualizing...")
            make_vids(
                osp.join(
                    video_path,
                    robot2smpl_vid_path(
                        robot_config.robot_type.name, data_idx, args.video_extension
                    ),
                ),
                smpl_data,
                motion_len,
                SMPL_PATH,
                NUM_BETAS,
                args.fps,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for fitting reachy to smpl")
    # fmt: off
    parser.add_argument("--fps", type=int, default=1)  # frame per second
    parser.add_argument("--video-result-path", "-vp", type=str, default=VIDEO_PATH)
    parser.add_argument("--video-extension", "-e", type=str, default="mp4")
    parser.add_argument("--visualize", "-viz", action="store_true")
    parser.add_argument(
        "--verbosity", "-ver",
        type=int, choices=range(0, 3), default=0,
        help="verbosity: 0: silent, 1: text, 2: text/visual. running 2 over ssh would need extra work"
    )
    parser.add_argument(
        "--robot-type", "-r",
        type=RobotType,
        required=True,
        help=f"Select the robot type: {RobotType._member_names_}"
    )
    # fmt: on

    args: Fit2SMPLArgs = parser.parse_args()

    main(args)
