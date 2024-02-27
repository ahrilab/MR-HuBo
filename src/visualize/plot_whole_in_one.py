"""
Usage:
    python src/visualize/plot_whole_in_one.py -r <robot_type> -i <data_idx (seed num)> -p <number of poses> -cf -e <out_extention>

Example:
    # Plot 0-20 pose of REACHY's seed number 0 data
    python src/visualize/plot_whole_in_one.py -r REACHY -i 0 -p 20 -e gif --fps 0.5

    # Plot 0-50 pose of NAO's seed number 3 data
    python src/visualize/plot_whole_in_one.py -r NAO -i 3 -p 50 -e mp4 --fps 1

    # Plot 0-20 pose of COMAN's seed number 0 data with collision free
    python src/visualize/plot_whole_in_one.py -r COMAN -i 0 -p 20 -cf -e gif --fps 0.5
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import os.path as osp
from matplotlib.animation import FuncAnimation
from human_body_prior.body_model.body_model import BodyModel
from body_visualizer.tools.vis_tools import render_smpl_params

sys.path.append("./src")
from utils.types import PlotWholeInOneArgs, RobotType
from utils.RobotConfig import RobotConfig
from utils.consts import *


def main(args: PlotWholeInOneArgs):
    robot_config = RobotConfig(args.robot_type)
    out_extention = args.out_extention

    # load data
    # fmt: off
    if args.collision_free:
        SMPL_PARAMS_PATH = osp.join(robot_config.CF_SMPL_PARAMS_PATH, f"params_{args.data_idx:04}.npz")
        ROBOT_XYZS_REPS_PATH = osp.join(robot_config.CF_XYZS_REPS_PATH, f"cf_xyzs+reps_{args.data_idx:04}.npz")
    else:
        SMPL_PARAMS_PATH = osp.join(robot_config.SMPL_PARAMS_PATH, f"params_{args.data_idx:04}.npz")
        ROBOT_XYZS_REPS_PATH = osp.join(robot_config.XYZS_REPS_PATH, f"xyzs+reps_{args.data_idx:04}.npz")

    # fmt: on

    # variables for smpl render
    smpl_params = np.load(SMPL_PARAMS_PATH)
    body_model = BodyModel(bm_fname=SMPL_PATH, num_betas=NUM_BETAS)
    smpl_dict: dict = np.load(SMPL_PATH)  # smpl body model

    # render smpl body model
    mean_pose_hand = np.repeat(
        np.concatenate([smpl_dict["hands_meanl"], smpl_dict["hands_meanr"]])[None],
        axis=0,
        repeats=args.pose_num,
    )
    body_parms = {
        **smpl_params,
        "betas": np.repeat(smpl_params["betas"][None], axis=0, repeats=args.pose_num),
        "pose_hand": mean_pose_hand,
    }
    body_parms = {
        k: torch.from_numpy(v[: args.pose_num])
        for k, v in body_parms.items()
        if k in ["root_orient", "trans", "pose_body", "pose_hand"]
    }
    front_body_imgs = render_smpl_params(body_model, body_parms, [-90, 0, 0])
    side_body_imgs = render_smpl_params(body_model, body_parms, [-80, 45, 0])

    # load robot xyzs
    xyzs4smpl = np.load(ROBOT_XYZS_REPS_PATH)["xyzs4smpl"][: args.pose_num]
    xyzs4robot = np.load(ROBOT_XYZS_REPS_PATH)["xyzs"][: args.pose_num]

    # Create a figure and 3D axis
    fig = plt.figure(figsize=(10, 15))
    ax1 = fig.add_subplot(321, projection="3d")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("Robot xyz - front")

    # Create the second set of axes (subplot 2)
    ax2 = fig.add_subplot(322, projection="3d")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("Robot xyz - side")

    # Create the third set of axes (subplot 3)
    ax3 = fig.add_subplot(323, projection="3d")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_title("SMPL xyz - front")

    # Create the forth set of axes (subplot 4)
    ax4 = fig.add_subplot(324, projection="3d")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_zlabel("Z")
    ax4.set_title("SMPL xyz - side")

    # Create the fifth, sixth set of axes for SMPL (subplot 5, 6)
    ax5 = fig.add_subplot(325)
    ax5.set_title("SMPL - front")

    ax6 = fig.add_subplot(326)
    ax6.set_title("SMPL - side")

    # Create an empty scatter plot (initialization)
    sc1 = ax1.scatter([], [], [])
    sc2 = ax2.scatter([], [], [])
    sc3 = ax3.scatter([], [], [])
    sc4 = ax4.scatter([], [], [])

    # Update function for animation
    def update(frame):
        sc1._offsets3d = (
            xyzs4robot[frame, :, 0],
            xyzs4robot[frame, :, 1],
            xyzs4robot[frame, :, 2],
        )
        sc2._offsets3d = (
            xyzs4robot[frame, :, 0],
            xyzs4robot[frame, :, 1],
            xyzs4robot[frame, :, 2],
        )
        sc3._offsets3d = (
            xyzs4smpl[frame, :, 0],
            xyzs4smpl[frame, :, 1],
            xyzs4smpl[frame, :, 2],
        )
        sc4._offsets3d = (
            xyzs4smpl[frame, :, 0],
            xyzs4smpl[frame, :, 1],
            xyzs4smpl[frame, :, 2],
        )
        ax5.imshow(front_body_imgs[frame])
        ax6.imshow(side_body_imgs[frame])
        return sc1, sc2, sc3, sc4

    # Set the number of frames (T) and interval between frames (in milliseconds)
    num_frames = args.pose_num
    interval = 1000

    # Set the initial view angle (elevation, azimuth)
    ax1.view_init(0, 0)
    ax2.view_init(20, -45)
    ax3.view_init(0, 0)
    ax4.view_init(20, -45)

    for ax in [ax1, ax2, ax3, ax4]:
        # Set axis limits
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 2])

        # Hide tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    for ax in [ax5, ax6]:
        # Hide tick labels
        ax.set_xticks([])
        ax.set_yticks([])

    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)

    ani.save(
        f"out/plot_spot/{robot_config.robot_type.name}{'_cf' if args.collision_free else ''}.{out_extention}",
        writer="imagemagick",
        fps=args.fps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for plot spots 3D")

    parser.add_argument("--robot-type", "-r", type=RobotType, default=RobotType.REACHY)
    parser.add_argument("--data-idx", "-i", type=int, default=0)
    parser.add_argument("--pose_num", "-p", type=int, default=20)
    parser.add_argument("--out-extention", "-e", type=str, default="gif")
    parser.add_argument("--fps", type=float, default=1)
    parser.add_argument("--collision-free", "-cf", action="store_true")

    args: PlotWholeInOneArgs = parser.parse_args()

    main(args)
