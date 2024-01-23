"""
Plot the base link positions of the robot

Usage:
    python plot_base_pos.py -r [robot_type] -rp

Example:
    python plot_base_pos.py -r NAO -rp
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import kinpy as kp
from kinpy import Chain

sys.path.append("src")
from utils.RobotConfig import RobotConfig
from utils.types import RobotType, PlotBaseArgs


def plot_base_pos(args: PlotBaseArgs):
    robot_config = RobotConfig(args.robot_type)

    joint_config = (
        {
            k: ((np.random.rand() * (v[1] - v[0])) + v[0])
            for k, v in robot_config.joi_range.items()
        }
        if args.random_pose
        else {}
    )

    # Load the URDF file & get positions of robot links
    chain: Chain = kp.build_chain_from_urdf(open(robot_config.URDF_PATH).read())
    fk_result = chain.forward_kinematics(joint_config)
    # fk_result = {link: {rot: [], pos: []}, ...}

    # fmt: on
    for e_link in robot_config.exclude_links:
        fk_result.pop(e_link)

    link_names = [link for link in fk_result]
    positions = np.array([link.pos for link in fk_result.values()])

    # Create a figure and 3D axis
    fig = plt.figure(figsize=(30, 15))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("front")

    # Create the second set of axes (subplot 2)
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("Side view")

    # Create an empty scatter plot (initialization)
    sc1 = ax1.scatter([], [], [])
    sc2 = ax2.scatter([], [], [])

    # Put the positions of the links into the scatter plot
    sc1._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
    sc2._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

    for i in range(len(link_names)):
        x, y, z = positions[i, 0], positions[i, 1], positions[i, 2]

        ax1.text(x, y, z, link_names[i], fontsize=10, color="black")
        ax2.text(x, y, z, link_names[i], fontsize=10, color="black")

    # Set the initial view angle (elevation, azimuth)
    ax1.view_init(0, 0)
    ax2.view_init(20, -45)

    for ax in [ax1, ax2]:
        # Set axis limits
        ax.set_xlim([-0.25, 0.25])
        ax.set_ylim([-0.25, 0.25])
        ax.set_zlim([-0, 0.5])

        # Hide tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    # Create save directory path if not exists
    save_path = f"out/base_pos/{args.robot_type.name}_base_pos.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the figure
    plt.savefig(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_type", "-r", type=RobotType, default=RobotType.NAO)
    parser.add_argument("--random_pose", "-rp", action="store_true")

    args: PlotBaseArgs = parser.parse_args()
    plot_base_pos(args)
