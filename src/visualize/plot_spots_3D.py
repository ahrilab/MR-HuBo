import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def main(args):
    robot = args.robot
    out_extention = args.out_extention

    # Generate some example data (replace this with your actual data)
    # T = 100  # Number of time steps
    # N = 50  # Number of points
    # data = np.random.rand(T, N, 3)
    data = np.load(f"data/{robot}/motions/raw/xyzs+reps_0000.npz")["xyzs4smpl"]

    # Create a figure and 3D axis
    fig = plt.figure(figsize=(10, 5))
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

    # Update function for animation
    def update(frame):
        sc1._offsets3d = (data[frame, :, 0], data[frame, :, 1], data[frame, :, 2])
        sc2._offsets3d = (data[frame, :, 0], data[frame, :, 1], data[frame, :, 2])
        return sc1, sc2

    # Set the number of frames (T) and interval between frames (in milliseconds)
    num_frames = len(data)
    interval = 1000

    # Set the initial view angle (elevation, azimuth)
    ax1.view_init(0, 0)
    ax2.view_init(20, -45)

    for ax in [ax1, ax2]:
        # Set axis limits
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 2])

        # Hide tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)

    # Create a writer to save the animation as a video file (MP4 format in this example)
    # writer = FFMpegWriter(fps=1, metadata=dict(artist="Me"), bitrate=1800)

    # Save the animation
    # ani.save(f"out/plot_spot/{robot}.mp4", writer=writer)

    ani.save(f"out/plot_spot/{robot}.{out_extention}", writer="imagemagick", fps=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for plot spots 3D")

    parser.add_argument("--robot", "-r", type=str, default="reachy")
    parser.add_argument("--out-extention", "-e", type=str, default="gif")

    args = parser.parse_args()

    main(args)
