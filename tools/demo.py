import os
import argparse
import imageio
import cv2
import pickle
import os.path as osp
from PIL import Image

import sys
sys.path.append("./src")

from model.infer_with_two_stage import infer_two_stage
from utils.RobotConfig import RobotConfig
from utils.infer_smpl_with_pymaf import infer_smpl_with_pymaf
from utils.types import RobotType
from visualize.pybullet_render import pybullet_render


def extract_frames(video_path: str) -> list:
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_list = []

    while True:
        # Read one frame at a time
        ret, frame = cap.read()

        # If no frame is returned, the video has ended
        if not ret:
            break

        # Convert the frame (BGR format) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the NumPy array to a PIL Image
        pil_image = Image.fromarray(frame_rgb)

        # Append the PIL image to the list (or process it directly)
        frame_list.append(pil_image)

    # Release the video capture object
    cap.release()

    return frame_list


if __name__ == "__main__":

    # Get arguments for demo
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_type", "-r", type=RobotType, default=RobotType.COMAN)
    parser.add_argument("--visualize", "-v", action="store_true")

    args = parser.parse_args()

    # 1. Load a video file and extract frames
    # input_video_path = "./data/demo.mp4"
    # images = extract_frames(input_video_path)
    image_dir = "./data/demo_images"
    images = [Image.open(osp.join(image_dir, img)) for img in sorted(os.listdir(image_dir))]
    
    # 2. PyMAF-X inferece
    smpl_pose = infer_smpl_with_pymaf(images)

    # 3. Motion Retarget to the robot
    robot_config = RobotConfig(args.robot_type)
    robot_motion: dict = infer_two_stage(robot_config, False, smpl_pose, "cuda")

    # 4. Save the robot motion to a file
    output_motion_path = f"./out/{robot_config.name}_demo_motion.pkl"
    with open(output_motion_path, "wb") as f:
        pickle.dump(robot_motion, f)

    # 5. Visualize with the PyBullet for COMAN and NAO
    if args.visualize:
        if args.robot_type == RobotType.REACHY:
            print("Visualizing Reachy robot is not supported.")

        else:
            frames = pybullet_render(robot_motion, robot_config, True)[1:]

            output_video_path = "./out/demo.mp4"
            imageio.mimsave(output_video_path, frames, fps=5)

