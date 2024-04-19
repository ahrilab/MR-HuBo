"""
Render the motion of the robot with pybullet simulator and save it as a gif or mp4 file.

Usage:
    python src/visualize/pybullet_render.py -r ROBOT_TYPE -v VIEW --fps FPS [-s] -rp ROBOT_POSE_PATH -op OUTPUT_PATH -e EXTENTION -gt -cf -mi MOTION_IDX
    python src/visualize/pybullet_render.py -r ROBOT_TYPE -v VIEW --fps FPS [-s] -mi MOTION_IDX [-gt] [-cf] [-ef]

Example:
    python src/visualize/pybullet_render.py -r COMAN -v front --fps 120 -s -mi 13_08 -ef -e mp4
    python src/visualize/pybullet_render.py -r COMAN -v front --fps 120 -s -mi 13_18 -a -e mp4

    python src/visualize/pybullet_render.py -r COMAN -v front --fps 120 -s -mi 13_18 -cf -ef -e gif
    python src/visualize/pybullet_render.py -r COMAN -v front --fps 120 -s -mi 13_18 -e mp4

    python src/visualize/pybullet_render.py -r COMAN -v front --fps 120 -rp ./out/pred_motions/COMAN/rep_only_13_18_stageii.pkl -op ./out/pybullet/rep_only_13_18.mp4
    python src/visualize/pybullet_render.py -r COMAN -v front --fps 120 -rp ./data/gt_motions/gt_coman_13_18_new.pkl -op ./out/pybullet/coman_gt_13_18_cf.mp4

    # render for GT
    python src/visualize/pybullet_render.py -r=COMAN -v front --fps 120 -s -gt -mi="13_08" -e mp4
    python src/visualize/pybullet_render.py -r=COMAN -v front --fps 120 -gt -cf -mi="13_18" -e mp4
"""

import pybullet as pb
import pybullet_data
import sys
import pickle
import imageio
import argparse
import numpy as np
import os
import os.path as osp
from typing import List, Dict
from scipy.signal import savgol_filter

sys.path.append("src/")
from utils.RobotConfig import RobotConfig
from utils.types import RobotType, PybulletRenderArgs
from utils.consts import *


def pybullet_render(motions, robot_config: RobotConfig, smooth: bool, view: str):

    # smooth
    if smooth:
        print("median filtering....")
        for ki, k in enumerate(motions[0].keys()):
            # for ki, k in enumerate(robot_config.joi_keys):
            values = np.array([th[k] for th in motions])
            if robot_config.robot_type == RobotType.COMAN:
                filter_window = 50
            elif robot_config.robot_type == RobotType.NAO:
                filter_window = 50
            values = savgol_filter(values, filter_window, 2)
            for thi, th in enumerate(motions):
                th[k] = values[thi]
        print("filtering done")

    # pybullet 시뮬레이터 초기화
    pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Set gravity
    pb.setGravity(0, 0, -11.7)
    pb.loadURDF("plane.urdf")

    # 로봇의 URDF 파일을 로드
    if robot_config.robot_type == RobotType.COMAN:
        robot_id = pb.loadURDF(robot_config.URDF_4_RENDER_PATH)
    else:
        robot_id = pb.loadURDF(robot_config.URDF_PATH)

    # Set the mass of the robot to 0, so that it is not affected by gravity (not rolling on the ground & not falling)
    pb.changeDynamics(robot_id, -1, mass=0)

    # Set the initial position of the robot on the ground
    if robot_config.robot_type == RobotType.COMAN:
        initial_position = [0, 0, 0.53]  # Adjust the values as needed
        camera_distance = 1
        camera_pitch = -15
    elif robot_config.robot_type == RobotType.NAO:
        initial_position = [0, 0, 0]
        camera_distance = 1
        camera_pitch = -30
    initial_orientation = pb.getQuaternionFromEuler([0, 0, 0])  # No initial rotation
    pb.resetBasePositionAndOrientation(robot_id, initial_position, initial_orientation)

    # pybullet 시뮬레이터 카메라 설정
    pb.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=90 if view == "front" else 0,
        cameraPitch=camera_pitch,
        cameraTargetPosition=initial_position,
    )

    # 로봇의 Joint {name: index} 매핑
    num_joints = pb.getNumJoints(robot_id)  # 로봇의 관절 개수 얻기
    joint_name_to_id = {}  # 관절 이름을 관절 인덱스로 매핑하기 위한 딕셔너리

    for i in range(num_joints):
        joint_info = pb.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode("utf-8")  # 관절 이름
        joint_name_to_id[joint_name] = joint_info[0]  # 관절 이름을 관절 인덱스로 매핑
        # joint_name_to_id = {"head_yaw": 0, "head_pitch": 1, "l_shoulder_pitch": 2, ...}

    frames = []

    for joi_angles in motions:
        # joint of interest의 각도를 pybullet 시뮬레이터에 설정
        for joi_name, angle in joi_angles.items():
            joi_index = joint_name_to_id[joi_name]
            pb.resetJointState(robot_id, joi_index, angle)  # 관절 각도 설정

        pb.stepSimulation()
        img = pb.getCameraImage(640, 480, renderer=pb.ER_BULLET_HARDWARE_OPENGL)

        frames.append(img[2])

    pb.disconnect()
    return frames


def main(args: PybulletRenderArgs):
    # robot의 정보를 가져옴
    robot_config = RobotConfig(args.robot_type)
    robot_name = robot_config.robot_type.name
    view = args.view

    # load motion data (joint) and set output path
    # if ground truth, load ground truth motion data
    if args.ground_truth:
        motions: List[Dict[str, float]] = pickle.load(open(GT_PATH, "rb"))
        motion_idx = args.motion_idx
        robot_name_for_gt = args.robot_type.name[0] + args.robot_type.name[1:].lower()
        if args.collision_free:
            motions = motions[robot_name_for_gt][motion_idx]["q_cf"]
        else:
            motions = motions[robot_name_for_gt][motion_idx]["q"]

    # if not ground truth, load predicted motion data
    else:
        # if robot_pose_path is given, load the motion data from the path
        if args.robot_pose_path:
            motions: List[Dict[str, float]] = pickle.load(
                open(args.robot_pose_path, "rb")
            )
        # if robot_pose_path is not given, load the motion data from the rule based path
        else:
            # fmt: off
            motions_dir = PRED_MOTIONS_DIR(robot_name, args.extreme_filter)
            motion_name = PRED_MOTION_NAME(robot_name, args.extreme_filter, args.motion_idx)
            motion_path = osp.join(motions_dir, motion_name)
            motions: List[Dict[str, float]] = pickle.load(open(motion_path, "rb"))
            # fmt: on

    frames = pybullet_render(motions, robot_config, args.smooth, view)

    # Save the frames as a gif or mp4 file
    if args.output_path is not None:
        output_path = args.output_path

    else:
        # fmt: off
        if args.ground_truth:
            output_dir = PYBULLET_GT_VID_DIR(robot_name)
            output_name = PYBULLET_GT_VID_NAME(robot_name, motion_idx, args.extention)
            output_path = osp.join(output_dir, output_name)
        else:
            output_dir = PYBULLET_PRED_VID_DIR(robot_name, args.extreme_filter)
            output_name = PYBULLET_PRED_VID_NAME(robot_name, args.extreme_filter, args.motion_idx, args.extention)
            output_path = osp.join(output_dir, output_name)
        # fmt: on

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    extension = output_path.split(".")[-1]
    if extension == "gif":
        imageio.mimwrite(output_path, frames, duration=1000 / args.fps)
    elif extension == "mp4":
        imageio.mimsave(output_path, frames, fps=args.fps)

    # save frames as images
    frame_dir = f"{output_dir}/{args.motion_idx}"
    os.makedirs(frame_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        imageio.imwrite(osp.join(frame_dir, f"frame_{i:04}.png"), frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for render with pybullet")
    parser.add_argument(
        "--view",
        "-v",
        type=str,
        default="front",
        choices=["front", "side"],
        help="view of camera (front or side)",
    )
    parser.add_argument("--robot-type", "-r", type=RobotType, default=RobotType.COMAN)
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--smooth", "-s", action="store_true")
    parser.add_argument("--robot-pose-path", "-rp", type=str)
    parser.add_argument("--output-path", "-op", type=str)
    parser.add_argument("--extention", "-e", type=str, default="gif")
    parser.add_argument("--ground-truth", "-gt", action="store_true")
    parser.add_argument("--collision-free", "-cf", action="store_true")
    parser.add_argument("--motion-idx", "-mi", type=str, default="02_05")
    parser.add_argument("--extreme-filter", "-ef", action="store_true")
    parser.add_argument("--arm-only", "-a", action="store_true")

    args: PybulletRenderArgs = parser.parse_args()
    main(args)
