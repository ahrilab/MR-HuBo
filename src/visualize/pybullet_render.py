"""
Render the motion of the robot with pybullet simulator and save it as a gif or mp4 file.

Usage:
    python src/visualize/pybullet_render.py -r ROBOT_TYPE -v VIEW --fps FPS [-s] -rp ROBOT_POSE_PATH -op OUTPUT_PATH -e EXTENTION -gt -cf -mi MOTION_IDX
    python src/visualize/pybullet_render.py -r ROBOT_TYPE -v VIEW --fps FPS [-s] -mi MOTION_IDX [-gt] [-cf] [-ef]

Example:
    python src/visualize/pybullet_render.py -r COMAN -v front --fps 120 -s -mi 13_18 -cf -ef -e gif
    python src/visualize/pybullet_render.py -r COMAN -v front --fps 120 -s -mi 13_18 -e mp4

    python src/visualize/pybullet_render.py -r COMAN -v front --fps 120 -rp ./out/pred_motions/COMAN/rep_only_13_18_stageii.pkl -op ./out/pybullet/rep_only_13_18.mp4
    python src/visualize/pybullet_render.py -r COMAN -v front --fps 120 -rp ./data/gt_motions/gt_coman_13_18_new.pkl -op ./out/pybullet/coman_gt_13_18_cf.mp4

    # render for GT
    python src/visualize/pybullet_render.py -r=COMAN -v front --fps 120 -s -gt -cf -mi="13_18" -e mp4
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
from typing import List, Dict
from scipy.signal import savgol_filter

sys.path.append("src/")
from utils.RobotConfig import RobotConfig
from utils.types import RobotType, PybulletRenderArgs
from utils.consts import *


def main(args: PybulletRenderArgs):
    # robot의 정보를 가져옴
    robot_config = RobotConfig(args.robot_type)
    view = args.view

    # load joint data
    if args.robot_pose_path is not None:
        motions: List[Dict[str, float]] = pickle.load(open(args.robot_pose_path, "rb"))
    else:
        if args.collision_free:
            if args.extreme_filter:
                motion_path = f"out/pred_motions/{args.robot_type.name}/cf/ef/{args.motion_idx}.pkl"
            else:
                motion_path = f"out/pred_motions/{args.robot_type.name}/cf/no_ef/{args.motion_idx}.pkl"
        else:
            if args.extreme_filter:
                motion_path = f"out/pred_motions/{args.robot_type.name}/no_cf/ef/{args.motion_idx}.pkl"
            else:
                motion_path = f"out/pred_motions/{args.robot_type.name}/no_cf/no_ef/{args.motion_idx}.pkl"

        motions: List[Dict[str, float]] = pickle.load(open(motion_path, "rb"))

    if args.ground_truth:
        motions: List[Dict[str, float]] = pickle.load(open(GT_PATH, "rb"))
        motion_idx = args.motion_idx
        robot_name_for_gt = args.robot_type.name[0] + args.robot_type.name[1:].lower()
        if args.collision_free:
            motions = motions[robot_name_for_gt][motion_idx]["q_cf"]
        else:
            motions = motions[robot_name_for_gt][motion_idx]["q"]

    # smooth
    if args.smooth:
        print("median filtering....")
        for ki, k in enumerate(motions[0].keys()):
            # for ki, k in enumerate(robot_config.joi_keys):
            values = np.array([th[k] for th in motions])
            if args.robot_type == RobotType.COMAN:
                filter_window = 50
            elif args.robot_type == RobotType.NAO:
                filter_window = 100
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
        camera_distance = 1.5
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

    if args.output_path is not None:
        output_path = args.output_path
    else:
        if args.collision_free:
            if args.extreme_filter:
                output_path = f"out/pybullet/{robot_config.robot_type.name}/cf/ef/{args.motion_idx}.{args.extention}"
            else:
                output_path = f"out/pybullet/{robot_config.robot_type.name}/cf/no_ef/{args.motion_idx}.{args.extention}"
        else:
            if args.extreme_filter:
                output_path = f"out/pybullet/{robot_config.robot_type.name}/no_cf/ef/{args.motion_idx}.{args.extention}"
            else:
                output_path = f"out/pybullet/{robot_config.robot_type.name}/no_cf/no_ef/{args.motion_idx}.{args.extention}"

        if args.ground_truth:
            robot_gt_path = f"out/pybullet/{robot_config.robot_type.name}/GT"
            os.makedirs(robot_gt_path, exist_ok=True)
            output_path = os.path.join(
                robot_gt_path,
                f"{args.motion_idx}{'_cf' if args.collision_free else ''}.{args.extention}",
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    extension = output_path.split(".")[-1]
    if extension == "gif":
        imageio.mimwrite(output_path, frames, duration=1000 / args.fps)
    elif extension == "mp4":
        imageio.mimsave(output_path, frames, fps=args.fps)

    pb.disconnect()


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

    args: PybulletRenderArgs = parser.parse_args()
    main(args)
