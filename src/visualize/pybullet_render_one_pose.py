"""
Render the motion of the robot with pybullet simulator and save it as a gif or mp4 file.

Usage:
    python src/visualize/pybullet_render_one_pose.py -r ROBOT_TYPE -di DATA_INDEX -pi POSE_INDEX

Example:
    python src/visualize/pybullet_render_one_pose.py -r COMAN -di 0 -pi 27
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

sys.path.append("src/")
from utils.RobotConfig import RobotConfig
from utils.types import RobotType, PybulletRenderOnePoseArgs
from utils.consts import *


def main(args: PybulletRenderOnePoseArgs):
    # robot의 정보를 가져옴
    robot_config = RobotConfig(args.robot_type)

    motion_path = osp.join(robot_config.ANGLES_PATH, f"angles_{args.data_index:04}.pkl")
    motions: List[Dict[str, float]] = pickle.load(open(motion_path, "rb"))

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
        camera_distance = 0.9
        camera_pitch = -15
        camera_yaw = 90
    elif robot_config.robot_type == RobotType.NAO:
        initial_position = [0, 0, 0]
        camera_distance = 1
        camera_pitch = -30
        camera_yaw = 90
    initial_orientation = pb.getQuaternionFromEuler([0, 0, 0])  # No initial rotation
    pb.resetBasePositionAndOrientation(robot_id, initial_position, initial_orientation)

    # pybullet 시뮬레이터 카메라 설정
    pb.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=camera_yaw,
        cameraPitch=camera_pitch,
        cameraTargetPosition=initial_position,
    )

    # # Create a projection matrix with the desired aspect ratio
    # screen_width = 640
    # screen_height = 640
    # aspect_ratio = float(screen_width / screen_height)
    # projection_matrix = pb.computeProjectionMatrixFOV(
    #     fov=60, aspect=aspect_ratio, nearVal=0.1, farVal=100
    # )

    # # Get the view matrix
    # view_matrix = pb.computeViewMatrixFromYawPitchRoll(
    #     cameraTargetPosition=initial_position,
    #     distance=camera_distance,
    #     yaw=camera_yaw,
    #     pitch=camera_pitch,
    #     roll=0,
    #     upAxisIndex=2,
    # )

    # # Set the projection matrix and view matrix
    # pb.getDebugVisualizerCamera()[1] = view_matrix
    # pb.getDebugVisualizerCamera()[2] = projection_matrix

    # 로봇의 Joint {name: index} 매핑
    num_joints = pb.getNumJoints(robot_id)  # 로봇의 관절 개수 얻기
    joint_name_to_id = {}  # 관절 이름을 관절 인덱스로 매핑하기 위한 딕셔너리

    for i in range(num_joints):
        joint_info = pb.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode("utf-8")  # 관절 이름
        joint_name_to_id[joint_name] = joint_info[0]  # 관절 이름을 관절 인덱스로 매핑
        # joint_name_to_id = {"head_yaw": 0, "head_pitch": 1, "l_shoulder_pitch": 2, ...}

    joi_angles = motions[args.pose_index]

    for i in range(5):
        # joint of interest의 각도를 pybullet 시뮬레이터에 설정
        for joi_name, angle in joi_angles.items():
            joi_index = joint_name_to_id[joi_name]
            pb.resetJointState(robot_id, joi_index, angle)  # 관절 각도 설정

        pb.stepSimulation()
        img = pb.getCameraImage(640, 480, renderer=pb.ER_BULLET_HARDWARE_OPENGL)
        img = img[2]

    output_path = f"out/pybullet/one_pose/{args.robot_type.name}/{args.data_index:04}_{args.pose_index:04}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # save image
    imageio.imsave(output_path, img)

    pb.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for render with pybullet")
    # fmt: off
    parser.add_argument("--robot_type", "-r", type=RobotType, default=RobotType.COMAN)
    parser.add_argument("--data-index", "-di", type=int, required=True, help="index of the data")
    parser.add_argument("--pose-index", "-pi", type=int, required=True, help="index of the pose")
    # fmt: on

    args: PybulletRenderOnePoseArgs = parser.parse_args()
    main(args)
