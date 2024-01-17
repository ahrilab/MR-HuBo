"""
Render the motion of the robot with pybullet simulator.

Usage:
    python src/visualize/pybullet_render.py -v VIEW --fps FPS [-s] -rp ROBOT_POSE_PATH -op OUTPUT_PATH

Example:
    python src/visualize/pybullet_render.py -v front --fps 120 -s -rp ./out/pred_motions/COMAN/rep_only_02_05.pkl -op ./out/pybullet/rep_only_02_05.mp4
"""

import pybullet as pb
import pybullet_data
import sys
import pickle
import imageio
import argparse
import numpy as np
from typing import List, Dict
from scipy.signal import savgol_filter

sys.path.append("src/")
from utils.RobotConfig import RobotConfig
from utils.types import RobotType, PybulletRenderArgs


def main(args: PybulletRenderArgs):
    # robot의 정보를 가져옴
    robot_config = RobotConfig(args.robot_type)
    view = args.view

    # load joint data
    if args.robot_pose_path is not None:
        motions: List[Dict[str, float]] = pickle.load(open(args.robot_pose_path, "rb"))
    else:
        motions: List[Dict[str, float]] = pickle.load(
            open("out/pymaf_COMAN_v1.pkl", "rb")
        )

    # smooth
    if args.smooth:
        print("median filtering....")
        for ki, k in enumerate(robot_config.joi_keys):
            values = np.array([th[k] for th in motions])
            values = savgol_filter(values, 50, 2)
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
        robot_id = pb.loadURDF(robot_config.URDF_4_RENDER_PATH, [0, 0, 0.53])
    else:
        robot_id = pb.loadURDF(robot_config.URDF_PATH, [0, 0, 0.53])

    # Set the mass of the robot to 0, so that it is not affected by gravity (not rolling on the ground & not falling)
    pb.changeDynamics(robot_id, -1, mass=0)

    # Set the initial position of the robot on the ground
    initial_position = [0, 0, 0.53]  # Adjust the values as needed
    initial_orientation = pb.getQuaternionFromEuler([0, 0, 0])  # No initial rotation
    pb.resetBasePositionAndOrientation(robot_id, initial_position, initial_orientation)

    # pybullet 시뮬레이터 카메라 설정
    pb.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=90 if view == "front" else 0,
        cameraPitch=-15,
        cameraTargetPosition=[0, 0, 0.53],
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
        output_path = f"out/pybullet/coman_{view}{'_s' if args.smooth else ''}.gif"

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
    parser.add_argument("--robot_type", "-r", type=RobotType, default=RobotType.COMAN)
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--smooth", "-s", action="store_true")
    parser.add_argument("--robot_pose_path", "-rp", type=str, required=False)
    parser.add_argument("--output_path", "-op", type=str, required=False)

    args: PybulletRenderArgs = parser.parse_args()
    main(args)
