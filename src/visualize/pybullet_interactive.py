"""
Render the motion of the robot with pybullet simulator and save it as a gif or mp4 file.

Usage:
    python src/visualize/pybullet_interactive.py -r [robot_type] -v [view]

Example:
    python src/visualize/pybullet_interactive.py -r COMAN -v front
"""

import pybullet as pb
import pybullet_data
import sys
import argparse

sys.path.append("src/")
from utils.RobotConfig import RobotConfig
from utils.types import RobotType, PybulletRenderArgs


def main(args: PybulletRenderArgs):
    # robot의 정보를 가져옴
    robot_config = RobotConfig(args.robot_type)
    view = args.view

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
        robot_id = pb.loadURDF(robot_config.URDF_PATH, [0, 0, 0])

    # Set the mass of the robot to 0, so that it is not affected by gravity (not rolling on the ground & not falling)
    pb.changeDynamics(robot_id, -1, mass=0)

    # Set the initial position of the robot on the ground
    initial_position = [0, 0, 0]  # Adjust the values as needed
    initial_orientation = pb.getQuaternionFromEuler([0, 0, 0])  # No initial rotation
    pb.resetBasePositionAndOrientation(robot_id, initial_position, initial_orientation)

    # pybullet 시뮬레이터 카메라 설정
    pb.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=90 if view == "front" else 0,
        cameraPitch=-15,
        cameraTargetPosition=[0, 0, 0],
    )

    # pybullet 시뮬레이터 실행
    pb.setTimeStep(1 / 240.0)
    pb.setRealTimeSimulation(1)

    # Run the simulation loop
    while True:
        pb.stepSimulation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for render with pybullet")
    parser.add_argument("--robot_type", "-r", type=RobotType, default=RobotType.COMAN)
    parser.add_argument("--view", "-v", type=str, default="front")

    args: PybulletRenderArgs = parser.parse_args()
    main(args)
