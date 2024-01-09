import pybullet as pb
import pybullet_data
import sys
import pickle
from typing import List, Dict
import imageio
import argparse
import numpy as np
from scipy.signal import savgol_filter

sys.path.append("src/")
from utils.RobotConfig import RobotConfig
from utils.types import RobotType, PybulletRenderArgs


def main(args: PybulletRenderArgs):
    # robot의 정보를 가져옴
    robot_config = RobotConfig(RobotType.COMAN)

    extention = args.extention
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
            values = savgol_filter(values, 25, 2)
            for thi, th in enumerate(motions):
                th[k] = values[thi]
        print("filtering done")

    # motions = motions[500:]

    # pybullet 시뮬레이터 초기화
    pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Set gravity
    pb.setGravity(0, 0, -11.7)
    pb.loadURDF("plane.urdf")

    # 로봇의 URDF 파일을 로드
    robot_id = pb.loadURDF(robot_config.URDF_PATH)

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

    if extention == "gif":
        imageio.mimsave(
            f"out/pybullet/coman_boxing_12_{view}{'_s' if args.smooth else ''}.{extention}",
            frames,
            duration=16.7,
        )
    elif extention == "mp4":
        imageio.mimsave(
            f"out/pybullet/coman_boxing_12_{view}{'_s' if args.smooth else ''}.{extention}",
            frames,
            fps=60,
        )

    # pb.stopStateRecording(video_recorder)
    pb.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for render with pybullet")
    parser.add_argument("--extention", "-e", type=str, default="gif")
    parser.add_argument(
        "--view",
        "-v",
        type=str,
        default="front",
        choices=["front", "side"],
        help="view of camera (front or side)",
    )
    parser.add_argument("--smooth", "-s", action="store_true")
    parser.add_argument("--robot_pose_path", "-rp", type=str, required=False)

    args: PybulletRenderArgs = parser.parse_args()
    main(args)
