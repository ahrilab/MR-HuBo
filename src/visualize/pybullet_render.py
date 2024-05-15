import pybullet as pb
import pybullet_data
import sys
import numpy as np
from scipy.signal import savgol_filter

sys.path.append("src/")
from utils.RobotConfig import RobotConfig
from utils.types import RobotType
from utils.consts import *


def pybullet_render(motions, robot_config: RobotConfig, smooth: bool):
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

    # Initialize the pybullet simulator
    pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Set gravity
    pb.setGravity(0, 0, -11.7)
    pb.loadURDF("plane.urdf")

    # load the robot URDF file
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

    # Set the camera position and orientation of the pybullet simulator
    pb.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=90,
        cameraPitch=camera_pitch,
        cameraTargetPosition=initial_position,
    )

    # map joint name to joint index ({joint_name: joint_index})
    num_joints = pb.getNumJoints(robot_id)  # number of joints in the robot
    joint_name_to_id = {}  # {joint_name: joint_index}

    for i in range(num_joints):
        joint_info = pb.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode("utf-8")
        joint_name_to_id[joint_name] = joint_info[0]
        # -> joint_name_to_id: {"head_yaw": 0, "head_pitch": 1, "l_shoulder_pitch": 2, ...}

    frames = []

    for joi_angles in motions:
        # Set the joint angles of the robot in the pybullet simulator
        for joi_name, angle in joi_angles.items():
            joi_index = joint_name_to_id[joi_name]
            pb.resetJointState(robot_id, joi_index, angle)  # set the joint angle

        pb.stepSimulation()
        img = pb.getCameraImage(640, 480, renderer=pb.ER_BULLET_HARDWARE_OPENGL)

        frames.append(img[2])

    pb.disconnect()
    return frames
