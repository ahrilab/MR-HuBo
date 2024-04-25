import numpy as np
import kinpy as kp
import sys

sys.path.append("./src")
from utils.consts import *
from utils.types import RobotType
from utils.RobotConfig import RobotConfig
from utils.forward_kinematics import forward_kinematics


def sample_robot_data(robot_type: RobotType, num_poses: int):
    """
    Sample random robot poses for each seed.

    Args:
        robot_type (RobotType): Robot type
        num_poses (int): Number of motions to be sampled

    Returns:
        angles_list (list): List of joint angles
        xyzs_array (ndarray): Numpy array of xyz positions of robot links
        reps_array (ndarray): Numpy array of 6D representations of robot links
        xyzs4smpl_array (ndarray): Numpy array of xyz positions of SMPL joints
    """

    # load the robot configurations and build a kinematic chain
    robot_config = RobotConfig(robot_type)
    chain = kp.build_chain_from_urdf(open(robot_config.URDF_PATH).read())

    angles_list = []
    xyzs_list = []
    reps_list = []
    xyzs4smpl_list = []

    # Sample robot poses as many as num_poses
    for i in range(num_poses):
        # fmt: off
        # angles: list of joints angle dicts (num_iter, joint_num) of {k: joint, v: angle}
        #         (roll, pitch, yaw of joints)
        # k: joint key, v[0]: MIN, v[1]: MAX,
        # v[1] - v[0]: Maximum range of each joint key.
        # np.random.rand() * (MAX - MIN) + MIN: Random value in range (MIN, MAX).
        angles = {
            k: ((np.random.rand() * (v[1] - v[0])) + v[0])
            for k, v in robot_config.joi_range.items()
        }
        # fmt: on

        xyzs, reps, xyzs4smpl = forward_kinematics(robot_config, chain, angles)

        angles_list.append(angles)
        xyzs_list.append(xyzs)
        reps_list.append(reps)
        xyzs4smpl_list.append(xyzs4smpl)

    # Convert the pos & reps lists into numpy arrays
    # fmt: off
    xyzs_array = np.asarray(xyzs_list)            # shape: (NUM_POSES, # of robot links, 3)
    reps_array = np.asarray(reps_list)            # shape: (NUM_POSES, # of robot links, 6)
    xyzs4smpl_array = np.asarray(xyzs4smpl_list)  # shape: (NUM_POSES, 21, 3)
    # fmt: on

    return angles_list, xyzs_array, reps_array, xyzs4smpl_array
