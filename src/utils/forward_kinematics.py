"""
Get the robot's kinematic chain information and robot joint angles of a single pose as input, 
and then use the forward kinematics to return the link positions and orientations.
"""

import kinpy as kp
import sys

sys.path.append("./src")
from utils.transform import quat2rep
from utils.consts import *
from utils.RobotConfig import RobotConfig


def forward_kinematics(robot_config: RobotConfig, chain: kp.Chain, angles: dict):
    """
    Input: robot's joint angles of a single pose (dict)
    Output: robot's link positions and orientations (list of xyzs, reps, xyzs4smpl)
    """

    # fk_result: forward kinematics result of chain (Robot)
    #            keys of each element: pos (xyz position), rot (rotation vector: quaternion representation)
    fk_result = chain.forward_kinematics(angles)

    xyzs = list()
    reps = list()

    for k, v in fk_result.items():

        # fmt: off
        # v.pos should change to 1 2 0 (when create curr_xyz)
        curr_xyz  = v.pos               # xyz position. shape: (3,)
        curr_quat = v.rot               # Quaternion Representation. shape: (4,)
        curr_rep  = quat2rep(curr_quat) # Transform Quaternion into 6D Rotation Representation
        # fmt: on

        xyzs.append(curr_xyz)
        reps.append(curr_rep)

    xyzs4smpl = robot_config.convert_xyzs(xyzs)

    return xyzs, reps, xyzs4smpl
