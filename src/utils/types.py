import argparse
from enum import Enum


class RobotType(Enum):
    """
    Enum Type of Robots that we use
    """

    REACHY = "REACHY"
    COMAN = "COMAN"


class SampleArgs(argparse.Namespace):
    """
    Arguments for Sampling Python Codes
    """

    num_seeds: int
    motions_per_seed: int
    robot_type: RobotType


class Fit2SMPLArgs(argparse.Namespace):
    """
    Arguments for Converting Robot Joints for SMPL Parameters Python Codes
    """

    robot_type: RobotType
    video_result_path: str
    visualize: int
    verbosity: int
    fps: int


class AdjustNeckArgs(argparse.Namespace):
    """
    Arguments for Adjust neck joints of robot Python Codes
    """

    robot_type: RobotType
