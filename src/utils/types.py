import argparse
from enum import Enum


class RobotType(Enum):
    """
    Enum Type of Robots that we use
    """

    REACHY = "REACHY"
    COMAN = "COMAN"
    NAO = "NAO"


# Argument Types
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
    video_result_dir: str
    video_extension: str
    visualize: bool
    verbosity: int
    fps: int
    restart_idx: int


class AdjustNeckArgs(argparse.Namespace):
    """
    Arguments for Adjust neck joints of robot Python Codes
    """

    robot_type: RobotType


class MakeRobotVideoArgs(argparse.Namespace):
    """
    Arguments for Making Robot Video Python Codes
    """

    motion_path: str
    result_path: str
    tmp_path: str
    fps: int
    resolution: int
    delete: bool
    smooth: bool
    robot_type: RobotType


class TrainArgs(argparse.Namespace):
    robot_type: RobotType
    wandb: bool


class TestArgs(argparse.Namespace):
    robot_type: RobotType
    human_pose_path: str
    robot_pose_result_path: str


class PybulletRenderArgs(argparse.Namespace):
    robot_type: RobotType
    smooth: bool
    view: str
    robot_pose_path: str
    output_path: str
    fps: int


class PlotBaseArgs(argparse.Namespace):
    robot_type: RobotType
    random_pose: bool
