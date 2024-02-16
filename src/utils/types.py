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
    multi_cpu: bool
    num_cores: int


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
    multi_cpu: bool
    num_cores: int
    collision_free: bool


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
    collision_free: bool
    extreme_filter: bool
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
    extention: str
    fps: int
    ground_truth: bool
    collision_free: bool
    motion_idx: str


class PlotBaseArgs(argparse.Namespace):
    robot_type: RobotType
    random_pose: bool


class EvaluateArgs(argparse.Namespace):
    robot_type: RobotType
    evaluate_type: str


class PlotWholeInOneArgs(argparse.Namespace):
    robot_type: RobotType
    data_idx: int
    pose_num: int
    out_extention: str
    fps: int


class ConvertMat2PklArgs(argparse.Namespace):
    robot_type: RobotType
