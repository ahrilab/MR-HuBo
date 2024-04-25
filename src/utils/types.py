import argparse
from enum import Enum


class RobotType(Enum):
    """
    Enum Type of Robots that we use
    """

    REACHY = "REACHY"
    COMAN = "COMAN"
    NAO = "NAO"


class EvaluateMode(Enum):
    """
    Enum Type of Evaluation
    """

    JOINT = "joint"
    LINK = "link"
    COS = "cos"


# Argument Types
class GenerateDataArgs(argparse.Namespace):
    """
    Arguments for Generating <Robot-Human> Paired Motion Data Python Codes
    """

    robot_type: RobotType
    num_seeds: int
    poses_per_seed: int
    device: str
    restart_idx: int


class TrainArgs(argparse.Namespace):
    robot_type: RobotType
    collision_free: bool
    extreme_filter: bool
    arm_only: bool
    wandb: bool
    device: str


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
    extreme_filter: bool
    motion_idx: str
    arm_only: bool


class PybulletRenderOnePoseArgs(argparse.Namespace):
    robot_type: RobotType
    data_index: int
    pose_index: int


class PlotBaseArgs(argparse.Namespace):
    robot_type: RobotType
    random_pose: bool


class PlotWholeInOneArgs(argparse.Namespace):
    robot_type: RobotType
    data_idx: int
    pose_num: int
    out_extention: str
    fps: int
    collision_free: bool


class ConvertMat2PklArgs(argparse.Namespace):
    robot_type: RobotType


class PickBestModelArgs(argparse.Namespace):
    robot_type: RobotType
    collision_free: bool
    extreme_filter: bool
    evaluate_mode: EvaluateMode
    device: str
    arm_only: bool


class EvaluateOnTestMotionsArgs(argparse.Namespace):
    robot_type: RobotType
    collision_free: bool
    extreme_filter: bool
    evaluate_mode: EvaluateMode
    save_pred_motion: bool
    device: str
    arm_only: bool
