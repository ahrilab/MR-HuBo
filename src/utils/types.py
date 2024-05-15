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
    """
    Arguments for Training the Model Python Codes
    """

    robot_type: RobotType
    extreme_filter: bool
    one_stage: bool
    wandb: bool
    device: str
    num_data: int


class EvaluateArgs(argparse.Namespace):
    """
    Arguments for Evaluating the Model Python Codes
    """

    robot_type: RobotType
    extreme_filter: bool
    one_stage: bool
    device: str
    evaluate_mode: EvaluateMode


class PybulletRenderArgs(argparse.Namespace):
    robot_type: RobotType
    smooth: bool
    extention: str
    fps: int
    ground_truth: bool
    extreme_filter: bool
    motion_idx: str
