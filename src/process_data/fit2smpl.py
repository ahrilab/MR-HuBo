import numpy as np
import sys

sys.path.append("./src")
from utils.hbp import run_ik_engine
from utils.consts import *
from utils.RobotConfig import RobotConfig


def fit2smpl(
    robot_config: RobotConfig,
    original_xyzs4smpl: np.ndarray,
    device: str,
    verbosity: int = 0,
) -> dict:
    """
    Fit robot's pose data to SMPL parameters by running VPoser's Inverse Kinematics Engine.

    Args:
        robot_config (RobotConfig): Robot configuration
        original_xyzs4smpl (np.ndarray): Original xyzs4smpl data
        device (str): Device for running the code
        verbosity (int): Verbosity level

    Returns:
        smpl_data (dict): SMPL parameters
    """

    # Convert (x, y, z) => (y, z, x)
    xyzs4smpl = np.zeros_like(original_xyzs4smpl)

    xyzs4smpl[:, :, 0] = original_xyzs4smpl[:, :, 1]
    xyzs4smpl[:, :, 1] = original_xyzs4smpl[:, :, 2]
    xyzs4smpl[:, :, 2] = original_xyzs4smpl[:, :, 0]

    # Run VPoser's Inverse Kinematics Engine to fit the robot's pose data to SMPL parameters
    smpl_data = run_ik_engine(
        motion=xyzs4smpl,
        batch_size=BATCH_SIZE,
        smpl_path=SMPL_PATH,
        vposer_path=VPOSER_PATH,
        num_betas=NUM_BETAS,
        device=device,
        verbosity=verbosity,
        smpl_joint_idx=robot_config.smpl_joint_idx,
    )

    return smpl_data
