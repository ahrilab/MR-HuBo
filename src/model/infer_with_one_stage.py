"""
Predict robot angles from SMPL parameters.
Load a pre-trained model and given SMPL parameters, predict robot angles.
"""

import torch
import sys
import os.path as osp

sys.path.append("./src")
from utils.RobotConfig import RobotConfig
from utils.types import EvaluateMode
from utils.consts import *
from utils.data import load_smpl_to_6D_reps
from model.net import MLP


def infer_one_stage(
    robot_config: RobotConfig,
    extreme_filter_off: bool,
    human_pose_path: str,
    device: str,
    evaluate_mode: EvaluateMode = EvaluateMode.LINK,
    weight_idx: int = -1,
):
    """
    Predict robot angles from SMPL parameters with motion retargeting model.

    Args:
        robot_config: RobotConfig
        extreme_filter: bool
        human_pose_path: str
        device: str
        evaluate_mode: EvaluateMode
        weight_idx: int

    Returns:
        robot_angles: List[dict]
    """

    # input & output dimensions
    # input: SMPL joint 6D representations (H)
    # output: robot joint angles (q)
    input_dim = SMPL_ARM_JOINT_REPS_DIM
    output_dim = robot_config.angles_dim

    # Load Model
    model = MLP(
        dim_input=input_dim,
        dim_output=output_dim,
        dim_hidden=HIDDEN_DIM,
    ).to(device)

    # Load weights
    # Firstly, set the weight directory
    robot_name = robot_config.robot_type.name
    weight_dir = MODEL_WEIGHTS_DIR(robot_name, True, extreme_filter_off)

    # Then, set the weight path
    # if weight_idx is not given, use the best weight
    if weight_idx == -1:
        model_name = MODEL_BEST_WEIGHT_NAME(robot_name, "os", evaluate_mode.value)

    # if weight_idx is given, use the weight with the given index
    else:
        model_name = MODEL_WEIGHT_NAME(robot_name, "os", weight_idx)

    model_path = osp.join(weight_dir, model_name)

    # Load the weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load SMPL parameters
    smpl_rep, _ = load_smpl_to_6D_reps(human_pose_path)

    # Predict robot angles
    with torch.no_grad():
        pred_angles = model(smpl_rep.to(device).float()).cpu().numpy()[:, :output_dim]

    # Convert the predicted robot angles to a list of dictionaries
    JOINT_KEYS = sorted(robot_config.joi_keys)
    robot_angles = []
    for p in pred_angles:
        robot_angles.append({k: p[i] for i, k in enumerate(JOINT_KEYS)})

    return robot_angles
