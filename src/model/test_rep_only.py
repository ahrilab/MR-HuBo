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


def infer_human2robot(
    robot_config: RobotConfig,
    collision_free: bool,
    extreme_filter: bool,
    human_pose_path: str,
    device: str,
    evaluate_mode: EvaluateMode = EvaluateMode.LINK,
    arm_only: bool = False,
    weight_idx: int = -1,
):
    """
    Predict robot angles from SMPL parameters with motion retargeting model.

    Args:
        robot_config: RobotConfig
        collision_free: bool
        extreme_filter: bool
        human_pose_path: str
        device: str
        evaluate_mode: EvaluateMode
        arm_only: bool
        weight_idx: int

    Returns:
        robot_angles: List[dict]
    """

    # input & output dimensions
    # input: SMPL joint 6D representations (H)
    # output: robot joint angles (q)
    if arm_only:
        input_dim = SMPL_ARM_JOINT_REPS_DIM
    else:
        input_dim = SMPL_JOINT_REPS_DIM

    if collision_free:
        output_dim = robot_config.cf_angles_dim
    else:
        output_dim = robot_config.angles_dim

    # Load Model
    model_pre = MLP(
        dim_input=input_dim,
        dim_output=robot_config.reps_dim,
        dim_hidden=HIDDEN_DIM,
    ).to(device)
    model_post = MLP(
        dim_input=robot_config.reps_dim,
        dim_output=output_dim,
        dim_hidden=HIDDEN_DIM,
    ).to(device)

    # Load weights
    # Firstly, set the weight directory
    if collision_free:
        if extreme_filter:
            weight_dir = f"out/models/{robot_config.robot_type.name}/cf/ex"
        else:
            weight_dir = f"out/models/{robot_config.robot_type.name}/cf/no_ex"
    else:
        if extreme_filter:
            weight_dir = f"out/models/{robot_config.robot_type.name}/no_cf/ex"
        else:
            weight_dir = f"out/models/{robot_config.robot_type.name}/no_cf/no_ex"
    if arm_only:
        if extreme_filter:
            weight_dir = f"out/models/{robot_config.robot_type.name}/arm_only/ex/"
        else:
            weight_dir = f"out/models/{robot_config.robot_type.name}/arm_only/no_ex/"

    # Then, set the weight path
    # if weight_idx is not given, use the best weight
    if weight_idx == -1:
        if arm_only:
            pre_model_path = osp.join(
                weight_dir,
                f"human2{robot_config.robot_type.name}_arm_only_pre_best_{evaluate_mode.value}.pth",
            )
            post_model_path = osp.join(
                weight_dir,
                f"human2{robot_config.robot_type.name}_arm_only_post_best_{evaluate_mode.value}.pth",
            )
        else:
            pre_model_path = osp.join(
                weight_dir,
                f"human2{robot_config.robot_type.name}_rep_only_pre_best_{evaluate_mode.value}.pth",
            )
            post_model_path = osp.join(
                weight_dir,
                f"human2{robot_config.robot_type.name}_rep_only_post_best_{evaluate_mode.value}.pth",
            )

    # if weight_idx is given, use the weight with the given index
    else:
        if arm_only:
            pre_model_path = osp.join(
                weight_dir,
                f"human2{robot_config.robot_type.name}_arm_only_pre_{weight_idx}.pth",
            )
            post_model_path = osp.join(
                weight_dir,
                f"human2{robot_config.robot_type.name}_arm_only_post_{weight_idx}.pth",
            )
        else:
            pre_model_path = osp.join(
                weight_dir,
                f"human2{robot_config.robot_type.name}_rep_only_pre_{weight_idx}.pth",
            )
            post_model_path = osp.join(
                weight_dir,
                f"human2{robot_config.robot_type.name}_rep_only_post_{weight_idx}.pth",
            )

    model_pre.load_state_dict(torch.load(pre_model_path, map_location=device))
    model_post.load_state_dict(torch.load(post_model_path, map_location=device))
    model_pre.eval()
    model_post.eval()

    smpl_rep, _ = load_smpl_to_6D_reps(human_pose_path, arm_only=arm_only)

    if collision_free:
        JOINT_KEYS = robot_config.cf_joi_keys
    else:
        JOINT_KEYS = robot_config.joi_keys

    with torch.no_grad():
        pre_pred = model_pre(smpl_rep.to(device).float())
        post_pred = model_post(pre_pred)
        post_pred = post_pred.detach().cpu().numpy()[:, :output_dim]

    JOINT_KEYS = sorted(JOINT_KEYS)
    robot_angles = []
    for p in post_pred:
        robot_angles.append({k: p[i] for i, k in enumerate(JOINT_KEYS)})

    return robot_angles
