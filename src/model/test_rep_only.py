"""
Predict robot angles from SMPL parameters.
Load a pre-trained model and SMPL parameter file, predict robot angles, and save it into a file.

Usage:
    python src/model/test_rep_only.py -r ROBOT_TYPE -hp HUMAN_POSE_PATH -rp ROBOT_POSE_RESULT_PATH

Example:
    python src/model/test_rep_only.py -r REACHY -hp ./data/gt_motions/amass_data/02_05_stageii.npz -rp ./out/pred_motions/REACHY/mr_result_REACHY_v1.pkl
"""

import torch
import sys
import argparse
import os.path as osp

sys.path.append("./src")
from utils.RobotConfig import RobotConfig
from utils.types import RobotType, TestArgs, EvaluateMode
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
    weight_idx: int = -1,
):
    """
    Predict robot angles from SMPL parameters with motion retargeting model.

    Args:
        robot_config: RobotConfig
        collision_free: bool
        extreme_filter: bool
        human_pose_path: str
        weight_idx: int

    Returns:
        robot_angles: List[dict]
    """

    if collision_free:
        ANGLES_DIM = robot_config.cf_angles_dim
    else:
        ANGLES_DIM = robot_config.angles_dim

    # Load Model
    model_pre = MLP(
        dim_input=SMPL_JOINT_REPS_DIM,
        dim_output=robot_config.reps_dim,
        dim_hidden=HIDDEN_DIM,
    ).to(device)
    model_post = MLP(
        dim_input=robot_config.reps_dim,
        dim_output=ANGLES_DIM,
        dim_hidden=HIDDEN_DIM,
    ).to(device)

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

    if weight_idx == -1:
        pre_model_path = osp.join(
            weight_dir,
            f"human2{robot_config.robot_type.name}_rep_only_pre_best_{evaluate_mode.value}.pth",
        )
        post_model_path = osp.join(
            weight_dir,
            f"human2{robot_config.robot_type.name}_rep_only_post_best_{evaluate_mode.value}.pth",
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

    smpl_rep, _ = load_smpl_to_6D_reps(human_pose_path)

    if collision_free:
        ANGLES_DIM = robot_config.cf_angles_dim
        JOINT_KEYS = robot_config.cf_joi_keys
    else:
        ANGLES_DIM = robot_config.angles_dim
        JOINT_KEYS = robot_config.joi_keys

    with torch.no_grad():
        pre_pred = model_pre(smpl_rep.to(device).float())
        post_pred = model_post(pre_pred)
        post_pred = post_pred.detach().cpu().numpy()[:, :ANGLES_DIM]

    JOINT_KEYS = sorted(JOINT_KEYS)
    robot_angles = []
    for p in post_pred:
        robot_angles.append({k: p[i] for i, k in enumerate(JOINT_KEYS)})

    return robot_angles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot_type",
        "-r",
        type=RobotType,
        default=RobotType.REACHY,
    )
    parser.add_argument("--human_pose_path", "-hp", type=str, default="./out/pymaf.pkl")
    parser.add_argument(
        "--robot_pose_result_path",
        "-rp",
        type=str,
        required=False,
    )
    args: TestArgs = parser.parse_args()

    infer_human2robot(args)
