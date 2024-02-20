"""
Predict robot angles from SMPL parameters.
Load a pre-trained model and SMPL parameter file, predict robot angles, and save it into a file.

Usage:
    python src/model/test_rep_only.py -r ROBOT_TYPE -hp HUMAN_POSE_PATH -rp ROBOT_POSE_RESULT_PATH

Example:
    python src/model/test_rep_only.py -r REACHY -hp ./data/gt_motions/amass_data/02_05_stageii.npz -rp ./out/pred_motions/REACHY/mr_result_REACHY_v1.pkl
"""

import joblib
import torch
import sys
import argparse
import os.path as osp
from pytorch3d.transforms import matrix_to_rotation_6d
from human_body_prior.tools.rotation_tools import aa2matrot

sys.path.append("./src")
from utils.RobotConfig import RobotConfig
from utils.types import RobotType, TestArgs
from utils.consts import *
from model.net import MLP


def load_smpl_reps(human_pose_path: str):
    """
    load SMPL parameters from a file and convert it to SMPL joint representations.
    """
    if human_pose_path.endswith(".pkl"):
        human_pose = joblib.load(open(human_pose_path, "rb"))["pose"][:, 3:66]
    elif human_pose_path.endswith(".npz"):
        human_pose = np.load(human_pose_path)["poses"][:, 3:66].astype(np.float32)

    length = len(human_pose)
    smpl_axis_angle = human_pose.reshape(length, -1, 3)
    num_joints = smpl_axis_angle.shape[1]
    smpl_axis_angle = smpl_axis_angle.reshape(length * num_joints, 3)

    smpl_rot = aa2matrot(torch.from_numpy(smpl_axis_angle))
    smpl_rep = matrix_to_rotation_6d(smpl_rot)

    smpl_rep = smpl_rep.reshape(length, num_joints, 6).reshape(length, -1)

    return smpl_rep


def infer_human2robot(
    robot_config: RobotConfig,
    collision_free: bool,
    extreme_filter: bool,
    human_pose_path: str,
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
    ).to(DEVICE)
    model_post = MLP(
        dim_input=robot_config.reps_dim,
        dim_output=ANGLES_DIM,
        dim_hidden=HIDDEN_DIM,
    ).to(DEVICE)

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
            weight_dir, f"human2{robot_config.robot_type.name}_rep_only_pre_best.pth"
        )
        post_model_path = osp.join(
            weight_dir, f"human2{robot_config.robot_type.name}_rep_only_post_best.pth"
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

    model_pre.load_state_dict(torch.load(pre_model_path))
    model_post.load_state_dict(torch.load(post_model_path))
    model_pre.eval()
    model_post.eval()

    smpl_rep = load_smpl_reps(human_pose_path)

    if collision_free:
        ANGLES_DIM = robot_config.cf_angles_dim
        JOINT_KEYS = robot_config.cf_joi_keys
    else:
        ANGLES_DIM = robot_config.angles_dim
        JOINT_KEYS = robot_config.joi_keys

    with torch.no_grad():
        pre_pred = model_pre(smpl_rep.to(DEVICE).float())
        post_pred = model_post(pre_pred)
        post_pred = post_pred.detach().cpu().numpy()[:, :ANGLES_DIM]

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
