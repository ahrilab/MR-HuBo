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
import pickle
import sys
import argparse
from pytorch3d.transforms import matrix_to_rotation_6d
from human_body_prior.tools.rotation_tools import aa2matrot

sys.path.append("./src")
from utils.RobotConfig import RobotConfig
from utils.types import RobotType, TestArgs
from utils.consts import *
from model.net import MLP


def infer_human2robot(args: TestArgs):
    robot_config = RobotConfig(args.robot_type)

    # Load Model
    model_pre = MLP(
        dim_input=robot_config.smpl_reps_dim,
        dim_output=robot_config.reps_dim,
        dim_hidden=HIDDEN_DIM,
    ).to(DEVICE)
    model_post = MLP(
        dim_input=robot_config.reps_dim,
        dim_output=robot_config.angles_dim,
        dim_hidden=HIDDEN_DIM,
    ).to(DEVICE)

    model_pre.load_state_dict(
        torch.load(
            f"out/models/{robot_config.robot_type.name}/human2{robot_config.robot_type.name}_rep_only_pre_v1.pth",
        )
    )
    model_post.load_state_dict(
        torch.load(
            f"out/models/{robot_config.robot_type.name}/human2{robot_config.robot_type.name}_rep_only_post_v1.pth",
        )
    )
    model_pre.eval()
    model_post.eval()

    ##################### get SMPL's 6D information of Detection result from pymaf.
    if args.human_pose_path.endswith(".pkl"):
        result = joblib.load(open(args.human_pose_path, "rb"))
        vid_pose = result["pose"][:, 3:66]
    elif args.human_pose_path.endswith(".npz"):
        result = np.load(args.human_pose_path)
        vid_pose = result["poses"][:, 3:66].astype(np.float32)

    # joi_keys = robot_config.joi_keys
    joint_keys = sorted([k for k, v in robot_config.joi_range.items()])

    length = len(vid_pose)
    smpl_aa = vid_pose.reshape(length, -1, 3)
    num_joints = smpl_aa.shape[1]
    smpl_aa = smpl_aa.reshape(length * num_joints, 3)

    smpl_rot = aa2matrot(torch.from_numpy(smpl_aa))
    smpl_rep = matrix_to_rotation_6d(smpl_rot)

    smpl_rep = smpl_rep.reshape(length, num_joints, 6).reshape(length, -1)

    with torch.no_grad():
        pre_pred = model_pre(smpl_rep.to(DEVICE).float())
        post_pred = model_post(pre_pred)
        post_pred = post_pred.detach().cpu().numpy()[:, : robot_config.angles_dim]

    robot_angles = []
    for p in post_pred:
        robot_angles.append({k: p[i] for i, k in enumerate(joint_keys)})

    if args.robot_pose_result_path:
        pickle.dump(
            robot_angles,
            open(args.robot_pose_result_path, "wb"),
        )
    else:
        pickle.dump(
            robot_angles,
            open(
                f"./out/{robot_config.robot_type.name}/mr_result_{robot_config.robot_type.name}_v1.pkl",
                "wb",
            ),
        )
    print("Finish!")


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
