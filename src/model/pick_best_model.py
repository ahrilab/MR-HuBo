"""
This module is used to pick the best model from the trained models.
The Best model is the one that has the lowest errors on the validation motion set of GT.

Usage:
    python src/model/pick_best_model.py -r <robot_type> -cf -ef -a -em <evaluate_mode> -d <device>

Example:
    python src/model/pick_best_model.py -r REACHY -cf -ef -em link
    python src/model/pick_best_model.py -r COMAN -ef -em joint
    python src/model/pick_best_model.py -r REACHY -a -ef -em link
"""

import pickle
import sys
import argparse
import os.path as osp
from shutil import copyfile
import matplotlib.pyplot as plt

sys.path.append("./src")
from utils.consts import *
from utils.types import RobotType, PickBestModelArgs, EvaluateMode
from utils.RobotConfig import RobotConfig
from model.test_rep_only import infer_human2robot
from utils.evaluate import evaluate


def pick_best_model(args: PickBestModelArgs):
    robot_config = RobotConfig(args.robot_type)

    robot_name_for_gt = args.robot_type.name[0] + args.robot_type.name[1:].lower()
    gt_motions = pickle.load(open(GT_PATH, "rb"))

    if args.extreme_filter:
        weight_num = EF_EPOCHS // MODEL_SAVE_EPOCH
    else:
        weight_num = NUM_EPOCHS // MODEL_SAVE_EPOCH
    all_motions_errors = []  # (2, 20)
    for val_motion_idx in VALID_GT_MOTION_IDXS:
        gt_motion = gt_motions[robot_name_for_gt][val_motion_idx]["q"]
        amass_data_path = osp.join(AMASS_DATA_PATH, f"{val_motion_idx}_stageii.npz")
        motion_errors = []  # (20,)

        for weight_idx in range(weight_num):
            pred_motion = infer_human2robot(
                robot_config=robot_config,
                collision_free=args.collision_free,
                extreme_filter=args.extreme_filter,
                human_pose_path=amass_data_path,
                device=args.device,
                arm_only=args.arm_only,
                weight_idx=weight_idx,
            )

            error: float = evaluate(
                robot_config=robot_config,
                evaluate_mode=args.evaluate_mode,
                pred_motion=pred_motion,
                gt_motion=gt_motion,
            )
            print(f"{weight_idx}: {error}")
            motion_errors.append(error)

        all_motions_errors.append(motion_errors)

    all_motions_errors = np.array(all_motions_errors)  # (2, 20)
    mean_errors = np.mean(all_motions_errors, axis=0)  # (20,)

    # Pick the best model
    if args.collision_free:
        if args.extreme_filter:
            weight_dir = f"out/models/{robot_config.robot_type.name}/cf/ex"
        else:
            weight_dir = f"out/models/{robot_config.robot_type.name}/cf/no_ex"
    else:
        if args.extreme_filter:
            weight_dir = f"out/models/{robot_config.robot_type.name}/no_cf/ex"
        else:
            weight_dir = f"out/models/{robot_config.robot_type.name}/no_cf/no_ex"

    if args.arm_only:
        if args.extreme_filter:
            weight_dir = f"out/models/{robot_config.robot_type.name}/arm_only/ex"
        else:
            weight_dir = f"out/models/{robot_config.robot_type.name}/arm_only/no_ex"

    # fmt: off
    # Plot the errors (motion 1, motion 2, mean)
    x = range(weight_num)
    plt.plot(x, all_motions_errors[0], label="motion 1")
    plt.plot(x, all_motions_errors[1], label="motion 2")
    plt.plot(x, mean_errors, label="mean")
    plt.legend()
    # save the plot
    fig_name = osp.join(weight_dir, f"{args.robot_type.name}_val_errors_{args.evaluate_mode.value}.png")
    plt.savefig(fig_name)
    print(f"Saved the plot: {fig_name}")

    best_model_idx = np.argmin(mean_errors)
    if args.arm_only:
        best_pre_model_weight_path = osp.join(weight_dir, f"human2{robot_config.robot_type.name}_arm_only_pre_{best_model_idx}.pth")
        best_post_model_weight_path = osp.join(weight_dir, f"human2{robot_config.robot_type.name}_arm_only_post_{best_model_idx}.pth")
        pre_model_save_path = osp.join(weight_dir, f"human2{robot_config.robot_type.name}_arm_only_pre_best_{args.evaluate_mode.value}.pth")
        post_model_save_path = osp.join(weight_dir, f"human2{robot_config.robot_type.name}_arm_only_post_best_{args.evaluate_mode.value}.pth")
    else:
        best_pre_model_weight_path  = osp.join(weight_dir, f"human2{robot_config.robot_type.name}_rep_only_pre_{best_model_idx}.pth")
        best_post_model_weight_path = osp.join(weight_dir, f"human2{robot_config.robot_type.name}_rep_only_post_{best_model_idx}.pth")
        pre_model_save_path  = osp.join(weight_dir, f"human2{robot_config.robot_type.name}_rep_only_pre_best_{args.evaluate_mode.value}.pth")
        post_model_save_path = osp.join(weight_dir, f"human2{robot_config.robot_type.name}_rep_only_post_best_{args.evaluate_mode.value}.pth")

    copyfile(best_pre_model_weight_path,  pre_model_save_path)
    copyfile(best_post_model_weight_path, post_model_save_path)
    # fmt: on


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-type",
        "-r",
        type=RobotType,
        choices=list(RobotType),
        default=RobotType.REACHY,
        help="Type of robot to use",
    )
    parser.add_argument(
        "--collision-free",
        "-cf",
        action="store_true",
        help="Use collision free model",
        default=False,
    )
    parser.add_argument(
        "--extreme-filter",
        "-ef",
        action="store_true",
        help="Use extreme filter model",
        default=False,
    )
    parser.add_argument(
        "--evaluate-mode",
        "-em",
        type=EvaluateMode,
        choices=list(EvaluateMode),
        default=EvaluateMode.LINK,
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--arm-only",
        "-a",
        action="store_true",
    )
    args: PickBestModelArgs = parser.parse_args()

    pick_best_model(args)
