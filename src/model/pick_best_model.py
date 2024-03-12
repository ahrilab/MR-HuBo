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
import matplotlib.pyplot as plt
from shutil import copyfile
from tqdm import tqdm

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

        for weight_idx in tqdm(range(weight_num)):
            pred_motion = infer_human2robot(
                robot_config=robot_config,
                extreme_filter=args.extreme_filter,
                human_pose_path=amass_data_path,
                device=args.device,
                weight_idx=weight_idx,
            )

            error: float = evaluate(
                robot_config=robot_config,
                evaluate_mode=args.evaluate_mode,
                pred_motion=pred_motion,
                gt_motion=gt_motion,
            )
            motion_errors.append(error)

        all_motions_errors.append(motion_errors)

    all_motions_errors = np.array(all_motions_errors)  # (2, 20)
    mean_errors = np.mean(all_motions_errors, axis=0)  # (20,)

    # Pick the best model
    if args.extreme_filter:
        weight_dir = f"out/models/{robot_config.robot_type.name}/same_epoch/ex"
    else:
        weight_dir = f"out/models/{robot_config.robot_type.name}/same_epoch/no_ex"

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

    # Save the best model
    best_model_idx = np.argmin(mean_errors)
    with open(osp.join(weight_dir, f"best_model_idx_{args.evaluate_mode.value}.txt"), "w") as f:
        f.write(f"Best model index: {best_model_idx}\n")
    best_pre_model_weight_path = osp.join(weight_dir, f"human2{robot_config.robot_type.name}_pre_{best_model_idx}.pth")
    best_post_model_weight_path = osp.join(weight_dir, f"human2{robot_config.robot_type.name}_post_{best_model_idx}.pth")
    pre_model_save_path = osp.join(weight_dir, f"human2{robot_config.robot_type.name}_pre_best_{args.evaluate_mode.value}.pth")
    post_model_save_path = osp.join(weight_dir, f"human2{robot_config.robot_type.name}_post_best_{args.evaluate_mode.value}.pth")

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
        "--extreme-filter",
        "-ef",
        action="store_true",
        help="Use extreme filter model",
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
    args: PickBestModelArgs = parser.parse_args()

    pick_best_model(args)
