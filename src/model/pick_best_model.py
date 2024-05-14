"""
This module is used to pick the best model from the trained models.
The Best model is the one that has the lowest errors on the validation motion set of GT.
"""

import pickle
import sys
import os.path as osp
import matplotlib.pyplot as plt
from shutil import copyfile
from tqdm import tqdm

sys.path.append("./src")
from model.inference_with_one_stage import infer_one_stage
from model.inference_with_two_stage import infer_two_stage
from utils.consts import *
from utils.RobotConfig import RobotConfig
from utils.types import EvaluateMode
from utils.calculate_error_from_motions import calculate_error


def pick_best_model(
    robot_config: RobotConfig,
    extreme_filter: bool,
    one_stage: bool,
    device: str,
    evaluate_mode: EvaluateMode,
) -> int:
    robot_name = robot_config.robot_type.name
    robot_name_for_gt = robot_name[0] + robot_name[1:].lower()
    gt_motions = pickle.load(open(GT_PATH, "rb"))

    if extreme_filter:
        weight_num = EF_EPOCHS // MODEL_SAVE_EPOCH
    else:
        weight_num = NUM_EPOCHS // MODEL_SAVE_EPOCH
    all_motions_errors = []  # (2, 20)
    for val_motion_idx in VALID_GT_MOTION_IDXS:
        gt_motion = gt_motions[robot_name_for_gt][val_motion_idx]["q"]
        amass_data_path = osp.join(AMASS_DATA_PATH, f"{val_motion_idx}_stageii.npz")
        motion_errors = []  # (20,)

        for weight_idx in tqdm(range(weight_num)):
            if one_stage:
                pred_motion = infer_one_stage(
                    robot_config=robot_config,
                    extreme_filter=extreme_filter,
                    human_pose_path=amass_data_path,
                    device=device,
                    weight_idx=weight_idx,
                )
            else:
                pred_motion = infer_two_stage(
                    robot_config=robot_config,
                    extreme_filter=extreme_filter,
                    human_pose_path=amass_data_path,
                    device=device,
                    weight_idx=weight_idx,
                )

            error: float = calculate_error(
                robot_config=robot_config,
                evaluate_mode=evaluate_mode,
                pred_motion=pred_motion,
                gt_motion=gt_motion,
            )
            motion_errors.append(error)

        all_motions_errors.append(motion_errors)

    all_motions_errors = np.array(all_motions_errors)  # (2, 20)
    mean_errors = np.mean(all_motions_errors, axis=0)  # (20,)
    best_model_idx = np.argmin(mean_errors)

    # Pick the best model
    weight_dir = MODEL_WEIGHTS_DIR(robot_name, one_stage, extreme_filter)

    # fmt: off
    # Save the best model
    with open(osp.join(weight_dir, f"best_model_idx_{evaluate_mode.value}.txt"), "w") as f:
        f.write(f"Best model index: {best_model_idx}\n")

    if one_stage:
        best_model_weight_path = osp.join(weight_dir, MODEL_WEIGHT_NAME(robot_name, "os", best_model_idx))
        model_save_path = osp.join(weight_dir, MODEL_BEST_WEIGHT_NAME(robot_name, "os", evaluate_mode.value))
        copyfile(best_model_weight_path, model_save_path)

    else:
        best_pre_model_weight_path = osp.join(weight_dir, MODEL_WEIGHT_NAME(robot_name, "pre", best_model_idx))
        best_post_model_weight_path = osp.join(weight_dir, MODEL_WEIGHT_NAME(robot_name, "post", best_model_idx))
        pre_model_save_path = osp.join(weight_dir, MODEL_BEST_WEIGHT_NAME(robot_name, "pre", evaluate_mode.value))
        post_model_save_path = osp.join(weight_dir, MODEL_BEST_WEIGHT_NAME(robot_name, "post", evaluate_mode.value))
        copyfile(best_pre_model_weight_path,  pre_model_save_path)
        copyfile(best_post_model_weight_path, post_model_save_path)

    # Plot the errors (motion 1, motion 2, mean)
    x = range(weight_num)
    plt.plot(x, all_motions_errors[0], label="motion 1")
    plt.plot(x, all_motions_errors[1], label="motion 2")
    plt.plot(x, mean_errors, label="mean")
    plt.legend()

    # save the plot
    fig_name = osp.join(weight_dir, f"{robot_name}_val_errors_{evaluate_mode.value}.png")
    plt.savefig(fig_name)
    print(f"Saved the plot: {fig_name}")
    # fmt: on

    return best_model_idx
