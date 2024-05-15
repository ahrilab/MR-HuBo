"""
Evalueate the model on the test motions.
"""

import sys
import pickle
import os
import os.path as osp
from tqdm import tqdm

sys.path.append("src")
from model.infer_with_one_stage import infer_one_stage
from model.infer_with_two_stage import infer_two_stage
from utils.types import EvaluateMode
from utils.RobotConfig import RobotConfig
from utils.consts import *
from utils.calculate_error_from_motions import calculate_error


def evaluate_on_test_motions(
    robot_config: RobotConfig,
    extreme_filter: bool,
    one_stage: bool,
    device: str,
    evaluate_mode: EvaluateMode,
    best_model_idx: int = -1,
):
    # store variables for motion paths
    robot_name = robot_config.robot_type.name
    gt_motions = pickle.load(open(GT_PATH, "rb"))
    robot_name_for_gt = robot_name[0] + robot_name[1:].lower()

    robot_pred_motion_dir = PRED_MOTIONS_DIR(robot_name, extreme_filter)
    os.makedirs(robot_pred_motion_dir, exist_ok=True)

    total_motion_errors = []
    for test_motion_idx in tqdm(TEST_GT_MOTION_IDXS):
        # load the ground truth motion and the human pose
        gt_motion = gt_motions[robot_name_for_gt][test_motion_idx]["q"]
        amass_data_path = osp.join(AMASS_DATA_PATH, f"{test_motion_idx}_stageii.npz")

        # predict the robot motion from the human pose
        if one_stage:
            pred_motion = infer_one_stage(
                robot_config=robot_config,
                extreme_filter=extreme_filter,
                human_pose_path=amass_data_path,
                device=device,
                evaluate_mode=evaluate_mode,
                weight_idx=best_model_idx,
            )
        else:
            pred_motion = infer_two_stage(
                robot_config=robot_config,
                extreme_filter=extreme_filter,
                human_pose_path=amass_data_path,
                device=device,
                evaluate_mode=evaluate_mode,
                weight_idx=best_model_idx,
            )

        # save the predicted motion
        pred_motion_path = osp.join(
            robot_pred_motion_dir,
            PRED_MOTION_NAME(robot_name, extreme_filter, test_motion_idx),
        )
        with open(pred_motion_path, "wb") as f:
            pickle.dump(pred_motion, f)

        # calculate the error between the predicted motion and the ground truth motion
        error = calculate_error(
            robot_config=robot_config,
            evaluate_mode=evaluate_mode,
            pred_motion=pred_motion,
            gt_motion=gt_motion,
        )

        total_motion_errors.append(error)

    # calculate the mean error
    total_motion_errors = np.array(total_motion_errors)
    mean_error = np.mean(total_motion_errors)

    # write the result to a file
    result_path = osp.join(
        robot_pred_motion_dir, EVAL_RESULT_TXT_NAME(evaluate_mode.name)
    )
    print(result_path)
    with open(result_path, "w") as f:
        f.write(f"Robot: {robot_name} EF: [{extreme_filter}]\n")
        f.write(f"Evaluate_mode: {evaluate_mode.name}\n")
        f.write(f"Mean_error: {mean_error}\n")
        f.write("===================================================\n")
        f.write("All errors:\n")
        for test_idx, error in zip(TEST_GT_MOTION_IDXS, total_motion_errors):
            f.write(f"{test_idx}: {error}\n")
