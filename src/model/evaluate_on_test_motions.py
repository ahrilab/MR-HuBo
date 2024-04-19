"""
Usage:
    python src/model/evaluate_on_test_motions.py -r [RobotType] -cf -ef -em [EvaluateMode] -s

Examples:
    python src/model/evaluate_on_test_motions.py -r REACHY -cf -ef -em link
    python src/model/evaluate_on_test_motions.py -r REACHY -em joint

    nohup python src/model/evaluate_on_test_motions.py -r REACHY -em link > log/reachy/evaluate_on_testset_240219_1954.log & \
    nohup python src/model/evaluate_on_test_motions.py -r REACHY -ef -em link > log/reachy/evaluate_on_testset_ef_240219_1954.log & \
    nohup python src/model/evaluate_on_test_motions.py -r REACHY -cf -em link > log/reachy/evaluate_on_testset_cf_240219_1954.log & \
    nohup python src/model/evaluate_on_test_motions.py -r REACHY -cf -ef -em link > log/reachy/evaluate_on_testset_cf_ef_240219_1954.log &
"""

import argparse
import sys
import pickle
import os
import os.path as osp
from tqdm import tqdm

sys.path.append("src")
from utils.types import RobotType, EvaluateOnTestMotionsArgs, EvaluateMode
from utils.RobotConfig import RobotConfig
from utils.consts import *
from model.test_two_stage import infer_human2robot
from utils.evaluate import evaluate


def main(args: EvaluateOnTestMotionsArgs):
    robot_config = RobotConfig(args.robot_type)
    robot_name = robot_config.robot_type.name

    gt_motions = pickle.load(open(GT_PATH, "rb"))
    robot_name_for_gt = args.robot_type.name[0] + args.robot_type.name[1:].lower()

    robot_pred_motion_dir = PRED_MOTIONS_DIR(robot_name, args.extreme_filter)
    os.makedirs(robot_pred_motion_dir, exist_ok=True)

    total_motion_errors = []
    if args.save_pred_motion:
        total_pred_motions = {}

    for test_motion_idx in tqdm(TEST_GT_MOTION_IDXS):
        gt_motion = gt_motions[robot_name_for_gt][test_motion_idx]["q"]
        amass_data_path = osp.join(AMASS_DATA_PATH, f"{test_motion_idx}_stageii.npz")
        pred_motion = infer_human2robot(
            robot_config=robot_config,
            extreme_filter=args.extreme_filter,
            human_pose_path=amass_data_path,
            device=args.device,
            evaluate_mode=args.evaluate_mode,
        )

        # save the predicted motion
        if args.save_pred_motion:
            pred_motion_path = osp.join(
                robot_pred_motion_dir,
                PRED_MOTION_NAME(robot_name, args.extreme_filter, test_motion_idx),
            )
            with open(pred_motion_path, "wb") as f:
                pickle.dump(pred_motion, f)
            total_pred_motions[test_motion_idx] = pred_motion

        error = evaluate(
            robot_config=robot_config,
            evaluate_mode=args.evaluate_mode,
            pred_motion=pred_motion,
            gt_motion=gt_motion,
        )

        total_motion_errors.append(error)

    total_motion_errors = np.array(total_motion_errors)
    mean_error = np.mean(total_motion_errors)

    if args.save_pred_motion:
        pred_motion_path = osp.join(
            robot_pred_motion_dir,
            f"pred_motions{'_ef' if args.extreme_filter else ''}.pkl",
        )
        with open(pred_motion_path, "wb") as f:
            pickle.dump(total_pred_motions, f)

    # write the result to a file
    result_path = osp.join(
        robot_pred_motion_dir, EVAL_RESULT_TXT_NAME(args.evaluate_mode.name)
    )
    print(result_path)
    with open(result_path, "w") as f:
        f.write(f"Robot: {args.robot_type.name} EF: [{args.extreme_filter}]\n")
        f.write(f"Evaluate_mode: {args.evaluate_mode.name}\n")
        f.write(f"Mean_error: {mean_error}\n")
        f.write("===================================================\n")
        f.write("All errors:\n")
        for test_idx, error in zip(TEST_GT_MOTION_IDXS, total_motion_errors):
            f.write(f"{test_idx}: {error}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model on test motions")
    parser.add_argument("--robot-type", "-r", type=RobotType, default=RobotType.REACHY)
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
        "--save-pred-motion",
        "-s",
        action="store_true",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda",
    )

    args: EvaluateOnTestMotionsArgs = parser.parse_args()
    main(args)
