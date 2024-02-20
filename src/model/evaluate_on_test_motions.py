"""
Usage:
    python src/model/evaluate_on_test_motions.py -r [RobotType] -cf -ef -em [EvaluateMode]

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
import os.path as osp

sys.path.append("src")
from utils.types import RobotType, EvaluateOnTestMotionsArgs, EvaluateMode
from utils.RobotConfig import RobotConfig
from utils.consts import *
from model.test_rep_only import infer_human2robot
from model.evaluate import evaluate


def main(args: EvaluateOnTestMotionsArgs):
    robot_config = RobotConfig(args.robot_type)

    gt_motions = pickle.load(open(GT_PATH, "rb"))
    robot_name_for_gt = args.robot_type.name[0] + args.robot_type.name[1:].lower()

    total_motion_errors = []
    for test_motion_idx in TEST_GT_MOTION_IDXS:
        gt_motion = gt_motions[robot_name_for_gt][test_motion_idx]["q"]
        amass_data_path = osp.join(AMASS_DATA_PATH, f"{test_motion_idx}_stageii.npz")
        pred_motion = infer_human2robot(
            robot_config=robot_config,
            collision_free=args.collision_free,
            extreme_filter=args.extreme_filter,
            human_pose_path=amass_data_path,
        )

        error = evaluate(
            robot_config=robot_config,
            evaluate_mode=args.evaluate_mode,
            pred_motion=pred_motion,
            gt_motion=gt_motion,
        )

        total_motion_errors.append(error)

    total_motion_errors = np.array(total_motion_errors)
    mean_error = np.mean(total_motion_errors)

    print(
        f"Robot: {args.robot_type.name} CF: [{args.collision_free}] EF: [{args.extreme_filter}]"
    )
    print(f"Evaluate_mode: {args.evaluate_mode.name}")
    print(f"Mean_error: {mean_error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model on test motions")
    parser.add_argument("--robot-type", "-r", type=RobotType, default=RobotType.REACHY)
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

    args = parser.parse_args()
    main(args)
