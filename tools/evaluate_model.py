"""
This script is used to evaluate the model.
Picks the best model on the validation set and evaluates it on the test motions.

# Usage
    python tools/evaluate_model.py -r ROBOT_TYPE [-ef] [-os] [-d DEVICE] [-em EVALUATE_MODE]

# Example
    python tools/evaluate_model.py -r REACHY
    python tools/evaluate_model.py -r REACHY -ef -os -d cuda -em joint
"""

import argparse
import sys

sys.path.append("./src")
from model.pick_best_model import pick_best_model
from model.evaluate_on_test_motions import evaluate_on_test_motions
from utils.types import EvaluateArgs, RobotType, EvaluateMode
from utils.RobotConfig import RobotConfig


def main(args: EvaluateArgs):
    robot_config = RobotConfig(args.robot_type)

    best_model_idx: int = pick_best_model(
        robot_config=robot_config,
        extreme_filter=args.extreme_filter,
        one_stage=args.one_stage,
        device=args.device,
        evaluate_mode=args.evaluate_mode,
    )

    print(f"Best model index: {best_model_idx}")

    evaluate_on_test_motions(
        robot_config=robot_config,
        extreme_filter=args.extreme_filter,
        one_stage=args.one_stage,
        device=args.device,
        evaluate_mode=args.evaluate_mode,
        best_model_idx=best_model_idx,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model")

    parser.add_argument(
        "--robot-type",
        "-r",
        type=RobotType,
        choices=list(RobotType),
        default=RobotType.REACHY,
    )
    parser.add_argument("--extreme-filter", "-ef", action="store_true")
    parser.add_argument("--one-stage", "-os", action="store_true")
    parser.add_argument("--device", "-d", type=str, default="cuda")
    parser.add_argument(
        "--evaluate-mode",
        "-em",
        type=EvaluateMode,
        choices=list(EvaluateMode),
        default=EvaluateMode.JOINT,
    )
    args: EvaluateArgs = parser.parse_args()
    main(args)
