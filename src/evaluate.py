"""
Evaluate the performance of the motion retargeting model.
Compare the predicted robot joint angles or link position distance with the ground truth in terms of the MSE.

Usage:
    python src/evaluate.py -r [robot_type] -e [evaluate_type]

    robot_type: RobotType
    evaluate_type: str, choices=["joint", "link"]

Example:
    python src/evaluate.py -r REACHY -e joint
    python src/evaluate.py -r COMAN -e link
"""

import pickle
import math
import argparse
import sys
import kinpy as kp
from typing import Dict, List

sys.path.append("src")
from utils.types import RobotType, EvaluateArgs
from utils.RobotConfig import RobotConfig


def main(args: EvaluateArgs):
    motions = [
        "02_05",
        "13_08",
        "13_15",
        "13_18",
        "13_21",
        "13_28",
        "15_08",
        "26_02",
        "54_16",
        "55_16",
        "56_02",
    ]

    for m_i, motion in enumerate(motions):
        robot_motion_path = (
            f"out/pred_motions/{args.robot_type.name}/rep_only_{motion}_stageii.pkl"
        )
        pred_motion: List[Dict[str, float]] = pickle.load(open(robot_motion_path, "rb"))

        robot_name_for_gt = args.robot_type.name[0] + args.robot_type.name[1:].lower()
        gt_motion_path = "data/gt_motions/mr_gt.pkl"
        gt_motion_set = pickle.load(open(gt_motion_path, "rb"))
        gt_motion = gt_motion_set[robot_name_for_gt][motion]["q_cf"]

        pred_motion_keys = list(pred_motion[0].keys())
        gt_motion_keys = list(gt_motion[0].keys())
        common_keys = list(set(pred_motion_keys).intersection(gt_motion_keys))

        if args.evaluate_type == "joint":
            if m_i == 0:
                print(f"common keys: {common_keys}")
                print(f"number of keys: {len(common_keys)}")
            motion_loss = 0.0
            for pose_idx in range(len(pred_motion)):
                pose_loss = 0.0

                for key in common_keys:
                    pred_value = pred_motion[pose_idx][key]
                    gt_value = gt_motion[pose_idx][key]

                    joint_loss = min(
                        (pred_value - gt_value) ** 2,
                        (2 * math.pi - abs(pred_value - gt_value)) ** 2,
                    )
                    pose_loss += joint_loss

                motion_loss += pose_loss

            motion_loss /= len(pred_motion)
            print(f"{motion} motion loss: {motion_loss}")

        elif args.evaluate_type == "link":
            robot_config = RobotConfig(args.robot_type)
            chain = kp.build_chain_from_urdf(open(robot_config.URDF_PATH).read())

            motion_loss = 0.0
            for pose_idx in range(len(pred_motion)):
                pose_loss = 0.0

                pred_joints = pred_motion[pose_idx]
                gt_joints = gt_motion[pose_idx]

                pred_fk_result = chain.forward_kinematics(pred_joints)
                gt_fk_result = chain.forward_kinematics(gt_joints)

                if m_i == 0 and pose_idx == 0:
                    print(f"number of links: {len(pred_fk_result)}")

                for link in pred_fk_result:
                    pred_value = pred_fk_result[link].pos
                    gt_value = gt_fk_result[link].pos

                    link_loss = (pred_value - gt_value) ** 2
                    link_loss = math.sqrt(link_loss.sum())
                    pose_loss += link_loss

                motion_loss += pose_loss

            motion_loss /= len(pred_motion)
            print(f"{motion} motion loss: {motion_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-type", "-r", type=RobotType, default=RobotType.REACHY)
    parser.add_argument(
        "--evaluate-type", "-e", type=str, default="joint", choices=["joint", "link"]
    )

    args: EvaluateArgs = parser.parse_args()
    main(args)
