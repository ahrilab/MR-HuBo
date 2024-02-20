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

import math
import sys
import kinpy as kp
from typing import List

sys.path.append("src")
from utils.types import EvaluateMode
from utils.RobotConfig import RobotConfig


def evaluate(
    robot_config: RobotConfig,
    evaluate_mode: EvaluateMode,
    pred_motion: List[dict],
    gt_motion: List[dict],
) -> float:
    # print(f"Evaluate Mode: {evaluate_mode.value}")

    pred_motion_joint_keys = list(pred_motion[0].keys())
    gt_motion_joint_keys = list(gt_motion[0].keys())
    common_joint_keys = list(
        set(pred_motion_joint_keys).intersection(gt_motion_joint_keys)
    )
    motion_error = 0.0

    if evaluate_mode == EvaluateMode.JOINT:
        # print(f"common keys: {common_joint_keys}")
        # print(f"number of keys: {len(common_joint_keys)}")
        for pose_idx in range(len(pred_motion)):
            pose_loss = 0.0

            for key in common_joint_keys:
                pred_value = pred_motion[pose_idx][key]
                gt_value = gt_motion[pose_idx][key]

                joint_loss = min(
                    (pred_value - gt_value) ** 2,
                    (2 * math.pi - abs(pred_value - gt_value)) ** 2,
                )
                pose_loss += joint_loss

            pose_loss /= len(common_joint_keys)
            motion_error += pose_loss
        motion_error /= len(pred_motion)

    elif evaluate_mode == EvaluateMode.LINK:
        chain = kp.build_chain_from_urdf(open(robot_config.URDF_PATH).read())

        for pose_idx in range(len(pred_motion)):
            pose_loss = 0.0

            pred_joints = pred_motion[pose_idx]
            gt_joints = gt_motion[pose_idx]

            pred_fk_result = chain.forward_kinematics(pred_joints)
            gt_fk_result = chain.forward_kinematics(gt_joints)

            if pose_idx == 0:
                print(f"number of links: {len(robot_config.evaluate_links)}")

            for link in robot_config.evaluate_links:
                pred_value = pred_fk_result[link].pos
                gt_value = gt_fk_result[link].pos

                link_loss = (pred_value - gt_value) ** 2
                link_loss = math.sqrt(link_loss.sum())
                pose_loss += link_loss

            pose_loss /= len(robot_config.evaluate_links)
            motion_error += pose_loss

        motion_error /= len(pred_motion)

    # print(f"motion error: {motion_error}")
    return motion_error
