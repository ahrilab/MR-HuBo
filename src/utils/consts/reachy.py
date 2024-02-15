import numpy as np
from enum import Enum
from typing import Callable, List

from utils.consts.smpl import SMPLX_JOINT_INDEX

# fmt: off
################################
#     Constants for Reachy     #
################################
# Reachy urdf: Definition of 31 joints, 31 links for reachy robot.
# theta: {roll, pitch, yaw of joints} -> len: 17

REACHY_URDF_PATH = "./data/reachy/reachy.urdf"

REACHY_ANGLES_PATH          = "./data/reachy/motions/original/robot/angles"
REACHY_XYZS_REPS_PATH       = "./data/reachy/motions/original/robot/xyzs+reps"
REACHY_SMPL_PARAMS_PATH     = "./data/reachy/motions/original/smpl_params"

REACHY_CF_MAT_PATH          = "./data/reachy/motions/cf/mats"
REACHY_CF_ANGLES_PATH       = "./data/reachy/motions/cf/robot/angles"
REACHY_CF_XYZS_REPS_PATH    = "./data/reachy/motions/cf/robot/xyzs+reps"
REACHY_CF_SMPL_PARAMS_PATH  = "./data/reachy/motions/cf/smpl_params"

# link index of reachy
REACHY_LINK_INDEX = Enum('REACHY_LINK_INDEX', [
    "pedestal",
    "torso",
    "r_shoulder",
    "r_shoulder_x",
    "r_upper_arm",
    "r_forearm",
    "r_wrist",
    "r_wrist2hand",
    "r_gripper_thumb",
    "r_gripper_finger",
    "right_tip",
    "l_shoulder",
    "l_shoulder_x",
    "l_upper_arm",
    "l_forearm",
    "l_wrist",
    "l_wrist2hand",
    "l_gripper_thumb",
    "l_gripper_finger",
    "left_tip",
    "head_x",
    "head_y",
    "head_z",
    "head",
    "r_antenna_link",
    "l_antenna_link",
    "left_camera",
    "right_camera",
    "top_neck_arm",
    "middle_neck_arm",
    "bottom_neck_arm",
], start=0)

# Reachy's Range of "Joints of Interest"
# joint list order: [(shoulder, elbow, forearm, wrist for R, L), neck]
REACHY_JOI = {
    "r_shoulder_pitch": {"range": [-2.618, 1.57]},
    "r_shoulder_roll":  {"range": [-3.14, 0.174]},
    "r_arm_yaw":        {"range": [-1.57, 1.57]},
    "r_elbow_pitch":    {"range": [-2.182, 0]},
    "r_forearm_yaw":    {"range": [-1.745, 1.745]},
    "r_wrist_pitch":    {"range": [-0.785, 0.785]},
    "r_wrist_roll":     {"range": [-0.785, 0.785]},
    "l_shoulder_pitch": {"range": [-2.618, 1.57]},
    "l_shoulder_roll":  {"range": [-0.174, 3.14]},
    "l_arm_yaw":        {"range": [-1.57, 1.57]},
    "l_elbow_pitch":    {"range": [-2.182, 0]},
    "l_forearm_yaw":    {"range": [-1.745, 1.745]},
    "l_wrist_pitch":    {"range": [-0.785, 0.785]},
    "l_wrist_roll":     {"range": [-0.785, 0.785]},
    "neck_roll":        {"range": [-0.4, 0.4]},
    "neck_pitch":       {"range": [-0.4, 0.55]},
    "neck_yaw":         {"range": [-1.4, 1.4]},
}

# RANGE: {k: joint, v: range} (e.g. {"r_shoulder_pitch": [-2.618, 1.57], ... })
REACHY_JOI_RANGE = dict((k, v["range"]) for k, v in REACHY_JOI.items())
REACHY_JOI_KEYS = REACHY_JOI.keys()

# SMPL-X Index for Reachy
# Total 21 joints
REACHY_SMPL_JOINT_IDX = [
    SMPLX_JOINT_INDEX.pelvis.value,
    SMPLX_JOINT_INDEX.right_hip.value,
    SMPLX_JOINT_INDEX.left_hip.value,
    SMPLX_JOINT_INDEX.right_knee.value,
    SMPLX_JOINT_INDEX.left_knee.value,
    SMPLX_JOINT_INDEX.spine3.value,
    SMPLX_JOINT_INDEX.neck.value,
    SMPLX_JOINT_INDEX.right_shoulder.value,
    SMPLX_JOINT_INDEX.right_elbow.value,
    SMPLX_JOINT_INDEX.right_wrist.value,
    SMPLX_JOINT_INDEX.right_thumb2.value,
    SMPLX_JOINT_INDEX.right_index1.value,
    SMPLX_JOINT_INDEX.right_index3.value,
    SMPLX_JOINT_INDEX.left_shoulder.value,
    SMPLX_JOINT_INDEX.left_elbow.value,
    SMPLX_JOINT_INDEX.left_wrist.value,
    SMPLX_JOINT_INDEX.left_thumb2.value,
    SMPLX_JOINT_INDEX.left_index1.value,
    SMPLX_JOINT_INDEX.left_index3.value,
    SMPLX_JOINT_INDEX.right_eye_smplhf.value,
    SMPLX_JOINT_INDEX.left_eye_smplhf.value,
]

# convert reachy's link xyzs (31) into smpl xyzs (21)
reachy_xyzs_to_smpl_xyzs: Callable[[List[np.ndarray]], List[np.ndarray]] = (
    lambda xyzs: [
        np.array([0.0, 0.0, 0.6]),                                                      # pelvis
        np.array([0.0, -0.04, 0.55]),                                                   # right hip
        np.array([0.0, 0.04, 0.55]),                                                    # left hip
        np.array([0.0, -0.08, 0.25]),                                                   # right knee
        np.array([0.0, 0.08, 0.25]),                                                    # left knee
        np.array([0.0, 0.0, 0.85]),                                                     # spine 3
        np.array([0.025, 0.0, 1.05]),                                                   # neck
        xyzs[REACHY_LINK_INDEX.r_shoulder.value],                                       # right_shoulder
        xyzs[REACHY_LINK_INDEX.r_forearm.value],                                        # right_elbow
        xyzs[REACHY_LINK_INDEX.r_wrist2hand.value],                                     # right_wrist
        xyzs[REACHY_LINK_INDEX.r_gripper_thumb.value],                                  # right_tumb2
        xyzs[REACHY_LINK_INDEX.r_gripper_finger.value],                                 # right_index1
        xyzs[REACHY_LINK_INDEX.right_tip.value],                                        # right_index3
        xyzs[REACHY_LINK_INDEX.l_shoulder.value],                                       # left_shoulder
        xyzs[REACHY_LINK_INDEX.l_forearm.value],                                        # left_elbow
        xyzs[REACHY_LINK_INDEX.l_wrist2hand.value],                                     # left_wrist
        xyzs[REACHY_LINK_INDEX.l_gripper_thumb.value],                                  # left_tumb2
        xyzs[REACHY_LINK_INDEX.l_gripper_finger.value],                                 # left_index1
        xyzs[REACHY_LINK_INDEX.left_tip.value],                                         # left_index3
        xyzs[REACHY_LINK_INDEX.right_camera.value] + np.array([-0.02, 0.01, 0.075]),    # right_eye_smplhf
        xyzs[REACHY_LINK_INDEX.left_camera.value] + np.array([-0.02, -0.01, 0.075]),    # left_eye_smplhf
    ]
)

REACHY_EVALUATE_LINKS = [
    "head",
    "top_neck_arm",
    "left_camera",
    "right_camera",

    "r_shoulder",
    "r_forearm",
    "r_wrist2hand",
    "r_gripper_thumb",
    "r_gripper_finger",
    "right_tip",

    "l_shoulder",
    "l_forearm",
    "l_wrist2hand",
    "l_gripper_thumb",
    "l_gripper_finger",
    "left_tip",
]

# Train Parameters
REACHY_XYZS_DIM         = len(REACHY_LINK_INDEX) * 3        # 31 links * 3 xyzs = 93
REACHY_REPS_DIM         = len(REACHY_LINK_INDEX) * 6        # 31 links * 6 reps = 186
REACHY_ANGLES_DIM       = len(REACHY_JOI)                   # 17 joints
REACHY_SMPL_REPS_DIM    = len(REACHY_SMPL_JOINT_IDX) * 6    # 21 joints * 6 reps = 126
