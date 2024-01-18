import numpy as np
from typing import List
from enum import Enum


##############################
#     Constants for NAO      #
##############################

# fmt: off
NAO_RAW_PATH  = "./data/nao/motions/raw"
BAO_SMPL_PATH = "./data/nao/motions/nao2smpl"
NAO_URDF_PATH = "./data/nao/nao.urdf"

NAO_LINK_INDEX = Enum('NAO_LINK_INDEX', [
    "world",
    "Neck",
    "Head",
    "LPelvis",
    "LHip",
    "LThigh",
    "LTibia",
    "LAnklePitch",
    "l_ankle",
    "RPelvis",
    "RHip",
    "RThigh",
    "RTibia",
    "RAnklePitch",
    "r_ankle",
    "base_link",
    "torso",
    "LShoulder",
    "LBicep",
    "LElbow",
    "LForeArm",
    "l_wrist",
    "l_gripper",
    "RShoulder",
    "RBicep",
    "RElbow",
    "RForeArm",
    "r_wrist",
    "r_gripper",
    "RFinger23_link",
    "RFinger13_link",
    "RFinger12_link",
    "LFinger21_link",
    "LFinger13_link",
    "LFinger11_link",
    "RFinger22_link",
    "LFinger22_link",
    "RFinger21_link",
    "LFinger12_link",
    "RFinger11_link",
    "LFinger23_link",
    "LThumb1_link",
    "RThumb1_link",
    "RThumb2_link",
    "LThumb2_link",
], start=0)


# Nao's Range of "Joints of Interest"
NAO_JOI = {
    "HeadYaw"       : {"range": [-2.08567, 2.08567]},
    "HeadPitch"     : {"range": [-0.671952, 0.514872]},
    "LShoulderPitch": {"range": [-2.08567, 2.08567]},
    "RShoulderPitch": {"range": [-2.08567, 2.08567]},
    "LShoulderRoll" : {"range": [-0.314159, 1.32645]},
    "RShoulderRoll" : {"range": [-1.32645, 0.314159]},
    "LElbowYaw"     : {"range": [-2.08567, 2.08567]},
    "RElbowYaw"     : {"range": [-2.08567, 2.08567]},
    "RElbowRoll"    : {"range": [0.0349066, 1.54462]},
    "LElbowRoll"    : {"range": [-1.54462, -0.0349066]},
    "LWristYaw"     : {"range": [-1.82387, 1.82387]},
    "RWristYaw"     : {"range": [-1.82387, 1.82387]},
}

# RANGE: {k: joint, v: range} (e.g. {"HeadYaw": [-2.08567, 2.08567], ... })
NAO_JOI_RANGE    = dict((k, v["range"]) for k, v in NAO_JOI.items())
NAO_JOI_KEYS     = NAO_JOI.keys()

# SMPL-X Index for NAO
# Total 23 joints
NAO_SMPL_JOINT_IDX = [

    # # base
    # SMPLX_JOINT_INDEX.pelvis.value,
    # # lower body
    # SMPLX_JOINT_INDEX.right_hip.value,
    # SMPLX_JOINT_INDEX.left_hip.value,
    # SMPLX_JOINT_INDEX.right_knee.value,
    # SMPLX_JOINT_INDEX.left_knee.value,
    # # Center Line
    # SMPLX_JOINT_INDEX.spine3.value,
    # SMPLX_JOINT_INDEX.neck.value,
    # SMPLX_JOINT_INDEX.right_eye_smplhf.value,
    # SMPLX_JOINT_INDEX.left_eye_smplhf.value,
    # # Right Arm
    # SMPLX_JOINT_INDEX.right_shoulder.value,
    # SMPLX_JOINT_INDEX.right_elbow.value,
    # SMPLX_JOINT_INDEX.right_wrist.value,
    # SMPLX_JOINT_INDEX.right_middle1.value,
    # SMPLX_JOINT_INDEX.right_thumb1.value,
    # SMPLX_JOINT_INDEX.right_pinky1.value,
    # # Left Arm
    # SMPLX_JOINT_INDEX.left_shoulder.value,
    # SMPLX_JOINT_INDEX.left_elbow.value,
    # SMPLX_JOINT_INDEX.left_wrist.value,
    # SMPLX_JOINT_INDEX.left_middle1.value,
    # SMPLX_JOINT_INDEX.left_thumb1.value,
    # SMPLX_JOINT_INDEX.left_pinky1.value,
]

# convert NAO's link xyzs (64) into smpl xyzs (25)
def nao_xyzs_to_smpl_xyzs(xyzs: List[np.ndarray]) -> List[np.ndarray]:
    smpl_xyzs = [
        # xyzs[NAO_LINK_INDEX.WASIT_IDX.value],                                                   # pelvis
        # xyzs[NAO_LINK_INDEX.R_HIP_MOT_IDX.value] + [0, -0.08, 0],                               # right hip
        # xyzs[NAO_LINK_INDEX.L_HIP_MOT_IDX.value] + [0, 0.08, 0],                                # left hip
        # xyzs[NAO_LINK_INDEX.R_LOW_LEG_IDX.value] + [0, -0.03, -0.05],                           # right knee
        # xyzs[NAO_LINK_INDEX.L_LOW_LEG_IDX.value] + [0, 0.03, -0.05],                            # left knee
        # xyzs[NAO_LINK_INDEX.TORSO_IDX.value] + [0, 0, -0.025],                                  # spine 3
        # xyzs[NAO_LINK_INDEX.TORSO_IDX.value] + [0, 0, 0.125],                                   # neck
        # xyzs[NAO_LINK_INDEX.GAZE_IDX.value] + [0, -0.03, 0.15],                                 # right_eye
        # xyzs[NAO_LINK_INDEX.GAZE_IDX.value] + [0, 0.03, 0.15],                                  # left_eye
        # xyzs[NAO_LINK_INDEX.R_SH_R_IDX.value] + [0, 0, 0.075],                                  # right_shoulder
        # (xyzs[NAO_LINK_INDEX.R_ELB_IDX.value] + [0, 0, 0.075]) * 1.25,                          # right_elbow
        # (xyzs[NAO_LINK_INDEX.R_WRIST_IDX.value] + [0, 0, 0.075]) * 1.25,                        # right_wrist
        # # (xyzs[NAO_LINK_INDEX.R_SOFT_HAND_IDX.value] + [0, 0, 0.075]) * 1.25,                  # right_index
        # ((xyzs[NAO_LINK_INDEX.R_HAND_UPPER_RIGHT_LINK_IDX.value] +                              # right_middle3
        # xyzs[NAO_LINK_INDEX.R_HAND_LOWER_RIGHT_LINK_IDX.value]) / 2 + [0, 0, 0.075]) * 1.25,
        # ((3 * xyzs[NAO_LINK_INDEX.R_HAND_UPPER_LEFT_LINK_IDX.value] +                           # right_thumb3
        # xyzs[NAO_LINK_INDEX.R_HAND_UPPER_RIGHT_LINK_IDX.value]) / 4 + [0, 0, 0.075]) * 1.25,
        # ((xyzs[NAO_LINK_INDEX.R_HAND_LOWER_RIGHT_LINK_IDX.value] +                              # right_pinky
        # xyzs[NAO_LINK_INDEX.R_HAND_LOWER_LEFT_LINK_IDX.value]) / 2 + [0, 0, 0.075]) * 1.25,
        # xyzs[NAO_LINK_INDEX.L_SH_R_IDX.value] + [0, 0, 0.075],                                  # left_shoulder
        # (xyzs[NAO_LINK_INDEX.L_ELB_IDX.value] + [0, 0, 0.075]) * 1.25,                          # left_elbow
        # (xyzs[NAO_LINK_INDEX.L_WRIST_IDX.value] + [0, 0, 0.075]) * 1.25,                        # left_wrist
        # # (xyzs[NAO_LINK_INDEX.L_SOFT_HAND_IDX.value] + [0, 0, 0.075]) * 1.25,                  # left_index
        # ((xyzs[NAO_LINK_INDEX.L_HAND_UPPER_RIGHT_LINK_IDX.value] +                              # left_middle3
        # xyzs[NAO_LINK_INDEX.L_HAND_LOWER_RIGHT_LINK_IDX.value]) / 2 + [0, 0, 0.075]) * 1.25,
        # ((3 * xyzs[NAO_LINK_INDEX.L_HAND_UPPER_LEFT_LINK_IDX.value] +                           # left_thumb3
        # xyzs[NAO_LINK_INDEX.L_HAND_UPPER_RIGHT_LINK_IDX.value]) / 4 + [0, 0, 0.075]) * 1.25,
        # ((xyzs[NAO_LINK_INDEX.L_HAND_LOWER_RIGHT_LINK_IDX.value] +                              # left_pinky
        # xyzs[NAO_LINK_INDEX.L_HAND_LOWER_LEFT_LINK_IDX.value]) / 2 + [0, 0, 0.075]) * 1.25,
    ]

    smpl_xyzs = [xyz + np.array([0, 0, 0.65]) for xyz in smpl_xyzs]
    return smpl_xyzs

# Train Parameters
NAO_XYZS_DIM = len(NAO_LINK_INDEX) * 3          # 64 links * 3 xyzs = 192
NAO_REPS_DIM = len(NAO_LINK_INDEX) * 6          # 64 links * 6 reps = 384
NAO_ANGLES_DIM = len(NAO_JOI)                   # 14 joints
NAO_SMPL_REPS_DIM = len(NAO_SMPL_JOINT_IDX) * 6 # 23 joints * 6 reps = 138
# fmt: on
