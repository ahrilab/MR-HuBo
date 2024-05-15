import numpy as np
from typing import List
from enum import Enum

from utils.consts.smpl import SMPLX_JOINT_INDEX

##############################
#     Constants for NAO      #
##############################

# fmt: off
NAO_URDF_PATH = "./data/nao/nao.urdf"

NAO_ANGLES_PATH          = "./data/nao/motions/robot/angles"
NAO_XYZS_REPS_PATH       = "./data/nao/motions/robot/xyzs+reps"
NAO_SMPL_PARAMS_PATH     = "./data/nao/motions/smpl_params"

# The Joint names of NAO
# ref: http://doc.aldebaran.com/1-14/family/robots/links_robot.html
NAO_LINK_INDEX = Enum('NAO_LINK_INDEX', [
    "world",
    "base_link",    #           xyz="0 0 0.33"
    "torso",        #           xyz="-0.00413 0 0.04342"
    "Neck",         #           xyz="-1e-05 0 -0.02742"
    "Head",         #           xyz="-0.00112 0 0.05258"

    "LPelvis",      #           xyz="-0.00781 -0.01114 0.02661"
    "LHip",         #           xyz="-0.01549 0.00029 -0.00515"
    "LThigh",       #           xyz="0.00138 0.00221 -0.05373"
    "LTibia",       #           xyz="0.00453 0.00225 -0.04936"
    "LAnklePitch",  #           xyz="0.00045 0.00029 0.00685"
    "l_ankle",      #           xyz="0.02542 0.0033 -0.03239"

    "RPelvis",      #           xyz="-0.00781 0.01114 0.02661"
    "RHip",         #           xyz="-0.01549 -0.00029 -0.00515"
    "RThigh",       #           xyz="0.00138 -0.00221 -0.05373"
    "RTibia",       #           xyz="0.00453 -0.00225 -0.04936"
    "RAnklePitch",  #           xyz="0.00045 -0.00029 0.00685"
    "r_ankle",      #           xyz="0.02542 -0.0033 -0.03239"

    "LShoulder",    #           xyz="-0.00165 -0.02663 0.00014"
    "LBicep",       #           xyz="0.02455 0.00563 0.0033"
    "LElbow",       #           xyz="-0.02744 0 -0.00014"
    "LForeArm",     #           xyz="0.02556 0.00281 0.00076"
    "l_wrist",      #           xyz="0.03434 -0.00088 0.00308"
    "l_gripper",

    'LFinger21_link',
    'LFinger22_link',
    'LFinger23_link',
    'LFinger11_link',
    'LFinger12_link',
    'LFinger13_link',
    'LThumb1_link',
    'LThumb2_link',

    "RShoulder",    #           xyz="-0.00165 0.02663 0.00014"
    "RBicep",       #           xyz="0.02455 -0.00563 0.0033"
    "RElbow",       #           xyz="-0.02744 0 -0.00014"
    "RForeArm",     #           xyz="0.02556 -0.00281 0.00076"
    "r_wrist",      #           xyz="0.03434 0.00088 0.00308"
    "r_gripper",

    'RFinger21_link',
    'RFinger22_link',
    'RFinger23_link',
    'RFinger11_link',
    'RFinger12_link',
    'RFinger13_link',
    'RThumb1_link',
    'RThumb2_link',
], start=0)


# Nao's Range of "Joints of Interest"
NAO_JOI = {
    "LShoulderPitch": {"range": [-2.08567, 2.08567]},
    "RShoulderPitch": {"range": [-2.08567, 2.08567]},
    "LShoulderRoll" : {"range": [-0.314159, 1.32645]},
    "RShoulderRoll" : {"range": [-1.32645, 0.314159]},
    "LElbowYaw"     : {"range": [-2.08567, 2.08567]},
    "RElbowYaw"     : {"range": [-2.08567, 2.08567]},
    "RElbowRoll"    : {"range": [0.0349066, 1.54462]},
    "LElbowRoll"    : {"range": [-1.54462, -0.0349066]},
}

# RANGE: {k: joint, v: range} (e.g. {"HeadYaw": [-2.08567, 2.08567], ... })
NAO_JOI_RANGE    = dict((k, v["range"]) for k, v in NAO_JOI.items())
NAO_JOI_KEYS     = NAO_JOI.keys()
NAO_CF_JOI_KEYS  = [
    'LShoulderPitch',
    'LShoulderRoll',
    'LElbowYaw',
    'LElbowRoll',
    'LWristYaw',
    'RShoulderPitch',
    'RShoulderRoll',
    'RElbowYaw',
    'RElbowRoll',
    'RWristYaw',
]

# SMPL-X Index for NAO
# Total 14 joints
NAO_SMPL_JOINT_IDX = [

    # base
    SMPLX_JOINT_INDEX.pelvis.value,

    # lower body
    SMPLX_JOINT_INDEX.right_hip.value,
    SMPLX_JOINT_INDEX.left_hip.value,
    SMPLX_JOINT_INDEX.right_knee.value,
    SMPLX_JOINT_INDEX.left_knee.value,

    # Center Line
    SMPLX_JOINT_INDEX.spine3.value,
    SMPLX_JOINT_INDEX.neck.value,
    SMPLX_JOINT_INDEX.right_eye_smplhf.value,
    SMPLX_JOINT_INDEX.left_eye_smplhf.value,

    # Right Arm
    SMPLX_JOINT_INDEX.right_shoulder.value,
    SMPLX_JOINT_INDEX.right_elbow.value,
    SMPLX_JOINT_INDEX.right_wrist.value,
    SMPLX_JOINT_INDEX.right_middle1.value,
    SMPLX_JOINT_INDEX.right_thumb1.value,

    # Left Arm
    SMPLX_JOINT_INDEX.left_shoulder.value,
    SMPLX_JOINT_INDEX.left_elbow.value,
    SMPLX_JOINT_INDEX.left_wrist.value,
    SMPLX_JOINT_INDEX.left_middle1.value,
    SMPLX_JOINT_INDEX.left_thumb1.value,
]

# convert NAO's link xyzs (45) into smpl xyzs (14)
def nao_xyzs_to_smpl_xyzs(xyzs: List[np.ndarray]) -> np.ndarray:

    r_sh2el = xyzs[NAO_LINK_INDEX.RElbow.value] - xyzs[NAO_LINK_INDEX.RShoulder.value]
    r_el2wr = xyzs[NAO_LINK_INDEX.r_gripper.value] - xyzs[NAO_LINK_INDEX.RElbow.value]
    l_sh2el = xyzs[NAO_LINK_INDEX.LElbow.value] - xyzs[NAO_LINK_INDEX.LShoulder.value]
    l_el2wr = xyzs[NAO_LINK_INDEX.l_gripper.value] - xyzs[NAO_LINK_INDEX.LElbow.value]
    sh2el_scale = 0.44
    el2wr_scale = 0.36

    smpl_xyzs = [
        (xyzs[NAO_LINK_INDEX.RPelvis.value] + xyzs[NAO_LINK_INDEX.LPelvis.value]) / 2,              # pelvis
        xyzs[NAO_LINK_INDEX.RTibia.value] + [0, 0.03, 0.06],                                        # right hip
        xyzs[NAO_LINK_INDEX.LTibia.value] + [0, -0.03, 0.06],                                       # left hip
        xyzs[NAO_LINK_INDEX.RTibia.value] + [0, 0.01, -0.08],                                       # right knee
        xyzs[NAO_LINK_INDEX.LTibia.value] + [0, -0.01, -0.08],                                      # left knee

        xyzs[NAO_LINK_INDEX.torso.value] + [0, 0, 0.03],                                            # spine 3
        xyzs[NAO_LINK_INDEX.Neck.value],                                                            # neck
        xyzs[NAO_LINK_INDEX.Head.value] + [0, -0.015, 0.07],                                        # right_eye
        xyzs[NAO_LINK_INDEX.Head.value] + [0, 0.015, 0.07],                                         # left_eye

        xyzs[NAO_LINK_INDEX.RShoulder.value],                                                       # right_shoulder
        xyzs[NAO_LINK_INDEX.RElbow.value] + (sh2el_scale * r_sh2el),                                # right_elbow
        xyzs[NAO_LINK_INDEX.r_gripper.value] + (sh2el_scale * r_sh2el) + (el2wr_scale * r_el2wr),   # right_wrist
        xyzs[NAO_LINK_INDEX.RFinger13_link.value] + (sh2el_scale * r_sh2el) + (el2wr_scale * r_el2wr), # right_middle1
        xyzs[NAO_LINK_INDEX.RThumb2_link.value] + (sh2el_scale * r_sh2el) + (el2wr_scale * r_el2wr),  # right_thumb1

        xyzs[NAO_LINK_INDEX.LShoulder.value],                                                       # left_shoulder
        xyzs[NAO_LINK_INDEX.LElbow.value] + (sh2el_scale * l_sh2el),                                # left_elbow
        xyzs[NAO_LINK_INDEX.l_gripper.value] + (sh2el_scale * l_sh2el) + (el2wr_scale * l_el2wr),   # left_wrist
        xyzs[NAO_LINK_INDEX.LFinger13_link.value] + (sh2el_scale * l_sh2el) + (el2wr_scale * l_el2wr), # left_middle1
        xyzs[NAO_LINK_INDEX.LThumb2_link.value] + (sh2el_scale * l_sh2el) + (el2wr_scale * l_el2wr),  # left_thumb1
    ]
    smpl_xyzs = np.array(smpl_xyzs) + [0, 0, 0.12]
    smpl_xyzs = smpl_xyzs * 2
    return smpl_xyzs

# Links to exclude while visualizing
NAO_EXCLUDE_LINKS = [
    "Head",                 # Neck과 겹침

    "world", "base_link",   # [0, 0, 0]

    "LHip", "LThigh",       # Pelvis와 겹침
    "RHip", "RThigh",

    "LAnklePitch",          # ankle과 겹침
    "RAnklePitch",

    # Bicep-Shoulder, ForeArm-Elbow, gripper-Hand 겹침
    # gripper는 thumb과 finger1 사이에 위치
    "LBicep", "LForeArm", "l_gripper",
    "RBicep", "RForeArm", "r_gripper",
]
# Nao의 finger는 3개이고, 엄지는 2개, 나머지는 3개의 link로 이루어져 있음
# 각각의 finger index는 숫자가 커질 수록 손가락의 끝으로 감
NAO_FINGER_LINKS = [
    "RFinger11_link",
    "RFinger12_link",
    "RFinger13_link",
    "RFinger21_link",
    "RFinger22_link",
    "RFinger23_link",
    "LFinger11_link",
    "LFinger12_link",
    "LFinger13_link",
    "LFinger21_link",
    "LFinger22_link",
    "LFinger23_link",
    "LThumb1_link",
    "RThumb1_link",
    "RThumb2_link",
    "LThumb2_link",
]
NAO_EXCLUDE_LINKS += NAO_FINGER_LINKS

NAO_EVALUATE_LINKS = [
    # 'LShoulder',
    'LElbow',
    'l_wrist',
    'l_gripper',

    # 'RShoulder',
    'RElbow',
    'r_wrist',
    'r_gripper',
]

NAO_JOINT_VECTORS = [
    {"from": "LShoulder",   "to": "LElbow"},
    {"from": "LElbow",      "to": "l_wrist"},
    {"from": "l_wrist",     "to": "l_gripper"},
    {"from": "RShoulder",   "to": "RElbow"},
    {"from": "RElbow",      "to": "r_wrist"},
    {"from": "r_wrist",     "to": "r_gripper"},
]

# Train Parameters
NAO_XYZS_DIM        = len(NAO_LINK_INDEX) * 3       # 45 links * 3 xyzs = 135
NAO_REPS_DIM        = len(NAO_LINK_INDEX) * 6       # 45 links * 6 reps = 270
NAO_ANGLES_DIM      = len(NAO_JOI)                  # 8 joints
NAO_SMPL_REPS_DIM   = len(NAO_SMPL_JOINT_IDX) * 6   # 14 joints * 6 reps = 84

NAO_CF_ANGLES_DIM   = 10
# fmt: on
