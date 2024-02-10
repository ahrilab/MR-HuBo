import numpy as np
from enum import Enum
from typing import List

from utils.consts.smpl import SMPLX_JOINT_INDEX

################################
#     Constants for COMAN      #
################################

# fmt: off
COMAN_RAW_PATH              = "./data/coman/motions/raw"
COMAN_SMPL_PATH             = "./data/coman/motions/coman2smpl"
COMAN_URDF_PATH             = "./data/coman/coman.urdf"
COMAN_URDF_4_RENDER_PATH    = "./data/coman/coman_nohands.urdf"

# link index of coman
class ComanLinkIndex(Enum):
    BASE_LINK_IDX               = 0
    WASIT_IDX                   = 1
    DWL_IDX                     = 2
    DWS_IDX                     = 3
    DWYTORSO_IDX                = 4
    TORSO_IDX                   = 5
    GAZE_IDX                    = 6
    R_SH_P_IDX                  = 7
    R_SH_R_IDX                  = 8
    R_SH_Y_IDX                  = 9
    R_ELB_IDX                   = 10
    R_FOREARM_IDX               = 11
    R_WR_MOT_2_IDX              = 12
    R_WR_MOT_3_IDX              = 13
    R_WRIST_IDX                 = 14
    R_SOFT_HAND_IDX             = 15
    R_HAND_UPPER_RIGHT_LINK_IDX = 16
    R_HAND_UPPER_LEFT_LINK_IDX  = 17
    R_HAND_LOWER_RIGHT_LINK_IDX = 18
    R_HAND_LOWER_LEFT_LINK_IDX  = 19
    R_ARM_FT_IDX                = 20
    L_SH_P_IDX                  = 21
    L_SH_R_IDX                  = 22
    L_SH_Y_IDX                  = 23
    L_ELB_IDX                   = 24
    L_FOREARM_IDX               = 25
    L_WR_MOT_2_IDX              = 26
    L_WR_MOT_3_IDX              = 27
    L_WRIST_IDX                 = 28
    L_SOFT_HAND_IDX             = 29
    L_HAND_UPPER_RIGHT_LINK_IDX = 30
    L_HAND_UPPER_LEFT_LINK_IDX  = 31
    L_HAND_LOWER_RIGHT_LINK_IDX = 32
    L_HAND_LOWER_LEFT_LINK_IDX  = 33
    L_ARM_FT_IDX                = 34
    R_HIP_MOT_IDX               = 35
    R_THIGH_UP_LEG_IDX          = 36
    R_THIGH_LOW_LEG_IDX         = 37
    R_LOW_LEG_IDX               = 38
    R_FOOT_MOT_IDX              = 39
    R_FOOT_IDX                  = 40
    R_ANKLE_IDX                 = 41
    R_SOLE_IDX                  = 42
    R_TOE_IDX                   = 43
    R_FOOT_UPPER_RIGHT_LINK_IDX = 44
    R_FOOT_UPPER_LEFT_LINK_IDX  = 45
    R_FOOT_LOWER_RIGHT_LINK_IDX = 46
    R_FOOT_LOWER_LEFT_LINK_IDX  = 47
    R_LEG_FT_IDX                = 48
    L_HIP_MOT_IDX               = 49
    L_THIGH_UP_LEG_IDX          = 50
    L_THIGH_LOW_LEG_IDX         = 51
    L_LOW_LEG_IDX               = 52
    L_FOOT_MOT_IDX              = 53
    L_FOOT_IDX                  = 54
    L_ANKLE_IDX                 = 55
    L_SOLE_IDX                  = 56
    L_TOE_IDX                   = 57
    L_FOOT_UPPER_RIGHT_LINK_IDX = 58
    L_FOOT_UPPER_LEFT_LINK_IDX  = 59
    L_FOOT_LOWER_RIGHT_LINK_IDX = 60
    L_FOOT_LOWER_LEFT_LINK_IDX  = 61
    L_LEG_FT_IDX                = 62
    IMU_LINK_IDX                = 63

# COMAN's Joints of Interest: Upper body actuated joints
COMAN_JOI = {
    "RShSag":           {"range": [-3.4034, 1.6581]},
    "RShLat":           {"range": [-2.094, 0.31415]},
    "RShYaw":           {"range": [-1.5708, 1.5708]},
    "RElbj":            {"range": [-2.3562, 0]},
    "LShSag":           {"range": [-3.4034, 1.6581]},
    "LShLat":           {"range": [-0.31415, 2.094]},
    "LShYaw":           {"range": [-1.5708, 1.5708]},
    "LElbj":            {"range": [-2.3562, 0]},
    "LForearmPlate":    {"range": [-1.5708, 1.5708]},
    "LWrj1":            {"range": [-0.524, 0.524]},
    "LWrj2":            {"range": [-0.1, 0.1]},     # {"range": [-0.785375, 1.395]},
    "RForearmPlate":    {"range": [-1.5708, 1.5708]},
    "RWrj1":            {"range": [-0.524, 0.524]},
    "RWrj2":            {"range": [-0.1, 0.1]},     # {"range": [-1.395, 0.785375]},
}

# RANGE: {k: joint, v: range} (e.g. {"RShSag": [-3.4034, 1.6581], ... })
COMAN_JOI_RANGE = dict((k, v["range"]) for k, v in COMAN_JOI.items())
COMAN_JOI_KEYS = COMAN_JOI.keys()


# SMPL-X Index for COMAN
# Total 21 joints
COMAN_SMPL_JOINT_IDX = [

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
    SMPLX_JOINT_INDEX.right_pinky1.value,
    # Left Arm
    SMPLX_JOINT_INDEX.left_shoulder.value,
    SMPLX_JOINT_INDEX.left_elbow.value,
    SMPLX_JOINT_INDEX.left_wrist.value,
    SMPLX_JOINT_INDEX.left_middle1.value,
    SMPLX_JOINT_INDEX.left_thumb1.value,
    SMPLX_JOINT_INDEX.left_pinky1.value,
]

# convert COMAN's link xyzs (64) into smpl xyzs (21)
def coman_xyzs_to_smpl_xyzs(xyzs: List[np.ndarray]) -> List[np.ndarray]:
    r_sh2el = xyzs[ComanLinkIndex.R_ELB_IDX.value] - xyzs[ComanLinkIndex.R_SH_R_IDX.value]
    r_el2wr = xyzs[ComanLinkIndex.R_WRIST_IDX.value] - xyzs[ComanLinkIndex.R_ELB_IDX.value]
    l_sh2el = xyzs[ComanLinkIndex.L_ELB_IDX.value] - xyzs[ComanLinkIndex.L_SH_R_IDX.value]
    l_el2wr = xyzs[ComanLinkIndex.L_WRIST_IDX.value] - xyzs[ComanLinkIndex.L_ELB_IDX.value]
    sh2el_scale = 0.25
    el2wr_scale = 0.15

    smpl_xyzs = [
        xyzs[ComanLinkIndex.WASIT_IDX.value],                                                   # pelvis
        xyzs[ComanLinkIndex.R_HIP_MOT_IDX.value] + [0, -0.08, 0],                               # right hip
        xyzs[ComanLinkIndex.L_HIP_MOT_IDX.value] + [0, 0.08, 0],                                # left hip
        xyzs[ComanLinkIndex.R_LOW_LEG_IDX.value] + [0, -0.03, -0.05],                           # right knee
        xyzs[ComanLinkIndex.L_LOW_LEG_IDX.value] + [0, 0.03, -0.05],                            # left knee

        xyzs[ComanLinkIndex.TORSO_IDX.value] + [0, 0, -0.025],                                  # spine 3
        xyzs[ComanLinkIndex.TORSO_IDX.value] + [0, 0, 0.125],                                   # neck
        xyzs[ComanLinkIndex.GAZE_IDX.value] + [0, -0.03, 0.15],                                 # right_eye
        xyzs[ComanLinkIndex.GAZE_IDX.value] + [0, 0.03, 0.15],                                  # left_eye

        xyzs[ComanLinkIndex.R_SH_R_IDX.value] + [0, 0, 0.075],                                  # right_shoulder
        (xyzs[ComanLinkIndex.R_ELB_IDX.value] + [0, 0, 0.075]) + (sh2el_scale * r_sh2el),       # right_elbow
        (xyzs[ComanLinkIndex.R_WRIST_IDX.value] + [0, 0, 0.075]) + (sh2el_scale * r_sh2el + el2wr_scale * r_el2wr), # right_wrist

        ((xyzs[ComanLinkIndex.R_HAND_UPPER_RIGHT_LINK_IDX.value] +                              # right_middle3
        xyzs[ComanLinkIndex.R_HAND_LOWER_RIGHT_LINK_IDX.value]) / 2 + [0, 0, 0.075]) + (sh2el_scale * r_sh2el + el2wr_scale * r_el2wr),
        ((3 * xyzs[ComanLinkIndex.R_HAND_UPPER_LEFT_LINK_IDX.value] +                           # right_thumb3
        xyzs[ComanLinkIndex.R_HAND_UPPER_RIGHT_LINK_IDX.value]) / 4 + [0, 0, 0.075]) + (sh2el_scale * r_sh2el + el2wr_scale * r_el2wr),
        ((xyzs[ComanLinkIndex.R_HAND_LOWER_RIGHT_LINK_IDX.value] +                              # right_pinky
        xyzs[ComanLinkIndex.R_HAND_LOWER_LEFT_LINK_IDX.value]) / 2 + [0, 0, 0.075]) + (sh2el_scale * r_sh2el + el2wr_scale * r_el2wr),

        xyzs[ComanLinkIndex.L_SH_R_IDX.value] + [0, 0, 0.075],                                  # left_shoulder
        (xyzs[ComanLinkIndex.L_ELB_IDX.value] + [0, 0, 0.075]) + (sh2el_scale * l_sh2el),                          # left_elbow
        (xyzs[ComanLinkIndex.L_WRIST_IDX.value] + [0, 0, 0.075]) + (sh2el_scale * l_sh2el + el2wr_scale * l_el2wr),                        # left_wrist

        ((xyzs[ComanLinkIndex.L_HAND_UPPER_RIGHT_LINK_IDX.value] +                              # left_middle3
        xyzs[ComanLinkIndex.L_HAND_LOWER_RIGHT_LINK_IDX.value]) / 2 + [0, 0, 0.075]) + (sh2el_scale * l_sh2el + el2wr_scale * l_el2wr),
        ((3 * xyzs[ComanLinkIndex.L_HAND_UPPER_LEFT_LINK_IDX.value] +                           # left_thumb3
        xyzs[ComanLinkIndex.L_HAND_UPPER_RIGHT_LINK_IDX.value]) / 4 + [0, 0, 0.075]) + (sh2el_scale * l_sh2el + el2wr_scale * l_el2wr),
        ((xyzs[ComanLinkIndex.L_HAND_LOWER_RIGHT_LINK_IDX.value] +                              # left_pinky
        xyzs[ComanLinkIndex.L_HAND_LOWER_LEFT_LINK_IDX.value]) / 2 + [0, 0, 0.075]) + (sh2el_scale * l_sh2el + el2wr_scale * l_el2wr),
    ]

    smpl_xyzs = np.array(smpl_xyzs) + [0, 0, 0.65]
    smpl_xyzs = smpl_xyzs * 1.1
    return smpl_xyzs

COMAN_EVALUATE_LINKS = [
    'RShp',
    'RShr',
    'RShy',
    'RElb',
    'RForearm',
    'r_wrist',
    'RSoftHand',

    'LShp',
    'LShr',
    'LShy',
    'LElb',
    'LForearm',
    'l_wrist',
    'LSoftHand'
]

# Train Parameters
COMAN_XYZS_DIM      = len(ComanLinkIndex) * 3           # 64 links * 3 xyzs = 192
COMAN_REPS_DIM      = len(ComanLinkIndex) * 6           # 64 links * 6 reps = 384
COMAN_ANGLES_DIM    = len(COMAN_JOI)                    # 14 joints
COMAN_SMPL_REPS_DIM = len(COMAN_SMPL_JOINT_IDX) * 6     # 21 joints * 6 reps = 126
