import numpy as np
from enum import Enum
from typing import Callable, List

# Constants for seeds
NUM_SEEDS = 1000
MOTION_PER_SEED = 2000

# Constants for training
HIDDEN_DIM = 512
BATCH_SIZE = 2048
LEARNING_RATE = 1e-4
DEVICE = "cuda"
NUM_EPOCHS = 1000

# fmt: off
# Constants for Human
VPOSER_PATH      = "./data/vposer_v2_05"
SMPL_PATH        = "./data/bodymodel/smplx/neutral.npz"
NUM_BETAS        = 16

# Path rules for data
robot_xyzs_reps_path = (lambda data_idx: f"xyzs+reps_{data_idx:04}.npz"
                        if type(data_idx) == int
                        else f"xyzs+reps_{data_idx}.npz")
robot_angles_path    = (lambda data_idx: f"angles_{data_idx:04}.pkl"
                        if type(data_idx) == int
                        else f"angles_{data_idx}.pkl")
smpl_params_path     = (lambda data_idx: f"params_{data_idx:04}.npz"
                        if type(data_idx) == int
                        else f"params_{data_idx}.npz")

# The Joint names of SMPL-X
# source: https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py

# Reference images
# SMPL joint index image: https://www.researchgate.net/profile/Pengpeng-Hu/publication/351179264/figure/fig2/AS:1018294295347201@1619791687651/Layout-of-23-joints-in-the-SMPL-models.ppm
# SMPL Joint index image is same until the 21th joint (0-21).
# Hand joint index reference: https://user-images.githubusercontent.com/11267794/97798641-e2a88600-1c62-11eb-887c-0dcab2e11811.png
# Hand joint index reference is not matched with SMPL-X joints, but you can see how the hand joints look like.

SMPLX_JOINT_INDEX = Enum('SMPLX_JOINT_INDEX', [
    ###### (0-21): SMPL ######
    "pelvis",           # 0
    "left_hip",         # 1
    "right_hip",        # 2
    "spine1",           # 3
    "left_knee",        # 4
    "right_knee",       # 5
    "spine2",           # 6
    "left_ankle",       # 7
    "right_ankle",      # 8
    "spine3",           # 9
    "left_foot",        # 10
    "right_foot",       # 11
    "neck",             # 12
    "left_collar",      # 13 (쇄골)
    "right_collar",     # 14
    "head",             # 15
    "left_shoulder",    # 16
    "right_shoulder",   # 17
    "left_elbow",       # 18
    "right_elbow",      # 19
    "left_wrist",       # 20
    "right_wrist",      # 21
    ###### (0-21): SMPL ######

    ############################
    "jaw",              # 22 -> Added in SMPL-X
    "left_eye_smplhf",  # 23 -> Added in SMPL-X
    "right_eye_smplhf", # 24 -> Added in SMPL-X
    ############################

    ### from 25 to 68: Added in SMPL-H ###
    # Left hand
    "left_index1",      # 25 (검지)
    "left_index2",
    "left_index3",
    "left_middle1",     # 28 (중지)
    "left_middle2",
    "left_middle3",
    "left_pinky1",      # 31 (새끼)
    "left_pinky2",
    "left_pinky3",
    "left_ring1",       # 34 (약지)
    "left_ring2",
    "left_ring3",
    "left_thumb1",      # 37 (엄지)
    "left_thumb2",
    "left_thumb3",
    # right hand
    "right_index1",     # 40
    "right_index2",
    "right_index3",
    "right_middle1",    # 43
    "right_middle2",
    "right_middle3",
    "right_pinky1",     # 46
    "right_pinky2",
    "right_pinky3",
    "right_ring1",      # 49
    "right_ring2",
    "right_ring3",
    "right_thumb1",     # 52
    "right_thumb2",
    "right_thumb3",
    ############
    "nose",             # 55
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",       # 65
    # Hand root
    "left_thumb",       # 66
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",      # 71
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",      # 75
    ### from 25 to 75: Added in SMPL-H ###

    "right_eye_brow1",  # 76
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",   # 81
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",            # 86
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",     # 90
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",       # 95
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",        # 101
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",    # 107
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",        # 110
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",
    "left_mouth_4",
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",      # 120
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
    # Face contour
    "right_contour_1",
    "right_contour_2",
    "right_contour_3",
    "right_contour_4",  # 130
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5",
    "left_contour_4",   # 140
    "left_contour_3",
    "left_contour_2",
    "left_contour_1",   # 143
], start=0)

################################
#     Constants for Reachy     #
################################
# Reachy urdf: Definition of 31 joints, 31 links for reachy robot.
# theta: {roll, pitch, yaw of joints} -> len: 17

REACHY_RAW_PATH  = "./data/reachy/motions/raw"
REACHY_FIX_PATH  = "./data/reachy/motions/fix"
REACHY_SMPL_PATH = "./data/reachy/motions/reachy2smpl"
REACHY_URDF_PATH = "./data/reachy/reachy.urdf"

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
        np.array([0.0, 0.0, 0.65]),                                        # pelvis
        np.array([0.0, -0.1, 0.65]),                                       # right hip
        np.array([0.0, 0.1, 0.65]),                                        # left hip
        np.array([0.0, -0.1, 0.36]),                                       # right knee
        np.array([0.0, 0.1, 0.36]),                                        # left knee
        np.array([0.0, 0.0, 0.9]),                                         # spine 3
        np.array([0.0, 0.0, 1.05]),                                        # neck
        xyzs[REACHY_LINK_INDEX.r_shoulder.value],                          # right_shoulder
        xyzs[REACHY_LINK_INDEX.r_forearm.value],                           # right_elbow
        xyzs[REACHY_LINK_INDEX.r_wrist2hand.value],                        # right_wrist
        xyzs[REACHY_LINK_INDEX.r_gripper_thumb.value],                     # right_tumb2
        xyzs[REACHY_LINK_INDEX.r_gripper_finger.value],                    # right_index1
        xyzs[REACHY_LINK_INDEX.right_tip.value],                           # right_index3
        xyzs[REACHY_LINK_INDEX.l_shoulder.value],                          # left_shoulder
        xyzs[REACHY_LINK_INDEX.l_forearm.value],                           # left_elbow
        xyzs[REACHY_LINK_INDEX.l_wrist2hand.value],                        # left_wrist
        xyzs[REACHY_LINK_INDEX.l_gripper_thumb.value],                     # left_tumb2
        xyzs[REACHY_LINK_INDEX.l_gripper_finger.value],                    # left_index1
        xyzs[REACHY_LINK_INDEX.left_tip.value],                            # left_index3
        np.array([xyzs[REACHY_LINK_INDEX.right_camera.value][0] - 0.01,
                xyzs[REACHY_LINK_INDEX.right_camera.value][1],
                xyzs[REACHY_LINK_INDEX.right_camera.value][2]]),           # right_eye_smplhf
        np.array([xyzs[REACHY_LINK_INDEX.left_camera.value][0] - 0.01,
                xyzs[REACHY_LINK_INDEX.left_camera.value][1],
                xyzs[REACHY_LINK_INDEX.left_camera.value][2]]),            # left_eye_smplhf
    ]
)

# Train Parameters
REACHY_XYZS_DIM = len(REACHY_LINK_INDEX) * 3            # 31 links * 3 xyzs = 93
REACHY_REPS_DIM = len(REACHY_LINK_INDEX) * 6            # 31 links * 6 reps = 186
REACHY_ANGLES_DIM = len(REACHY_JOI)                     # 17 joints
REACHY_SMPL_REPS_DIM = len(REACHY_SMPL_JOINT_IDX) * 6   # 21 joints * 6 reps = 126

# fmt: on

################################
#     Constants for COMAN      #
################################

# fmt: off
COMAN_RAW_PATH  = "./data/coman/motions/raw"
COMAN_FIX_PATH  = "./data/coman/motions/fix"
COMAN_SMPL_PATH = "./data/coman/motions/coman2smpl"
COMAN_URDF_PATH = "./data/coman/coman.urdf"

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
# Total 23 joints
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

# convert COMAN's link xyzs (64) into smpl xyzs (25)
def coman_xyzs_to_smpl_xyzs(xyzs: List[np.ndarray]) -> List[np.ndarray]:
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
        (xyzs[ComanLinkIndex.R_ELB_IDX.value] + [0, 0, 0.075]) * 1.25,                          # right_elbow
        (xyzs[ComanLinkIndex.R_WRIST_IDX.value] + [0, 0, 0.075]) * 1.25,                        # right_wrist
        # (xyzs[ComanLinkIndex.R_SOFT_HAND_IDX.value] + [0, 0, 0.075]) * 1.25,                  # right_index
        ((xyzs[ComanLinkIndex.R_HAND_UPPER_RIGHT_LINK_IDX.value] +                              # right_middle3
        xyzs[ComanLinkIndex.R_HAND_LOWER_RIGHT_LINK_IDX.value]) / 2 + [0, 0, 0.075]) * 1.25,
        ((3 * xyzs[ComanLinkIndex.R_HAND_UPPER_LEFT_LINK_IDX.value] +                           # right_thumb3
        xyzs[ComanLinkIndex.R_HAND_UPPER_RIGHT_LINK_IDX.value]) / 4 + [0, 0, 0.075]) * 1.25,
        ((xyzs[ComanLinkIndex.R_HAND_LOWER_RIGHT_LINK_IDX.value] +                              # right_pinky
        xyzs[ComanLinkIndex.R_HAND_LOWER_LEFT_LINK_IDX.value]) / 2 + [0, 0, 0.075]) * 1.25,
        xyzs[ComanLinkIndex.L_SH_R_IDX.value] + [0, 0, 0.075],                                  # left_shoulder
        (xyzs[ComanLinkIndex.L_ELB_IDX.value] + [0, 0, 0.075]) * 1.25,                          # left_elbow
        (xyzs[ComanLinkIndex.L_WRIST_IDX.value] + [0, 0, 0.075]) * 1.25,                        # left_wrist
        # (xyzs[ComanLinkIndex.L_SOFT_HAND_IDX.value] + [0, 0, 0.075]) * 1.25,                  # left_index
        ((xyzs[ComanLinkIndex.L_HAND_UPPER_RIGHT_LINK_IDX.value] +                              # left_middle3
        xyzs[ComanLinkIndex.L_HAND_LOWER_RIGHT_LINK_IDX.value]) / 2 + [0, 0, 0.075]) * 1.25,
        ((3 * xyzs[ComanLinkIndex.L_HAND_UPPER_LEFT_LINK_IDX.value] +                           # left_thumb3
        xyzs[ComanLinkIndex.L_HAND_UPPER_RIGHT_LINK_IDX.value]) / 4 + [0, 0, 0.075]) * 1.25,
        ((xyzs[ComanLinkIndex.L_HAND_LOWER_RIGHT_LINK_IDX.value] +                              # left_pinky
        xyzs[ComanLinkIndex.L_HAND_LOWER_LEFT_LINK_IDX.value]) / 2 + [0, 0, 0.075]) * 1.25,
    ]

    smpl_xyzs = [xyz + np.array([0, 0, 0.65]) for xyz in smpl_xyzs]
    return smpl_xyzs

# Train Parameters
COMAN_XYZS_DIM = len(ComanLinkIndex) * 3            # 64 links * 3 xyzs = 192
COMAN_REPS_DIM = len(ComanLinkIndex) * 6            # 64 links * 6 reps = 384
COMAN_ANGLES_DIM = len(COMAN_JOI)                   # 14 joints
COMAN_SMPL_REPS_DIM = len(COMAN_SMPL_JOINT_IDX) * 6 # 23 joints * 6 reps = 138

# fmt: on

################################
#     Constants for YUMI      #
################################

# fmt: off
YUMI_RAW_PATH  = "./data/yumi/motions/raw"
YUMI_FIX_PATH  = "./data/yumi/motions/fix"
YUMI_URDF_PATH = "./data/yumi/yumi.urdf"
# fmt: on

################################

# Constants for Others
VIDEO_PATH = "./out/vids"
TMP_FRAME_PATH = "./out/tmp"
robot2smpl_vid_path: Callable[
    [str, int, str], str
] = lambda robot_name, index, extention: f"{robot_name}2smpl_{index}.{extention}"


### 혹시 몰라서 남김...
# Joint Index of Reachy
# fmt: off
# REACHY_PEDSTAL_IDX          = 0
# REACHY_R_ANTENNA_IDX        = 1
# REACHY_L_ANTENNA_IDX        = 2
# REACHY_R_SHOULDER_PITCH_IDX = 3
# REACHY_R_SHOULDER_ROLL_IDX  = 4
# REACHY_R_ARM_YAW_IDX        = 5
# REACHY_R_ELBOW_PITCH_IDX    = 6
# REACHY_R_FOREARM_YAW_IDX    = 7
# REACHY_R_WRIST_PITCH_IDX    = 8
# REACHY_R_WRIST_ROLL_IDX     = 9
# REACHY_R_GRIPPER_IDX        = 10
# REACHY_R_TIP                = 11
# REACHY_L_SHOULDER_PITCH_IDX = 12
# REACHY_L_SHOULDER_ROLL_IDX  = 13
# REACHY_L_ARM_YAW_IDX        = 14
# REACHY_L_ELBOW_PITCH_IDX    = 15
# REACHY_L_FOREARM_YAW_IDX    = 16
# REACHY_L_WRIST_PITCH_IDX    = 17
# REACHY_L_WRIST_ROLL_IDX     = 18
# REACHY_L_GRIPPER_IDX        = 19
# REACHY_L_TIP                = 20
# REACHY_NECK_ROLL_IDX        = 21
# REACHY_NECK_PITCH_IDX       = 22
# REACHY_NECK_YAW_IDX         = 23
# REACHY_NECK_FIXED_IDX       = 24
# REACHY_NECK_TOP_IDX         = 25
# REACHY_NECK_MIDDLE_IDX      = 26
# REACHY_NECK_BOTTOM_IDX      = 27
# REACHY_LEFT_CAM_FIXED_IDX   = 28
# REACHY_RIGHT_CAM_FIXED_IDX  = 29
# fmt: on
