import numpy as np
from enum import Enum
from typing import Callable, List

# Constants for seeds
NUM_SEEDS = 500
MOTION_PER_SEED = 2000

# fmt: off
# Constants for Human
HUMAN_PARAM_PATH = "./data/human"
VPOSER_PATH      = "./data/vposer_v2_05"
SMPL_PATH        = "./data/bodymodel/smplx/neutral.npz"
SMPL_NECK_IDX    = 12
NUM_BETAS        = 16

# Path rules for data
robot_xyzs_reps_path = (lambda data_idx: f"xyzs+reps_{data_idx:04}.npz"
                        if type(data_idx) == int
                        else f"xyzs+reps_{data_idx}.npz")
robot_angles_path    = (lambda data_idx: f"angles_{data_idx:04}.pkl"
                        if type(data_idx) == int
                        else f"angles_{data_idx}.pkl")
human_params_path    = (lambda data_idx: f"params_{data_idx:04}.npz"
                        if type(data_idx) == int
                        else f"params_{data_idx}.npz")


################################
#     Constants for Reachy     #
################################
# Reachy urdf: Definition of 31 joints, 31 links for reachy robot.
# theta: {roll, pitch, yaw of joints} -> len: 17

REACHY_RAW_PATH  = "./data/reachy/motions/raw"
REACHY_FIX_PATH  = "./data/reachy/motions/fix"
REACHY_URDF_PATH = "./data/reachy/reachy.urdf"

# convert reachy's xyzs into smpl xyzs
convert_reachy_xyzs_to_smpl_xyzs: Callable[[List[np.ndarray]], List[np.ndarray]] = (
    lambda xyzs: [
        np.array([0.0, 0.0, 0.65]),     # pelvis
        np.array([0.0, -0.1, 0.65]),    # right hip
        np.array([0.0, 0.1, 0.65]),     # left hip
        np.array([0.0, -0.1, 0.36]),    # right knee
        np.array([0.0, 0.1, 0.36]),     # left knee
        np.array([0.0, 0.0, 0.9]),      # spine 3
        np.array([0.0, 0.0, 1.05]),     # neck
        xyzs[ReachyLinkIndex.REACHY_R_SHOULDER_IDX.value],
        xyzs[ReachyLinkIndex.REACHY_R_FOREARM_IDX.value],
        xyzs[ReachyLinkIndex.REACHY_R_WIRST2HAND_IDX.value],
        xyzs[ReachyLinkIndex.REACHY_R_GRIPPER_THUMB_IDX.value],
        xyzs[ReachyLinkIndex.REACHY_R_GRIPPER_FINGER_IDX.value],
        xyzs[ReachyLinkIndex.REACHY_RIGHT_TIP_IDX.value],
        xyzs[ReachyLinkIndex.REACHY_L_SHOULDER_IDX.value],
        xyzs[ReachyLinkIndex.REACHY_L_FOREARM_IDX.value],
        xyzs[ReachyLinkIndex.REACHY_L_WIRST2HAND_IDX.value],
        xyzs[ReachyLinkIndex.REACHY_L_GRIPPER_THUMB_IDX.value],
        xyzs[ReachyLinkIndex.REACHY_L_GRIPPER_FINGER_IDX.value],
        xyzs[ReachyLinkIndex.REACHY_LEFT_TIP_IDX.value],
        np.array([xyzs[ReachyLinkIndex.REACHY_RIGHT_CAMERA_IDX.value][0] - 0.01,
                xyzs[ReachyLinkIndex.REACHY_RIGHT_CAMERA_IDX.value][1],
                xyzs[ReachyLinkIndex.REACHY_RIGHT_CAMERA_IDX.value][2]]),  # righ camera
        np.array([xyzs[ReachyLinkIndex.REACHY_LEFT_CAMERA_IDX.value][0] - 0.01,
                xyzs[ReachyLinkIndex.REACHY_LEFT_CAMERA_IDX.value][1],
                xyzs[ReachyLinkIndex.REACHY_LEFT_CAMERA_IDX.value][2]]),  # left camera
    ]
)

# link index of reachy
class ReachyLinkIndex(Enum):
    REACHY_PEDESTAL_IDX         = 0
    REACHY_TORSO_IDX            = 1
    REACHY_R_SHOULDER_IDX       = 2
    REACHY_R_SHOULDER_X_IDX     = 3
    REACHY_R_UPPER_ARM_IDX      = 4
    REACHY_R_FOREARM_IDX        = 5
    REACHY_R_WRIST_IDX          = 6
    REACHY_R_WIRST2HAND_IDX     = 7
    REACHY_R_GRIPPER_THUMB_IDX  = 8
    REACHY_R_GRIPPER_FINGER_IDX = 9
    REACHY_RIGHT_TIP_IDX        = 10
    REACHY_L_SHOULDER_IDX       = 11
    REACHY_L_SHOULDER_X_IDX     = 12
    REACHY_L_UPPER_ARM_IDX      = 13
    REACHY_L_FOREARM_IDX        = 14
    REACHY_L_WRIST_IDX          = 15
    REACHY_L_WIRST2HAND_IDX     = 16
    REACHY_L_GRIPPER_THUMB_IDX  = 17
    REACHY_L_GRIPPER_FINGER_IDX = 18
    REACHY_LEFT_TIP_IDX         = 19
    REACHY_HEAD_X_IDX           = 20
    REACHY_HEAD_Y_IDX           = 21
    REACHY_HEAD_Z_IDX           = 22
    REACHY_HEAD_IDX             = 23
    REACHY_R_ANTENNA_LINK_IDX   = 24
    REACHY_L_ANTENNA_LINK_IDX   = 25
    REACHY_LEFT_CAMERA_IDX      = 26
    REACHY_RIGHT_CAMERA_IDX     = 27
    REACHY_TOP_NECK_ARM_IDX     = 28
    REACHY_MIDDLE_NECK_ARM_IDX  = 29
    REACHY_BOTTOM_NECK_ARM_IDX  = 30

# Joint range of reachy
# joint list order: [(shoulder, elbow, forearm, wrist for R, L), neck]
REACHY_JOINT_RANGE = {
    "r_shoulder_pitch"  : [-2.618, 1.57],
    "r_shoulder_roll"   : [-3.14, 0.174],
    "r_arm_yaw"         : [-1.57, 1.57],
    "r_elbow_pitch"     : [-2.182, 0],
    "r_forearm_yaw"     : [-1.745, 1.745],
    "r_wrist_pitch"     : [-0.785, 0.785],
    "r_wrist_roll"      : [-0.785, 0.785],
    "l_shoulder_pitch"  : [-2.618, 1.57],
    "l_shoulder_roll"   : [-0.174, 3.14],
    "l_arm_yaw"         : [-1.57, 1.57],
    "l_elbow_pitch"     : [-2.182, 0],
    "l_forearm_yaw"     : [-1.745, 1.745],
    "l_wrist_pitch"     : [-0.785, 0.785],
    "l_wrist_roll"      : [-0.785, 0.785],
    "neck_roll"         : [-0.4, 0.4],
    "neck_pitch"        : [-0.4, 0.55],
    "neck_yaw"          : [-1.4, 1.4],
}
# fmt: on

################################
#     Constants for COMAN      #
################################


################################

# Constants for Others
VIDEO_PATH = "./out/vids"
reachy2smpl_vid_path: Callable[[int], str] = lambda index: f"reachy2smpl_{index}.mp4"


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
