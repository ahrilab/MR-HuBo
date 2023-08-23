import numpy as np

REACHY_RAW_PATH = "./data/reachy/raw"
REACHY_FIX_PATH = "./data/reachy/fix"
REACHY_URDF_PATH = "./data/reachy.urdf"
HUMAN_PARAM_PATH = "./data/human"

# fmt: off
reachy_xyzs_reps_path = (lambda data_idx: f"xyz+reps_{data_idx:03}.npz"
                         if type(data_idx) == int
                         else f"xyz+reps_{data_idx}.npz")
reachy_angles_path = (lambda data_idx: f"angles_{data_idx:03}.pkl"
                      if type(data_idx) == int
                      else f"angles_{data_idx}.pkl")
human_params_path = (lambda data_idx: f"params_{data_idx:03}.npz"
                     if type(data_idx) == int
                     else f"params_{data_idx}.npz")
# fmt: on

VPOSER_PATH = "./data/vposer_v2_05"
SMPL_PATH = "./data/bodymodel/smplx/neutral.npz"

VIDEO_PATH = "./out/vids"
reachy2smpl_vid_path = lambda index: f"reachy2smpl_{index}.mp4"

# fmt: off
# I tried to map XYZ points of robot body to SMPL joints. The joint order needs to be aligned properly.
# TODO: plot the robot XYZ positions to be saved, and find out whether the robot joints are well aligned with SMPL joints.
get_xyzs4smpl = lambda xyzs: [
    np.array([0.0, 0.0, 0.65]),     # pelvis
    np.array([0.0, -0.1, 0.65]),    # right hip
    np.array([0.0, 0.1, 0.65]),     # left hip
    np.array([0.0, -0.1, 0.36]),    # right knee
    np.array([0.0, 0.1, 0.36]),     # left knee
    np.array([0.0, 0.0, 0.9]),      # spine 3
    np.array([0.0, 0.0, 1.05]),     # neck
    xyzs[REACHY_R_SHOULDER_IDX],
    xyzs[REACHY_R_FOREARM_IDX],
    xyzs[REACHY_R_WIRST2HAND_IDX],
    xyzs[REACHY_R_GRIPPER_THUMB_IDX],
    xyzs[REACHY_R_GRIPPER_FINGER_IDX],
    xyzs[REACHY_RIGHT_TIP_IDX],
    xyzs[REACHY_L_SHOULDER_IDX],
    xyzs[REACHY_L_FOREARM_IDX],
    xyzs[REACHY_L_WIRST2HAND_IDX],
    xyzs[REACHY_L_GRIPPER_THUMB_IDX],
    xyzs[REACHY_L_GRIPPER_FINGER_IDX],
    xyzs[REACHY_LEFT_TIP_IDX],
    np.array(
        [xyzs[REACHY_RIGHT_CAMERA_IDX][0] - 0.01, xyzs[REACHY_RIGHT_CAMERA_IDX][1], xyzs[REACHY_RIGHT_CAMERA_IDX][2]]
    ),  # righ camera
    np.array(
        [xyzs[REACHY_LEFT_CAMERA_IDX][0] - 0.01, xyzs[REACHY_LEFT_CAMERA_IDX][1], xyzs[REACHY_LEFT_CAMERA_IDX][2]]
    ),  # left camera
]
# fmt: on

SMPL_NECK_IDX = 12

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

# link index of reachy
# fmt: off
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
# fmt: on
