from typing import Callable

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
# fmt: on

################################

# Constants for Generating videos
VIDEO_PATH = "./out/vids"
TMP_FRAME_PATH = "./out/tmp"
robot2smpl_vid_path: Callable[[str, int, str], str] = (
    lambda robot_name, index, extention: f"{robot_name}2smpl_{index}.{extention}"
)

GT_PATH = "./data/gt_motions/mr_gt.pkl"

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
