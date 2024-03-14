from typing import Callable

# Constants for seeds
NUM_SEEDS = 1000
MOTION_PER_SEED = 2000

# Constants for training
DATA_SPLIT_RATIO = 50
HIDDEN_DIM = 512
BATCH_SIZE = 2048
EF_BATCH_SIZE = 6000
LEARNING_RATE = 1e-4
DEVICE = "cuda"
NUM_EPOCHS = 100
EF_EPOCHS = 300

MODEL_SAVE_EPOCH = 5

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

# Constants for model weights
MODEL_WEIGHTS_DIR: Callable[[str, bool], str] = (
    lambda robot_name, ex: f"./out/models/{robot_name}/final/{"ex" if ex else "no_ex"}"
)
PRE_MODEL_WEIGHT_NAME: Callable[[str, int], str] = (
    lambda robot_name, weight_idx: f"human2{robot_name}_pre_{weight_idx}.pth"
)
PRE_MODEL_BEST_WEIGHT_NAME: Callable[[str, str], str] = (
    lambda robot_name, evaluation_mode: f"human2{robot_name}_pre_best_{evaluation_mode}.pth"
)
POST_MODEL_WEIGHT_NAME: Callable[[str, int], str] = (
    lambda robot_name, weight_idx: f"human2{robot_name}_post_{weight_idx}.pth"
)
POST_MODEL_BEST_WEIGHT_NAME: Callable[[str, str], str] = (
    lambda robot_name, evaluation_mode: f"human2{robot_name}_post_best_{evaluation_mode}.pth"
)

# Constants for evaluation path
PRED_MOTIONS_DIR: Callable[[str, bool], str] = (
    lambda robot_name, extreme_filter:
        f"./out/pred_motions/{robot_name}/final/{'ex' if extreme_filter else 'no_ex'}"
)
EVAL_RESULT_TXT_NAME: Callable[[str], str] = lambda evaluation_mode: f"result_{evaluation_mode}.txt"
PRED_MOTION_NAME: Callable[[str, bool, str], str] = (
    lambda robot_name, extreme_filter, motion_idx:
        f"pred_{robot_name}_{'ex' if extreme_filter else 'no_ex'}_{motion_idx}.pkl"
)

# Constants for rendered Videos
PYBULLET_PRED_VID_DIR: Callable[[str, bool], str] = (
    lambda robot_name, extreme_filter:
        f"./out/pybullet/{robot_name}/final/{'ex' if extreme_filter else 'no_ex'}"
)
PYBULLET_GT_VID_DIR: Callable[[str], str] = lambda robot_name: f"./out/pybullet/{robot_name}/gt"

PYBULLET_PRED_VID_NAME: Callable[[str, bool, str], str] = (
    lambda robot_name, extreme_filter, motion_idx, extention:
        f"{robot_name}_{'ex' if extreme_filter else 'no_ex'}_{motion_idx}.{extention}"
)
PYBULLET_GT_VID_NAME: Callable[[str, str, str], str] = (
    lambda robot_name, motion_idx, extention: f"{robot_name}_gt_{motion_idx}.{extention}"
)


################################

# Constants for Generating videos
VIDEO_PATH = "./out/vids"
TMP_FRAME_PATH = "./out/tmp"
robot2smpl_vid_path: Callable[[str, int, str], str] = (
    lambda robot_name, index, extention: f"{robot_name}2smpl_{index}.{extention}"
)

# Constants for Ground Truth Motions
GT_PATH = "./data/gt_motions/mr_gt.pkl"

GT_MOTION_IDXS = [
    "02_05",  # punch strike
    "13_08",  # unscrew bottlecap, drink soda, screw on bottlecap
    "13_15",  # laugh
    "13_18",  # boxing
    "13_21",  # wash windows
    "13_28",  # direct traffic, wave, point
    "15_08",  # hand signals - horizontally revolve forearms
    "26_02",  # basketball signals
    "54_16",  # superhero
    "55_16",  # panda (human subject)
    "56_02",  # vignettes - fists up, wipe window, yawn, stretch, angrily grab, smash against wall
]

VALID_GT_MOTION_IDXS = [
    "13_28",  # direct traffic, wave, point
    "54_16",  # superhero
]

TEST_GT_MOTION_IDXS = [idx for idx in GT_MOTION_IDXS if idx not in VALID_GT_MOTION_IDXS]

AMASS_DATA_PATH = "./data/gt_motions/amass_data"

# Constants for Evaluation
PRED_MOTION_PATH = "./out/pred_motions"
