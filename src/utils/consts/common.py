from typing import Callable

# Constants for seeds
NUM_SEEDS = 1000
POSE_PER_SEED = 2000

# Constants for training
DATA_SPLIT_RATIO = 50
HIDDEN_DIM = 512
LEARNING_RATE = 1e-4
DEVICE = "cuda"
EF_OFF_BATCH_SIZE = 2048
EF_BATCH_SIZE = 6000
EF_OFF_NUM_EPOCHS = 100
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

# Constants for model weights
MODEL_WEIGHTS_DIR: Callable[[str, bool, bool], str] = (
    lambda robot_name, one_stage, extreme_filter_off:
        f"./out/models/{robot_name}/{'os' if one_stage else 'ts'}/{'no_ex' if extreme_filter_off else 'ex'}"
)
MODEL_WEIGHT_NAME: Callable[[str, str, int], str] = (
    lambda robot_name, model_type, weight_idx: f"human2{robot_name}_{model_type}_{weight_idx}.pth"
)
MODEL_BEST_WEIGHT_NAME: Callable[[str, str, str], str] = (
    lambda robot_name, model_type, evaluation_mode: f"human2{robot_name}_{model_type}_best_{evaluation_mode}.pth"
)

# Constants for evaluation path
PRED_MOTIONS_DIR: Callable[[str, bool, bool], str] = (
    lambda robot_name, one_stage, extreme_filter_off:
        f"./out/pred_motions/{robot_name}/{'os' if one_stage else 'ts'}/{'no_ex' if extreme_filter_off else 'ex'}"
)
EVAL_RESULT_TXT_NAME: Callable[[str], str] = lambda evaluation_mode: f"result_{evaluation_mode}.txt"
PRED_MOTION_NAME: Callable[[str, bool, str], str] = (
    lambda robot_name, extreme_filter_off, motion_idx:
        f"pred_{robot_name}_{'no_ex' if extreme_filter_off else 'ex'}_{motion_idx}.pkl"
)

# Constants for rendered video files
PYBULLET_PRED_VID_DIR: Callable[[str, bool, bool], str] = (
    lambda robot_name, one_stage, extreme_filter_off:
        f"./out/pybullet/{robot_name}/{'os' if one_stage else 'ts'}/{'noex' if extreme_filter_off else 'ex'}"
)
PYBULLET_GT_VID_DIR: Callable[[str], str] = lambda robot_name: f"./out/pybullet/{robot_name}/gt"

PYBULLET_PRED_VID_NAME: Callable[[str, bool, str, str], str] = (
    lambda robot_name, extreme_filter_off, motion_idx, extention:
        f"{robot_name}_{'no_ex' if extreme_filter_off else 'ex'}_{motion_idx}.{extention}"
)
PYBULLET_GT_VID_NAME: Callable[[str, str, str], str] = (
    lambda robot_name, motion_idx, extention: f"{robot_name}_gt_{motion_idx}.{extention}"
)
# fmt: on

################################

# Constants for Ground Truth Motions
GT_PATH = "./data/gt_motions/mr_gt.pkl"
AMASS_DATA_PATH = "./data/gt_motions/amass_data"

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
