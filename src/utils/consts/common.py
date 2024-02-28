from typing import Callable

# Constants for seeds
NUM_SEEDS = 1000
MOTION_PER_SEED = 2000

# Constants for training
HIDDEN_DIM = 512
BATCH_SIZE = 2048
EF_BATCH_SIZE = 10000
LEARNING_RATE = 1e-4
DEVICE = "cuda"
NUM_EPOCHS = 1000
EF_EPOCHS = 5000

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
