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
    xyzs[2], xyzs[5], xyzs[7], xyzs[8],
    xyzs[9], xyzs[10], xyzs[11], xyzs[14],
    xyzs[16], xyzs[17], xyzs[18], xyzs[19],
    np.array([xyzs[27][0] - 0.01, xyzs[27][1], xyzs[27][2]]),  # righ camera
    np.array([xyzs[26][0] - 0.01, xyzs[26][1], xyzs[26][2]]),  # left camera
]
# fmt: on

SMPL_NECK_IDX = 12
