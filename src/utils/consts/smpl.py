from enum import Enum

# fmt: off
# Constants for SMPL
VPOSER_PATH         = "./data/vposer_v2_05"
SMPL_PATH           = "./data/bodymodel/smplx/neutral.npz"
NUM_BETAS           = 16
VPOSER_BATCH_SIZE   = 500

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
