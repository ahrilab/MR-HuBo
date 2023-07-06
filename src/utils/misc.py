"""
Miscellaneous functions and variables.
"""

# fmt: off
reachy_to_smpl_idx = [
    0, 2, 1, 5, 4, 9, 12, 17, 19, 21, 53, 40, 42, 16, 18, 20, 38, 25, 27, 24, 23,
]   # length: 21, smpl의 index: 23, smpl-x Body: 21
# fmt: on

joint_range = {
    "r_shoulder_pitch": [-2.618, 1.57],
    "r_shoulder_roll": [-3.14, 0.174],
    "r_arm_yaw": [-1.57, 1.57],
    "r_elbow_pitch": [-2.182, 0],
    "r_forearm_yaw": [-1.745, 1.745],
    "r_wrist_pitch": [-0.785, 0.785],
    "r_wrist_roll": [-0.785, 0.785],
    "l_shoulder_pitch": [-2.618, 1.57],
    "l_shoulder_roll": [-0.174, 3.14],
    "l_arm_yaw": [-1.57, 1.57],
    "l_elbow_pitch": [-2.182, 0],
    "l_forearm_yaw": [-1.745, 1.745],
    "l_wrist_pitch": [-0.785, 0.785],
    "l_wrist_roll": [-0.785, 0.785],
    "neck_roll": [-0.4, 0.4],
    "neck_pitch": [-0.4, 0.55],
    "neck_yaw": [-1.4, 1.4],
}
"""
Possible range of each joint's roll, pitch, yaw angles. -> maybe?

joint list: [(shoulder, elbow, forearm, wrist for R, L), neck]
"""

reachy_to_smpl_idx = [0, 2, 1, 5, 4, 9, 12, 17, 19, 21, 53, 40, 42, 16, 18, 20, 38, 25, 27, 24, 23]


ret_keys = [
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
]
smplx_jname = [
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]
