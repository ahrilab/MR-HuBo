import numpy as np
import os.path as osp
import pickle
import torch
import random
import joblib
import sys
from tqdm import tqdm
from torch.utils.data import Dataset
from pytorch3d.transforms import matrix_to_rotation_6d
from human_body_prior.tools.rotation_tools import aa2matrot
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from sklearn.metrics import mean_squared_error as mse

sys.path.append("./src")
from utils.consts import *


def draw(probs):
    val = random.random()
    csum = np.cumsum(probs)
    i = sum(1 - np.array(csum >= val, dtype=int))
    if i == 2000:
        return -1
    else:
        return i


def load_smpl_to_6D_reps(human_pose_path: str, arm_only: bool = False):
    """
    load SMPL parameters from a file and convert it to SMPL joint 6D representations.
    """
    # fmt: off
    if human_pose_path.endswith(".pkl"):
        original_pose: np.ndarray = joblib.load(open(human_pose_path, "rb"))["pose_body"]
    elif human_pose_path.endswith(".npz"):
        original_pose: np.ndarray = np.load(human_pose_path)["pose_body"]
    # fmt: on

    # We only get the last 18 values, which is the axis-angle of the arm joints.
    if arm_only:
        human_pose = original_pose[:, -18:]
    else:
        human_pose = original_pose

    num_poses = len(human_pose)

    # human pose as the axis-angle format
    smpl_axis_angle = human_pose.reshape(num_poses, -1, 3)
    num_joints = smpl_axis_angle.shape[1]
    smpl_axis_angle = smpl_axis_angle.reshape(num_poses * num_joints, 3)

    # convert axis-angle to rotation matrix
    smpl_rot = aa2matrot(torch.from_numpy(smpl_axis_angle))

    # convert rotation matrix to 6D representation
    smpl_rep = matrix_to_rotation_6d(smpl_rot)
    smpl_rep = smpl_rep.reshape(num_poses, num_joints, 6).reshape(num_poses, -1)

    return smpl_rep, original_pose


def load_and_split_train_test(
    input_path: str,
    reps_path: str,
    target_path: str,
    num_data: int,
    split_ratio: int = 10,
    collision_free: bool = False,
    extreme_filter: bool = False,
    arm_only: bool = False,
):
    """
    Load SMPL parameters, robot xyzs, reps, and joint angles, and split them into train and test.

    Args:
    ----------
    input_path (str): path to load SMPL parameters
    reps_path (str): path to load robot xyzs and reps
    target_path (str): path to load robot joint angles
    num_data (int): number of total data
    split_ratio (int): ratio of train/test split
    collision_free (bool): whether to apply self-collision handling to the data
    extreme_filter (bool): whether to apply extreme filter to the data
    """
    if extreme_filter:
        vp, _ = load_model(
            VPOSER_PATH,
            model_code=VPoser,
            remove_words_in_model_weights="vp_model.",
            disable_grad=True,
        )
        vp = vp.to(DEVICE)

    test_num = num_data // split_ratio

    all_robot_xyzs = {"train": [], "test": []}
    all_robot_reps = {"train": [], "test": []}
    all_robot_angles = {"train": [], "test": []}
    all_smpl_reps = {"train": [], "test": []}
    all_smpl_probs = {"train": [], "test": []}

    print("Loading data...")
    for idx in tqdm(range(num_data)):
        # print(idx, "/", num_data)

        # Human data processing
        smpl_rep, smpl_pose = load_smpl_to_6D_reps(
            osp.join(input_path, f"params_{idx:04}.npz"), arm_only
        )
        num_poses = len(smpl_rep)

        if extreme_filter:
            z: torch.Tensor = vp.encode(torch.from_numpy(smpl_pose[:]).to(DEVICE))
            z_mean = z.mean

            reconstructed_smpl: torch.Tensor = (
                vp.decode(z_mean)["pose_body"].contiguous().view(-1, 63)
            )
            rec_errors = []

            for i in range(num_poses):  # 2000
                # fmt: off
                original_pose = smpl_pose[i]                        # Tensor shaped (63,)
                reconstructed_pose = reconstructed_smpl[i].cpu()    # Tensor shaped (63,)

                rec_error = mse(original_pose, reconstructed_pose)  # float value (non-negative value)
                rec_errors.append(rec_error)                        # List of float values
                # fmt: on

            rec_errors = np.array(rec_errors)
            rec_errors = torch.from_numpy(rec_errors)

            # Threshold value for the reconstruction error.
            # If the reconstruction error is greater than this value, it is considered as an extreme value.
            # threshold = 0.005

            # probs = 1 - (rec_errors / threshold)
            # probs[probs < 0] = 0
            # probs[probs > 0] = 1

            p = lambda e: torch.sigmoid((0.003 - e) * 1000) + 0.04
            probs = p(rec_errors)

            smpl_probs = probs.numpy()

        # Robot data processing
        if collision_free:
            angles_file_name = f"cf_angles_{idx:04}.pkl"
            xyzs_reps_file_name = f"cf_xyzs+reps_{idx:04}.npz"
        else:
            angles_file_name = f"angles_{idx:04}.pkl"
            xyzs_reps_file_name = f"xyzs+reps_{idx:04}.npz"

        robot_angle = pickle.load(open(osp.join(target_path, angles_file_name), "rb"))
        angle_chunk = []
        for ra in robot_angle:
            values = []
            for k in sorted(list(ra.keys())):
                values.append(ra[k])
            angle_chunk.append(np.array(values))
        angle_chunk = np.asarray(angle_chunk)

        robot_xyzrep = np.load(osp.join(reps_path, xyzs_reps_file_name))
        robot_xyzs: np.ndarray = robot_xyzrep["xyzs"]
        robot_reps: np.ndarray = robot_xyzrep["reps"]

        robot_xyzs = robot_xyzs.reshape(num_poses, -1)
        robot_reps = robot_reps.reshape(num_poses, -1)

        # Split the data into train and test
        if idx < test_num:
            target = "test"
        else:
            target = "train"

        all_robot_xyzs[target].append(robot_xyzs)
        all_robot_reps[target].append(robot_reps)
        all_robot_angles[target].append(angle_chunk)
        all_smpl_reps[target].append(smpl_rep)
        if extreme_filter:
            all_smpl_probs[target].append(smpl_probs)

    for target in ["test", "train"]:
        all_robot_xyzs[target] = np.concatenate(all_robot_xyzs[target], axis=0)
        all_robot_reps[target] = np.concatenate(all_robot_reps[target], axis=0)
        all_robot_angles[target] = np.concatenate(all_robot_angles[target], axis=0)
        all_smpl_reps[target] = np.concatenate(all_smpl_reps[target], axis=0)
        if extreme_filter:
            all_smpl_probs[target] = np.concatenate(all_smpl_probs[target], axis=0)

    return (
        all_robot_xyzs,
        all_robot_reps,
        all_robot_angles,
        all_smpl_reps,
        all_smpl_probs,
    )


class H2RMotionData(Dataset):
    def __init__(
        self,
        robot_xyz,
        robot_rep,
        robot_angle,
        smpl_rep,
        smpl_prob,
        extreme_filter=False,
    ):
        self.robot_xyz = robot_xyz
        self.robot_rep = robot_rep
        self.robot_angle = robot_angle
        self.smpl_rep = smpl_rep
        self.smpl_prob = smpl_prob
        self.extreme_filter = extreme_filter

    def __len__(self):
        return len(self.smpl_rep)

    def __getitem__(self, idx):
        sample = dict()

        sample["robot_xyz"] = self.robot_xyz[idx]
        sample["robot_rep"] = self.robot_rep[idx]
        sample["robot_angle"] = self.robot_angle[idx]
        sample["smpl_rep"] = self.smpl_rep[idx]
        if self.extreme_filter:
            sample["smpl_prob"] = self.smpl_prob[idx]

        return sample
