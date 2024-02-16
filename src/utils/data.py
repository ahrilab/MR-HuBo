import numpy as np
import os.path as osp
import pickle
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from pytorch3d.transforms import matrix_to_rotation_6d
from human_body_prior.tools.rotation_tools import aa2matrot
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from torch.distributions import multivariate_normal
from sklearn.metrics import mean_squared_error as mse
import sys

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


def split_train_test(
    input_path: str,
    reps_path: str,
    target_path: str,
    num_data: int,
    split_ratio: int = 10,
    sample_vposer: bool = True,
    collision_free: bool = False,
    extreme_filter: bool = False,
):
    """
    Args:
    ----------
    input_path (str): path to load SMPL parameters
    reps_path (str): path to load robot xyzs and reps
    target_path (str): path to load robot joint angles
    num_data (int): number of total data
    split_ratio (int): ratio of train/test split
    sample_vposer (bool): whether to sample from VPoser latent space
    collision_free (bool): whether to apply self-collision handling to the data
    extreme_filter (bool): whether to apply extreme filter to the data
    """
    if extreme_filter:
        vp, ps = load_model(
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
    all_smpl_rots = {"train": [], "test": []}
    all_smpl_probs = {"train": [], "test": []}

    print("Loading data...")
    for idx in tqdm(range(num_data)):
        # print(idx, "/", num_data)
        smpl = np.load(osp.join(input_path, "params_{:04}.npz".format(idx)))
        smpl_pose_body = smpl["pose_body"]
        num_poses = len(smpl_pose_body)

        if sample_vposer:
            z = vp.encode(torch.from_numpy(smpl_pose_body).to(DEVICE))
            z_mean = z.mean.detach().cpu()  # 2000 32
            dim_z = z_mean.shape[1]
            dist = multivariate_normal.MultivariateNormal(
                loc=torch.zeros(dim_z), covariance_matrix=torch.eye(dim_z)
            )
            z_prob = torch.exp(dist.log_prob(z_mean))
            z_prob = z_prob / torch.sum(z_prob)

            n_sample = 500
            sample_idx = []
            while len(sample_idx) < n_sample:
                i = draw(z_prob)
                if i not in sample_idx and i > 0:
                    sample_idx.append(i)

        if extreme_filter:
            min_v = 0
            max_v = 0.005
            p_low = 1  # Higher probability for lower values
            p_high = 0  # Lower probability for higher values

            z: torch.Tensor = vp.encode(torch.from_numpy(smpl_pose_body[:]).to(DEVICE))
            z_mean = z.mean

            reconstructed_smpl: torch.Tensor = (
                vp.decode(z_mean)["pose_body"].contiguous().view(-1, 63)
            )
            rec_errors = []

            for i in range(num_poses):  # 2000
                # fmt: off
                original_pose = smpl["pose_body"][i]                # Tensor shaped (63,)
                reconstructed_pose = reconstructed_smpl[i].cpu()    # Tensor shaped (63,)

                rec_error = mse(original_pose, reconstructed_pose)  # float value
                rec_errors.append(rec_error)                        # List of float values
                # fmt: on

            rec_errors = np.array(rec_errors)
            rec_errors = torch.from_numpy(rec_errors)

            probs = 1 - (
                (rec_errors - min_v) / (max_v - min_v) * (p_low - p_high) + p_high
            )
            probs[probs < 0] = 0

            smpl_probs = probs.numpy()

        smpl_aa = smpl_pose_body.reshape(num_poses, -1, 3)
        num_smpl_joints = smpl_aa.shape[1]

        smpl_rot = aa2matrot(
            torch.from_numpy(smpl_aa.reshape(num_poses * num_smpl_joints, 3))
        )
        smpl_rep = matrix_to_rotation_6d(smpl_rot)

        smpl_rot = smpl_rot.reshape(num_poses, num_smpl_joints, 3, 3)
        smpl_rep = smpl_rep.reshape(num_poses, num_smpl_joints, 6)

        smpl_rot = smpl_rot.numpy().reshape(num_poses, -1)
        smpl_rep = smpl_rep.numpy().reshape(num_poses, -1)

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

        if idx < test_num:
            target = "test"
        else:
            target = "train"

        if sample_vposer:
            all_robot_xyzs[target].append(robot_xyzs[sample_idx])
            all_robot_reps[target].append(robot_reps[sample_idx])
            all_robot_angles[target].append(angle_chunk[sample_idx])
            all_smpl_reps[target].append(smpl_rep[sample_idx])
            all_smpl_rots[target].append(smpl_rot[sample_idx])
            if extreme_filter:
                all_smpl_probs[target].append(smpl_probs[sample_idx])
        else:
            all_robot_xyzs[target].append(robot_xyzs)
            all_robot_reps[target].append(robot_reps)
            all_robot_angles[target].append(angle_chunk)
            all_smpl_reps[target].append(smpl_rep)
            all_smpl_rots[target].append(smpl_rot)
            if extreme_filter:
                all_smpl_probs[target].append(smpl_probs)

    for target in ["test", "train"]:
        all_robot_xyzs[target] = np.concatenate(all_robot_xyzs[target], axis=0)
        all_robot_reps[target] = np.concatenate(all_robot_reps[target], axis=0)
        all_robot_angles[target] = np.concatenate(all_robot_angles[target], axis=0)
        all_smpl_reps[target] = np.concatenate(all_smpl_reps[target], axis=0)
        all_smpl_rots[target] = np.concatenate(all_smpl_rots[target], axis=0)
        if extreme_filter:
            all_smpl_probs[target] = np.concatenate(all_smpl_probs[target], axis=0)

    return (
        all_robot_xyzs,
        all_robot_reps,
        all_robot_angles,
        all_smpl_reps,
        all_smpl_rots,
        all_smpl_probs,
    )


class H2RMotionData(Dataset):
    def __init__(
        self,
        robot_xyz,
        robot_rep,
        robot_angle,
        smpl_rep,
        smpl_rot,
        smpl_prob,
        extreme_filter=False,
    ):
        self.robot_xyz = robot_xyz
        self.robot_rep = robot_rep
        self.robot_angle = robot_angle
        self.smpl_rep = smpl_rep
        self.smpl_rot = smpl_rot
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
        sample["smpl_rot"] = self.smpl_rot[idx]
        if self.extreme_filter:
            sample["smpl_prob"] = self.smpl_prob[idx]

        return sample
